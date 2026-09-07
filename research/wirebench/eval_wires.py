# 3DGSレンダリング結果の細線領域を評価するスクリプト
# テスト視点ごとに 全体PSNR / 細線領域PSNR / 背景PSNR を計り、
# 細線が消えている様子の比較図（GT | render | 差分）を書き出す。
# さらに「消失率」メトリクス（線検出recall）を計算する：
#   GT細線マスク M に対し、画像 I（GT画像 or レンダ）を Canny→HoughLinesP で線検出し、
#   バンド dilate(M, band_radius) 内に限定した検出マップ DET(I) を作る。
#   マスク画素が「被覆」される ⇔ DET(I) から距離 tau px 以内（distanceTransformで判定）。
#   Recall(I) = 被覆画素数 / マスク画素総数。R_gt=Recall(GT画像) は検出器自体の妥当性、
#   R_ren=Recall(レンダ) が本命。正規化recall DR = R_ren / max(R_gt, 1e-6)、消失率 = 1 - DR。
#   検出は常に「画像」に対して行い、マスクは正解位置としてのみ使う（マスク自体には検出をかけない）。
#
# 使い方: python3 eval_wires.py <実験dir> [iteration] [モデル出力dir名] [オプション]
#   例: python3 eval_wires.py . 7000            → <実験dir>/output/test/ours_7000 を評価
#       python3 eval_wires.py . 30000 output_30k → <実験dir>/output_30k/test/ours_30000 を評価
#   オプション（消失率メトリクス用。全て既定値ありで省略可、既定値のまま従来どおり動く）:
#     --tau PX             被覆判定の距離しきい値[px]（既定2）
#     --band-radius PX     検出をマスク周辺のこのバンド内に限定[px]（既定2）
#     --canny-lo/--canny-hi  Cannyの低/高しきい値
#     --hough-thresh        HoughLinesPの投票数しきい値
#     --min-len             HoughLinesPの最小線分長[px]
#     --max-gap             HoughLinesPの最大ギャップ[px]（断線をこの距離まで繋ぐ）

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fresh(p):
    # 実験dirを cp -al で複製すると gt/ 配下の成果物が元dirとinode共有になり、
    # 上書き保存が元実験の結果を巻き添えで壊す。書く前にunlinkしてリンクを切る。
    p = Path(p)
    p.unlink(missing_ok=True)
    return p


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp", help="実験ディレクトリ")
    p.add_argument("iteration", nargs="?", default="7000", help="学習iteration（既定7000）")
    p.add_argument("model_dir", nargs="?", default="output", help="モデル出力dir名（既定output）")
    p.add_argument("--tau", type=float, default=2.0,
                   help="被覆判定の距離しきい値[px]（既定2）")
    p.add_argument("--band-radius", type=int, default=2,
                   help="検出をマスク周辺のこのバンド内に限定[px]（既定2）")
    p.add_argument("--mask-dilate", type=int, default=2,
                   help="PSNR用細線マスクの膨張半径[px]（既定2=従来の5×5。0で膨張なし。"
                        "マスク希釈の寄与分析用。消失率側のband-radiusとは独立）")
    p.add_argument("--canny-lo", type=float, default=10.0)
    p.add_argument("--canny-hi", type=float, default=30.0)
    p.add_argument("--hough-thresh", type=int, default=8)
    p.add_argument("--min-len", type=float, default=8.0)
    p.add_argument("--max-gap", type=float, default=25.0)
    return p


def psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return np.nan
        mse = d[mask].mean()
    else:
        mse = d.mean()
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf


def detect_line_map(img_bgr, band_mask, canny_lo, canny_hi, hough_thresh, min_len, max_gap):
    """画像から Canny→HoughLinesP で線分を検出し、太さ1pxでラスタ化。band_mask内に限定する。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                             minLineLength=min_len, maxLineGap=max_gap)
    det = np.zeros(gray.shape, np.uint8)
    if lines is not None:
        # OpenCV 4系は (N,1,4)、5系は (N,4) を返す。どちらでも動くよう正規化する
        # （2026-09-04: コンテナのcv2が5.0.0になり、旧来の lines[:,0] が壊れたため）
        for l in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(v) for v in l)
            cv2.line(det, (x1, y1), (x2, y2), 255, 1)
    return cv2.bitwise_and(det, det, mask=band_mask)


def covered_fraction(det_map, mask_bool, tau):
    """mask_bool画素のうちdet_mapからtau px以内にある画素の割合。マスクが空ならnan。"""
    total = int(mask_bool.sum())
    if total == 0:
        return np.nan, 0, 0
    not_det = (det_map == 0).astype(np.uint8)
    dist = cv2.distanceTransform(not_det, cv2.DIST_L2, 5)
    covered = (dist <= tau) & mask_bool
    n_covered = int(covered.sum())
    return n_covered / total, n_covered, total


def load_wire_bar_splitter(exp):
    """細線マスクを『電線』と『フェンス棒』に分ける関数を返す。GTが無ければ None。

    細線マスク（pass_index=1）には電線とフェンス縦棒の両方が入る。棒の半径は固定で
    電線だけが --wire_radius で変わるため、太さスイープでマスクの構成比が大きく変わり、
    「太いほど悪い」という交絡を生んでいた（2026-09-06 の監査で発覚・§9ab）。
    GTの電線中心線点群を各視点へ投影し、マスク画素を最近傍ラベルで分類する。
    """
    try:
        import json as _json
        from scipy.spatial import cKDTree
        poses = _json.load(open(exp / "gt" / "poses.json"))
        d = np.load(exp / "gt" / "wire_points.npz")
        P, LB = d["points"], d["labels"]
    except Exception:
        return None

    def split(name, mask_bool):
        if name not in poses.get("frames", {}):
            return None
        C = np.array(poses["frames"][name]); W2C = np.linalg.inv(C)
        Xc = (W2C[:3, :3] @ P.T + W2C[:3, 3:4]).T
        zc = -Xc[:, 2]
        u = poses["cx"] + poses["fx"] * Xc[:, 0] / np.maximum(zc, 1e-6)
        v = poses["cy"] - poses["fy"] * Xc[:, 1] / np.maximum(zc, 1e-6)
        H, W = mask_bool.shape
        ok = (zc > 0.1) & (u > -50) & (u < W + 50) & (v > -50) & (v < H + 50)
        ys, xs = np.where(mask_bool)
        if ok.sum() < 2 or ys.size == 0:
            return None
        _, idx = cKDTree(np.stack([u[ok], v[ok]], 1)).query(np.stack([xs, ys], 1))
        cls = LB[ok][idx]
        wire = np.zeros_like(mask_bool); wire[ys[cls == 1], xs[cls == 1]] = True
        bar = np.zeros_like(mask_bool); bar[ys[cls == 2], xs[cls == 2]] = True
        return wire, bar
    return split


def band_from_mask(mask_bool, band_radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_radius + 1, 2 * band_radius + 1))
    return cv2.dilate(mask_bool.astype(np.uint8) * 255, kernel)


def main():
    args = build_argparser().parse_args()
    exp = Path(args.exp)
    it = args.iteration
    model_dir = args.model_dir
    rdir = exp / model_dir / "test" / f"ours_{it}"
    out_dir = exp / "gt" / f"wire_eval_{model_dir}_{it}" if model_dir != "output" else exp / "gt" / f"wire_eval_{it}"
    if args.mask_dilate != 2:
        # 非標準の膨張半径では標準評価(summary.json等)を上書きしない（マスク希釈分析用の別dir）
        out_dir = out_dir.with_name(out_dir.name + f"_md{args.mask_dilate}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # llffhold=8: テスト視点は名前順の 0,8,16,... 番目。
    # **ただし 3DGS が並べるのは COLMAP に登録された画像**であって input/ の全ファイルではない。
    # 登録に失敗した画像があると input/ 由来の並びとずれ、**別フレームのマスクで評価してしまう**。
    # （2026-09-06 のコード監査で発覚。street f40 は5視点中3視点、crossing f40 は2視点がずれていた。）
    # よって sparse/0/images.bin の登録名から算出し、無い場合だけ input/ にフォールバックする。
    names = None
    img_bin = exp / "sparse" / "0" / "images.bin"
    if img_bin.exists():
        try:
            import sys as _sys
            _sys.path.insert(0, "/opt/gaussian-splatting")
            from scene.colmap_loader import read_extrinsics_binary
            names = sorted(Path(v.name).stem for v in read_extrinsics_binary(str(img_bin)).values())
        except Exception as e:
            print(f"[警告] images.bin を読めなかったので input/ から算出します: {e}")
    if not names:
        names = sorted(p.stem for p in (exp / "input").glob("*.png"))
    test_names = names[::8]

    splitter = load_wire_bar_splitter(exp)
    rows = []              # PSNR系（従来どおり）
    split_rows = []        # 電線のみ / フェンス棒（2026-09-06 追加）
    recall_rows = []       # 消失率系（新規）: name, r_gt, r_ren, n_covered_gt, n_covered_ren, n_total
    k = args.mask_dilate
    kernel5 = np.ones((2 * k + 1, 2 * k + 1), np.uint8) if k > 0 else None
    for i, name in enumerate(test_names):
        ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
        gt = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
        frame_no = name.split("_")[1]                       # frame_0001 → 0001
        mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{frame_no}.png"), 0)
        if ren is None or gt is None or mask is None:
            continue
        if mask.shape != gt.shape[:2]:
            mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        mb = mask > 127
        m = cv2.dilate(mb.astype(np.uint8), kernel5).astype(bool) if kernel5 is not None else mb
        rows.append((name, psnr(ren, gt), psnr(ren, gt, np.repeat(m[..., None], 3, 2)),
                     psnr(ren, gt, np.repeat(~m[..., None], 3, 2)), m.mean() * 100))
        if splitter is not None:
            sp = splitter(name, m)
            if sp is not None:
                wmask, bmask = sp
                split_rows.append((
                    psnr(ren, gt, np.repeat(wmask[..., None], 3, 2)) if wmask.any() else np.nan,
                    psnr(ren, gt, np.repeat(bmask[..., None], 3, 2)) if bmask.any() else np.nan,
                    int(wmask.sum()), int(bmask.sum())))

        band = band_from_mask(mb, args.band_radius)
        det_gt = detect_line_map(gt, band, args.canny_lo, args.canny_hi,
                                  args.hough_thresh, args.min_len, args.max_gap)
        det_ren = detect_line_map(ren, band, args.canny_lo, args.canny_hi,
                                   args.hough_thresh, args.min_len, args.max_gap)
        r_gt, cov_gt, tot = covered_fraction(det_gt, mb, args.tau)
        r_ren, cov_ren, _ = covered_fraction(det_ren, mb, args.tau)
        recall_rows.append((name, r_gt, r_ren, cov_gt, cov_ren, tot))

    print(f"{'view':12s} {'全体PSNR':>9s} {'細線PSNR':>9s} {'背景PSNR':>9s} {'細線px%':>8s}")
    for n, p_all, p_wire, p_bg, frac in rows:
        print(f"{n:12s} {p_all:9.2f} {p_wire:9.2f} {p_bg:9.2f} {frac:8.2f}")
    arr = np.array([[r[1], r[2], r[3]] for r in rows])
    print(f"{'平均':12s} {arr[:,0].mean():9.2f} {arr[:,1].mean():9.2f} {arr[:,2].mean():9.2f}")
    print(f"\n→ 背景PSNRと細線PSNRの差: {arr[:,2].mean()-arr[:,1].mean():.2f} dB（大きいほど細線だけ壊れている）")

    # ── 電線 / フェンス棒の分離（本命指標は「電線のみ」。§9ab）──
    wire_only = bar_only = gap_wire = None
    if split_rows:
        sa = np.array([[r[0], r[1]] for r in split_rows], dtype=float)
        wpx = sum(r[2] for r in split_rows); bpx = sum(r[3] for r in split_rows)
        wire_only = float(np.nanmean(sa[:, 0])); bar_only = float(np.nanmean(sa[:, 1]))
        gap_wire = float(arr[:, 2].mean() - wire_only)
        print(f"\n★ 電線のみPSNR: {wire_only:.2f}　フェンス棒PSNR: {bar_only:.2f}"
              f"（マスク中の電線比 {wpx/max(wpx+bpx,1)*100:.1f}%）")
        print(f"★ 背景と電線のみの差: {gap_wire:.2f} dB ← **本命指標**"
              f"（従来の細線PSNRは電線と棒の混合なので過大評価になる）")

    # ── 消失率（線検出recall）の集計。視点ごとの画素数で重み付けした全体recallを本命値とする ──
    tot_sum = sum(r[5] for r in recall_rows)
    cov_gt_sum = sum(r[3] for r in recall_rows)
    cov_ren_sum = sum(r[4] for r in recall_rows)
    recall_gt = cov_gt_sum / tot_sum if tot_sum else float("nan")
    recall_ren = cov_ren_sum / tot_sum if tot_sum else float("nan")
    recall_norm = recall_ren / max(recall_gt, 1e-6)
    disappearance_rate = 1 - recall_norm
    ren_vals = [r[2] for r in recall_rows if not np.isnan(r[2])]
    recall_ren_worst = min(ren_vals) if ren_vals else float("nan")

    print(f"\n{'view':12s} {'R_gt':>8s} {'R_ren':>8s}")
    for n, r_gt_i, r_ren_i, *_ in recall_rows:
        print(f"{n:12s} {r_gt_i:8.3f} {r_ren_i:8.3f}")
    print(f"{'合計':12s} {recall_gt:8.3f} {recall_ren:8.3f}")
    print(f"\n→ 正規化recall DR = R_ren/R_gt = {recall_norm:.3f}　消失率 = 1-DR = {disappearance_rate:.3f}"
          f"　(tau={args.tau}px, band={args.band_radius}px)")

    # 集計用JSON（run_sweep.pyが読む。既存キーは変えず追記のみ）
    json.dump({"iteration": int(it), "model_dir": model_dir,
               "psnr_all": float(arr[:, 0].mean()), "psnr_wire": float(arr[:, 1].mean()),
               "psnr_bg": float(arr[:, 2].mean()),
               "gap_db": float(arr[:, 2].mean() - arr[:, 1].mean()),
               "psnr_wireonly": wire_only, "psnr_bar": bar_only, "gap_wireonly_db": gap_wire,
               "psnr_wire_min": float(arr[:, 1].min()), "n_test_views": len(rows),
               "recall_gt": float(recall_gt), "recall_ren": float(recall_ren),
               "recall_norm": float(recall_norm), "disappearance_rate": float(disappearance_rate),
               "recall_ren_worst": float(recall_ren_worst),
               "tau_px": args.tau, "band_px": args.band_radius,
               "mask_dilate_px": args.mask_dilate,
               "hough_params": {"canny_lo": args.canny_lo, "canny_hi": args.canny_hi,
                                 "hough_thresh": args.hough_thresh, "min_len": args.min_len,
                                 "max_gap": args.max_gap}},
              open(fresh(out_dir / "summary.json"), "w"), indent=1)

    # ── 比較図: 細線が最も壊れている視点(PSNR基準)で GT|render|差分 と拡大クロップ ──────
    worst = int(np.argmin([r[2] for r in rows]))
    name = rows[worst][0]
    i = worst
    ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
    gt = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
    frame_no = name.split("_")[1]
    mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{frame_no}.png"), 0)
    if mask.shape != gt.shape[:2]:
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    full = np.hstack([gt, ren])
    cv2.putText(full, "GT", (10, 40), 0, 1.2, (0, 0, 255), 3)
    cv2.putText(full, "3DGS render", (gt.shape[1] + 10, 40), 0, 1.2, (0, 0, 255), 3)
    cv2.imwrite(str(fresh(out_dir / f"compare_{name}.png")), full)

    # 細線マスクの重心まわりを2倍拡大でクロップ（上位2クラスタ）
    ys, xs = np.where(mask > 127)
    if len(ys) > 0:
        for ci, q in enumerate([0.3, 0.7]):
            cy, cx = int(np.quantile(ys, q)), int(np.quantile(xs, q))
            h, w = 140, 240
            y0, x0 = max(cy - h // 2, 0), max(cx - w // 2, 0)
            gtc = gt[y0:y0 + h, x0:x0 + w]
            rnc = ren[y0:y0 + h, x0:x0 + w]
            z = 3
            crop = np.hstack([cv2.resize(gtc, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST),
                              np.full((h * z, 8, 3), 255, np.uint8),
                              cv2.resize(rnc, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)])
            cv2.imwrite(str(fresh(out_dir / f"crop{ci}_{name}.png")), crop)
    print(f"\n比較図を保存: {out_dir}/（最悪視点 {name}）")

    # ── 消失率の診断図: recallが最も低い(最も消えている)視点で GT線=緑/レンダ検出=赤/被覆=黄 ──
    valid_idx = [idx for idx, r in enumerate(recall_rows) if not np.isnan(r[2])]
    if valid_idx:
        worst_r = min(valid_idx, key=lambda idx: recall_rows[idx][2])
        rname = recall_rows[worst_r][0]
        i = worst_r
        ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
        frame_no = rname.split("_")[1]
        mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{frame_no}.png"), 0)
        if mask.shape != ren.shape[:2]:
            mask = cv2.resize(mask, (ren.shape[1], ren.shape[0]), interpolation=cv2.INTER_NEAREST)
        mb = mask > 127
        band = band_from_mask(mb, args.band_radius)
        det_ren = detect_line_map(ren, band, args.canny_lo, args.canny_hi,
                                   args.hough_thresh, args.min_len, args.max_gap)
        not_det = (det_ren == 0).astype(np.uint8)
        dist = cv2.distanceTransform(not_det, cv2.DIST_L2, 5)
        covered = (dist <= args.tau) & mb

        vis = ren.copy()
        vis[mb] = (0, 255, 0)          # 緑: GT細線マスク（正解位置）
        vis[det_ren > 0] = (0, 0, 255)  # 赤: レンダ画像からの検出線
        vis[covered] = (0, 255, 255)    # 黄: 被覆（正解位置かつ検出でtau px以内）
        cv2.putText(vis, f"detect overlay (worst R_ren={recall_rows[worst_r][2]:.2f})",
                    (10, 30), 0, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(fresh(out_dir / f"detect_overlay_{rname}.png")), vis)
        print(f"消失率診断図を保存: {out_dir}/detect_overlay_{rname}.png（最悪視点 {rname}, R_ren={recall_rows[worst_r][2]:.3f}）")


if __name__ == "__main__":
    main()
