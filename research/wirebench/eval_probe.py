# 「帯 vs 半平面」検定シーン（make_probe_scene.py）の評価。物体ごと・テスト視点ごとに
# 「境界の単位長さあたりの二乗誤差を境界の本数で割ったもの」（境界1本あたりの誤差密度）を測る。
# 設計: memo/design_20260922_band_vs_halfplane.md §4。研究メモ §9as。
#
# 測り方（1視点・1物体）:
#   - 物体マスク（>0.5）の境界画素の ±2px を測定領域にする。細い帯は2本の境界の領域が重なり、帯全体が1つの領域になる
#   - 除外：他の物体のマスク（>0.05）から4px以内、画像端4px、円柱の端（キャップ）と板の角の近傍
#   - 境界の長さは幾何から出す：円柱は軸を600点に分けて投影し、シルエットの左右の境界それぞれについて測れる区間の
#     画面上の長さを足す（片側だけ他物体に隠れる場合も正しく数える）。板は4辺を各300点に分けて投影し、残った区間の長さの和
#   - 誤差密度 e = 領域内の二乗誤差の和（RGB平均・[0,1]） / 境界の総延長 bnd_len_px  [単位: 二乗誤差・px / px]
#   - 見かけ幅 = 残った軸サンプルでの 2R·f/深度 の平均
# 比較基準として、GT をガウシアンでぼかしただけの擬似モデル（--blur-ref σ...）も同じ指標で測る。
# 線形なぼかしでも細い帯では両側の境界の残差が帯の内側で同符号に重なり e が増えるので、
# 「帯特有の難しさ」はモデルの e がぼかし基準の e をどれだけ超えるかで判定する（collect_probe.py）。
#
# 使い方:
#   python3 eval_probe.py <exp> <iter> <model_dir名> [...]        # 学習済みモデル（test/ours_<iter>/renders）
#   python3 eval_probe.py <exp> --blur-ref 0.5 1.0 1.5 2.0          # ぼかし基準（学習不要。テスト視点のGTから作る）
#   --split train を付けると学習視点（train/ours_<iter>/renders、render.py を --skip_test で実行しておく）で測る。
#     テスト視点では空色のフローターが物体を覆う視点があり（§9as）、学習視点＝表現できるかだけを測る切り分け用
# 出力: <exp>/gt/probe_eval/<model>_<iter>.csv（ぼかし基準は blur_s<σ>.csv。--split train は probe_eval_train/）。1行＝1視点×1物体
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from testviews import test_view_names   # noqa: E402

BAND = 2          # 境界の周り ±BAND px
EXCL_OTHER = 4    # 他物体からの除外距離 px
MARGIN = 4        # 画像端の除外 px
N_AX = 600        # 円柱の軸のサンプル数
N_EDGE = 300      # 板の1辺のサンプル数


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def project(P, M, f, W, H):
    """world 点群 → 画素座標と深度（Blender/OpenGL の cam2world M）"""
    pc = (M[:3, :3].T @ (P - M[:3, 3]).T).T
    dep = -pc[:, 2]
    u = f * pc[:, 0] / np.maximum(dep, 1e-6) + W / 2
    v = H / 2 - f * pc[:, 1] / np.maximum(dep, 1e-6)
    return np.stack([u, v], 1), dep


def seg_samples(a, b, n):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def eval_view(gt, pred, frame_no, exp, objs, poses, f):
    """1視点の全物体の行を返す"""
    H, W = gt.shape[:2]
    M = np.array(poses["frames"][f"frame_{frame_no:04d}"])
    masks = {}
    for o in objs:
        m = cv2.imread(str(exp / "gt" / "mask_obj" / f"obj{o['id']:02d}" / f"{frame_no:04d}.png"), cv2.IMREAD_GRAYSCALE)
        masks[o["id"]] = m.astype(np.float32) / 255.0
    err = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(axis=2)   # [0,1]^2
    inner = np.zeros((H, W), np.uint8)
    inner[MARGIN:H - MARGIN, MARGIN:W - MARGIN] = 1
    rows = []
    for o in objs:
        m = (masks[o["id"]] > 0.5).astype(np.uint8)
        if m.sum() < 10:
            continue
        others = np.zeros((H, W), np.uint8)
        for oid, mm in masks.items():
            if oid != o["id"]:
                others |= (mm > 0.05).astype(np.uint8)
        excl = cv2.dilate(others, disk(EXCL_OTHER))
        bnd = m - cv2.erode(m, disk(1))
        region = cv2.dilate(bnd, disk(BAND)) & inner & (1 - excl)

        if o["kind"] == "cylinder":
            P = seg_samples(o["p0"], o["p1"], N_AX)
            segs = [np.arange(N_AX)]
            n_bnd = 2
            uv, dep = project(P, M, f, W, H)
            r_px = o["radius_m"] * f / np.maximum(dep, 1e-6)
            # キャップ近傍の除外（端点から 見かけ半径×1.5＋6px）
            for idx in (0, N_AX - 1):
                if dep[idx] > 0:
                    cv2.circle(region, (int(round(uv[idx, 0])), int(round(uv[idx, 1]))),
                               int(np.ceil(r_px[idx] * 1.5 + 6)), 0, -1)
        else:
            C = o["corners"]
            P = np.concatenate([seg_samples(C[i], C[(i + 1) % 4], N_EDGE) for i in range(4)])
            segs = [np.arange(i * N_EDGE, (i + 1) * N_EDGE) for i in range(4)]
            n_bnd = 1
            uv, dep = project(P, M, f, W, H)
            r_px = np.zeros(len(P))
            for i in range(4):
                c = uv[i * N_EDGE]
                if dep[i * N_EDGE] > 0:
                    cv2.circle(region, (int(round(c[0])), int(round(c[1]))), 8, 0, -1)
            # 板が真横に近く、向かい合う辺の間隔が狭い視点は「半平面」でないので捨てる
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            if dt.max() < 2 * BAND + 4:
                continue
        ui = lambda q: np.clip(np.round(q).astype(int), 0, [W - 1, H - 1])
        ys, xs = np.nonzero(region)
        if len(ys) == 0:
            continue
        pix = np.stack([xs, ys], 1).astype(np.float64)
        _, nn = cKDTree(uv).query(pix)
        if o["kind"] == "cylinder":
            # シルエットの左右の境界を別々に扱う：軸サンプルから画面上の法線方向に見かけ半径だけ離れた点が
            # 除外域に入っていなければその側の境界は「測れる」。片側だけ他物体に隠れる場合も正しく数える
            t = np.gradient(uv, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            nrm = np.stack([-t[:, 1], t[:, 0]], 1)
            ok_ax = dep > 0
            for idx in (0, N_AX - 1):
                ok_ax &= np.linalg.norm(uv - uv[idx], axis=1) > r_px[idx] * 1.5 + 6
            avail = inner & (1 - excl)
            keep_side = {}
            for sgn in (1, -1):
                q = ui(uv + sgn * r_px[:, None] * nrm)
                keep_side[sgn] = ok_ax & (avail[q[:, 1], q[:, 0]] > 0)
            seglen = np.linalg.norm(np.diff(uv, axis=0), axis=1)
            L = sum(float((seglen * (k[1:] & k[:-1])).sum()) for k in keep_side.values())
            side = np.where(((pix - uv[nn]) * nrm[nn]).sum(1) >= 0, 1, -1)
            sel = np.where(side > 0, keep_side[1][nn], keep_side[-1][nn])
            denom = L                                   # 左右の境界の長さの和（＝本数×長さ）
            keep = keep_side[1] | keep_side[-1]
        else:
            q = ui(uv)
            near_m = cv2.dilate(m, disk(2))
            keep = (dep > 0) & (region[q[:, 1], q[:, 0]] > 0) & (near_m[q[:, 1], q[:, 0]] > 0)
            L = 0.0
            for sg in segs:
                k = keep[sg]
                d = np.linalg.norm(np.diff(uv[sg], axis=0), axis=1)
                L += float((d * (k[1:] & k[:-1])).sum())
            sel = keep[nn]
            denom = n_bnd * L
        if denom < 20 or sel.sum() < 10:
            continue
        ys, xs = ys[sel], xs[sel]
        sse = float(err[ys, xs].sum())
        mse = sse / len(ys)
        rows.append({"frame": frame_no, "obj": o["id"], "kind": o["kind"],
                     "radius_mm": round(o.get("radius_m", 0) * 1000, 2), "tilt_deg": o.get("tilt_deg"),
                     "width_px": round(float((2 * r_px[keep]).mean()), 3) if o["kind"] == "cylinder" else None,
                     "bnd_len_px": round(denom, 1), "n_boundaries": n_bnd, "n_px": int(len(ys)),
                     "sse": sse, "e_per_len": sse / denom,
                     "psnr_region": 10 * np.log10(1.0 / max(mse, 1e-12))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("iteration", nargs="?", type=int)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--blur-ref", type=float, nargs="*", help="GT をガウシアン σ[px] でぼかした擬似モデルを評価")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    objs = json.load(open(exp / "gt" / "probe_objects.json"))
    poses = json.load(open(exp / "gt" / "poses.json"))
    f = poses["fx"]
    names = test_view_names(exp)
    if a.split == "train":
        # 学習側の並び（COLMAP 登録名の名前順）からテスト視点を除いたもの。render.py の train/ も同じ順
        tv = set(names)
        names = [n for n in test_view_names(exp, hold=1) if n not in tv]
    out_dir = exp / "gt" / ("probe_eval" if a.split == "test" else "probe_eval_train")
    out_dir.mkdir(exist_ok=True)

    jobs = []
    if a.blur_ref:
        for s in a.blur_ref:
            jobs.append((f"blur_s{s:g}", None, s))
    for m in a.models:
        jobs.append((f"{m}_{a.iteration}", exp / m / a.split / f"ours_{a.iteration}", None))
    for tag, tdir, sigma in jobs:
        rows = []
        if tdir is not None:
            rd = tdir / "renders"
            preds = sorted(rd.glob("*.png"))
            if len(preds) != len(names):
                sys.exit(f"{tag}: レンダ {len(preds)} 枚とテスト視点 {len(names)} 枚が合わない")
        for i, name in enumerate(names):
            frame_no = int(name.split("_")[-1])
            gt = cv2.imread(str(exp / "input" / f"{name}.png"))[:, :, ::-1] / 255.0
            if sigma is not None:
                pred = cv2.GaussianBlur(gt, (0, 0), sigma)
            else:
                pred = cv2.imread(str(preds[i]))[:, :, ::-1] / 255.0
                if pred.shape != gt.shape:
                    sys.exit(f"{tag}: 画像サイズが GT と違う {pred.shape} vs {gt.shape}")
            for r in eval_view(gt, pred, frame_no, exp, objs, poses, f):
                rows.append({"model": tag, "view": name, **r})
        p = out_dir / f"{tag}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        n_obj = len({r["obj"] for r in rows})
        print(f"{tag}: {len(rows)} 行（{len(names)} 視点・{n_obj} 物体）→ {p}")


if __name__ == "__main__":
    main()
