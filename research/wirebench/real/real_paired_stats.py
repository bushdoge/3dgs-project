# 実データの手法差の統計検定（監査 memo/review_20260924.md R-01・修正計画B5）。
# 空背景／構造背景の電線マスク（wire_struct_masks.build。基準モデルの GT から1回だけ作る）で、
# 各モデル群（seed 違いの run を glob でまとめる）のテスト視点ごとの電線PSNRを出し、基準群との差を
#   (1) seed レベル：run ごとの平均PSNR を群間で Welch t 検定（seed 分散を含む本来の検定。各群 n≥2 のとき）
#   (2) 視点レベル：seed 平均した視点ごとの値の対差を t 検定・Wilcoxon（seed 分散を含まないので甘い参考値）
# で報告する。n=1 の群は (1) を出さない。
# --compare A-B で基準群以外どうしの対比（例 lodgs-mip30k）も同じ2つの検定で出す（2026-10-05 追加。§9ar-11 の
# LOD-GS−Mip はそれまでインライン計算のみだった）。seed Welch p には、その領域で出した全比較（基準群対比＋--compare）を
# 1つの族として Holm 補正した値も併記する。
# 使い方:
#   python3 research/wirebench/real/real_paired_stats.py                       # railside の既定群
#   python3 research/wirebench/real/real_paired_stats.py --exp 20260903_wire_street_01 \
#       --group 3dgs30k=output_3dgs30k_r* --group mip30k=output_mip30k_r* [--compare mip30k-3dgs30k ...]
# 出力: 標準出力＋ <実験dir>/real_stats.json
import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wire_struct_masks import build   # noqa: E402

E = Path("/workspace/experiments")
DEFAULT_GROUPS = ["3dgs30k=output_30k,output_30k_r*", "mip30k=output_mip30k,output_mip30k_r*",
                  "mip_np10=output_mip30k_np10,output_mip30k_np10_r*",
                  "mip_nprel024=output_mip30k_nprel024_r*", "lodgs=output_lodgs_30k,output_lodgs_30k_r*",
                  "mip_np05=output_mip30k_np05", "mip_np03=output_mip30k_np03", "mip_ks005=output_mip30k_ks005"]
DEFAULT_COMPARES = ["lodgs-mip30k", "lodgs-mip_np10"]
MIN_PX = 300


def holm(ps):
    """Holm の逐次補正（ステップダウン、単調化あり）。NaN はそのまま返す"""
    idx = [i for i, p in enumerate(ps) if not np.isnan(p)]
    order = sorted(idx, key=lambda i: ps[i])
    out, run = list(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - k) * ps[i]))
        out[i] = run
    return out


def compare(seedmean, viewmean, a, b):
    """群 a − 群 b の seed Welch と視点対検定"""
    sa, sb = seedmean[a], seedmean[b]
    wp = stats.ttest_ind(sa, sb, equal_var=False).pvalue if len(sa) > 1 and len(sb) > 1 else np.nan
    dv = viewmean[a] - viewmean[b]
    tp = stats.ttest_1samp(dv, 0).pvalue
    wx = stats.wilcoxon(dv).pvalue if len(dv) > 5 and np.any(dv != 0) else np.nan
    return sa.mean() - sb.mean(), wp, dv, tp, wx


def dirs_of(model_dir, it):
    base = model_dir / "test" / f"ours_{it}"
    for g, r in (("gt", "renders"), ("gt_-1", "test_preds_-1")):
        if (base / g).is_dir() and (base / r).is_dir() and any((base / r).glob("*.png")):
            return base / g, base / r
    return None


def psnr(a, b, m):
    d = ((a.astype(np.float64) - b.astype(np.float64)) ** 2)[m > 0]
    return np.nan if d.size == 0 else 10 * np.log10(255.0 ** 2 / max(d.mean(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="20260823_wire_railside_01")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--group", action="append", help="名前=glob[,glob...]（最初の群が基準）")
    ap.add_argument("--compare", action="append", help="基準群以外の対比 A-B（群名。--group 指定時の既定は無し）")
    args = ap.parse_args()
    exp = E / args.exp
    groups = {}
    for spec in (args.group or DEFAULT_GROUPS):
        name, pats = spec.split("=", 1)
        runs = []
        for p in pats.split(","):
            runs += [Path(x) for x in sorted(glob.glob(str(exp / p)))]
        runs = [r for r in dict.fromkeys(runs) if dirs_of(r, args.iteration)]
        if runs:
            groups[name] = runs
    base_name = next(iter(groups))
    compares = args.compare if args.compare is not None else ([] if args.group else DEFAULT_COMPARES)

    # マスクは基準群の最初の run の GT から1回だけ作る（GT は全モデルで画素一致することを監査で確認済み）
    gtd0, _ = dirs_of(groups[base_name][0], args.iteration)
    masks = {}
    for p in sorted(gtd0.glob("*.png")):
        d = build(cv2.imread(str(p)))
        masks[p.name] = {r: cv2.dilate(d[r], np.ones((3, 3), np.uint8)) for r in ("open_v", "struct_v")
                         if int(d[r].sum()) >= MIN_PX}
    print(f"実験 {args.exp} / iter {args.iteration} / 基準群 {base_name}")
    for reg in ("open_v", "struct_v"):
        print(f"  {reg}: 有効視点 {sum(reg in m for m in masks.values())}/{len(masks)}")

    # 各 run の視点ごとの PSNR
    per_run = {}
    for g, runs in groups.items():
        for r in runs:
            gtd, rd = dirs_of(r, args.iteration)
            vals = {"open_v": {}, "struct_v": {}}
            for gp, rp in zip(sorted(gtd.glob("*.png")), sorted(rd.glob("*.png"))):
                G, Rr = cv2.imread(str(gp)), cv2.imread(str(rp))
                if G.shape != Rr.shape:
                    Rr = cv2.resize(Rr, (G.shape[1], G.shape[0]))
                for reg, m in masks.get(gp.name, {}).items():
                    vals[reg][gp.name] = psnr(G, Rr, m)
            per_run[(g, r.name)] = vals

    out = {"exp": args.exp, "iteration": args.iteration, "base": base_name,
           "groups": {g: [r.name for r in rs] for g, rs in groups.items()}, "regions": {}}
    for reg, jp in (("open_v", "空背景"), ("struct_v", "構造背景")):
        views = sorted(set.intersection(*[set(v[reg]) for v in per_run.values()])) if per_run else []
        if not views:
            continue
        print(f"\n== {jp}（{reg}・{len(views)}視点）==")
        print(f"{'群':12s} {'n':>2s} {'平均':>7s} {'run間sd':>7s} | {'差':>6s} {'Welch p':>6s} {'Holm':>6s} | {'視点t p':>6s} {'Wilcoxon p':>10s} {'正の視点':>7s}")
        seedmean = {g: np.array([np.mean([per_run[(g, r.name)][reg][v] for v in views]) for r in rs])
                    for g, rs in groups.items()}
        viewmean = {g: np.array([np.mean([per_run[(g, r.name)][reg][v] for r in rs]) for v in views])
                    for g, rs in groups.items()}
        res = {}
        for g in groups:
            sm = seedmean[g]
            sd = sm.std(ddof=1) if len(sm) > 1 else np.nan
            if g == base_name:
                d, wp, dv, tp, wx = 0.0, np.nan, np.zeros(len(views)), np.nan, np.nan
            else:
                d, wp, dv, tp, wx = compare(seedmean, viewmean, g, base_name)
            res[g] = {"n_seeds": len(sm), "seed_means": sm.round(3).tolist(), "mean": float(sm.mean()), "sd": sd,
                      "diff_vs_base": float(d), "welch_p_seed": wp, "view_paired_t_p": tp, "view_wilcoxon_p": wx,
                      "n_pos_views": int((dv > 0).sum())}
        cmps = {}
        for spec in compares:
            a, b = spec.split("-", 1)
            if a in groups and b in groups:
                d, wp, dv, tp, wx = compare(seedmean, viewmean, a, b)
                cmps[spec] = {"diff": float(d), "welch_p_seed": wp, "view_paired_t_p": tp, "view_wilcoxon_p": wx,
                              "n_pos_views": int((dv > 0).sum())}
        # Holm：この領域で出した seed Welch の全比較を1つの族とする
        fam = [(res, g) for g in res if g != base_name] + [(cmps, k) for k in cmps]
        for (dct, k), ph in zip(fam, holm([dct[k]["welch_p_seed"] for dct, k in fam])):
            dct[k]["welch_p_seed_holm"] = ph
        for g, x in res.items():
            print(f"{g:12s} {x['n_seeds']:2d} {x['mean']:7.2f} {x['sd']:7.2f} | {x['diff_vs_base']:+6.2f} "
                  f"{x['welch_p_seed']:6.3f} {x.get('welch_p_seed_holm', np.nan):6.3f} | "
                  f"{x['view_paired_t_p']:6.3f} {x['view_wilcoxon_p']:10.3f} {x['n_pos_views']:3d}/{len(views)}")
        for k, x in cmps.items():
            print(f"  {k:22s} {x['diff']:+6.2f} {x['welch_p_seed']:6.3f} {x['welch_p_seed_holm']:6.3f} | "
                  f"{x['view_paired_t_p']:6.3f} {x['view_wilcoxon_p']:10.3f} {x['n_pos_views']:3d}/{len(views)}")
        nan2none = lambda dct: {k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in x.items()}
                                for k, x in dct.items()}
        res, cmps = nan2none(res), nan2none(cmps)
        out["regions"][reg] = {"n_views": len(views), "groups": res, "compares": cmps}
    json.dump(out, open(exp / "real_stats.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n保存: {exp / 'real_stats.json'}")


if __name__ == "__main__":
    main()
