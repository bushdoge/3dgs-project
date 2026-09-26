# 実データの手法差の統計検定（監査 memo/review_20260924.md R-01・修正計画B5）。
# 空背景／構造背景の電線マスク（wire_struct_masks.build。基準モデルの GT から1回だけ作る）で、
# 各モデル群（seed 違いの run を glob でまとめる）のテスト視点ごとの電線PSNRを出し、基準群との差を
#   (1) seed レベル：run ごとの平均PSNR を群間で Welch t 検定（seed 分散を含む本来の検定。各群 n≥2 のとき）
#   (2) 視点レベル：seed 平均した視点ごとの値の対差を t 検定・Wilcoxon（seed 分散を含まないので甘い参考値）
# で報告する。n=1 の群は (1) を出さない。
# 使い方:
#   python3 research/wirebench/real/real_paired_stats.py                       # railside の既定群
#   python3 research/wirebench/real/real_paired_stats.py --exp 20260903_wire_street_01 \
#       --group 3dgs30k=output_3dgs30k_r* --group mip30k=output_mip30k_r*
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
MIN_PX = 300


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
        print(f"{'群':12s} {'n':>2s} {'平均':>7s} {'run間sd':>7s} | {'差':>6s} {'seed Welch p':>12s} | {'視点対差':>7s} {'t p':>6s} {'Wilcoxon p':>10s} {'正の視点':>7s}")
        seedmean = {g: np.array([np.mean([per_run[(g, r.name)][reg][v] for v in views]) for r in rs])
                    for g, rs in groups.items()}
        viewmean = {g: np.array([np.mean([per_run[(g, r.name)][reg][v] for r in rs]) for v in views])
                    for g, rs in groups.items()}
        res = {}
        for g in groups:
            sm, b = seedmean[g], seedmean[base_name]
            d = sm.mean() - b.mean()
            wp = stats.ttest_ind(sm, b, equal_var=False).pvalue if g != base_name and len(sm) > 1 and len(b) > 1 else np.nan
            dv = viewmean[g] - viewmean[base_name]
            tp = stats.ttest_1samp(dv, 0).pvalue if g != base_name else np.nan
            wx = stats.wilcoxon(dv).pvalue if g != base_name and len(dv) > 5 and np.any(dv != 0) else np.nan
            sd = sm.std(ddof=1) if len(sm) > 1 else np.nan
            print(f"{g:12s} {len(sm):2d} {sm.mean():7.2f} {sd:7.2f} | {d:+6.2f} {wp:12.3f} | {dv.mean():+7.2f} {tp:6.3f} {wx:10.3f} {int((dv > 0).sum()):3d}/{len(dv)}")
            res[g] = {"n_seeds": len(sm), "seed_means": sm.round(3).tolist(), "mean": float(sm.mean()),
                      "diff_vs_base": float(d), "welch_p_seed": None if np.isnan(wp) else float(wp),
                      "view_paired_t_p": None if np.isnan(tp) else float(tp),
                      "view_wilcoxon_p": None if np.isnan(wx) else float(wx)}
        out["regions"][reg] = {"n_views": len(views), "groups": res}
    json.dump(out, open(exp / "real_stats.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n保存: {exp / 'real_stats.json'}")


if __name__ == "__main__":
    main()
