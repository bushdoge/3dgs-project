# 裾の空間分布診断：誤差上位断面（tail）が「どの視点」「線上のどこ」に集中するかを可視化する
# （§9jで残差の約75%が上位5〜10%断面に集中と判明。その裾が視点固定か・線分上で連続か・
#   散発かはPhase 2のロス設計（視点適応 vs 画素適応 vs 幾何補正）を分ける決定的情報）
#   1. 視点別: 各テスト視点の誤差総量シェアと上位5%断面の個数分布
#   2. 連続性: 上位5%断面の隣接クラスタリング率（ランダム散布の期待値と比較）
#   3. 可視化: 各テスト視点のGT画像に断面誤差をヒートマップ重畳（上位5%は強調）
# 使い方: python3 tail_map.py <実験dir> [iteration] [モデル出力dir名] [--win 14]
# 出力: 標準出力＋ <実験dir>/gt/tail_map_<model_dir>_<iteration>.json / .png

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fresh(p):
    p = Path(p)
    p.unlink(missing_ok=True)
    return p


def dip_centroid(prof, win):
    """プロファイルのディップ重心（error_budget.pyと同じ定義）。測れなければNone"""
    bg = np.median(np.concatenate([prof[:3], prof[-3:]]))
    dip = bg - prof
    dmax = dip.max()
    if dmax < 8:
        return None
    above = dip >= dmax / 2
    c = win if above[win] else int(np.argmax(dip))
    lo = c
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = c
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    seg = dip[lo:hi + 1]
    return lo + float((seg * np.arange(len(seg))).sum() / seg.sum())


parser = argparse.ArgumentParser()
parser.add_argument("exp")
parser.add_argument("iteration", nargs="?", default="7000")
parser.add_argument("model_dir", nargs="?", default="output")
parser.add_argument("--win", type=int, default=14)
args = parser.parse_args()

exp = Path(args.exp)
rdir = exp / args.model_dir / "test" / f"ours_{args.iteration}"
names = test_view_names(exp)
win = args.win

# 断面ごとの (視点idx, cx, cy, 二乗誤差, 位置補正後誤差) を収集
secs = []          # dict のリスト
gts = {}           # 可視化用にGT画像を保持
for i, name in enumerate(names):
    ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
    gt = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
    mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{name.split('_')[1]}.png"), 0)
    if ren is None or gt is None or mask is None:
        continue
    if mask.shape != gt.shape[:2]:
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    mb = mask > 127
    gts[i] = gt
    gray_r = cv2.cvtColor(ren, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
    gray_g = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
    h, w = mb.shape
    for cx in range(0, w, 4):
        col = mb[:, cx]
        if not col.any():
            continue
        ys = np.where(col)[0]
        for run in np.split(ys, np.where(np.diff(ys) > 1)[0] + 1):
            if len(run) > win:
                continue
            cy = int(run.mean())
            y0, y1 = cy - win, cy + win + 1
            if y0 < 0 or y1 > h:
                continue
            pg, pr = gray_g[y0:y1, cx], gray_r[y0:y1, cx]
            e0 = float(((pr - pg) ** 2).sum())
            cg, cr_ = dip_centroid(pg, win), dip_centroid(pr, win)
            if cg is None or cr_ is None:
                e1 = e0
            else:
                x = np.arange(len(pr), dtype=np.float64)
                pr_shift = np.interp(x, x - (cr_ - cg), pr)
                e1 = min(e0, float(((pr_shift - pg) ** 2).sum()))
            secs.append({"view": i, "cx": cx, "cy": cy, "err": e0, "err_shifted": e1})

err = np.array([s["err"] for s in secs])
view = np.array([s["view"] for s in secs])
cxs = np.array([s["cx"] for s in secs])
cys = np.array([s["cy"] for s in secs])
tot = err.sum()
n_top = max(1, len(secs) // 20)
top_idx = np.argsort(err)[::-1][:n_top]
is_top = np.zeros(len(secs), dtype=bool)
is_top[top_idx] = True

# 1. 視点別の集中度
views = sorted(gts.keys())
per_view = []
for v in views:
    m = view == v
    per_view.append({"view": int(v), "name": names[v], "n_sections": int(m.sum()),
                     "err_share": round(float(err[m].sum() / tot), 3),
                     "n_top5pct": int(is_top[m].sum()),
                     "top5pct_share_of_view_err":
                         round(float(err[m & is_top].sum() / max(err[m].sum(), 1e-9)), 3)})

# 2. 連続性: 上位断面の隣接率（同一視点・cx差4px以内・cy差3px以内を隣接と定義）
#    ランダム散布なら隣接率 ≈ 5% 前後になるはず。大幅に高ければ「裾は線分単位で連続」
def adjacency_rate(flag):
    hit = 0
    cand = np.where(flag)[0]
    for i in cand:
        m = (view == view[i]) & (np.abs(cxs - cxs[i]) == 4) & (np.abs(cys - cys[i]) <= 3)
        if flag[m].any():
            hit += 1
    return hit / max(len(cand), 1)

adj_obs = adjacency_rate(is_top)
rng = np.random.default_rng(0)
adj_rand = float(np.mean([adjacency_rate(rng.permutation(is_top)) for _ in range(20)]))

# 上位断面の連続ラン長（同一視点で cx が4刻みに連続する上位断面のかたまり）
runs = []
for v in views:
    idx = np.where((view == v) & is_top)[0]
    if len(idx) == 0:
        continue
    order = idx[np.argsort(cxs[idx])]
    cur = 1
    for a, b in zip(order[:-1], order[1:]):
        if cxs[b] - cxs[a] == 4 and abs(int(cys[b]) - int(cys[a])) <= 3:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
runs = np.array(runs) if runs else np.array([1])

# 位置ずれ補正の寄与（上位断面 vs 全体。裾の中身が位置ずれか形状かの切り分け）
shift_gain_all = float((err - np.array([s["err_shifted"] for s in secs])).sum() / tot)
top_err = err[top_idx]
top_shift = np.array([secs[i]["err_shifted"] for i in top_idx])
shift_gain_top = float((top_err - top_shift).sum() / top_err.sum())

res = {"model_dir": args.model_dir, "iteration": int(args.iteration), "win_px": win,
       "n_sections": len(secs), "n_top5pct": int(n_top),
       "per_view": per_view,
       "adjacency_rate_top5pct": round(adj_obs, 3),
       "adjacency_rate_random": round(adj_rand, 3),
       "top_run_length_median": float(np.median(runs)),
       "top_run_length_max": int(runs.max()),
       "position_shift_gain_all": round(shift_gain_all, 3),
       "position_shift_gain_top5pct": round(shift_gain_top, 3)}

print(f"[1] 視点別誤差シェア（{len(views)}視点）:")
for pv in sorted(per_view, key=lambda x: -x["err_share"]):
    print(f"    view {pv['view']:2d} ({pv['name']}): 誤差シェア {pv['err_share']*100:4.1f}% "
          f"/ 上位5%断面 {pv['n_top5pct']:3d}個 / 視点内で上位断面が占める誤差 {pv['top5pct_share_of_view_err']*100:.0f}%")
print(f"[2] 上位5%断面の隣接率: 観測 {adj_obs*100:.0f}% vs ランダム期待 {adj_rand*100:.0f}% "
      f"/ 連続ラン長 中央値 {np.median(runs):.0f}・最大 {runs.max()}")
print(f"[3] 位置ずれ補正で消える誤差: 全体 {shift_gain_all*100:.0f}% / 上位5%断面内 {shift_gain_top*100:.0f}%")

# 3. 可視化: GTに断面誤差を重畳（対数スケール色、上位5%は白丸で強調）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent)); from testviews import test_view_names

ncol = min(5, len(views))
nrow = (len(views) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
axes = np.atleast_1d(axes).ravel()
le = np.log10(err + 1)
vmin, vmax = np.percentile(le, [5, 99.5])
for ax, v in zip(axes, views):
    m = view == v
    ax.imshow(cv2.cvtColor(gts[v], cv2.COLOR_BGR2RGB))
    sc = ax.scatter(cxs[m], cys[m], c=le[m], s=3, cmap="inferno", vmin=vmin, vmax=vmax)
    mt = m & is_top
    ax.scatter(cxs[mt], cys[mt], facecolors="none", edgecolors="white", s=28, linewidths=0.7)
    pv = next(p for p in per_view if p["view"] == v)
    ax.set_title(f"view {v} — err share {pv['err_share']*100:.0f}%", fontsize=9)
    ax.axis("off")
for ax in axes[len(views):]:
    ax.axis("off")
fig.suptitle(f"{exp.name} / {args.model_dir} @ {args.iteration} — wire section error map "
             f"(white circles = top 5%)", fontsize=11)
fig.tight_layout()
png = fresh(exp / "gt" / f"tail_map_{args.model_dir}_{args.iteration}.png")
fig.savefig(png, dpi=110)
out = fresh(exp / "gt" / f"tail_map_{args.model_dir}_{args.iteration}.json")
json.dump(res, open(out, "w"), indent=1)
print(f"保存: {out} / {png}")
