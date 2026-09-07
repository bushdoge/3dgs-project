# 残差の誤差収支分析：細線マスク内の二乗誤差を「輝度/色」「位置ずれ/形状」「裾集中度」に分解する
# （§9i追記2の最終段。残差10dBが何にいくら払われているかの収支を出す）
#   1. 輝度 vs 色: YCrCb空間でマスク内二乗誤差をY成分とCr/Cb成分に分ける
#   2. 位置 vs 形状: 同一断面ペアの輝度プロファイルで、測定済みの重心差ぶんレンダ側を
#      サブピクセルシフトして二乗誤差がどれだけ減るか（減った分＝位置ずれの寄与）
#   3. 裾集中度: 断面ごとの二乗誤差を降順に並べ、上位5%/10%断面が全体の何%を占めるか
# 使い方: python3 error_budget.py <実験dir> [iteration] [モデル出力dir名] [--win 14]
# 出力: 標準出力＋ <実験dir>/gt/error_budget_<model_dir>_<iteration>.json

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent)); from testviews import test_view_names


def fresh(p):
    p = Path(p)
    p.unlink(missing_ok=True)
    return p


def dip_centroid(prof, win):
    """プロファイルのディップ重心（measure_wire_width.pyと同じ定義）。測れなければNone"""
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

E_y = E_c = 0.0                     # 1. 輝度/色
sec_orig, sec_shift = [], []        # 2. 断面ごとの (元の二乗誤差, シフト後の二乗誤差)
for i, name in enumerate(names):
    ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
    gt = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
    mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{name.split('_')[1]}.png"), 0)
    if ren is None or gt is None or mask is None:
        continue
    if mask.shape != gt.shape[:2]:
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    mb = mask > 127

    ycc_r = cv2.cvtColor(ren, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    ycc_g = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    d = (ycc_r - ycc_g)[mb]
    E_y += float((d[:, 0] ** 2).sum())
    E_c += float((d[:, 1:] ** 2).sum())

    gray_r, gray_g = ycc_r[..., 0], ycc_g[..., 0]
    h, w = mb.shape
    win = args.win
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
            cg, cr_ = dip_centroid(pg, win), dip_centroid(pr, win)
            e0 = float(((pr - pg) ** 2).sum())
            if cg is None or cr_ is None:
                sec_orig.append(e0)
                sec_shift.append(e0)      # 位置補正不能な断面はシフト効果なし扱い
                continue
            x = np.arange(len(pr), dtype=np.float64)
            pr_shift = np.interp(x, x - (cr_ - cg), pr)   # レンダをGT重心位置へ寄せる
            e1 = float(((pr_shift - pg) ** 2).sum())
            sec_orig.append(e0)
            sec_shift.append(min(e0, e1))  # シフトで悪化する断面は位置寄与ゼロと数える

sec_orig, sec_shift = np.array(sec_orig), np.array(sec_shift)
tot = sec_orig.sum()
pos_share = float((sec_orig - sec_shift).sum() / tot) if tot else float("nan")
order = np.sort(sec_orig)[::-1]
top5 = float(order[:max(1, len(order) // 20)].sum() / tot)
top10 = float(order[:max(1, len(order) // 10)].sum() / tot)

res = {"model_dir": args.model_dir, "iteration": int(args.iteration), "win_px": args.win,
       "luma_share": round(E_y / (E_y + E_c), 3), "chroma_share": round(E_c / (E_y + E_c), 3),
       "n_sections": len(sec_orig),
       "position_share_of_luma_section_err": round(pos_share, 3),
       "top5pct_sections_err_share": round(top5, 3),
       "top10pct_sections_err_share": round(top10, 3)}
print(f"[1] マスク内二乗誤差の内訳: 輝度 {res['luma_share']*100:.0f}% / 色 {res['chroma_share']*100:.0f}%")
print(f"[2] 断面輝度誤差のうち位置ずれ補正で消える分: {pos_share*100:.0f}%（対応断面 {len(sec_orig)}）")
print(f"[3] 裾集中度: 上位5%断面が全体の {top5*100:.0f}% / 上位10%が {top10*100:.0f}% を占める")
out = fresh(exp / "gt" / f"error_budget_{args.model_dir}_{args.iteration}.json")
json.dump(res, open(out, "w"), indent=1)
print(f"保存: {out}")
