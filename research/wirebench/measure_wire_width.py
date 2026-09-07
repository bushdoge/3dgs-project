# レンダ上の電線の「見かけの太さ」を測る（残差分解の後半：フットプリント太りの直接定量）
# 方法：GT細線マスクの各列（電線がほぼ水平なので列断面が線を横切る）について、
#   マスクの連結ランごとに中心yを取り、グレースケール輝度の縦プロファイルから
#   「背景輝度 − 輝度」のディップの FWHM（半値全幅, px）と深さ（コントラスト比）を測る。
#   GT画像とレンダの両方に同じ測定をかけ、太り倍率 = width_render / width_gt を出す。
# 使い方: python3 measure_wire_width.py <実験dir> [iteration] [モデル出力dir名] [--win 14]
# 出力: 標準出力＋ <実験dir>/gt/wire_width_<model_dir>_<iteration>.json

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


def profile_stats(gray, cx, cy, win):
    """列cxの cy±win の縦プロファイルからディップの (FWHM, 深さ, 重心y[サブピクセル]) を返す"""
    h = gray.shape[0]
    y0, y1 = cy - win, cy + win + 1
    if y0 < 0 or y1 > h:
        return None
    prof = gray[y0:y1, cx].astype(np.float64)
    bg = np.median(np.concatenate([prof[:3], prof[-3:]]))       # 窓の両端=背景
    dip = bg - prof
    dmax = dip.max()
    if dmax < 8:                                                 # 線が見えない（8/255未満）
        return None
    above = dip >= dmax / 2
    # 中心付近の連結領域だけをFWHMとして数える
    c = win
    if not above[c]:
        c = int(np.argmax(dip))
    lo = c
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = c
    while hi < len(above) - 1 and above[hi + 1]:
        hi += 1
    seg = dip[lo:hi + 1]
    centroid = y0 + lo + float((seg * np.arange(len(seg))).sum() / seg.sum())
    return float(hi - lo + 1), float(dmax / max(bg, 1e-6)), centroid


def measure_paired(gt_bgr, ren_bgr, mask_bool, win, step=4):
    """同一断面でGTとレンダの両方を測る。戻り値: (gt統計, ren統計, |重心差|のリスト[px])"""
    gray_gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
    gray_ren = cv2.cvtColor(ren_bgr, cv2.COLOR_BGR2GRAY)
    out = {"gt": ([], []), "ren": ([], [])}
    dc = []
    h, w = mask_bool.shape
    for cx in range(0, w, step):
        col = mask_bool[:, cx]
        if not col.any():
            continue
        ys = np.where(col)[0]
        runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)   # 列内の連結ラン=線1本ずつ
        for run in runs:
            if len(run) > win:                                   # 縦に長すぎるラン（棒等）は除外
                continue
            cy = int(run.mean())
            r_gt = profile_stats(gray_gt, cx, cy, win)
            r_ren = profile_stats(gray_ren, cx, cy, win)
            if r_gt is not None:
                out["gt"][0].append(r_gt[0])
                out["gt"][1].append(r_gt[1])
            if r_ren is not None:
                out["ren"][0].append(r_ren[0])
                out["ren"][1].append(r_ren[1])
            if r_gt is not None and r_ren is not None:
                dc.append(abs(r_ren[2] - r_gt[2]))               # 線中心のサブピクセル位置ずれ
    return out, dc


parser = argparse.ArgumentParser()
parser.add_argument("exp")
parser.add_argument("iteration", nargs="?", default="7000")
parser.add_argument("model_dir", nargs="?", default="output")
parser.add_argument("--win", type=int, default=14, help="プロファイル半窓[px]（既定14）")
args = parser.parse_args()

exp = Path(args.exp)
rdir = exp / args.model_dir / "test" / f"ours_{args.iteration}"
names = test_view_names(exp)
agg = {"gt": ([], []), "ren": ([], [])}
center_diffs = []
for i, name in enumerate(names):
    ren = cv2.imread(str(rdir / "renders" / f"{i:05d}.png"))
    gtimg = cv2.imread(str(rdir / "gt" / f"{i:05d}.png"))
    mask = cv2.imread(str(exp / "gt" / "mask_thin" / f"{name.split('_')[1]}.png"), 0)
    if ren is None or gtimg is None or mask is None:
        continue
    if mask.shape != gtimg.shape[:2]:
        mask = cv2.resize(mask, (gtimg.shape[1], gtimg.shape[0]), interpolation=cv2.INTER_NEAREST)
    out, dc = measure_paired(gtimg, ren, mask > 127, args.win)
    for key in ["gt", "ren"]:
        agg[key][0].extend(out[key][0])
        agg[key][1].extend(out[key][1])
    center_diffs.extend(dc)

res = {"model_dir": args.model_dir, "iteration": int(args.iteration), "win_px": args.win}
for key, label in [("gt", "GT"), ("ren", "render")]:
    ws, ds = np.array(agg[key][0]), np.array(agg[key][1])
    res[f"n_{key}"] = len(ws)
    res[f"width_{key}_q25_50_75"] = [round(float(np.percentile(ws, p)), 2) for p in (25, 50, 75)] if len(ws) else None
    res[f"depth_{key}_q50"] = round(float(np.median(ds)), 3) if len(ds) else None
    print(f"{label:7s}: 測定断面 {len(ws):5d}  FWHM[q25/50/75]={res[f'width_{key}_q25_50_75']}px  ディップ深さq50={res[f'depth_{key}_q50']}")
if res["width_gt_q25_50_75"] and res["width_ren_q25_50_75"]:
    ratio = res["width_ren_q25_50_75"][1] / max(res["width_gt_q25_50_75"][1], 1e-6)
    res["fatten_ratio_q50"] = round(ratio, 3)
    print(f"→ 太り倍率（FWHM中央値比 render/GT）: {ratio:.2f}")
if center_diffs:
    d = np.array(center_diffs)
    res["n_paired"] = len(d)
    res["center_offset_px_q25_50_75"] = [round(float(np.percentile(d, p)), 3) for p in (25, 50, 75)]
    res["center_offset_frac_gt_0.5px"] = round(float((d > 0.5).mean()), 3)
    res["center_offset_frac_gt_1px"] = round(float((d > 1.0).mean()), 3)
    print(f"→ 線中心の位置ずれ |Δ| [q25/50/75] = {res['center_offset_px_q25_50_75']} px"
          f"（>0.5pxが{res['center_offset_frac_gt_0.5px']*100:.0f}%・>1pxが{res['center_offset_frac_gt_1px']*100:.0f}%・対応断面{len(d)}）")
out = fresh(exp / "gt" / f"wire_width_{args.model_dir}_{args.iteration}.json")
json.dump(res, open(out, "w"), indent=1)
print(f"保存: {out}")
