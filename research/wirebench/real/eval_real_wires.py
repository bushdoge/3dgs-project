# 実データ用の細線評価。GTマスクが無いので、GT画像から「青空を背景にした細い暗線」を
# 検出して細線マスクとし、テスト視点で 細線PSNR / 背景PSNR / 落差 / 線コントラスト保持率 を測る。
# 合成の eval_wires.py の実データ代用版（GT深度・poses・wire_pointsを使わない）。
import cv2, numpy as np
from pathlib import Path

EXP = Path("/workspace/experiments/20260823_wire_railside_01")
SETS = {"3dgs": (EXP/"output/test/ours_7000/gt", EXP/"output/test/ours_7000/renders"),
        "mip":  (EXP/"output_mip/test/ours_7000/gt_-1", EXP/"output_mip/test/ours_7000/test_preds_-1")}

def wire_mask(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sky = ((b - r > 18) & (v > 150)).astype(np.uint8)
    frac = cv2.blur(sky.astype(np.float32), (21, 21))
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    w = ((bh > 25) & (frac > 0.85) & (sky == 0)).astype(np.uint8)
    return cv2.morphologyEx(w, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), sky, frac

def psnr(a, b, m=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if m is not None:
        d = d[m > 0]
        if d.size == 0: return np.nan
    mse = d.mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))

for tag, (gtd, rend) in SETS.items():
    gts = sorted(gtd.glob("*.png")); res = sorted(rend.glob("*.png"))
    rows = []
    for gp, rp in zip(gts, res):
        gt = cv2.imread(str(gp)); re_ = cv2.imread(str(rp))
        if gt.shape != re_.shape:
            re_ = cv2.resize(re_, (gt.shape[1], gt.shape[0]))
        wm, sky, frac = wire_mask(gt)
        if wm.sum() < 200: continue
        wmd = cv2.dilate(wm, np.ones((3, 3), np.uint8))
        bgm = ((frac > 0.85) & (wmd == 0)).astype(np.uint8)   # 空背景側の非線画素＝背景
        gg = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float64)
        rg = cv2.cvtColor(re_, cv2.COLOR_BGR2GRAY).astype(np.float64)
        base = np.percentile(gg[sky > 0], 50) if sky.sum() > 0 else 255
        c_gt = (base - gg[wm > 0]).mean() / base * 100
        c_re = (base - rg[wm > 0]).mean() / base * 100
        rows.append((psnr(gt, re_), psnr(gt, re_, wmd), psnr(gt, re_, bgm), c_gt, c_re))
    a = np.array(rows)
    print(f"[{tag}]  テスト{len(rows)}枚")
    print(f"   全体PSNR {a[:,0].mean():6.2f}   細線PSNR {a[:,1].mean():6.2f}   背景(空側)PSNR {a[:,2].mean():6.2f}"
          f"   落差 {a[:,2].mean()-a[:,1].mean():5.2f}dB")
    print(f"   線コントラスト GT {a[:,3].mean():5.1f}% → レンダ {a[:,4].mean():5.1f}%"
          f"  （保持率 {a[:,4].mean()/a[:,3].mean()*100:.0f}%）")
    print(f"   細線PSNR最悪視点 {a[:,1].min():.2f}")
