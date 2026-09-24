# 実データ第1波の細線評価（7k/30k 横断版）。research/wirebench/real/eval_real_wires.py と同じマスク・同じ指標で、
# 対象を 7k・30k の全モデルに広げただけのもの（§9t-2の数値と直接比較できるよう検出パラメータは不変）。
# GTマスクが無いので GT画像から「青空を背景にした細い暗線」を検出して細線マスク代用とする。
import cv2, numpy as np
from pathlib import Path

EXP = Path("/workspace/experiments/20260823_wire_railside_01")
# tag: (GT画像dir, レンダdir)  ※素3DGSは gt/renders、Mipは gt_-1/test_preds_-1
SETS = {
    "3dgs 7k":        (EXP/"output/test/ours_7000/gt",            EXP/"output/test/ours_7000/renders"),
    "mip(ks0.1) 7k":  (EXP/"output_mip/test/ours_7000/gt_-1",     EXP/"output_mip/test/ours_7000/test_preds_-1"),
    "3dgs 30k":       (EXP/"output_30k/test/ours_30000/gt",       EXP/"output_30k/test/ours_30000/renders"),
    "mip(ks0.1) 30k": (EXP/"output_mip30k/test/ours_30000/gt_-1", EXP/"output_mip30k/test/ours_30000/test_preds_-1"),
    "mip(ks0.05) 30k":(EXP/"output_mip30k_ks005/test/ours_30000/gt_-1",
                       EXP/"output_mip30k_ks005/test/ours_30000/test_preds_-1"),
}

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
    return 10 * np.log10(255.0 ** 2 / max(d.mean(), 1e-9))

print(f"{'model':16s} {'n':>3s} {'全体':>7s} {'細線':>7s} {'背景':>7s} {'落差':>7s} {'保持率':>7s} {'細線最悪':>8s}")
for tag, (gtd, rend) in SETS.items():
    if not (gtd.is_dir() and rend.is_dir()):
        print(f"{tag:16s}  -- 未作成（レンダ待ち）")
        continue
    gts, res = sorted(gtd.glob("*.png")), sorted(rend.glob("*.png"))
    rows = []
    for gp, rp in zip(gts, res):
        gt, re_ = cv2.imread(str(gp)), cv2.imread(str(rp))
        if gt.shape != re_.shape:
            re_ = cv2.resize(re_, (gt.shape[1], gt.shape[0]))
        wm, sky, frac = wire_mask(gt)
        if wm.sum() < 200: continue
        wmd = cv2.dilate(wm, np.ones((3, 3), np.uint8))
        bgm = ((frac > 0.85) & (wmd == 0)).astype(np.uint8)
        gg = cv2.cvtColor(gt,  cv2.COLOR_BGR2GRAY).astype(np.float64)
        rg = cv2.cvtColor(re_, cv2.COLOR_BGR2GRAY).astype(np.float64)
        base = np.percentile(gg[sky > 0], 50) if sky.sum() > 0 else 255
        rows.append((psnr(gt, re_), psnr(gt, re_, wmd), psnr(gt, re_, bgm),
                     (base - gg[wm > 0]).mean() / base * 100,
                     (base - rg[wm > 0]).mean() / base * 100))
    a = np.array(rows)
    print(f"{tag:16s} {len(rows):3d} {a[:,0].mean():7.2f} {a[:,1].mean():7.2f} {a[:,2].mean():7.2f} "
          f"{a[:,2].mean()-a[:,1].mean():7.2f} {a[:,4].mean()/a[:,3].mean()*100:6.0f}% {a[:,1].min():8.2f}")
