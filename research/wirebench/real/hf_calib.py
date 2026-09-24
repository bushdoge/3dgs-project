# 高周波テクスチャ保持率（レンダ/GT）の定義の較正。
# memo §1c/§9af の「実データでMipが高周波を 59.5%→31.1% に半減」「合成 3DGS30k 84.7% / Mip30k 93.0% /
# LOD-GS 90.5%」は算出スクリプトが残っていなかったため、候補定義を並べて既存値を再現できるものを探す。
# 使い方: python3 research/wirebench/real/hf_calib.py <実験dir> [<実験dir> ...]
import sys, cv2, numpy as np
from pathlib import Path

def pairs(out):
    """output*/test/ours_*/ から (tag, gtdir, renderdir) を列挙（3DGS系: gt/renders、Mip: gt_-1/test_preds_-1）"""
    for od in sorted(out.glob("output*/test/ours_*")):
        tag = f"{od.parent.parent.name}@{od.name.split('_')[1]}"
        for g, r in (("gt", "renders"), ("gt_-1", "test_preds_-1")):
            if (od/g).is_dir() and (od/r).is_dir():
                yield tag, od/g, od/r
                break

def metrics(g, r):
    out = {}
    lg, lr = cv2.Laplacian(g, cv2.CV_64F), cv2.Laplacian(r, cv2.CV_64F)
    out["lap_E"] = (lr**2).mean() / (lg**2).mean()
    out["lap_abs"] = np.abs(lr).mean() / np.abs(lg).mean()
    for s in (1.0, 2.0):
        hg, hr = g - cv2.GaussianBlur(g, (0, 0), s), r - cv2.GaussianBlur(r, (0, 0), s)
        out[f"hp{s:g}_E"] = (hr**2).mean() / (hg**2).mean()
    # 半径方向の周波数でナイキストの 1/4 以上のパワー
    Fg, Fr = np.abs(np.fft.fft2(g))**2, np.abs(np.fft.fft2(r))**2
    fy, fx = np.fft.fftfreq(g.shape[0])[:, None], np.fft.fftfreq(g.shape[1])[None, :]
    rad = np.sqrt(fx**2 + fy**2) / 0.5
    for c in (0.25, 0.5):
        m = rad >= c
        out[f"fft{c:g}"] = Fr[m].sum() / Fg[m].sum()
    return out

for exp in map(Path, sys.argv[1:]):
    print(f"== {exp.name}")
    for tag, gd, rd in pairs(exp):
        acc = []
        for gp, rp in zip(sorted(gd.glob("*.png")), sorted(rd.glob("*.png"))):
            g = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE).astype(np.float64)
            r = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).astype(np.float64)
            if g.shape != r.shape:
                r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
            acc.append(metrics(g, r))
        if not acc:
            continue
        keys = acc[0].keys()
        print(f"  {tag:32s} n={len(acc):3d} " + " ".join(f"{k}={np.mean([a[k] for a in acc])*100:5.1f}%" for k in keys))
