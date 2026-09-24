# 実データ「構造背景の細線」マスク。
# 空背景で検出した電線画素にRANSACで2次曲線を当て（電柱付近で線が交差するため連結成分では分離できない）、
# 建物・樹木側へ外挿して構造背景区間のマスクを作る。
# （空でない領域を素朴に細線検出すると窓枠・レール・電柱を拾うため。§9t の失敗の回避）
import cv2, numpy as np, sys
from pathlib import Path

def sky_and_wire(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sky = ((b - r > 18) & (v > 150)).astype(np.uint8)
    frac = cv2.blur(sky.astype(np.float32), (21, 21))
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    w = ((bh > 25) & (frac > 0.85) & (sky == 0)).astype(np.uint8)
    w = cv2.morphologyEx(w, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return w, sky, frac

def ransac_wires(wmask, W, n_iter=3000, tol=2.0, min_inl=300, min_span=350, max_lines=20, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.where(wmask > 0)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    fits = []
    while len(fits) < max_lines and len(pts) >= min_inl:
        best = None
        for _ in range(n_iter):
            idx = rng.choice(len(pts), 3, replace=False)
            sx, sy = pts[idx, 0], pts[idx, 1]
            if sx.max() - sx.min() < min_span:            # x方向に広くサンプル
                continue
            try:
                c = np.polyfit(sx, sy, 2)
            except Exception:
                continue
            d = np.abs(np.polyval(c, pts[:, 0]) - pts[:, 1])
            inl = d < tol
            k = int(inl.sum())
            if best is None or k > best[0]:
                best = (k, c, inl)
        if best is None or best[0] < min_inl:
            break
        k, c, inl = best
        px, py = pts[inl, 0], pts[inl, 1]
        if px.max() - px.min() < min_span:
            pts = pts[~inl]
            continue
        c = np.polyfit(px, py, 2)                          # 全インライアで再フィット
        res = float(np.median(np.abs(np.polyval(c, px) - py)))
        fits.append(dict(c=c, xa=int(px.min()), xb=int(px.max()), n=k, res=res))
        pts = pts[~inl]
    return fits

def masks_from_fits(fits, shape, frac, sky_thr=0.85, half=1):
    """戻り: (空背景マスク, 構造背景マスク) いずれも線の太さ 2*half+1 px"""
    H, W = shape
    open_m = np.zeros((H, W), np.uint8)
    struct_m = np.zeros((H, W), np.uint8)
    for f in fits:
        xs = np.arange(W)
        yv = np.polyval(f["c"], xs)
        for dy in range(-half, half + 1):
            yy = np.round(yv + dy).astype(int)
            ok = (yy >= 0) & (yy < H)
            sel_x, sel_y = xs[ok], yy[ok]
            issky = frac[sel_y, sel_x] > sky_thr
            open_m[sel_y[issky], sel_x[issky]] = 1
            struct_m[sel_y[~issky], sel_x[~issky]] = 1
    return open_m, struct_m

if __name__ == "__main__":
    p = Path(sys.argv[1])
    im = cv2.imread(str(p)); H, W = im.shape[:2]
    wm, sky, frac = sky_and_wire(im)
    fits = ransac_wires(wm, W)
    om, sm = masks_from_fits(fits, (H, W), frac)
    print(f"{p.name}: 線 {len(fits)} 本  空背景 {om.sum()}px  構造背景 {sm.sum()}px")
    for f in fits:
        print(f"   x[{f['xa']:4d},{f['xb']:4d}] inl={f['n']:5d} res={f['res']:.2f}")
    vis = im.copy()
    vis[om > 0] = (0, 0, 255)      # 赤=空背景（既存評価の対象）
    vis[sm > 0] = (0, 255, 0)      # 緑=構造背景（外挿）
    cv2.imwrite("/workspace/tmp/_extrap_vis.png", vis)
