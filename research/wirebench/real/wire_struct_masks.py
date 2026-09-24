# 実データ: 電線を「空背景区間」と「構造背景区間（建物・樹木にかぶる区間）」に分けたマスクを作る。
#  1) 空背景で検出した電線画素にRANSACで2次曲線を当てる（電柱付近で交差するので連結成分では分離できない）
#  2) 全幅に外挿し、各画素を線の上下±off pxの「背景の色」で 空/構造 に分類する
#     （21x21の空占有率だと電線が密集する箇所で空を構造と誤判定するため）
#  3) GT上で実際に上下より暗いかを検証して選別（電柱・建物による遮蔽区間はここで落ちる）
import warnings, cv2, numpy as np, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wire_extrap import sky_and_wire, ransac_wires

OFF, MARGIN = 6, 5

def _line_pixels(fits, H, W, half=1):
    xs_all, ys_all = [], []
    for f in fits:
        xs = np.arange(W)
        yv = np.polyval(f["c"], xs)
        for dy in range(-half, half + 1):
            yy = np.round(yv + dy).astype(int)
            ok = (yy >= 0) & (yy < H)
            xs_all.append(xs[ok]); ys_all.append(yy[ok])
    if not xs_all:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(xs_all), np.concatenate(ys_all)

def build(gt_bgr):
    H, W = gt_bgr.shape[:2]
    gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b, g, r = [c.astype(np.int16) for c in cv2.split(gt_bgr)]
    is_sky = ((b - r > 18) & (gray > 150))            # 画素単位の空判定
    wm, sky, frac = sky_and_wire(gt_bgr)
    fits = ransac_wires(wm, W)
    xs, ys = _line_pixels(fits, H, W)
    yu, yd = np.clip(ys - OFF, 0, H - 1), np.clip(ys + OFF, 0, H - 1)
    open_bg = is_sky[yu, xs] & is_sky[yd, xs]         # 上下とも空 → 空背景区間
    dark    = gray[ys, xs] <= np.minimum(gray[yu, xs], gray[yd, xs]) - MARGIN
    out = {}
    for name, sel in (("open_raw", open_bg), ("struct_raw", ~open_bg),
                      ("open_v", open_bg & dark), ("struct_v", (~open_bg) & dark)):
        m = np.zeros((H, W), np.uint8); m[ys[sel], xs[sel]] = 1; out[name] = m
    out.update(fits=fits, is_sky=is_sky.astype(np.uint8), frac=frac)
    return out

if __name__ == "__main__":
    from pathlib import Path
    gtd = Path("/workspace/experiments/20260823_wire_railside_01/output_30k/test/ours_30000/gt")
    print(f"{'view':6s} {'線':>3s} {'空raw':>7s} {'空検証':>7s} {'構造raw':>8s} {'構造検証':>8s} {'残存率':>7s}")
    tot = np.zeros(4)
    for p in sorted(gtd.glob("*.png")):
        d = build(cv2.imread(str(p)))
        a, b_, c, e = [int(d[k].sum()) for k in ("open_raw", "open_v", "struct_raw", "struct_v")]
        tot += [a, b_, c, e]
        print(f"{p.stem:6s} {len(d['fits']):3d} {a:7d} {b_:7d} {c:8d} {e:8d} {(e/max(c,1)*100):6.0f}%")
    print(f"{'合計':6s} {'':3s} {tot[0]:7.0f} {tot[1]:7.0f} {tot[2]:8.0f} {tot[3]:8.0f} {tot[3]/max(tot[2],1)*100:6.0f}%")
