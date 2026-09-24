# 実データ第1波の追加評価：同じ電線を「空背景区間」と「構造背景区間（建物にかぶる区間）」に分けて測る。
# 合成ベンチの open / enclosed 対比の実データ版。線の太さ・材質が同一で背景だけ違う＝統制が効く。
# マスクは research/wirebench/real/wire_struct_masks.py（GTは5モデルで同一なので1回だけ作って共有）。
import warnings, cv2, numpy as np, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wire_struct_masks import build, OFF
from pathlib import Path

EXP = Path("/workspace/experiments/20260823_wire_railside_01")
SETS = {
    "3dgs 7k":         (EXP/"output/test/ours_7000/gt",              EXP/"output/test/ours_7000/renders"),
    "mip(ks0.1) 7k":   (EXP/"output_mip/test/ours_7000/gt_-1",       EXP/"output_mip/test/ours_7000/test_preds_-1"),
    "3dgs 30k":        (EXP/"output_30k/test/ours_30000/gt",         EXP/"output_30k/test/ours_30000/renders"),
    "mip(ks0.1) 30k":  (EXP/"output_mip30k/test/ours_30000/gt_-1",   EXP/"output_mip30k/test/ours_30000/test_preds_-1"),
    "mip(ks0.05) 30k": (EXP/"output_mip30k_ks005/test/ours_30000/gt_-1", EXP/"output_mip30k_ks005/test/ours_30000/test_preds_-1"),
    "mip+prune0.5 30k":(EXP/"output_mip30k_np05/test/ours_30000/gt_-1",   EXP/"output_mip30k_np05/test/ours_30000/test_preds_-1"),
    "LOD-GS 30k":      (EXP/"output_lodgs_30k/test/ours_30000/gt",        EXP/"output_lodgs_30k/test/ours_30000/renders"),
}
MIN_PX = 300          # マスクが小さすぎる視点は捨てる（統計にならない）

def psnr(a, b, m):
    d = ((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    d = d[m > 0]
    return np.nan if d.size == 0 else 10 * np.log10(255.0**2 / max(d.mean(), 1e-9))

def contrast(gray, ys, xs, bg):
    """線画素の局所コントラスト（背景比%）。bg は GT から取った局所背景輝度（全モデル共通の基準）"""
    return float(((bg - gray[ys, xs]) / np.maximum(bg, 1e-6)).mean() * 100)

# --- マスクを1回だけ構築 -------------------------------------------------
gtd0 = EXP/"output_30k/test/ours_30000/gt"
M = {}
for p in sorted(gtd0.glob("*.png")):
    gt = cv2.imread(str(p)); d = build(gt)
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H = g.shape[0]
    ent = {}
    for reg in ("open_v", "struct_v"):
        ys, xs = np.where(d[reg] > 0)
        if ys.size < MIN_PX:
            continue
        bg = (g[np.clip(ys-OFF,0,H-1), xs] + g[np.clip(ys+OFF,0,H-1), xs]) / 2.0
        m3 = cv2.dilate(d[reg], np.ones((3,3), np.uint8))     # PSNRは±1px膨張で測る
        ent[reg] = dict(ys=ys, xs=xs, bg=bg, m3=m3, gtc=contrast(g, ys, xs, bg))
    M[p.name] = ent
for reg, jp in (("open_v","空背景"), ("struct_v","構造背景")):
    ok = [k for k,v in M.items() if reg in v]
    npx = sum(int(M[k][reg]['m3'].sum()) for k in ok)
    print(f"{jp}: 有効視点 {len(ok)}/10  マスク画素(±1px膨張) {npx}  "
          f"GTコントラスト {np.mean([M[k][reg]['gtc'] for k in ok]):.1f}%")
print()

# --- モデルごとに測る ----------------------------------------------------
hdr = f"{'model':17s}"
for jp in ("空背景","構造背景"):
    hdr += f" | {jp+' PSNR':>10s} {'保持率':>6s} {'最悪':>6s}"
print(hdr); print("-" * 78)
rows = {}
for tag, (gtd, rend) in SETS.items():
    if not (gtd.is_dir() and rend.is_dir()):
        print(f"{tag:17s} | 未作成（レンダ待ち）"); continue
    res = sorted(rend.glob("*.png")); gts = sorted(gtd.glob("*.png"))
    acc = {r: [] for r in ("open_v","struct_v")}
    for gp, rp in zip(gts, res):
        gt, re_ = cv2.imread(str(gp)), cv2.imread(str(rp))
        if gt.shape != re_.shape:
            re_ = cv2.resize(re_, (gt.shape[1], gt.shape[0]))
        rg = cv2.cvtColor(re_, cv2.COLOR_BGR2GRAY).astype(np.float64)
        for reg, e in M[gp.name].items():
            acc[reg].append((psnr(gt, re_, e['m3']),
                             contrast(rg, e['ys'], e['xs'], e['bg']) / e['gtc'] * 100))
    line = f"{tag:17s}"; rows[tag] = {}
    for reg in ("open_v","struct_v"):
        a = np.array(acc[reg])
        line += f" | {a[:,0].mean():10.2f} {a[:,1].mean():5.0f}% {a[:,0].min():6.2f}"
        rows[tag][reg] = (a[:,0].mean(), a[:,1].mean(), a[:,0].min())
    print(line)
print()
for reg, jp in (("open_v","空背景"), ("struct_v","構造背景")):
    b = rows["3dgs 30k"][reg]
    for tag in ("mip(ks0.1) 30k", "mip+prune0.5 30k"):
        if tag not in rows: continue
        m = rows[tag][reg]
        print(f"{jp}30k: {tag} − 素3DGS = {m[0]-b[0]:+.2f} dB / 保持率 {b[1]:.0f}%→{m[1]:.0f}%")
