# 近カメラprune 半径感度（0.3 / 0.5 / 1.0）の一括評価。§9t-4 の表を prune 半径方向へ拡張する。
#
# 測るのは np0.5 の時と完全に同じ3点（research/wirebench/real/analyze_np05.py と research/wirebench/real/eval_real_open_vs_struct.py を統合）:
#   (1) ガウシアンの最近傍カメラ距離分布 …… 近カメラフローターがどこまで消えたか
#   (2) テスト視点ごとの全体PSNR      …… 破綻視点(00001/00004/00009)が回復しているか
#   (3) 空背景/構造背景に分けた細線PSNR・保持率 …… 細線の利得が維持されているか
# 半径を上げすぎると実構造まで削る（SfM点の1%ileが1.267）ので、(1)(2)が良くて(3)が落ちる半径＝削りすぎ。
#
# 使い方: python3 research/wirebench/real/eval_prune_sweep.py            （レンダ未完のモデルは自動でスキップ）
import warnings, sys, glob
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/gaussian-splatting")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, cv2
from pathlib import Path
from plyfile import PlyData
from scipy.spatial import cKDTree
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from wire_struct_masks import build, OFF

EXP = Path("/workspace/experiments/20260823_wire_railside_01")
MIN_PX = 300     # マスクが小さすぎる視点は統計にならないので捨てる（eval_real_open_vs_struct.py と同値）

# tag -> (出力dir, レンダのサブdir, GTのサブdir)。mip系は render.py の命名が素3DGSと違う。
MODELS = [
    ("3dgs 30k",        "output_30k",           "renders",         "gt"),
    ("mip 30k",         "output_mip30k",        "test_preds_-1",   "gt_-1"),
    ("mip+prune0.3",    "output_mip30k_np03",   "test_preds_-1",   "gt_-1"),
    ("mip+prune0.5",    "output_mip30k_np05",   "test_preds_-1",   "gt_-1"),
    ("mip+prune1.0",    "output_mip30k_np10",   "test_preds_-1",   "gt_-1"),
]
IT = 30000

def psnr_masked(a, b, m):
    d = ((a.astype(np.float64) - b.astype(np.float64)) ** 2)[m > 0]
    return np.nan if d.size == 0 else 10 * np.log10(255.0**2 / max(d.mean(), 1e-9))

def psnr_full(a, b):
    return 10 * np.log10(255.0**2 / max(((a.astype(np.float64) - b.astype(np.float64))**2).mean(), 1e-9))

def contrast(gray, ys, xs, bg):
    """線画素の局所コントラスト（背景比%）。bg は GT 由来＝全モデル共通の基準"""
    return float(((bg - gray[ys, xs]) / np.maximum(bg, 1e-6)).mean() * 100)

present = [(t, s, r, g) for t, s, r, g in MODELS
           if (EXP/s/f"test/ours_{IT}"/r).is_dir() and (EXP/s/f"test/ours_{IT}"/g).is_dir()]
missing = [t for t, s, r, g in MODELS if (t, s, r, g) not in present]
if missing:
    print(f"※ 未作成（学習/レンダ待ち）: {', '.join(missing)}\n")

# ── (1) ガウシアンの最近傍カメラ距離分布 ────────────────────────────────
C = np.array([-qvec2rotmat(v.qvec).T @ v.tvec
              for v in read_extrinsics_binary(str(EXP/"sparse/0/images.bin")).values()])
tree = cKDTree(C)
print("=== (1) ガウシアンの最近傍カメラ距離分布（SfM実構造は 0.1%ile=0.376 / 1%ile=1.267）===")
print(f"{'model':14s} {'個数':>7s}  " + "  ".join(f"<{T}" for T in (0.3, 0.5, 1.0, 1.6)))
for tag, sub, _, _ in MODELS:
    f = EXP/sub/f"point_cloud/iteration_{IT}/point_cloud.ply"
    if not f.exists():
        print(f"{tag:14s} 未作成"); continue
    v = PlyData.read(str(f))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    d, _ = tree.query(xyz); n = len(xyz)
    print(f"{tag:14s} {n/1e6:6.2f}M  " + "  ".join(f"{(d < T).sum()/n*100:6.2f}%" for T in (0.3, 0.5, 1.0, 1.6)))
    del v, xyz, d

# ── (2)(3) 画像系。GTは全モデル共通なので素3DGS側の並びを基準に使う ───────
if not present:
    sys.exit(0)
gt_ref = sorted((EXP/"output_30k"/f"test/ours_{IT}/gt").glob("*.png"))

print("\n=== (2) テスト視点ごとの全体PSNR ===")
full = {t: [] for t, *_ in present}
for i, gp in enumerate(gt_ref):
    gt = cv2.imread(str(gp)); line = f"{i:4d}  "
    for tag, sub, rend, _ in present:
        rs = sorted((EXP/sub/f"test/ours_{IT}"/rend).glob("*.png"))
        r = cv2.imread(str(rs[i]))
        if r.shape != gt.shape: r = cv2.resize(r, (gt.shape[1], gt.shape[0]))
        p = psnr_full(gt, r); full[tag].append(p); line += f"{p:14.2f}"
    if i == 0: print("view  " + "".join(f"{t:>14s}" for t, *_ in present))
    print(line)
print("平均  " + "".join(f"{np.mean(full[t]):14.2f}" for t, *_ in present))
print("最悪  " + "".join(f"{np.min(full[t]):14.2f}" for t, *_ in present))

# 細線マスク（GT由来なので1回だけ作って全モデルで共有）
M = {}
for p in gt_ref:
    gt = cv2.imread(str(p)); d = build(gt)
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float64); H = g.shape[0]
    ent = {}
    for reg in ("open_v", "struct_v"):
        ys, xs = np.where(d[reg] > 0)
        if ys.size < MIN_PX: continue
        bg = (g[np.clip(ys-OFF, 0, H-1), xs] + g[np.clip(ys+OFF, 0, H-1), xs]) / 2.0
        ent[reg] = dict(ys=ys, xs=xs, bg=bg,
                        m3=cv2.dilate(d[reg], np.ones((3, 3), np.uint8)),   # PSNRは±1px膨張で測る
                        gtc=contrast(g, ys, xs, bg))
    M[p.name] = ent

print("\n=== (3) 細線PSNR・保持率（空背景 / 構造背景）===")
print(f"{'model':14s}" + "".join(f" | {j+' PSNR':>10s} {'保持率':>6s} {'最悪':>6s}" for j in ("空背景", "構造背景")))
print("-" * 76)
rows = {}
for tag, sub, rend, gtsub in present:
    gts = sorted((EXP/sub/f"test/ours_{IT}"/gtsub).glob("*.png"))
    res = sorted((EXP/sub/f"test/ours_{IT}"/rend).glob("*.png"))
    acc = {r: [] for r in ("open_v", "struct_v")}
    for gp, rp in zip(gts, res):
        gt, re_ = cv2.imread(str(gp)), cv2.imread(str(rp))
        if gt.shape != re_.shape: re_ = cv2.resize(re_, (gt.shape[1], gt.shape[0]))
        rg = cv2.cvtColor(re_, cv2.COLOR_BGR2GRAY).astype(np.float64)
        for reg, e in M[gp.name].items():
            acc[reg].append((psnr_masked(gt, re_, e["m3"]),
                             contrast(rg, e["ys"], e["xs"], e["bg"]) / e["gtc"] * 100))
    line = f"{tag:14s}"; rows[tag] = {}
    for reg in ("open_v", "struct_v"):
        a = np.array(acc[reg])
        line += f" | {a[:,0].mean():10.2f} {a[:,1].mean():5.0f}% {a[:,0].min():6.2f}"
        rows[tag][reg] = (a[:, 0].mean(), a[:, 1].mean(), a[:, 0].min())
    print(line)

print("\n=== 素3DGS 30k 比 ===")
for reg, jp in (("open_v", "空背景"), ("struct_v", "構造背景")):
    b = rows["3dgs 30k"][reg]
    for tag in rows:
        if tag == "3dgs 30k": continue
        m = rows[tag][reg]
        print(f"{jp}: {tag:14s} {m[0]-b[0]:+6.2f} dB / 保持率 {b[1]:.0f}%→{m[1]:.0f}% "
              f"({m[1]-b[1]:+.0f}pt) / 全体PSNR {np.mean(full[tag]):.2f}")
