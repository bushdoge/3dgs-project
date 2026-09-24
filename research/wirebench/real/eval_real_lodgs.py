# LOD-GS 実データ本命検証（§9af 案(a)・48GB 再実行）の統合評価。
# 問い：実データ（Mipがテクスチャを失い、歩き撮りでサンプリングレートが変わる条件）で、
#       LOD-GS は「細線の改善」と「テクスチャ保持」を両立できるか。
# 既存の実データ30kモデルと LOD-GS を**同じ物差し**で並べる：
#   (1) 細線（空背景マスク・§9t-2と同じ検出）: 全体/細線/背景/落差/保持率/細線最悪   ← research/wirebench/real/eval_real_wires30k.py と同じ
#   (2) 空背景 vs 構造背景（§9t-4のRANSAC外挿マスク）                                ← research/wirebench/real/eval_real_open_vs_struct.py と同じ
#   (3) 高周波テクスチャ保持率（レンダ/GT）。旧定義のスクリプトが残っていないため research/wirebench/real/hf_calib.py で較正し、
#       実データの既報値（mip30k 31.1%）を再現する lap_abs（平均|ラプラシアン|の比）を主、hp2_E を副とする
#   (4) テスト視点ごとの全体PSNR（近カメラフローターによる破綻視点 00001/00004/00009 の有無）
#   (5) ガウシアンの最近傍学習カメラ距離の分布（近カメラフローター率）               ← research/wirebench/real/analyze_np05.py と同じ
#   SSIM/LPIPS は本家3DGSの metrics.py と同じ関数（utils.loss_utils.ssim / lpipsPyTorch VGG）で全モデルを計算し直す
#   （LOD-GS 同梱の metrics.py はマスク必須・視点番号決め打ちの改変版で使えず、ks005/np10 は results.json が無いため）。
import sys, hashlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent)); sys.path.insert(0, "/opt/gaussian-splatting")
import cv2, numpy as np, torch
from pathlib import Path
from wire_struct_masks import build, OFF
from utils.loss_utils import ssim as ssim_fn
from lpipsPyTorch import lpips as lpips_fn

def tens(bgr):
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().cuda() / 255.0

EXP = Path("/workspace/experiments/20260823_wire_railside_01")
# tag: (出力dir, GT subdir, レンダ subdir)
SETS = {
    "3dgs 30k":         ("output_30k",          "gt",    "renders"),
    "mip 30k":          ("output_mip30k",       "gt_-1", "test_preds_-1"),
    "mip ks0.05 30k":   ("output_mip30k_ks005", "gt_-1", "test_preds_-1"),
    "mip+prune0.5 30k": ("output_mip30k_np05",  "gt_-1", "test_preds_-1"),
    "mip+prune1.0 30k": ("output_mip30k_np10",  "gt_-1", "test_preds_-1"),
    "LOD-GS 30k":       ("output_lodgs_30k",    "gt",    "renders"),
    # 2026-09-22 追加：同じ学習済みモデルを LOD 補正なしでレンダしたもの（research/wirebench/real/lodgs_lod_ablation.py）
    "LOD-GS 30k noLOD": ("output_lodgs_30k_noLOD", "gt", "renders"),
}
IT = "ours_30000"

def load(tag):
    od, g, r = SETS[tag]
    gd, rd = EXP/od/"test"/IT/g, EXP/od/"test"/IT/r
    if not (gd.is_dir() and rd.is_dir()):
        return None
    return sorted(gd.glob("*.png")), sorted(rd.glob("*.png"))

def psnr(a, b, m=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if m is not None:
        d = d[m > 0]
        if d.size == 0: return np.nan
    return 10 * np.log10(255.0 ** 2 / max(d.mean(), 1e-9))

# ---- (1) の細線マスク（research/wirebench/real/eval_real_wires30k.py と同一）----
def wire_mask(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sky = ((b - r > 18) & (v > 150)).astype(np.uint8)
    frac = cv2.blur(sky.astype(np.float32), (21, 21))
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    w = ((bh > 25) & (frac > 0.85) & (sky == 0)).astype(np.uint8)
    return cv2.morphologyEx(w, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), sky, frac

def contrast(gray, ys, xs, bg):
    return float(((bg - gray[ys, xs]) / np.maximum(bg, 1e-6)).mean() * 100)

def hf(g, r):
    lg, lr = cv2.Laplacian(g, cv2.CV_64F), cv2.Laplacian(r, cv2.CV_64F)
    hg, hr = g - cv2.GaussianBlur(g, (0, 0), 2.0), r - cv2.GaussianBlur(r, (0, 0), 2.0)
    return np.abs(lr).mean() / np.abs(lg).mean(), (hr**2).mean() / (hg**2).mean()

# ---- GT が全モデルで同一かを確認（§9t-4 と同じ手続き）----
ref = load("3dgs 30k")[0]
md5 = lambda p: hashlib.md5(cv2.imread(str(p)).tobytes()).hexdigest()
ref_md5 = [md5(p) for p in ref]
for tag in SETS:
    L = load(tag)
    if L is None: continue
    same = [md5(p) for p in L[0]] == ref_md5
    print(f"GT一致 {tag:17s}: {'OK' if same else '** 不一致 **'}  (n={len(L[0])})")
print()

# ---- (2) の空/構造背景マスクを1回だけ構築（research/wirebench/real/eval_real_open_vs_struct.py と同一）----
MIN_PX = 300
M = {}
for p in ref:
    gt = cv2.imread(str(p)); d = build(gt)
    g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float64); H = g.shape[0]
    ent = {}
    for reg in ("open_v", "struct_v"):
        ys, xs = np.where(d[reg] > 0)
        if ys.size < MIN_PX: continue
        bg = (g[np.clip(ys-OFF, 0, H-1), xs] + g[np.clip(ys+OFF, 0, H-1), xs]) / 2.0
        ent[reg] = dict(ys=ys, xs=xs, bg=bg, m3=cv2.dilate(d[reg], np.ones((3, 3), np.uint8)),
                        gtc=contrast(g, ys, xs, bg))
    M[p.name] = ent

R = {}
for tag in SETS:
    L = load(tag)
    if L is None:
        print(f"{tag}: 未作成"); continue
    w1, os_, hfs, full, sl = [], {"open_v": [], "struct_v": []}, [], [], []
    for gp, rp in zip(*L):
        gt, re_ = cv2.imread(str(gp)), cv2.imread(str(rp))
        if gt.shape != re_.shape:
            re_ = cv2.resize(re_, (gt.shape[1], gt.shape[0]))
        gg = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float64)
        rg = cv2.cvtColor(re_, cv2.COLOR_BGR2GRAY).astype(np.float64)
        full.append(psnr(gt, re_)); hfs.append(hf(gg, rg))
        with torch.no_grad():
            tr, tg = tens(re_), tens(gt)
            sl.append((ssim_fn(tr, tg).item(), lpips_fn(tr, tg, net_type="vgg").item()))
        wm, sky, frac = wire_mask(gt)
        if wm.sum() >= 200:
            wmd = cv2.dilate(wm, np.ones((3, 3), np.uint8))
            bgm = ((frac > 0.85) & (wmd == 0)).astype(np.uint8)
            base = np.percentile(gg[sky > 0], 50) if sky.sum() > 0 else 255
            w1.append((psnr(gt, re_, wmd), psnr(gt, re_, bgm),
                       (base - gg[wm > 0]).mean() / base * 100, (base - rg[wm > 0]).mean() / base * 100))
        for reg, e in M[gp.name].items():
            os_[reg].append((psnr(gt, re_, e["m3"]), contrast(rg, e["ys"], e["xs"], e["bg"]) / e["gtc"] * 100))
    a, h, s = np.array(w1), np.array(hfs), np.array(sl)
    R[tag] = dict(full=np.mean(full), full_min=np.min(full), full_v=full,
                  thin=a[:, 0].mean(), bg=a[:, 1].mean(), keep=a[:, 3].mean() / a[:, 2].mean() * 100,
                  thin_min=a[:, 0].min(), n1=len(a),
                  open=np.array(os_["open_v"]), struct=np.array(os_["struct_v"]),
                  hf_lap=h[:, 0].mean() * 100, hf_hp=h[:, 1].mean() * 100,
                  ssim=s[:, 0].mean(), lpips=s[:, 1].mean())

print("=== (1) 細線（空背景の暗線検出マスク・§9t-2と同一） ＋ 全体 ＋ テクスチャ ===")
print(f"{'model':17s} {'全体':>6s} {'最悪':>6s} {'SSIM':>6s} {'LPIPS':>6s} | {'細線':>6s} {'背景':>6s} {'落差':>6s} "
      f"{'保持率':>5s} {'細線最悪':>7s} | {'高周波lap':>8s} {'hp2E':>6s}")
for t, r in R.items():
    print(f"{t:17s} {r['full']:6.2f} {r['full_min']:6.2f} {r['ssim']:6.4f} {r['lpips']:6.4f} | {r['thin']:6.2f} "
          f"{r['bg']:6.2f} {r['bg']-r['thin']:6.2f} {r['keep']:4.0f}% {r['thin_min']:7.2f} | "
          f"{r['hf_lap']:7.1f}% {r['hf_hp']:5.1f}%")

print("\n=== (2) 同じ電線を背景で二分（§9t-4のRANSACマスク） ===")
print(f"{'model':17s} | {'空背景PSNR':>10s} {'保持率':>5s} {'最悪':>6s} | {'構造背景PSNR':>12s} {'保持率':>5s} {'最悪':>6s}")
for t, r in R.items():
    o, s = r["open"], r["struct"]
    print(f"{t:17s} | {o[:,0].mean():10.2f} {o[:,1].mean():4.0f}% {o[:,0].min():6.2f} | "
          f"{s[:,0].mean():12.2f} {s[:,1].mean():4.0f}% {s[:,0].min():6.2f}")
b = R["3dgs 30k"]
for t in R:
    if t == "3dgs 30k": continue
    r = R[t]
    print(f"  {t:17s} − 素3DGS: 空背景 {r['open'][:,0].mean()-b['open'][:,0].mean():+.2f} dB"
          f" / 構造背景 {r['struct'][:,0].mean()-b['struct'][:,0].mean():+.2f} dB / 全体 {r['full']-b['full']:+.2f} dB"
          f" / 高周波 {b['hf_lap']:.1f}→{r['hf_lap']:.1f}%")

print("\n=== (4) テスト視点ごとの全体PSNR ===")
print("view " + " ".join(f"{t[:12]:>12s}" for t in R))
for i in range(len(ref)):
    print(f"{i:4d} " + " ".join(f"{R[t]['full_v'][i]:12.2f}" for t in R))

print("\n=== (5) ガウシアンの最近傍学習カメラ距離（近カメラフローター率） ===")
from plyfile import PlyData
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from scipy.spatial import cKDTree
C = np.array([-qvec2rotmat(v.qvec).T @ v.tvec
              for v in read_extrinsics_binary(str(EXP/"sparse/0/images.bin")).values()])
kd = cKDTree(C)
for t in R:
    f = EXP/SETS[t][0]/"point_cloud/iteration_30000/point_cloud.ply"
    if not f.exists():
        print(f"{t:17s} plyなし"); continue
    v = PlyData.read(str(f))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    d, _ = kd.query(xyz); n = len(xyz)
    print(f"{t:17s} {n/1e6:5.2f}M  " + " ".join(f"<{T}:{(d<T).sum()/n*100:6.2f}%" for T in (0.3, 0.5, 1.0)))
    del v, xyz, d
