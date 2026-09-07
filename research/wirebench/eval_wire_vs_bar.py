# コード監査で発覚した交絡の検証：細線マスク（pass_index=1）は**電線とフェンス縦棒の両方**を含む。
# 電線の半径は --wire_radius で変わるがフェンス棒は BAR_R=0.012 固定なので、
# 太さスイープでマスクの構成比が大きく変わる（推定：r08 は棒8割、r60 は電線2/3）。
# 「太いほど落差が拡大」という §9 の知見が、この構成比の変化で説明できてしまう可能性がある。
#
# ここでは GT の電線中心線点群（gt/wire_points.npz。label 1=電線 / 2=フェンス棒）を各テスト視点へ
# 投影し、マスク画素を**最近傍のラベル**で電線／棒に分類して、それぞれのPSNRを別々に測る。
# 電線と棒は空間的に大きく離れている（棒は地面付近・電線は電柱間の高所）ので最近傍分類で十分。
#
# 使い方: python3 eval_wire_vs_bar.py <実験dir> <iteration> [model_dir]
import sys, json
from pathlib import Path
import numpy as np, cv2
from scipy.spatial import cKDTree

EXP = Path(sys.argv[1]).resolve()
IT = int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "output"

poses = json.load(open(EXP / "gt" / "poses.json"))
fx, fy, cx0, cy0 = poses["fx"], poses["fy"], poses["cx"], poses["cy"]
d = np.load(EXP / "gt" / "wire_points.npz")
P, LB = d["points"], d["labels"]

# テスト視点は eval_wires と同じ導出（COLMAP登録名の8個おき）
sys.path.insert(0, "/opt/gaussian-splatting")
from scene.colmap_loader import read_extrinsics_binary
names = sorted(Path(v.name).stem for v in
               read_extrinsics_binary(str(EXP / "sparse/0/images.bin")).values())[::8]

md = EXP / MODEL / "test" / f"ours_{IT}"
acc = {1: [], 2: []}
px = {1: 0, 2: 0}
for i, name in enumerate(names):
    gt = cv2.imread(str(md / "gt" / f"{i:05d}.png"))
    ren = cv2.imread(str(md / "renders" / f"{i:05d}.png"))
    mask = cv2.imread(str(EXP / "gt" / "mask_thin" / f"{name.split('_')[1]}.png"), 0)
    if gt is None or ren is None or mask is None:
        continue
    if ren.shape != gt.shape:
        ren = cv2.resize(ren, (gt.shape[1], gt.shape[0]))
    C = np.array(poses["frames"][name]); W2C = np.linalg.inv(C)
    Xc = (W2C[:3, :3] @ P.T + W2C[:3, 3:4]).T
    zc = -Xc[:, 2]
    u = cx0 + fx * Xc[:, 0] / np.maximum(zc, 1e-6)
    v = cy0 - fy * Xc[:, 1] / np.maximum(zc, 1e-6)
    H, W = mask.shape
    ok = (zc > 0.1) & (u > -50) & (u < W + 50) & (v > -50) & (v < H + 50)
    if ok.sum() < 2:
        continue
    tree = cKDTree(np.stack([u[ok], v[ok]], 1))
    lab_ok = LB[ok]
    ys, xs = np.where(mask > 127)
    if ys.size == 0:
        continue
    _, idx = tree.query(np.stack([xs, ys], 1))
    cls = lab_ok[idx]
    err = ((gt.astype(np.float64) - ren.astype(np.float64)) ** 2).mean(2)
    for c in (1, 2):
        sel = cls == c
        if sel.sum():
            acc[c].append(err[ys[sel], xs[sel]]); px[c] += int(sel.sum())

print(f"実験 {EXP.name} / model {MODEL} / iter {IT} / テスト{len(names)}視点")
print(f"{'領域':14s}{'PSNR':>8s}{'画素/視点':>11s}{'マスク内比':>11s}")
tot = sum(px.values())
for c, nm in ((1, "電線(wire)"), (2, "フェンス棒")):
    if not acc[c]:
        continue
    e = np.concatenate(acc[c])
    print(f"{nm:14s}{10*np.log10(255.0**2/e.mean()):8.2f}{px[c]/len(names):11.0f}{px[c]/tot*100:10.1f}%")
