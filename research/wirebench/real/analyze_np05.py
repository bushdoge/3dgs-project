# near_prune版 mip30k の事後分析：
#   (1) ガウシアンが学習カメラからどの距離に分布しているか（近カメラフローターが消えたか）
#   (2) テスト視点ごとの全体PSNR（視点00001/00004/00009の破綻が消えたか）
# 細線指標は research/wirebench/real/eval_real_open_vs_struct.py 側で測る。
import sys; sys.path.insert(0, '/opt/gaussian-splatting')
import numpy as np, cv2, glob
from plyfile import PlyData
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from scipy.spatial import cKDTree
from pathlib import Path

E = Path('/workspace/experiments/20260823_wire_railside_01')
C = np.array([-qvec2rotmat(v.qvec).T @ v.tvec
              for v in read_extrinsics_binary(str(E/'sparse/0/images.bin')).values()])

print("=== ガウシアンの最近傍カメラ距離分布 ===")
for name, sub in [('3dgs30k','output_30k'), ('mip30k','output_mip30k'), ('mip+prune0.5','output_mip30k_np05')]:
    f = E/sub/'point_cloud/iteration_30000/point_cloud.ply'
    if not f.exists():
        print(f"{name:14s} 未作成"); continue
    v = PlyData.read(str(f))['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], 1)
    d, _ = cKDTree(C).query(xyz); n = len(xyz)
    row = " ".join(f"<{T}:{(d<T).sum()/n*100:5.2f}%" for T in (0.3, 0.5, 1.0, 1.6))
    print(f"{name:14s} {n/1e6:5.2f}M  {row}")
    del v, xyz, d

print("\n=== テスト視点ごとの全体PSNR ===")
S = {'3dgs30k': 'output_30k/test/ours_30000/renders',
     'mip30k': 'output_mip30k/test/ours_30000/test_preds_-1',
     'mip+prune0.5': 'output_mip30k_np05/test/ours_30000/test_preds_-1'}
gts = sorted(glob.glob(str(E/'output_30k/test/ours_30000/gt/*.png')))
def psnr(a, b): return 10*np.log10(255.0**2/max(((a.astype(np.float64)-b.astype(np.float64))**2).mean(), 1e-9))
avail = {k: sorted(glob.glob(str(E/v/'*.png'))) for k, v in S.items()}
avail = {k: v for k, v in avail.items() if v}
print("view  " + " ".join(f"{k:>13s}" for k in avail))
acc = {k: [] for k in avail}
for i, gp in enumerate(gts):
    gt = cv2.imread(gp); line = f"{i:4d}  "
    for k, ps in avail.items():
        r = cv2.imread(ps[i])
        if r.shape != gt.shape: r = cv2.resize(r, (gt.shape[1], gt.shape[0]))
        p = psnr(gt, r); acc[k].append(p); line += f"{p:13.2f} "
    print(line)
print("平均  " + " ".join(f"{np.mean(acc[k]):13.2f}" for k in avail))
print("最悪  " + " ".join(f"{np.min(acc[k]):13.2f}" for k in avail))
