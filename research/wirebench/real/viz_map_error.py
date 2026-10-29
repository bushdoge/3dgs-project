# 実データで「確実に空いている点」が Splat-Nav の地図では障害物の中になっている例を図にする（研究メモ §9az）。
#   左：その点が空の画素に写る学習視点の写真（＝その方向は空まで何も無い証拠）。点（赤×）、Splat-Nav が寄せた先（青＋）、
#       その点のセルを占有にしたガウシアン（1σ 楕円体を写真に投影した輪郭。色＝不透明度）を重ねる
#   右：Splat-Nav の格子（元実装と一致する高速版・0.15m・機体半径 0.1m）の、点の高さでの水平断面。灰＝占有。
#       占有の原因のガウシアンの 1σ 軸平行外接箱（Splat-Nav が占有にする範囲、橙の枠）と 1σ 楕円（赤の線）
# 「原因のガウシアン」＝ 1σ 外接箱が、点のセルの周り（機体膨張の 3×3×3 の範囲）に重なるもの。点そのものが楕円体（＋κ）に入るかも併記。
# 実行は tools/splatnav/venv の python。GPU 不要。
# 使い方: python3 real/viz_map_error.py <exp> <model> [--pair N --end start|goal]（省略時は寄せた距離が最大の端点）
#   → research/wirebench/figs/map_error_<exp>_<model>_p<N><start|goal>.png
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402
from matplotlib.patches import Ellipse, Rectangle     # noqa: E402
import numpy as np                                    # noqa: E402
import pycolmap                                       # noqa: E402
import torch                                          # noqa: E402
from scipy.spatial.transform import Rotation          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import splatnav_wires as SW                           # noqa: E402
import sky_carve as SC                                # noqa: E402
from real_splatnav import camera_frame                # noqa: E402
from initialization.grid_utils import GSplatVoxel     # noqa: E402

FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
font_manager.fontManager.addfont(FONT)
# 日本語フォント（Droid Sans Fallback）は英数字を持たないので、英数字は DejaVu Sans、無い字を日本語フォントで補う
plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=FONT).get_name()]
GRID, KAPPA = 0.15, 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("model")
    ap.add_argument("--pair", type=int)
    ap.add_argument("--end", choices=["start", "goal"])
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    al, _, _ = camera_frame(rec, 1.4)
    z = np.load(exp / "splatnav_full_pairs.npz")
    lo, hi, pairs = z["lower"], z["upper"], z["pairs"]
    SW.LOWER, SW.UPPER = lo, hi
    SW.use_fast_voxel()
    mu, q, sc, op = SW.load_ply_gt(exp / a.model / "point_cloud" / "iteration_30000" / "point_cloud.ply", al)
    c = SW.crop(mu, sc, pad=1.0)
    mu, q, sc, op = mu[c], q[c], sc[c], op[c]
    res = torch.tensor(np.ceil((hi - lo) / GRID).astype(int))
    vox = GSplatVoxel(SW.Map(mu, q, sc), lower_bound=torch.tensor(lo, dtype=torch.float32), upper_bound=torch.tensor(hi, dtype=torch.float32),
                      resolution=res, radius=KAPPA, device=SW.DEV)
    occ = vox.non_navigable_grid.numpy()
    centers = vox.grid_centers.numpy()

    # 端点の選択（障害物の中で、寄せた距離が最大のもの）
    cands = []
    for pi, (s, g) in enumerate(pairs):
        for name, p in (("start", s), ("goal", g)):
            i = vox.get_indices(torch.tensor(p, dtype=torch.float32)).numpy()
            if occ[tuple(i)]:
                qp = vox.find_closest_navigable(torch.tensor(p, dtype=torch.float32)).numpy()
                cands.append((float(np.linalg.norm(qp - p)), pi, name, p, qp, i))
    if a.pair is not None:
        cands = [x for x in cands if x[1] == a.pair and (a.end is None or x[2] == a.end)]
    if not cands:
        sys.exit("障害物の中の端点が無い")
    dmove, pi, name, p, qp, idx = max(cands, key=lambda x: x[0])

    # 原因のガウシアン：1σ 外接箱が点のセルの 3×3×3 近傍に重なるもの
    Rw = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    scc = np.maximum(sc, SW.MIN_SCALE)
    half = np.sqrt(np.einsum("nij,nj->ni", Rw ** 2, scc ** 2))
    cc = centers[tuple(idx)]
    culp = np.nonzero(np.all(np.abs(mu - cc) <= half + 1.5 * GRID, 1))[0]
    cov = (Rw[culp] * scc[culp][:, None, :] ** 2) @ Rw[culp].transpose(0, 2, 1)
    d = p - mu[culp]
    m2 = np.einsum("ni,nij,nj->n", d, np.linalg.inv(cov + KAPPA ** 2 * np.eye(3)), d)
    inside = m2 <= 1.0

    # 写真：点が安全な空の画素に写る視点のうち、カメラが最も近いもの
    sa = argparse.Namespace(bh_thr=12, wire_dilate=6, erode=4, border=8, obs_dilate=40, close=9)
    obs = SC.observations(rec)
    best = None
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        A = cfw.rotation.matrix() @ al["R"].T / al["s"]
        b = np.asarray(cfw.translation) - A @ al["t"]
        pc = A @ p + b
        if pc[2] <= 0.5:
            continue
        u, v = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
        if not (20 <= u < cam.width - 20 and 20 <= v < cam.height - 20):
            continue
        dist = pc[2] * al["s"] / al["s"]
        if best is not None and dist >= best[0]:
            continue
        bgr = cv2.imread(str(exp / "images" / im.name))
        s = SC.safe_sky(bgr, sa, obs.get(im.name))
        if s[int(v), int(u)]:
            best = (dist, im.name, A, b, fx, fy, cx, cy, bgr)
    if best is None:
        sys.exit("点が空の画素に写る視点が見つからない")
    _, iname, A, b, fx, fy, cx, cy, bgr = best

    fig, ax = plt.subplots(1, 2, figsize=(16, 8.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax[0].imshow(bgr[:, :, ::-1])

    def proj(X):
        pc_ = A @ X + b
        return np.array([fx * pc_[0] / pc_[2] + cx, fy * pc_[1] / pc_[2] + cy]), pc_
    cmap = plt.get_cmap("viridis")
    for k, j in enumerate(culp):
        uv, pc_ = proj(mu[j])
        if pc_[2] <= 0.3:
            continue
        S = A @ cov[k] @ A.T
        J = np.array([[fx / pc_[2], 0, -fx * pc_[0] / pc_[2] ** 2], [0, fy / pc_[2], -fy * pc_[1] / pc_[2] ** 2]])
        S2 = J @ S @ J.T
        w, V = np.linalg.eigh(S2)
        w = np.sqrt(np.maximum(w, 1e-6))
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ax[0].add_patch(Ellipse(uv, 2 * w[1], 2 * w[0], angle=ang, fill=False, lw=1.2, ec=cmap(float(op[j]))))
    up, _ = proj(p)
    uq, _ = proj(qp)
    ax[0].plot(*up, "x", color="red", ms=16, mew=3, label="確実に空いている点（この方向は空まで何も無い）")
    ax[0].plot(*uq, "+", color="deepskyblue", ms=16, mew=3, label=f"Splat-Nav が寄せた先（{dmove:.2f}m 先）")
    ax[0].set_xlim(0, bgr.shape[1])
    ax[0].set_ylim(bgr.shape[0], 0)
    ax[0].axis("off")
    ax[0].legend(loc="lower left", fontsize=11, framealpha=0.85)
    ax[0].set_title(f"撮影した写真 {iname}\n原因のガウシアン {len(culp)} 個の1σ楕円を投影（色＝不透明度：紫 0 → 黄 1）", fontsize=12)

    # 右：水平断面
    zi = idx[2]
    sl = occ[:, :, zi].T
    ext = [lo[0], lo[0] + GRID * occ.shape[0], lo[1], lo[1] + GRID * occ.shape[1]]
    ax[1].imshow(sl, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    for k, j in enumerate(culp):
        ax[1].add_patch(Rectangle((mu[j, 0] - half[j, 0], mu[j, 1] - half[j, 1]), 2 * half[j, 0], 2 * half[j, 1], fill=False, ec="orange", lw=1.0))
        S = cov[k][:2, :2]
        w, V = np.linalg.eigh(S)
        w = np.sqrt(np.maximum(w, 1e-9))
        ax[1].add_patch(Ellipse(mu[j, :2], 2 * w[1], 2 * w[0], angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])), fill=False, ec="red", lw=1.0))
    ax[1].plot(p[0], p[1], "x", color="red", ms=16, mew=3)
    ax[1].plot(qp[0], qp[1], "+", color="deepskyblue", ms=16, mew=3)
    r = max(3.0, 1.5 * dmove)
    ax[1].set_xlim(p[0] - r, p[0] + r)
    ax[1].set_ylim(p[1] - r, p[1] + r)
    ax[1].set_xlabel("x [m]（進行方向）")
    ax[1].set_ylabel("y [m]")
    ax[1].set_title(f"Splat-Nav の格子の水平断面（灰＝障害物扱い）\n橙＝原因のガウシアンの1σ軸平行外接箱、赤＝1σ楕円の上から見た形", fontsize=12)

    big = scc[culp].max(1)
    S = np.array([pt.xyz for pt in rec.points3D.values()])
    S = (al["s"] * (al["R"] @ S.T)).T + al["t"]
    from scipy.spatial import cKDTree
    d_sfm = cKDTree(S).query(p)[0]
    txt = (f"{exp.name} / {a.model} / 組{pi} の{'始点' if name == 'start' else '終点'}：一番近い実物（SfM の3D点）まで {d_sfm:.2f}m。"
           f"Splat-Nav はこの点を {dmove:.2f}m 動かしてから計画した。\n"
           f"原因のガウシアン {len(culp)} 個：最大軸 中央値 {np.median(big):.2f}m（最大 {big.max():.2f}m）、不透明度 中央値 {np.median(op[culp]):.2f}。"
           f"点がその楕円体（＋機体半径）の中に入るもの {int(inside.sum())} 個＝障害物扱いは軸平行の外接箱によるもの" if inside.sum() == 0 else
           f"{exp.name} / {a.model} / 組{pi}：一番近い実物まで {d_sfm:.2f}m、原因 {len(culp)} 個、楕円体（＋機体半径）が点を含むもの {int(inside.sum())} 個")
    fig.text(0.01, 0.01, txt, fontsize=11)
    out = HERE.parent / "figs" / f"map_error_{exp.name}_{a.model}_p{pi}{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out, dpi=110)
    print(out)
    print(txt)


if __name__ == "__main__":
    main()
