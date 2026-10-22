# 後処理で「空のガウシアン」を地図から除く（学習し直さない。研究メモ §9ax）。
# 各ガウシアンの中心を学習視点（COLMAP の全登録画像、--views even なら偶数番だけ）に投影し、写る画素が
#   空（sky_carve.safe_sky：色＋SfM 特徴点の除外＋上端連結＋電線の周りを除く）           → 空の票
#   空でない（色の判定で空でない画素＝建物・電線・地面など）                              → 空でない票
#   どちらでもない（色は空だが安全側の判定で除いた画素）                                → 数えない
# とし、空の票が --min-votes 以上かつ 空の票/(空+空でない) ≥ --ratio のガウシアンを除く（既定は ratio 0＝空間彫刻：
# 空の画素の視線はその先の空まで何にも当たらないので、その視線上にあるガウシアンは実在の物体でない）。
# 実在の物体（電線・建物）は自分の画素に写るので空の画素には写らない。空のマスクは電線・暗い細線・SfM 特徴点の周りを除いた安全側。
# 当初の「空の票の割合 ≥0.9」は、地平線近くの白っぽい空を色の判定が「空でない」に数えるため1個も除けなかった（合成アンカー）。
# 出力：<exp>/<model>_skyprune/point_cloud/iteration_<iter>/point_cloud.ply（元の ply から行を落としただけ）と除いた数の要約。
# 実データの評価（sky_carve.py）は空の画素を使うので、除去に使った視点と評価に使う視点を分けたいときは --views even を使い、
# 評価側は sky_carve.py --views odd で測る（循環を避ける）。
# 使い方: python3 sky_prune.py <exp> <model_dir名> [...] [--views all|even] [--ratio 0.9 --min-votes 2]
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "real"))
from wire_extrap import sky_and_wire     # noqa: E402


def safe_sky_args():
    return argparse.Namespace(bh_thr=12, wire_dilate=6, erode=4, border=8, obs_dilate=40, close=9)


def image_path(exp, name):
    for d in ("images", "input"):
        p = exp / d / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--views", choices=["all", "even"], default="all")
    ap.add_argument("--ratio", type=float, default=0.0)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--suffix", default="_skyprune")
    a = ap.parse_args()
    # safe_sky は sky_carve にあるが、Splat-Nav を import しないよう関数だけ読み込む
    src = (HERE / "sky_carve.py").read_text()
    ns = {"cv2": cv2, "np": np, "sky_and_wire": sky_and_wire}
    exec("def safe_sky" + src.split("def safe_sky")[1].split("\ndef ")[0], ns)
    safe_sky = ns["safe_sky"]
    sa = safe_sky_args()

    exp = Path(a.exp).resolve()
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    ims = sorted(rec.images.values(), key=lambda i: i.name)
    if a.views == "even":
        ims = ims[::2]
    views = []
    for im in ims:
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        bgr = cv2.imread(str(image_path(exp, im.name)))
        obs = np.array([p.xy for p in im.points2D if p.has_point3D()]) if len(im.points2D) else np.zeros((0, 2))
        _, sky_color, _ = sky_and_wire(bgr)
        s = safe_sky(bgr, sa, obs)
        lab = np.zeros(sky_color.shape, np.int8)     # 1＝空、−1＝空でない、0＝数えない
        lab[s > 0] = 1
        lab[sky_color == 0] = -1
        views.append((cfw.rotation.matrix(), np.asarray(cfw.translation), fx, fy, cx, cy, lab))
    print(f"{exp.name}: 視点 {len(views)}（{a.views}）", flush=True)
    for m in a.models:
        ply = exp / m / "point_cloud" / f"iteration_{a.iter}" / "point_cloud.ply"
        pd_ = PlyData.read(str(ply))
        v = pd_["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
        sky_v = np.zeros(len(xyz), int)
        non_v = np.zeros(len(xyz), int)
        for R, t, fx, fy, cx, cy, lab in views:
            pc = xyz @ R.T + t
            z = pc[:, 2]
            ok = z > 1e-3
            u = np.full(len(xyz), -1.0)
            w = np.full(len(xyz), -1.0)
            u[ok] = fx * pc[ok, 0] / z[ok] + cx
            w[ok] = fy * pc[ok, 1] / z[ok] + cy
            H, W = lab.shape
            ok &= (u >= 0) & (u < W) & (w >= 0) & (w < H)
            l = np.zeros(len(xyz), np.int8)
            l[ok] = lab[w[ok].astype(int), u[ok].astype(int)]
            sky_v += l == 1
            non_v += l == -1
        rm = (sky_v >= a.min_votes) & (sky_v / np.maximum(sky_v + non_v, 1) >= a.ratio)
        out_dir = exp / f"{m}{a.suffix}" / "point_cloud" / f"iteration_{a.iter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(v.data[~rm], "vertex")], text=False).write(str(out_dir / "point_cloud.ply"))
        op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
        sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3) if f"scale_{i}" in v.data.dtype.names], 1)).max(1)
        print(f"  {m}: {rm.sum()}/{len(xyz)} 個を除去（{rm.mean()*100:.1f}%）・除いたものの最大軸中央 {np.median(sc[rm]) if rm.any() else 0:.3f}・"
              f"不透明度中央 {np.median(op[rm]) if rm.any() else 0:.2f} → {out_dir / 'point_cloud.ply'}", flush=True)


if __name__ == "__main__":
    main()
