# 空の後処理（sky_prune.py、§9ax）で電線の被覆が減った原因を調べる（研究メモ §9bc）。
#   元の地図で電線の点（wire3d.py の復元）を覆っていて、後処理で取り除かれたガウシアン（＝被覆を失わせたもの）を取り出し、
#   (a) 形：大きさ・細長さ・電線の向きとのそろい方・不透明度・中心と電線の距離
#   (b) 取り除かれた理由：中心が「確かな空」に写った視点で、投影位置から
#       ・その写真で検出した電線／暗い細線の画素（safe_sky が除外に使うもの）までの距離 [px]
#       ・復元した電線の点を同じ写真に投影した位置までの距離 [px]
#   を出す。候補の原因：中心が電線からずれている／その写真で電線が検出されていない／電線の所を通る大きなフローター／復元の誤差。
# 使い方: python3 real/why_prune_hits_wires.py <exp> <model>   （sky_prune.py --views even で <model>_skyprune を作ってあること）
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycolmap
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from real_splatnav import camera_frame          # noqa: E402
from wire_extrap import sky_and_wire           # noqa: E402
import sky_prune as SP                         # noqa: E402


def load(ply, al):
    v = PlyData.read(str(ply))["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    sc = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    if "filter_3D" in names:
        f2 = np.asarray(v["filter_3D"], np.float64)[:, None] ** 2
        op = op * np.sqrt(np.prod(sc ** 2 / (sc ** 2 + f2), axis=1))
        sc = np.sqrt(sc ** 2 + f2)
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    R = al["R"][None] @ Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    mu = (al["s"] * (al["R"] @ xyz.T)).T + al["t"]
    return xyz, mu, R, sc * al["s"], op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("model")
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    rec = pycolmap.Reconstruction(str(exp / "sparse" / "0"))
    al, _, _ = camera_frame(rec, 1.4)
    W = np.load(exp / "wire3d_bright.npz")["points"]
    wt = cKDTree(W)
    xyz0, mu0, R0, s0, op0 = load(exp / a.model / "point_cloud" / "iteration_30000" / "point_cloud.ply", al)
    xyz1, *_ = load(exp / f"{a.model}_skyprune" / "point_cloud" / "iteration_30000" / "point_cloud.ply", al)
    # sky_prune.py は元の並び順のまま行を落とすので、先頭から突き合わせれば取り除かれた行が分かる
    removed = np.ones(len(xyz0), bool)
    j = 0
    for i in range(len(xyz0)):
        if j < len(xyz1) and xyz0[i, 0] == xyz1[j, 0] and xyz0[i, 1] == xyz1[j, 1] and xyz0[i, 2] == xyz1[j, 2]:
            removed[i] = False
            j += 1
    assert j == len(xyz1), "突き合わせに失敗"
    # 電線の点を覆っていた（1σ楕円体＋2cm）取り除かれたガウシアン
    idx = np.nonzero(removed & (wt.query(mu0)[0] < s0.max(1) + 0.05))[0]
    cov = (R0[idx] * s0[idx][:, None, :] ** 2) @ R0[idx].transpose(0, 2, 1)
    inv = np.linalg.inv(cov + 0.02 ** 2 * np.eye(3))
    cover = []
    for k, j in enumerate(idx):
        nb = wt.query_ball_point(mu0[j], s0[j].max() + 0.05)
        if not nb:
            cover.append(0)
            continue
        d = W[nb] - mu0[j]
        cover.append(int((np.einsum("ni,ij,nj->n", d, inv[k], d) <= 1).sum()))
    cover = np.array(cover)
    sel = idx[cover > 0]
    print(f"取り除かれたガウシアン {removed.sum()} 個のうち、電線の点を覆っていたもの {len(sel)} 個（覆っていた電線の点 延べ {cover.sum()}）")
    if not len(sel):
        return
    # 形：電線の局所の向き（近くの電線の点の主成分）と、ガウシアンの一番長い軸のなす角
    rows = []
    for j in sel:
        nb = wt.query_ball_point(mu0[j], 0.3)
        Wn = W[nb] if len(nb) >= 3 else W[wt.query(mu0[j], 5)[1]]
        u, sv, vt = np.linalg.svd(Wn - Wn.mean(0))
        wdir = vt[0]
        ax = R0[j][:, np.argmax(s0[j])]
        ang = np.degrees(np.arccos(min(1.0, abs(float(ax @ wdir)))))
        ss = np.sort(s0[j])[::-1]
        rows.append({"gid": int(j), "max_m": ss[0], "mid_m": ss[1], "min_m": ss[2], "elong": ss[0] / max(ss[1], 1e-6),
                     "angle_to_wire_deg": ang, "opacity": op0[j], "center_to_wire_m": float(wt.query(mu0[j])[0])})
    df = pd.DataFrame(rows)
    # 取り除かれた理由：中心が確かな空に写った視点での、電線画素・復元電線の投影までの距離
    src = (HERE.parent / "sky_carve.py").read_text()
    ns = {"cv2": cv2, "np": np, "sky_and_wire": sky_and_wire}
    exec("def safe_sky" + src.split("def safe_sky")[1].split("\ndef ")[0], ns)
    sa = SP.safe_sky_args()
    ims = sorted(rec.images.values(), key=lambda i: i.name)[::2]      # sky_prune --views even と同じ
    votes = {int(j): [] for j in sel}
    for im in ims:
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Rc, tc = cfw.rotation.matrix(), np.asarray(cfw.translation)
        bgr = cv2.imread(str(SP.image_path(exp, im.name)))
        obs = np.array([p.xy for p in im.points2D if p.has_point3D()])
        sky = ns["safe_sky"](bgr, sa, obs) > 0
        wm, _, _ = sky_and_wire(bgr)
        v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        thin = ((wm > 0) | (bh > sa.bh_thr)).astype(np.uint8)
        dt_thin = cv2.distanceTransform(1 - thin, cv2.DIST_L2, 5)
        # 復元電線の投影（COLMAP 座標へ戻す）
        Wc = (al["R"].T @ ((W - al["t"]) / al["s"]).T).T
        pw = Wc @ Rc.T + tc
        okw = pw[:, 2] > 0.3
        uw = np.c_[fx * pw[okw, 0] / pw[okw, 2] + cx, fy * pw[okw, 1] / pw[okw, 2] + cy]
        tw = cKDTree(uw) if len(uw) else None
        H, Wd = sky.shape
        for j in sel:
            pc = Rc @ xyz0[j] + tc
            if pc[2] <= 1e-3:
                continue
            u, vv = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
            if not (0 <= u < Wd and 0 <= vv < H) or not sky[int(vv), int(u)]:
                continue
            votes[int(j)].append((float(dt_thin[int(vv), int(u)]), float(tw.query([u, vv])[0]) if tw else np.nan))
    df["n_votes"] = [len(votes[g]) for g in df.gid]
    df["px_to_detected_thin_med"] = [np.median([x[0] for x in votes[g]]) if votes[g] else np.nan for g in df.gid]
    df["px_to_recon_wire_med"] = [np.median([x[1] for x in votes[g]]) if votes[g] else np.nan for g in df.gid]
    pd.set_option("display.width", 220)
    print(df.describe(percentiles=[0.1, 0.5, 0.9]).T[["count", "10%", "50%", "90%"]].round(3).to_string())
    out = exp / f"why_prune_hits_wires_{a.model}.csv"
    df.to_csv(out, index=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
