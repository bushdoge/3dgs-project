# wirebench の解析スクリプトが共通で使う幾何ユーティリティ。
#   - COLMAP座標 → GT(Blender)座標の相似変換（Umeyama）。SfMが一部破綻している実験のために、
#     素朴な推定の RMSE が大きいときは外れカメラを除いてロバストに再推定し、外れカメラを報告する（監査R-07）
#   - 電線スパン（電柱近傍を除いた電線の中心線）の取得と、折れ線への線分距離（監査C2/R-13。
#     GTサンプルは13.7cm間隔と粗いので、点への最近傍距離だと電線上の点を約4割取りこぼす）
#   - Mip-Splatting の ply に保存された 3Dフィルタ（filter_3D）を反映した実効不透明度・実効スケール
#     （Mip のレンダラが実際に使う値。生の opacity を読むと Mip と素3DGSが同じ土俵にならない。監査R-06）
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ALIGN_OK_RMSE = 0.02   # [m] これを超えたら位置合わせ不良として扱う（研究メモの鉄則は1cm）


def umeyama(src, dst):
    """src→dst の相似変換 (s, R, t)。dst ≈ s R src + t。鏡映は除外する。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (sc ** 2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return s, R, t


def camera_pairs(exp, rec):
    """COLMAP再構成 rec と gt/poses.json から、画像名で対応するカメラ中心の組を返す。"""
    gt = json.load(open(Path(exp) / "gt" / "poses.json"))
    names, cc, gc = [], [], []
    for img in rec.images.values():
        stem = Path(img.name).stem
        if stem in gt["frames"]:
            names.append(stem)
            cc.append(img.projection_center())
            gc.append(np.array(gt["frames"][stem])[:3, 3])
    return names, np.array(cc), np.array(gc), gt


def align_colmap_to_gt(exp, rec):
    """カメラ中心から COLMAP→GT の相似変換を推定する。

    素朴な Umeyama の RMSE が ALIGN_OK_RMSE 以下ならそれをそのまま使う（従来と同一の結果）。
    超えた場合は、残差が max(5cm, 3×中央値) を超えるカメラを除いて再推定を繰り返し、
    インライアだけで推定した変換を返す。戻り値の ok は「インライアの RMSE が閾値以下か」。
    """
    names, C, G, gt = camera_pairs(exp, rec)
    s, R, t = umeyama(C, G)
    res = np.linalg.norm((s * (R @ C.T)).T + t - G, axis=1)
    rmse_naive = float(np.sqrt((res ** 2).mean()))
    keep = np.ones(len(C), bool)
    if rmse_naive > ALIGN_OK_RMSE:
        for _ in range(10):
            s, R, t = umeyama(C[keep], G[keep])
            res = np.linalg.norm((s * (R @ C.T)).T + t - G, axis=1)
            new = res < max(0.05, 3 * np.median(res))
            if new.sum() < 3 or (new == keep).all():
                break
            keep = new
    rmse = float(np.sqrt((res[keep] ** 2).mean()))
    return {"s": s, "R": R, "t": t, "rmse": rmse, "rmse_naive": rmse_naive,
            "ok": rmse <= ALIGN_OK_RMSE, "n_cams": len(C), "n_inliers": int(keep.sum()),
            "outlier_frames": [n for n, k in zip(names, keep) if not k], "gt": gt}


def apply(al, X):
    return (al["s"] * (al["R"] @ np.asarray(X, np.float64).T)).T + al["t"]


def wire_span(exp, min_pole_dist=0.3):
    """電線スパン部（電柱から min_pole_dist 以上離れた電線の中心線サンプル）と線半径を返す。"""
    exp = Path(exp)
    sp = json.load(open(exp / "gt" / "scene_params.json")) if (exp / "gt" / "scene_params.json").exists() else {}
    wp = np.load(exp / "gt" / "wire_points.npz")
    pts, labels = wp["points"], wp["labels"]
    poles = np.array(sp.get("pole_positions", [[-4.0, -1.8], [4.0, -1.2]]), dtype=float)[:, :2]
    wire = pts[labels == 1]
    far = np.min(np.linalg.norm(wire[:, None, :2] - poles[None], axis=2), axis=1) > min_pole_dist
    return wire[far], float(sp.get("wire_radius", 0.015)), far


def polyline_distance(query, poly_pts, gap=0.5, max_exact=None):
    """query の各点から、poly_pts が表す折れ線群（連続点間距離が gap を超える所で分割）への最短距離。

    max_exact を与えると、サンプル点への最近傍距離が max_exact を超える点は厳密計算を省き
    その最近傍距離（＝線分距離の上界）を返す（ガウシアン数百万個でも速く回すため）。
    """
    query = np.asarray(query, np.float64)
    best = np.full(len(query), np.inf)
    idx = np.arange(len(query))
    if max_exact is not None:
        nn = cKDTree(poly_pts).query(query)[0]
        best = nn.copy()
        idx = np.where(nn <= max_exact)[0]
    q = query[idx]
    sub = np.full(len(q), np.inf)
    breaks = np.where(np.linalg.norm(np.diff(poly_pts, axis=0), axis=1) > gap)[0] + 1
    for seg in np.split(poly_pts, breaks):
        if len(seg) < 2:
            continue
        a, b = seg[:-1], seg[1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-12)
        for k in range(0, len(q), 20000):        # (N,M) の一時配列を抑えるため分割
            qq = q[k:k + 20000]
            w = qq[:, None, :] - a[None]
            u = np.clip((w * ab[None]).sum(2) / L2[None], 0.0, 1.0)
            proj = a[None] + u[..., None] * ab[None]
            sub[k:k + 20000] = np.minimum(sub[k:k + 20000], np.linalg.norm(qq[:, None, :] - proj, axis=2).min(1))
    best[idx] = sub
    return best


def gaussian_params(vertex):
    """ply の vertex から (xyz, 生の不透明度, 生のスケール, 実効不透明度, 実効スケール, has_filter) を返す。

    Mip-Splatting は filter_3D (=f) を持ち、レンダ時に
      スケール → sqrt(s² + f²)、不透明度 → op × sqrt(Π s² / Π (s² + f²))
    を使う（gaussian_model.py の get_scaling_with_3D_filter / get_opacity_with_3D_filter）。
    filter_3D が無い ply（素3DGS等）では実効値＝生の値。
    """
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(vertex["opacity"], np.float64)))
    sc = np.exp(np.stack([vertex[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    names = vertex.data.dtype.names
    if "filter_3D" in names:
        f2 = np.asarray(vertex["filter_3D"], np.float64)[:, None] ** 2
        s2 = sc ** 2
        op_eff = op * np.sqrt(np.prod(s2 / (s2 + f2), axis=1))
        sc_eff = np.sqrt(s2 + f2)
        return xyz, op, sc, op_eff, sc_eff, True
    return xyz, op, sc, op, sc, False
