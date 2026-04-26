# 3DGS PLYファイルをブラウザ内で3D表示するページ
# point_cloud.ply から位置・色・不透明度を読み取り Plotly で可視化する

import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go

st.title("🔭 PLY ビューア")
st.caption("学習済みの 3D Gaussian Splat をブラウザ内で確認できます")

st.divider()

# ─── PLY リーダー ─────────────────────────────────────────────────────────────

def read_3dgs_ply(path: Path):
    """
    3DGS フォーマットの PLY を読み込み、numpy structured array を返す。
    binary_little_endian / binary_big_endian / ascii に対応。
    """
    with open(path, "rb") as f:
        # ヘッダー解析
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break

        header = header_bytes.decode("ascii", errors="ignore")
        lines  = header.splitlines()

        # エンコーディング
        encoding = "binary_little_endian"
        for l in lines:
            if l.startswith("format"):
                encoding = l.split()[1]
                break

        # 頂点数
        n_verts = 0
        for l in lines:
            if l.startswith("element vertex"):
                n_verts = int(l.split()[-1])
                break

        # プロパティ定義
        _dtype_map = {
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
            "int": "i4",   "int32": "i4",   "uint": "u4", "uint32": "u4",
            "short": "i2", "ushort": "u2",
            "char": "i1",  "uchar": "u1",
        }
        props = []
        in_vertex = False
        for l in lines:
            if l.startswith("element vertex"):
                in_vertex = True
            elif l.startswith("element") and not l.startswith("element vertex"):
                in_vertex = False
            elif l.startswith("property") and in_vertex:
                parts = l.split()
                dtype = _dtype_map.get(parts[1], "f4")
                props.append((parts[2], dtype))

        np_dtype = np.dtype(props)

        # データ読み込み
        if encoding == "ascii":
            data = np.zeros(n_verts, dtype=np_dtype)
            for i in range(n_verts):
                vals = f.readline().split()
                for j, (name, _) in enumerate(props):
                    data[name][i] = float(vals[j])
        elif encoding == "binary_big_endian":
            raw  = f.read(n_verts * np_dtype.itemsize)
            data = np.frombuffer(raw, dtype=np_dtype.newbyteorder(">"))
        else:
            raw  = f.read(n_verts * np_dtype.itemsize)
            data = np.frombuffer(raw, dtype=np_dtype)

    return data


def extract_colors(data):
    """DC 球面調和係数から RGB を計算する"""
    SH_C0 = 0.28209479177387814
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    r = np.clip(sigmoid(0.5 + SH_C0 * data["f_dc_0"].astype(np.float32)), 0, 1)
    g = np.clip(sigmoid(0.5 + SH_C0 * data["f_dc_1"].astype(np.float32)), 0, 1)
    b = np.clip(sigmoid(0.5 + SH_C0 * data["f_dc_2"].astype(np.float32)), 0, 1)
    a = np.clip(sigmoid(data["opacity"].astype(np.float32)), 0, 1)
    return r, g, b, a


# ─── 実験・イテレーション選択 ─────────────────────────────────────────────────

experiments_dir = Path("/workspace/experiments")
exp_list = sorted(
    [p for p in experiments_dir.iterdir() if p.is_dir()],
    reverse=True,
) if experiments_dir.exists() else []

if not exp_list:
    st.warning("experiments/ に実験が見つかりません。")
    st.stop()

sel_exp = st.selectbox(
    "実験フォルダ",
    exp_list,
    format_func=lambda p: p.name,
)

pc_dir   = sel_exp / "output" / "point_cloud"
iter_dirs = sorted(
    [d for d in pc_dir.iterdir() if (d / "point_cloud.ply").exists()],
    key=lambda d: d.name,
) if pc_dir.exists() else []

if not iter_dirs:
    st.info("学習済み point_cloud.ply が見つかりません。先に 3DGS 学習を完了してください。")
    st.stop()

sel_iter = st.selectbox(
    "イテレーション",
    iter_dirs,
    index=len(iter_dirs) - 1,
    format_func=lambda d: d.name.replace("iteration_", "iter "),
)
ply_path = sel_iter / "point_cloud.ply"

# ─── 表示設定 ─────────────────────────────────────────────────────────────────
vc1, vc2, vc3 = st.columns(3)
with vc1:
    max_pts = st.slider("最大表示点数", 10_000, 300_000, 100_000, 10_000,
                        help="多いほど詳細・重くなります")
with vc2:
    opacity_thr = st.slider("不透明度フィルタ", 0.0, 0.5, 0.05, 0.01,
                             help="これ以下のガウシアンは非表示")
with vc3:
    pt_size = st.slider("点サイズ", 1, 5, 2)

# ─── ロード & 描画 ────────────────────────────────────────────────────────────
ply_mb = ply_path.stat().st_size / 1e6
st.caption(f"PLYファイル: `{ply_path.relative_to(sel_exp)}`　({ply_mb:.1f} MB)")

if st.button("🔭 3D表示", type="primary", use_container_width=True):
    with st.spinner("PLY を読み込んでいます…"):
        try:
            data = read_3dgs_ply(ply_path)
        except Exception as e:
            st.error(f"読み込みエラー: {e}")
            st.stop()

    st.session_state["ply_data"]    = data
    st.session_state["ply_ply_path"] = str(ply_path)

# キャッシュ済みデータがあれば描画
if "ply_data" in st.session_state and st.session_state.get("ply_ply_path") == str(ply_path):
    data = st.session_state["ply_data"]
    n_total = len(data)

    # 不透明度フィルタ
    with st.spinner("色・不透明度を計算中…"):
        r, g, b, alpha = extract_colors(data)

    mask  = alpha > opacity_thr
    x_f   = data["x"][mask].astype(np.float32)
    y_f   = data["y"][mask].astype(np.float32)
    z_f   = data["z"][mask].astype(np.float32)
    r_f, g_f, b_f, a_f = r[mask], g[mask], b[mask], alpha[mask]
    n_filtered = len(x_f)

    # サブサンプリング（不透明度で重み付け）
    if n_filtered > max_pts:
        probs = a_f / a_f.sum()
        idx   = np.random.choice(n_filtered, max_pts, replace=False, p=probs)
        x_f, y_f, z_f = x_f[idx], y_f[idx], z_f[idx]
        r_f, g_f, b_f = r_f[idx], g_f[idx], b_f[idx]

    n_show = len(x_f)

    # 統計表示
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("総ガウシアン数", f"{n_total:,}")
    mc2.metric("フィルタ後",     f"{n_filtered:,}")
    mc3.metric("表示点数",       f"{n_show:,}")
    mc4.metric("PLYサイズ",      f"{ply_mb:.1f} MB")

    # 色配列（ベクトル演算で高速化）
    with st.spinner("描画中…"):
        rgb_int  = np.stack([r_f, g_f, b_f], axis=1)
        rgb_int  = (rgb_int * 255).astype(np.uint8)
        hex_list = [
            "#{:02x}{:02x}{:02x}".format(ri, gi, bi)
            for ri, gi, bi in rgb_int
        ]

        fig = go.Figure(data=[go.Scatter3d(
            x=x_f, y=y_f, z=z_f,
            mode="markers",
            marker=dict(
                size=pt_size,
                color=hex_list,
                opacity=0.85,
            ),
            hoverinfo="none",
        )])
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                bgcolor="#0a0e18",
                aspectmode="data",
            ),
            paper_bgcolor="#0a0e18",
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
        )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("マウスドラッグで回転 / スクロールでズーム / 右ドラッグで平行移動")
else:
    st.info("「🔭 3D表示」ボタンを押すと点群が表示されます。")

# ─── 固定フッター ─────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
