# COLMAPを使ってカメラ姿勢推定（Structure from Motion）を実行するページ
# gaussian-splatting/convert.py を経由して実行する。入力は experiment/input/、出力は experiment/sparse/0/
# COLMAP完了後にカメラ位置の3D可視化も表示する

import sys
import streamlit as st
import subprocess
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, "/opt/gaussian-splatting")
from scene.colmap_loader import read_extrinsics_binary, read_points3D_binary, qvec2rotmat

st.set_page_config(page_title="COLMAP実行", page_icon="📷", layout="wide")

st.title("📷 COLMAP実行")
st.caption("フレーム画像からカメラ姿勢を推定します（Structure from Motion）")

st.divider()

# ── 入力設定 ──────────────────────────────────────────────────────────────────
st.subheader("入力設定")

experiments_dir = Path("/workspace/experiments")
exp_dirs = []
if experiments_dir.exists():
    # input/ フォルダが存在し、かつ画像が入っている実験ディレクトリを列挙
    for p in sorted(experiments_dir.iterdir()):
        if p.is_dir():
            inp = p / "input"
            if inp.exists() and (list(inp.glob("*.jpg")) or list(inp.glob("*.png"))):
                n = len(list(inp.glob("*.jpg"))) + len(list(inp.glob("*.png")))
                exp_dirs.append((str(p), n))

col1, col2 = st.columns(2)

with col1:
    if exp_dirs:
        labels = [f"{Path(d).name}  （{n}枚）" for d, n in exp_dirs]
        idx = st.selectbox("実験フォルダ（input/ フォルダを含むもの）",
                           range(len(exp_dirs)), format_func=lambda i: labels[i])
        source_path = exp_dirs[idx][0]
    else:
        st.warning("experiments/ 配下にフレームが入った input/ フォルダが見つかりません。\n"
                   "先にフレーム抽出を実行してください。")
        source_path = st.text_input("実験フォルダのパスを直接入力",
                                    placeholder="/workspace/experiments/20240101_120000_scene1")

with col2:
    camera_model = st.selectbox(
        "カメラモデル",
        ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
        help="通常はOPENCV推奨。360度変換済みや合成画像はPINHOLE。",
    )

# ── 詳細設定 ──────────────────────────────────────────────────────────────────
with st.expander("詳細設定"):
    use_gpu = st.checkbox("GPU使用（SIFT特徴点抽出）", value=True)

# ── 出力先の説明 ──────────────────────────────────────────────────────────────
if source_path:
    st.info(
        f"**出力先（自動）:**\n"
        f"- `{source_path}/sparse/0/` — カメラ姿勢（疎な点群）\n"
        f"- `{source_path}/images/` — アンディストート済み画像"
    )

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")

cmd_args = [
    "python /workspace/scripts/run_colmap.py",
    f'--source_path "{source_path}"',
    f"--camera_model {camera_model}",
]
if not use_gpu:
    cmd_args.append("--no_gpu")

st.code(" \\\n  ".join(cmd_args), language="bash")

# ── 既存結果の確認 ────────────────────────────────────────────────────────────
if source_path:
    sparse_done = Path(source_path) / "sparse" / "0"
    if sparse_done.exists():
        st.success("✅ すでにCOLMAPの出力（sparse/0/）が存在します。再実行すると上書きされます。")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()
st.warning("⚠️ COLMAPはGPUを使用します。フレーム数によって数分〜数十分かかります。")

if st.button("▶ COLMAPを実行", type="primary", disabled=not source_path):
    input_dir = Path(source_path) / "input"
    if not input_dir.exists():
        st.error(f"input/ フォルダが見つかりません: {input_dir}")
    else:
        st.info("COLMAPを実行中です。しばらくお待ちください...")

        run_args = ["python", "/workspace/scripts/run_colmap.py",
                    "--source_path", source_path,
                    "--camera_model", camera_model]
        if not use_gpu:
            run_args.append("--no_gpu")

        with st.spinner("COLMAP実行中（数分〜数十分かかります）..."):
            result = subprocess.run(run_args, capture_output=True, text=True)

        if result.returncode == 0:
            sparse_dir = Path(source_path) / "sparse" / "0"
            if sparse_dir.exists():
                n_cameras = len(list(sparse_dir.glob("*.bin")))
                st.success("COLMAP が正常に完了しました！")
                st.metric("sparse/0/ のファイル数", n_cameras)
                st.info(f"次のステップ：「🧠 3DGS学習実行」ページで `{source_path}` を選択してください。")
            else:
                st.warning("完了しましたが sparse/0/ が見つかりません。ログを確認してください。")
        else:
            st.error("COLMAPの実行中にエラーが発生しました。")

        with st.expander("ログを表示"):
            st.text(result.stdout or "（出力なし）")
            if result.stderr:
                st.text("STDERR:\n" + result.stderr)

# ── COLMAP結果の可視化 ────────────────────────────────────────────────────────
if source_path:
    sparse_dir = Path(source_path) / "sparse" / "0"
    images_bin = sparse_dir / "images.bin"
    points_bin = sparse_dir / "points3D.bin"

    if images_bin.exists():
        st.divider()
        st.subheader("📍 COLMAP結果の可視化")
        st.caption("推定されたカメラ位置（青）と疎な点群（灰）を3D表示します")

        try:
            import plotly.graph_objects as go

            # カメラ位置の取得
            images_data = read_extrinsics_binary(str(images_bin))
            cam_positions = []
            for img in images_data.values():
                R = qvec2rotmat(img.qvec)
                t = np.array(img.tvec)
                pos = -R.T @ t
                cam_positions.append(pos)
            cam_positions = np.array(cam_positions)

            traces = []

            # 疎な点群
            if points_bin.exists():
                pts = read_points3D_binary(str(points_bin))
                if pts:
                    xyz = np.array([p.xyz for p in pts.values()])
                    rgb = np.array([p.rgb for p in pts.values()])
                    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]

                    # 点が多い場合は間引く
                    max_pts = 50000
                    if len(xyz) > max_pts:
                        idx = np.random.choice(len(xyz), max_pts, replace=False)
                        xyz = xyz[idx]
                        colors = [colors[i] for i in idx]

                    traces.append(go.Scatter3d(
                        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                        mode="markers",
                        marker=dict(size=1, color=colors, opacity=0.5),
                        name=f"点群（{len(xyz):,}点）",
                    ))

            # カメラ位置
            traces.append(go.Scatter3d(
                x=cam_positions[:, 0],
                y=cam_positions[:, 1],
                z=cam_positions[:, 2],
                mode="markers+text",
                marker=dict(size=5, color="blue", symbol="circle"),
                name=f"カメラ（{len(cam_positions)}台）",
            ))

            fig = go.Figure(data=traces)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(x=0, y=1),
                scene=dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                    aspectmode="data",
                ),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("推定カメラ数", len(cam_positions))
            if points_bin.exists() and pts:
                c2.metric("疎な点群数", f"{len(pts):,} 点")

        except Exception as e:
            st.warning(f"可視化に失敗しました: {e}")
