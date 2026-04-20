# 3DGS学習結果（レンダリング画像・ログ・ファイル構造）を確認するページ

import streamlit as st
import re
from pathlib import Path

st.set_page_config(page_title="結果確認", page_icon="🖼️", layout="wide")

st.title("🖼️ 結果確認")
st.caption("学習結果のレンダリング画像・ログ・ファイル構造を確認します")

st.divider()

# ── 実験フォルダ選択 ──────────────────────────────────────────────────────────
experiments_dir = Path("/workspace/experiments")
experiment_list = []
if experiments_dir.exists():
    experiment_list = sorted(
        [str(p) for p in experiments_dir.iterdir() if p.is_dir()],
        reverse=True,
    )

if not experiment_list:
    st.warning("experiments/ フォルダに実験結果が見つかりません。")
    st.stop()

selected_exp = st.selectbox(
    "実験フォルダを選択",
    experiment_list,
    format_func=lambda x: Path(x).name,
)
exp_path = Path(selected_exp)

st.divider()

# ── フォルダ内の状態サマリー ──────────────────────────────────────────────────
st.subheader("📁 パイプラインの進捗")

def count_images(folder):
    p = Path(folder)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".png", ".jpeg"})

col1, col2, col3, col4 = st.columns(4)

with col1:
    n_input = count_images(exp_path / "input")
    st.metric("入力フレーム", f"{n_input} 枚" if n_input else "未実行")

with col2:
    has_colmap = (exp_path / "sparse" / "0").exists()
    st.metric("COLMAP", "✅ 完了" if has_colmap else "❌ 未実行")

with col3:
    has_output = (exp_path / "output").exists()
    # 最新のpoint_cloudを探す
    ply_files = list((exp_path / "output").rglob("*.ply")) if has_output else []
    st.metric("3DGS学習", f"✅ {len(ply_files)}ply" if ply_files else ("⏳ フォルダあり" if has_output else "❌ 未実行"))

with col4:
    n_renders = count_images(exp_path / "renders")
    # gaussian-splatting の render.py 出力先も探す
    if n_renders == 0 and has_output:
        n_renders = sum(
            count_images(p)
            for p in (exp_path / "output").rglob("renders")
            if p.is_dir()
        )
    st.metric("レンダリング画像", f"{n_renders} 枚" if n_renders else "なし")

# ── point_cloud 情報 ─────────────────────────────────────────────────────────
if has_output:
    st.divider()
    st.subheader("💾 保存済みpoint_cloud")

    pc_dir = exp_path / "output" / "point_cloud"
    if pc_dir.exists():
        iters = sorted(pc_dir.iterdir(), key=lambda p: p.name)
        cols = st.columns(len(iters)) if iters else []
        for col, it_dir in zip(cols, iters):
            ply = it_dir / "point_cloud.ply"
            size_mb = ply.stat().st_size / 1e6 if ply.exists() else 0
            col.metric(it_dir.name, f"{size_mb:.1f} MB" if ply.exists() else "なし")
    else:
        st.info("point_cloud/ フォルダがまだありません。")

# ── 学習ログとPSNR ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 学習ログ・メトリクス")

log_files = (
    list((exp_path / "output").rglob("*.txt")) +
    list((exp_path / "logs").rglob("*.txt")) +
    list((exp_path / "output").rglob("*.log"))
) if (exp_path / "output").exists() or (exp_path / "logs").exists() else []

# train_log.txt を優先
train_logs = [f for f in log_files if "train_log" in f.name]
log_files = train_logs + [f for f in log_files if f not in train_logs]

if log_files:
    selected_log = st.selectbox("ログファイルを選択", log_files,
                                format_func=lambda x: str(x.relative_to(exp_path)))
    log_content = selected_log.read_text(errors="replace")

    # PSNR を抽出して表示
    psnr_records = []
    for m in re.finditer(
        r"\[ITER (\d+)\] Evaluating (\w+): L1 [^\s]+ PSNR [^(\n]*\(([\d.]+)\)",
        log_content,
    ):
        psnr_records.append({
            "iteration": int(m.group(1)),
            "split": m.group(2),
            "PSNR": float(m.group(3)),
        })

    if psnr_records:
        import pandas as pd
        st.subheader("📈 PSNR推移")
        df = pd.DataFrame(psnr_records)
        pivot = df.pivot(index="iteration", columns="split", values="PSNR")
        st.line_chart(pivot)
        st.dataframe(df, use_container_width=True)

    with st.expander("ログ全文"):
        st.text_area("", log_content, height=300, label_visibility="collapsed")
else:
    st.info("ログファイルが見つかりません。")

# ── レンダリング画像 ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🖼️ レンダリング画像")

# renders/ または output/**/renders/ を探す
render_dirs = [exp_path / "renders"]
if has_output:
    render_dirs += list((exp_path / "output").rglob("renders"))

images = []
for rd in render_dirs:
    if rd.is_dir():
        images += sorted(rd.rglob("*.png")) + sorted(rd.rglob("*.jpg"))

if images:
    cols_per_row = st.slider("1行あたりの表示枚数", 2, 6, 4)
    for i in range(0, min(len(images), 24), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(images):
                col.image(str(images[i + j]),
                          caption=images[i + j].name,
                          use_container_width=True)
else:
    st.info("レンダリング画像が見つかりません。render.py を実行すると生成されます。")

# ── config.yaml ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("⚙️ 実験設定（config.yaml）")

config_path = exp_path / "config.yaml"
if config_path.exists():
    with open(config_path) as f:
        st.code(f.read(), language="yaml")
else:
    # gaussian-splatting が生成する cfg_args も探す
    cfg_args = list((exp_path / "output").rglob("cfg_args")) if has_output else []
    if cfg_args:
        st.code(cfg_args[0].read_text(), language="text")
    else:
        st.info("config ファイルが見つかりません。")
