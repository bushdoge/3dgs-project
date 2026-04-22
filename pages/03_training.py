# 3D Gaussian Splattingの学習をtrain.pyで実行するページ
# リアルタイムでログ・Loss・PSNRを表示する。学習中断ボタン付き。

import os
import re
import signal
import time
import subprocess
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="3DGS学習実行", page_icon="🧠", layout="wide")

# ── セッション状態の初期化 ────────────────────────────────────────────────────
if "train_proc" not in st.session_state:
    st.session_state.train_proc = None       # Popen オブジェクト
    st.session_state.train_log_path = None   # ログファイルパス
    st.session_state.train_model_path = None # モデル出力先
    st.session_state.train_source = None     # 実験ディレクトリ
    st.session_state.train_iterations = 30000


def is_training():
    proc = st.session_state.train_proc
    return proc is not None and proc.poll() is None


def parse_log(log_text):
    """ログからLossとPSNRを抽出してリストで返す"""
    # tqdmのキャリッジリターン除去
    cleaned = "\n".join(
        line.split("\r")[-1] for line in log_text.split("\n")
    )

    # Loss抽出（tqdmの進捗バーから）
    losses = []
    for m in re.finditer(r"Loss:\s*([\d.]+)", cleaned):
        losses.append(float(m.group(1)))

    # PSNR抽出（評価ステップのログ行から）
    psnr_records = []
    for m in re.finditer(
        r"\[ITER (\d+)\] Evaluating (\w+): L1 [^\s]+ PSNR [^(\n]*\(([\d.]+)\)",
        cleaned,
    ):
        psnr_records.append({
            "iteration": int(m.group(1)),
            "split": m.group(2),
            "PSNR": float(m.group(3)),
        })

    return losses, psnr_records, cleaned


# ════════════════════════════════════════════════════════════════════════════
#  学習中の進捗ビュー
# ════════════════════════════════════════════════════════════════════════════
if is_training() or (
    st.session_state.train_proc is not None
    and st.session_state.train_proc.poll() is not None
):
    proc = st.session_state.train_proc
    running = proc.poll() is None

    if running:
        st.title("🧠 3DGS 学習実行")
        st.markdown(
            '<span style="color:#00cc66">● 学習中</span>',
            unsafe_allow_html=True,
        )
    else:
        st.title("🧠 3DGS 学習実行")
        if proc.returncode == 0:
            st.success("✅ 学習が完了しました！")
        else:
            st.error(f"❌ エラーで終了しました（終了コード: {proc.returncode}）")

    # ── ログ読み込み ──
    log_path = Path(st.session_state.train_log_path)
    log_text = ""
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")

    losses, psnr_records, cleaned_log = parse_log(log_text)

    # ── 進捗バー ──
    iters = st.session_state.train_iterations
    if losses:
        # tqdmのパーセント表示から現在ステップを推定
        m = re.search(r"(\d+)/(\d+)", cleaned_log.split("\n")[-50:][0] if cleaned_log else "")
        current_iter = 0
        for line in reversed(cleaned_log.split("\n")):
            m2 = re.search(r"(\d+)/" + str(iters), line)
            if m2:
                current_iter = int(m2.group(1))
                break
        if current_iter > 0:
            st.progress(min(current_iter / iters, 1.0),
                        text=f"{current_iter:,} / {iters:,} ステップ")

    # ── Loss グラフ ──
    if losses:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("📉 Loss（直近500点）")
            display_losses = losses[-500:]
            st.line_chart(
                {"Loss": display_losses},
                height=200,
            )

        # ── PSNR グラフ ──
        with col_g2:
            if psnr_records:
                st.subheader("📈 PSNR（評価ステップ）")
                import pandas as pd
                df = pd.DataFrame(psnr_records)
                pivot = df.pivot(index="iteration", columns="split", values="PSNR")
                st.line_chart(pivot, height=200)
            else:
                st.subheader("📈 PSNR（評価ステップ）")
                st.caption("評価タイミング（test_iterations）になると表示されます")

    # ── 最新ログ ──
    with st.expander("📋 学習ログ（末尾）", expanded=False):
        tail = "\n".join(cleaned_log.split("\n")[-80:])
        st.text(tail or "（まだログがありません）")

    # ── 操作ボタン ──
    st.divider()
    col_b1, col_b2 = st.columns([1, 5])
    with col_b1:
        if running:
            if st.button("⏹ 学習を中断", type="secondary"):
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
                st.session_state.train_proc = None
                st.rerun()
        else:
            if st.button("← 設定画面に戻る"):
                st.session_state.train_proc = None
                st.rerun()

    # 学習中は自動リフレッシュ
    if running:
        time.sleep(3)
        st.rerun()

    st.stop()


# ════════════════════════════════════════════════════════════════════════════
#  設定ビュー（学習前）
# ════════════════════════════════════════════════════════════════════════════
st.title("🧠 3DGS 学習実行")
st.caption("3D Gaussian Splatting の学習を実行します（gaussian-splatting/train.py 使用）")

st.divider()

# ── 入力設定 ──────────────────────────────────────────────────────────────────
st.subheader("入力設定（COLMAPの出力フォルダを指定）")

experiments_dir = Path("/workspace/experiments")
ready_dirs = []
if experiments_dir.exists():
    for p in sorted(experiments_dir.iterdir()):
        if p.is_dir() and (p / "sparse" / "0").exists():
            ready_dirs.append(str(p))

if ready_dirs:
    source_path = st.selectbox(
        "実験フォルダ（sparse/0/ が含まれるもの）",
        ready_dirs,
        format_func=lambda x: Path(x).name,
    )
else:
    st.warning("COLMAPの出力（sparse/0/）が見つかりません。先にCOLMAP実行を完了してください。")
    source_path = st.text_input(
        "実験フォルダのパスを直接入力",
        placeholder="/workspace/experiments/20240101_120000_scene1",
    )

# ── 学習パラメータ ────────────────────────────────────────────────────────────
st.subheader("学習パラメータ")

col1, col2, col3 = st.columns(3)

with col1:
    iterations = st.number_input(
        "学習ステップ数", min_value=1000, max_value=100000,
        value=30000, step=1000,
        help="デフォルトは30000。短時間で試すなら7000程度でも可。",
    )

with col2:
    save_iterations = st.text_input(
        "保存タイミング（カンマ区切り）", value="7000,30000",
        help="指定ステップ数ごとにチェックポイントを保存します",
    )

with col3:
    test_iterations = st.text_input(
        "評価タイミング（カンマ区切り）", value="7000,30000",
        help="PSNR等の評価を行うステップ数",
    )

# ── 出力設定 ──────────────────────────────────────────────────────────────────
st.subheader("出力設定")

default_model = str(Path(source_path) / "output") if source_path else ""
model_path = st.text_input("モデル出力フォルダ", value=default_model)

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")

save_list = [s.strip() for s in save_iterations.split(",") if s.strip()]
test_list = [t.strip() for t in test_iterations.split(",") if t.strip()]

cmd_parts = [
    "python /workspace/scripts/run_train.py",
    f'--source "{source_path}"',
    f'--model_path "{model_path}"',
    f"--iterations {iterations}",
]
if save_list:
    cmd_parts.append("--save_iterations " + " ".join(save_list))
if test_list:
    cmd_parts.append("--test_iterations " + " ".join(test_list))

st.code(" \\\n  ".join(cmd_parts), language="bash")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()

st.error("⚠️ 学習はGPUを長時間占有します。他の処理が動いていないか確認してから実行してください。")

confirm = st.checkbox("確認しました。学習を開始します。")

if st.button("▶ 学習を開始", type="primary",
             disabled=not (source_path and model_path and confirm)):
    if not os.path.exists(source_path):
        st.error(f"フォルダが見つかりません: {source_path}")
    else:
        os.makedirs(model_path, exist_ok=True)
        log_path = str(Path(model_path) / "train_log.txt")

        run_args = [
            "python", "/workspace/scripts/run_train.py",
            "--source", source_path,
            "--model_path", model_path,
            "--iterations", str(iterations),
        ]
        if save_list:
            run_args += ["--save_iterations"] + save_list
        if test_list:
            run_args += ["--test_iterations"] + test_list

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            run_args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        st.session_state.train_proc = proc
        st.session_state.train_log_path = log_path
        st.session_state.train_model_path = model_path
        st.session_state.train_source = source_path
        st.session_state.train_iterations = iterations

        st.rerun()

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
