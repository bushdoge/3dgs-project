# フレーム抽出→COLMAP→3DGS学習を一括で自動実行するパイプラインページ
# 各ステップの完了を自動検知して次のステップに進む

import os
import signal
import time
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="パイプライン実行", page_icon="🚀", layout="wide")

# ── セッション状態の初期化 ────────────────────────────────────────────────────
DEFAULT_PIPELINE = {
    "active": False,
    "step": "setup",       # setup / extracting / colmap / training / done / failed
    "step_status": {},     # {step_name: "running" | "done" | "error"}
    "experiment_dir": None,
    "video_path": None,
    "fps": 2.0,
    "camera_model": "OPENCV",
    "iterations": 30000,
    "save_iterations": [7000, 30000],
    "test_iterations": [7000, 30000],
    "proc": None,          # 現在実行中の Popen
    "log_path": None,
    "error_msg": None,
    "start_time": None,
    "step_times": {},
}

if "pipeline" not in st.session_state:
    st.session_state.pipeline = DEFAULT_PIPELINE.copy()

pl = st.session_state.pipeline


def step_badge(name, status):
    colors = {
        "waiting": ("#888", "⏳"),
        "running": ("#00aaff", "🔄"),
        "done": ("#00cc66", "✅"),
        "error": ("#ff4444", "❌"),
    }
    color, icon = colors.get(status, ("#888", "⏳"))
    st.markdown(
        f'<span style="color:{color};font-weight:bold">{icon} {name}</span>',
        unsafe_allow_html=True,
    )


def get_log_tail(log_path, n=30):
    p = Path(log_path)
    if not p.exists():
        return ""
    lines = p.read_text(errors="replace").split("\n")
    return "\n".join(lines[-n:])


# ════════════════════════════════════════════════════════════════════════════
#  ステップ進行ロジック
# ════════════════════════════════════════════════════════════════════════════
def advance_pipeline():
    """現在のステップを確認し、完了していれば次のステップを開始する"""
    proc = pl["proc"]
    if proc is None:
        return

    retcode = proc.poll()
    if retcode is None:
        return  # まだ実行中

    step = pl["step"]

    if retcode != 0:
        pl["step_status"][step] = "error"
        pl["error_msg"] = f"ステップ「{step}」がエラーで終了しました（終了コード: {retcode}）"
        pl["step"] = "failed"
        pl["proc"] = None
        return

    pl["step_status"][step] = "done"
    pl["step_times"][step] = time.time()

    if step == "extracting":
        start_colmap()
    elif step == "colmap":
        start_training()
    elif step == "training":
        pl["step"] = "done"
        pl["proc"] = None


def start_colmap():
    exp_dir = pl["experiment_dir"]
    log_path = str(Path(exp_dir) / "colmap_log.txt")
    pl["log_path"] = log_path
    pl["step"] = "colmap"
    pl["step_status"]["colmap"] = "running"

    cmd = [
        "python", "/workspace/scripts/run_colmap.py",
        "--source_path", exp_dir,
        "--camera_model", pl["camera_model"],
    ]
    log_file = open(log_path, "w")
    pl["proc"] = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def start_training():
    exp_dir = pl["experiment_dir"]
    model_path = str(Path(exp_dir) / "output")
    log_path = str(Path(model_path) / "train_log.txt")
    os.makedirs(model_path, exist_ok=True)
    pl["log_path"] = log_path
    pl["step"] = "training"
    pl["step_status"]["training"] = "running"

    cmd = [
        "python", "/workspace/scripts/run_train.py",
        "--source", exp_dir,
        "--model_path", model_path,
        "--iterations", str(pl["iterations"]),
        "--save_iterations", *[str(i) for i in pl["save_iterations"]],
        "--test_iterations", *[str(i) for i in pl["test_iterations"]],
    ]
    log_file = open(log_path, "w")
    pl["proc"] = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


# ════════════════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════════════════
st.title("🚀 パイプライン実行")
st.caption("フレーム抽出 → COLMAP → 3DGS学習 を自動で順番に実行します")

# ── ステップ進捗バー ──────────────────────────────────────────────────────────
steps = ["extracting", "colmap", "training"]
step_labels = ["① フレーム抽出", "② COLMAP", "③ 3DGS学習"]

col_steps = st.columns(3)
for col, step, label in zip(col_steps, steps, step_labels):
    with col:
        status = pl["step_status"].get(step, "waiting")
        if pl["step"] == step and status != "done":
            status = "running"
        step_badge(label, status)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
#  実行中・完了ビュー
# ════════════════════════════════════════════════════════════════════════════
if pl["active"]:
    advance_pipeline()

    current_step = pl["step"]

    if current_step == "done":
        st.success("🎉 パイプラインが完了しました！")
        exp_dir = pl["experiment_dir"]
        st.info(f"実験フォルダ: `{exp_dir}`\n\n「🖼️ 結果確認」ページで結果を確認してください。")

        elapsed = time.time() - pl["start_time"]
        st.metric("総実行時間", f"{elapsed/60:.1f} 分")

        if st.button("🔄 新しいパイプラインを開始"):
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()
        st.stop()

    elif current_step == "failed":
        st.error(f"❌ {pl['error_msg']}")
        if pl["log_path"]:
            with st.expander("ログを確認"):
                st.text(get_log_tail(pl["log_path"], 50))
        if st.button("🔄 リセットして最初から"):
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()
        st.stop()

    else:
        step_name_ja = {"extracting": "フレーム抽出", "colmap": "COLMAP", "training": "3DGS学習"}.get(current_step, current_step)
        st.info(f"**{step_name_ja}** を実行中...")

        # ログ表示
        if pl["log_path"]:
            log_tail = get_log_tail(pl["log_path"], 20)
            st.text_area("最新ログ", log_tail, height=200, label_visibility="visible")

        # 中断ボタン
        if st.button("⏹ パイプラインを中断"):
            proc = pl["proc"]
            if proc and proc.poll() is None:
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()

        time.sleep(3)
        st.rerun()

    st.stop()


# ════════════════════════════════════════════════════════════════════════════
#  設定ビュー（パイプライン開始前）
# ════════════════════════════════════════════════════════════════════════════
st.subheader("🎬 入力動画の選択")

data_dir = Path("/workspace/data")
video_files = []
for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
    video_files += [str(p.relative_to("/workspace")) for p in data_dir.rglob(ext)]
video_files = sorted(video_files)

col1, col2 = st.columns(2)
with col1:
    if video_files:
        sel_video = st.selectbox("動画ファイル", video_files)
        video_path = f"/workspace/{sel_video}"
    else:
        st.warning("data/ 配下に動画が見つかりません。")
        video_path = st.text_input("動画ファイルのパスを入力",
                                   placeholder="/workspace/data/scene1/video.mp4")

with col2:
    fps = st.number_input("抽出FPS", min_value=0.1, max_value=30.0, value=2.0, step=0.5)

st.subheader("🏷️ 実験名の設定")

col3, col4 = st.columns(2)
with col3:
    scene_name = Path(video_path).parent.name if video_path else "scene"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{timestamp}_{scene_name}"
    exp_name = st.text_input("実験フォルダ名", value=default_name)
    experiment_dir = f"/workspace/experiments/{exp_name}"
    st.caption(f"作成先: `{experiment_dir}`")

st.subheader("⚙️ COLMAP・学習パラメータ")

col5, col6, col7 = st.columns(3)
with col5:
    camera_model = st.selectbox("カメラモデル", ["OPENCV", "PINHOLE", "SIMPLE_RADIAL"])
with col6:
    iterations = st.number_input("学習ステップ数", min_value=1000, max_value=100000,
                                 value=30000, step=1000)
with col7:
    save_iters_str = st.text_input("保存タイミング", value="7000,30000")

save_iters = [int(s.strip()) for s in save_iters_str.split(",") if s.strip().isdigit()]

st.divider()

st.error("⚠️ 学習はGPUを長時間占有します。他の処理が動いていないか確認してください。")
confirm = st.checkbox("確認しました。パイプラインを開始します。")

if st.button("🚀 パイプラインを開始", type="primary",
             disabled=not (video_path and exp_name and confirm)):
    if not os.path.exists(video_path):
        st.error(f"動画ファイルが見つかりません: {video_path}")
    else:
        input_dir = str(Path(experiment_dir) / "input")
        log_path = str(Path(experiment_dir) / "extract_log.txt")
        os.makedirs(input_dir, exist_ok=True)

        cmd = [
            "python", "/workspace/scripts/extract_frames.py",
            "--input", video_path,
            "--output", input_dir,
            "--fps", str(fps),
        ]
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        st.session_state.pipeline = {
            **DEFAULT_PIPELINE,
            "active": True,
            "step": "extracting",
            "step_status": {"extracting": "running"},
            "experiment_dir": experiment_dir,
            "video_path": video_path,
            "fps": fps,
            "camera_model": camera_model,
            "iterations": iterations,
            "save_iterations": save_iters or [7000, 30000],
            "test_iterations": save_iters or [7000, 30000],
            "proc": proc,
            "log_path": log_path,
            "start_time": time.time(),
        }
        st.rerun()
