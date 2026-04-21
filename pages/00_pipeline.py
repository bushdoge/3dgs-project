# フレーム抽出→COLMAP→3DGS学習を一括で自動実行するパイプラインページ
# 各ステップの完了を自動検知して次のステップに進む

import json
import os
import re
import signal
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="パイプライン実行", page_icon="🚀", layout="wide")

PIPELINE_STATE_FILE = "/workspace/tmp/pipeline_state.json"

# ── パイプライン状態の永続化 ─────────────────────────────────────────────────

def save_pipeline_state(state: dict):
    """Popenオブジェクトを除いたパイプライン状態をJSONに保存する"""
    serializable = {k: v for k, v in state.items() if k != "proc"}
    Path(PIPELINE_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(PIPELINE_STATE_FILE).write_text(
        json.dumps(serializable, ensure_ascii=False), encoding="utf-8"
    )

def load_pipeline_state() -> dict:
    """保存されたパイプライン状態を読み込む。なければNoneを返す"""
    try:
        if Path(PIPELINE_STATE_FILE).exists():
            data = json.loads(Path(PIPELINE_STATE_FILE).read_text(encoding="utf-8"))
            data["proc"] = None  # 再起動後はPopenなし、pidで代替
            return data
    except Exception:
        pass
    return None

def clear_pipeline_state():
    """パイプライン状態ファイルを削除する"""
    p = Path(PIPELINE_STATE_FILE)
    if p.exists():
        p.unlink()

# ── セッション状態の初期化 ────────────────────────────────────────────────────
DEFAULT_PIPELINE = {
    "active": False,
    "step": "setup",       # setup / extracting / colmap / training / done / failed
    "step_status": {},     # {step_name: "running" | "done" | "error"}
    "experiment_dir": None,
    "video_path": None,
    # フレーム抽出
    "fps": 2.0,
    "is_360": False,
    "fov": 90,
    "out_w": 1024,
    "out_h": 1024,
    "angles": [(0, 0), (90, 0), (180, 0), (270, 0)],
    # 姿勢推定
    "use_hloc": False,
    "feature_type": "superpoint_max",
    "matcher_type": "superpoint+lightglue",
    "camera_model": "OPENCV",
    "use_gpu": True,
    # 学習
    "iterations": 30000,
    "save_iterations": [7000, 30000],
    "test_iterations": [7000, 30000],
    "proc": None,
    "pid": None,       # サブプロセスのPID（再起動後の復元用）
    "log_path": None,
    "error_msg": None,
    "start_time": None,
    "step_times": {},
}

if "pipeline" not in st.session_state:
    saved = load_pipeline_state()
    if saved and saved.get("active"):
        st.session_state.pipeline = saved
        st.session_state._pipeline_restored = True
    else:
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

def _pid_returncode(pid) -> int | None:
    """PIDのみでプロセスの終了を確認する（Streamlit再起動後の復元用）。
    実行中: None、終了（成功）: 0、終了（エラー）: 1 を返す。"""
    if not pid:
        return None
    try:
        os.kill(pid, 0)
        return None  # プロセスはまだ生きている
    except (ProcessLookupError, OSError):
        pass
    # プロセスが消えた → ログ末尾でエラー判定
    log_path = pl.get("log_path", "")
    if log_path and Path(log_path).exists():
        tail = Path(log_path).read_text(errors="replace").split("\n")[-15:]
        for line in tail:
            if "ERROR:" in line or ("error" in line.lower() and "failed" in line.lower()):
                return 1
    return 0


def advance_pipeline():
    """現在のステップを確認し、完了していれば次のステップを開始する"""
    proc = pl["proc"]
    pid  = pl.get("pid")

    # プロセスの終了コードを取得（Popenあり / PIDのみ の両方に対応）
    if proc is not None:
        retcode = proc.poll()
    else:
        retcode = _pid_returncode(pid)

    if retcode is None:
        return  # まだ実行中

    step = pl["step"]

    if retcode != 0:
        pl["step_status"][step] = "error"
        pl["error_msg"] = f"ステップ「{step}」がエラーで終了しました（終了コード: {retcode}）"
        pl["step"] = "failed"
        pl["proc"] = None
        save_pipeline_state(pl)
        return

    pl["step_status"][step] = "done"
    pl["step_times"][step]  = time.time()
    pl["proc"] = None

    if step == "extracting":
        start_colmap()
    elif step == "colmap":
        start_training()
    elif step == "training":
        pl["step"] = "done"
        pl["pid"]  = None
        save_pipeline_state(pl)


def start_colmap():
    exp_dir = pl["experiment_dir"]
    log_path = str(Path(exp_dir) / "colmap_log.txt")
    pl["log_path"] = log_path
    pl["step"] = "colmap"
    pl["step_status"]["colmap"] = "running"

    if pl["use_hloc"]:
        cmd = [
            sys.executable, "/workspace/scripts/run_hloc.py",
            "--source_path", exp_dir,
            "--feature_type", pl["feature_type"],
            "--matcher_type", pl["matcher_type"],
        ]
    else:
        cmd = [
            sys.executable, "/workspace/scripts/run_colmap.py",
            "--source_path", exp_dir,
            "--camera_model", pl["camera_model"],
        ]
        if not pl["use_gpu"]:
            cmd.append("--no_gpu")

    log_file = open(log_path, "w")
    pl["proc"] = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    pl["pid"]  = pl["proc"].pid
    save_pipeline_state(pl)


def start_training():
    exp_dir = pl["experiment_dir"]
    model_path = str(Path(exp_dir) / "output")
    log_path = str(Path(model_path) / "train_log.txt")
    os.makedirs(model_path, exist_ok=True)
    pl["log_path"] = log_path
    pl["step"] = "training"
    pl["step_status"]["training"] = "running"

    cmd = [
        sys.executable, "/workspace/scripts/run_train.py",
        "--source", exp_dir,
        "--model_path", model_path,
        "--iterations", str(pl["iterations"]),
        "--save_iterations", *[str(i) for i in pl["save_iterations"]],
        "--test_iterations", *[str(i) for i in pl["test_iterations"]],
    ]
    log_file = open(log_path, "w")
    pl["proc"] = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    pl["pid"]  = pl["proc"].pid
    save_pipeline_state(pl)


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
if st.session_state.get("_pipeline_restored"):
    st.info("🔄 Streamlit 再起動前のパイプライン実行状態を復元しました。そのまま監視を継続します。")
    del st.session_state["_pipeline_restored"]

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
            clear_pipeline_state()
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()
        st.stop()

    elif current_step == "failed":
        st.error(f"❌ {pl['error_msg']}")
        if pl["log_path"]:
            with st.expander("ログを確認"):
                st.text(get_log_tail(pl["log_path"], 50))
        if st.button("🔄 リセットして最初から"):
            clear_pipeline_state()
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()
        st.stop()

    else:
        step_name_ja = {"extracting": "フレーム抽出", "colmap": "COLMAP", "training": "3DGS学習"}.get(current_step, current_step)
        st.info(f"**{step_name_ja}** を実行中...")

        # ── ステップ別進捗バー ──────────────────────────────────────────────
        if pl["log_path"] and Path(pl["log_path"]).exists():
            _content = Path(pl["log_path"]).read_text(errors="replace")
            _pct, _bar_label = None, ""
            _colmap_step_names = {1: "特徴点抽出", 2: "マッチング", 3: "3D再構成", 4: "undistortion"}

            if current_step == "extracting":
                if pl.get("is_360"):
                    _m = re.findall(r'\[(\d+)/(\d+)\]', _content)
                    if _m:
                        _cur, _tot = int(_m[-1][0]), int(_m[-1][1])
                        _pct = min(_cur / _tot, 1.0)
                        _bar_label = f"フレーム変換: {_cur} / {_tot} 枚 ({_pct*100:.0f}%)"
                else:
                    _tm = re.search(r'PROGRESS_TOTAL (\d+)', _content)
                    _pm = re.findall(r'PROGRESS (\d+)/(\d+)', _content)
                    if _tm and _pm:
                        _tot = int(_tm.group(1))
                        _cur = int(_pm[-1][0])
                        if _tot > 0:
                            _pct = min(_cur / _tot, 1.0)
                            _bar_label = f"フレーム抽出: {_cur} / {_tot} 枚 ({_pct*100:.0f}%)"

            elif current_step == "colmap":
                if pl.get("use_hloc"):
                    _m = re.findall(r'\[(\d+)/4\]', _content)
                else:
                    _m = re.findall(r'\[COLMAP (\d+)/4\]', _content)
                if _m:
                    _cur = int(_m[-1])
                    _pct = min(_cur / 4, 1.0)
                    _bar_label = (f"ステップ {_cur}/4: {_colmap_step_names.get(_cur, '')} "
                                  f"({_pct*100:.0f}%)")

            elif current_step == "training":
                _total = pl.get("iterations", 30000)
                _tm = re.findall(rf'(\d+)/{_total}', _content)
                if not _tm:
                    _tm = re.findall(r'\[ITER\s+(\d+)\]', _content)
                if _tm:
                    _cur = int(_tm[-1])
                    _pct = min(_cur / _total, 1.0)
                    _bar_label = f"学習進捗: {_cur:,} / {_total:,} iter ({_pct*100:.0f}%)"

            if _pct is not None:
                st.caption(_bar_label)
                st.progress(_pct)
            else:
                st.caption("進捗を取得中...")

        # ログ表示
        if pl["log_path"]:
            log_tail = get_log_tail(pl["log_path"], 20)
            st.text_area("最新ログ", log_tail, height=200, label_visibility="visible")

        # 中断ボタン
        if st.button("⏹ パイプラインを中断"):
            proc = pl["proc"]
            pid  = pl.get("pid")
            kill_pid = None
            if proc and proc.poll() is None:
                kill_pid = proc.pid
            elif pid:
                kill_pid = pid
            if kill_pid:
                try:
                    os.kill(kill_pid, signal.SIGTERM)
                except Exception:
                    pass
            clear_pipeline_state()
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
for subdir in ("360movies", "movies"):
    for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        video_files += [str(p.relative_to("/workspace")) for p in (data_dir / subdir).rglob(ext)]
video_files = sorted(video_files)

if video_files:
    sel_video = st.selectbox("動画ファイル（data/360movies/ または data/movies/）", video_files)
    video_path = f"/workspace/{sel_video}"
else:
    st.warning("data/360movies/ または data/movies/ に動画が見つかりません。")
    video_path = st.text_input("動画ファイルのパスを入力",
                               placeholder="/workspace/data/movies/scene1.mp4")

is_360 = st.checkbox(
    "360度動画（ピンホール変換する）",
    help="等距円筒形式の360度動画の場合にチェックしてください。",
)

sel_angles = [(0, 0), (90, 0), (180, 0), (270, 0)]
fov_val, out_w_val, out_h_val = 90, 1024, 1024

if is_360:
    st.info("各フレームをピンホールカメラ視点に変換してからCOLMAPに渡します。")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fov_val = st.slider("水平視野角（FOV）", 60, 120, 90, 5)
    with col_b:
        out_size = st.selectbox("出力解像度", ["512×512", "1024×1024", "2048×2048"], index=1)
        out_w_val = out_h_val = int(out_size.split("×")[0])
    with col_c:
        fps = st.number_input("抽出FPS", min_value=0.1, max_value=10.0, value=1.0, step=0.5)

    # 8×3 方向選択グリッド（水平45°刻み × 垂直-30°/0°/+30°）
    st.markdown("**変換する方向を選択（水平角 × 垂直角）**")
    YAW_ANGLES   = [0, 45, 90, 135, 180, 225, 270, 315]
    PITCH_ANGLES = [30, 0, -30]
    YAW_SHORT    = ["前\n0°", "45°", "右\n90°", "135°", "後\n180°", "225°", "左\n270°", "315°"]
    PITCH_LABELS = {30: "上 +30°", 0: "水平  0°", -30: "下 −30°"}

    header_cols = st.columns([1.5] + [1] * 8)
    header_cols[0].write("")
    for i, label in enumerate(YAW_SHORT):
        header_cols[i + 1].markdown(label)

    angle_checks = {}
    for pitch in PITCH_ANGLES:
        row_cols = st.columns([1.5] + [1] * 8)
        row_cols[0].markdown(f"**{PITCH_LABELS[pitch]}**")
        for i, yaw in enumerate(YAW_ANGLES):
            angle_checks[(yaw, pitch)] = row_cols[i + 1].checkbox(
                "　",
                value=(pitch == 0),
                key=f"pl_360_y{yaw}_p{pitch}",
                label_visibility="collapsed",
            )

    sel_angles = [(y, p) for (y, p), v in angle_checks.items() if v]
else:
    fps = st.number_input("抽出FPS", min_value=0.1, max_value=30.0, value=2.0, step=0.5)

st.subheader("🏷️ 実験名の設定")

col3, col4 = st.columns(2)
with col3:
    scene_name = Path(video_path).stem if video_path else "scene"
    # 動画が変わったときだけデフォルト名を再生成（それ以外はユーザー編集を維持）
    if st.session_state.get("_exp_scene") != scene_name or "exp_name" not in st.session_state:
        st.session_state["_exp_scene"] = scene_name
        st.session_state["exp_name"] = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{scene_name}"
    exp_name = st.text_input("実験フォルダ名", key="exp_name")
    experiment_dir = f"/workspace/experiments/{exp_name}"
    st.caption(f"作成先: `{experiment_dir}`")

st.subheader("⚙️ 姿勢推定（Step 2）設定")

# プリセット
PRESETS = [
    ("COLMAP\n（標準）",         False, None,             None),
    ("SuperPoint\n+LightGlue",  True,  "superpoint_max", "superpoint+lightglue"),
    ("SuperPoint\n+SuperGlue",  True,  "superpoint_max", "superglue"),
    ("DISK\n+LightGlue",        True,  "disk",           "disk+lightglue"),
    ("SIFT\n+NN",               True,  "sift",           "NN-ratio"),
]
st.caption("▼ プリセット（クリックで下の設定に反映）")
preset_cols = st.columns(len(PRESETS))
for col, (label, hloc, feat, matcher) in zip(preset_cols, PRESETS):
    if col.button(label, use_container_width=True, key=f"pl_preset_{label}"):
        st.session_state["pl_use_hloc"] = hloc
        if feat:
            st.session_state["pl_feature"] = feat
        if matcher:
            st.session_state["pl_matcher"] = matcher
        st.rerun()

col_sfm1, col_sfm2, col_sfm3 = st.columns(3)

FEATURE_OPTIONS = ["superpoint_max", "superpoint_aachen", "disk", "aliked-n16",
                   "sift", "r2d2", "d2net-ss"]
MATCHER_OPTIONS = ["superpoint+lightglue", "disk+lightglue", "aliked+lightglue",
                   "superglue", "superglue-fast", "NN-superpoint", "NN-ratio",
                   "NN-mutual", "adalam"]

with col_sfm1:
    use_hloc = st.checkbox("HLocを使用", value=st.session_state.get("pl_use_hloc", False))
    st.session_state["pl_use_hloc"] = use_hloc
    if not use_hloc:
        camera_model = st.selectbox("カメラモデル", ["OPENCV", "PINHOLE", "SIMPLE_RADIAL"])
        use_gpu = st.checkbox("GPU使用", value=True)
    else:
        camera_model = "OPENCV"
        use_gpu = True

with col_sfm2:
    if use_hloc:
        feat_default = st.session_state.get("pl_feature", "superpoint_max")
        feat_idx = FEATURE_OPTIONS.index(feat_default) if feat_default in FEATURE_OPTIONS else 0
        feature_type = st.selectbox("特徴点抽出器", FEATURE_OPTIONS, index=feat_idx)
        st.session_state["pl_feature"] = feature_type
    else:
        st.selectbox("特徴点抽出器", ["SIFT（COLMAP内蔵）"], disabled=True)
        feature_type = "superpoint_max"

with col_sfm3:
    if use_hloc:
        match_default = st.session_state.get("pl_matcher", "superpoint+lightglue")
        match_idx = MATCHER_OPTIONS.index(match_default) if match_default in MATCHER_OPTIONS else 0
        matcher_type = st.selectbox("マッチャー", MATCHER_OPTIONS, index=match_idx)
        st.session_state["pl_matcher"] = matcher_type
    else:
        st.selectbox("マッチャー", ["SIFT最近傍（COLMAP内蔵）"], disabled=True)
        matcher_type = "superpoint+lightglue"

st.subheader("⚙️ 学習（Step 3）設定")

col6, col7, col8 = st.columns(3)
with col6:
    iterations = st.number_input("学習ステップ数", min_value=1000, max_value=100000,
                                 value=30000, step=1000)
with col7:
    save_iters_str = st.text_input("保存タイミング", value="7000,30000")
with col8:
    test_iters_str = st.text_input("評価タイミング", value="7000,30000")

save_iters = [int(s.strip()) for s in save_iters_str.split(",") if s.strip().isdigit()]
test_iters = [int(s.strip()) for s in test_iters_str.split(",") if s.strip().isdigit()]

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

        if is_360:
            cmd = [
                sys.executable, "/workspace/scripts/convert_360.py",
                "--input", video_path,
                "--output", input_dir,
                "--fov", str(fov_val),
                "--width", str(out_w_val),
                "--height", str(out_h_val),
                "--fps", str(fps),
                "--angles", *[f"{y},{p}" for y, p in sel_angles],
            ]
        else:
            cmd = [
                sys.executable, "/workspace/scripts/extract_frames.py",
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
            "is_360": is_360,
            "fov": fov_val,
            "out_w": out_w_val,
            "out_h": out_h_val,
            "angles": sel_angles,
            "use_hloc": use_hloc,
            "feature_type": feature_type,
            "matcher_type": matcher_type,
            "camera_model": camera_model,
            "use_gpu": use_gpu,
            "iterations": iterations,
            "save_iterations": save_iters or [7000, 30000],
            "test_iterations": test_iters or [7000, 30000],
            "proc": proc,
            "pid": proc.pid,
            "log_path": log_path,
            "start_time": time.time(),
        }
        save_pipeline_state(st.session_state.pipeline)
        st.rerun()
