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
    "pair_method": "exhaustive",
    "retrieval_model": "netvlad",
    "num_matched": 20,
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
    # /proc/<pid>/status でゾンビ(Z)を検出する。
    # os.kill(pid, 0) はゾンビに対しても成功してしまうため信頼できない。
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("State:"):
                    if "Z" in line:
                        break  # ゾンビ = 終了済み → ログ判定へ
                    return None  # 生きているプロセス
        except OSError:
            pass
    # プロセスが消えた or ゾンビ → ログ末尾でエラー判定
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


def _write_settings_header(log_file, step_label: str):
    """ログファイルの先頭に実験設定を書き込む"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    angles_str = "  ".join(f"y={y}/p={p}" for y, p in pl.get("angles", []))
    save_iters = ", ".join(str(i) for i in pl.get("save_iterations", []))
    test_iters = ", ".join(str(i) for i in pl.get("test_iterations", []))
    colmap_method = "HLoc" if pl.get("use_hloc") else "COLMAP"

    lines = [
        "=" * 64,
        f"  実験設定  [{step_label}]  {now}",
        "=" * 64,
        f"  実験ディレクトリ : {pl.get('experiment_dir', '-')}",
        f"  入力動画/ソース  : {pl.get('video_path', '-')}",
        "",
        "  [フレーム抽出]",
    ]
    if pl.get("is_360"):
        lines += [
            f"    360度変換 : あり  FOV={pl.get('fov')}°  {pl.get('out_w')}x{pl.get('out_h')}",
            f"    向き      : {angles_str if angles_str else '-'}",
        ]
    else:
        lines.append("    360度変換 : なし（通常動画）")
    lines.append(f"    FPS       : {pl.get('fps')}")
    lines += [
        "",
        f"  [カメラ推定 ({colmap_method})]",
    ]
    if pl.get("use_hloc"):
        lines += [
            f"    特徴量    : {pl.get('feature_type')}",
            f"    マッチング: {pl.get('matcher_type')}",
            f"    ペアリスト: {pl.get('pair_method')}  "
            f"(retrieval={pl.get('retrieval_model')}, {pl.get('num_matched')} pairs)",
        ]
    else:
        lines += [
            f"    カメラモデル: {pl.get('camera_model')}",
            f"    GPU         : {'あり' if pl.get('use_gpu') else 'なし'}",
        ]
    lines += [
        "",
        "  [3DGS学習]",
        f"    イテレーション : {pl.get('iterations')}",
        f"    保存           : {save_iters}",
        f"    テスト         : {test_iters}",
        "=" * 64,
        "",
    ]
    log_file.write("\n".join(lines) + "\n")
    log_file.flush()


def start_colmap():
    exp_dir = pl["experiment_dir"]
    log_path = str(Path(exp_dir) / "colmap_log.txt")
    pl["log_path"] = log_path
    pl["step"] = "colmap"
    pl["step_status"]["colmap"] = "running"

    if pl["use_hloc"]:
        cmd = [
            sys.executable, "/workspace/scripts/run_hloc.py",
            "--source_path",    exp_dir,
            "--feature_type",   pl["feature_type"],
            "--matcher_type",   pl["matcher_type"],
            "--pair_method",    pl.get("pair_method", "exhaustive"),
            "--retrieval_model", pl.get("retrieval_model", "netvlad"),
            "--num_matched",    str(pl.get("num_matched", 20)),
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
    _write_settings_header(log_file, "カメラ推定")
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
    _write_settings_header(log_file, "3DGS学習")
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

        # ── 全ステップ進捗バー ──────────────────────────────────────────────
        try:
            from pipeline_widget import (
                _parse_colmap_substeps, _render_substep_bars, _parse_progress,
            )
            _exp_dir = pl.get("experiment_dir", "")
            _step_logs = {
                "extracting": str(Path(_exp_dir) / "extract_log.txt"),
                "colmap":     str(Path(_exp_dir) / "colmap_log.txt"),
                "training":   str(Path(_exp_dir) / "output" / "train_log.txt"),
            }
            _STEP_LABELS = {
                "extracting": "① フレーム抽出",
                "colmap":     "② COLMAP / HLoc",
                "training":   "③ 3DGS 学習",
            }
            _COLOR = {"done": "#00cc66", "running": "#00aaff",
                      "error": "#ff4444", "waiting": "#334455"}
            _ICON  = {"done": "✅", "running": "🔄",
                      "error": "❌", "waiting": "⏳"}

            for _sk in ("extracting", "colmap", "training"):
                _st = pl["step_status"].get(_sk, "waiting")
                if current_step == _sk and _st != "done":
                    _st = "running"
                _c   = _COLOR.get(_st, "#334455")
                _ico = _ICON.get(_st, "⏳")
                st.markdown(
                    f'<span style="color:{_c};font-size:0.82rem;font-weight:bold;">'
                    f'{_ico} {_STEP_LABELS[_sk]}</span>',
                    unsafe_allow_html=True,
                )
                if _st == "waiting":
                    st.progress(0.0)
                    continue
                if _st == "done":
                    st.progress(1.0)
                    continue
                # running: ログから進捗を取得
                _pl_s = {**pl, "log_path": _step_logs[_sk], "step": _sk}
                if _sk == "colmap":
                    _subs = _parse_colmap_substeps(_pl_s)
                    if _subs:
                        _render_substep_bars(_subs)
                    else:
                        st.caption("ログを解析中...")
                else:
                    _pct, _lbl = _parse_progress(_pl_s)
                    if _pct is not None:
                        st.caption(_lbl)
                        st.progress(_pct)
                    else:
                        st.caption("進捗を取得中...")
        except Exception:
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

# ── HLoc ペアリスト生成方式 ────────────────────────────────────────────────────
RETRIEVAL_MODELS = ["netvlad", "openibl", "dir", "megaloc"]
RETRIEVAL_DESC = {
    "netvlad":  "VGG16-NetVLAD（定番・屋外向け）",
    "openibl":  "OpenIBL（NetVLAD改良版）",
    "dir":      "DIR（屋外シーン向け）",
    "megaloc":  "MegaLoc（大規模シーン向け）",
}
if use_hloc:
    st.markdown("**ペアリスト生成方式**")
    pair_cols = st.columns([1, 1])
    with pair_cols[0]:
        pair_method = st.radio(
            "方式",
            ["exhaustive", "retrieval"],
            index=0 if st.session_state.get("pl_pair_method", "exhaustive") == "exhaustive" else 1,
            format_func=lambda x: "Exhaustive（全ペア・精度優先）" if x == "exhaustive"
                                  else "Retrieval（類似画像のみ・速度優先）",
            label_visibility="collapsed",
        )
        st.session_state["pl_pair_method"] = pair_method
    with pair_cols[1]:
        if pair_method == "retrieval":
            ret_default = st.session_state.get("pl_retrieval_model", "netvlad")
            ret_idx = RETRIEVAL_MODELS.index(ret_default) if ret_default in RETRIEVAL_MODELS else 0
            retrieval_model = st.selectbox(
                "検索モデル", RETRIEVAL_MODELS, index=ret_idx,
                format_func=lambda x: RETRIEVAL_DESC.get(x, x),
            )
            st.session_state["pl_retrieval_model"] = retrieval_model
            num_matched = st.number_input(
                "top-K（1画像あたりのペア数）", min_value=5, max_value=100,
                value=st.session_state.get("pl_num_matched", 20), step=5,
            )
            st.session_state["pl_num_matched"] = num_matched
            n_images_est = len(list(Path("/workspace/experiments").rglob("input/*.jpg")))
            pairs_est = n_images_est * num_matched if n_images_est else 0
            if pairs_est:
                st.caption(f"推定ペア数: 約 {pairs_est:,}")
        else:
            retrieval_model = "netvlad"
            num_matched = 20
            st.caption("全ペア総当たり。\n画像が多いと非常に時間がかかります。")
else:
    pair_method     = "exhaustive"
    retrieval_model = "netvlad"
    num_matched     = 20

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

        # ヘッダー書き込みのために先にplを更新する（_write_settings_headerがplを参照するため）
        pl.update({
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
            "pair_method": pair_method,
            "retrieval_model": retrieval_model,
            "num_matched": num_matched,
            "camera_model": camera_model,
            "use_gpu": use_gpu,
            "iterations": iterations,
            "save_iterations": save_iters or [7000, 30000],
            "test_iterations": test_iters or [7000, 30000],
        })
        log_file = open(log_path, "w")
        _write_settings_header(log_file, "フレーム抽出")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        st.session_state.pipeline = {
            **DEFAULT_PIPELINE,
            **pl,
            "active": True,
            "step": "extracting",
            "step_status": {"extracting": "running"},
            "proc": proc,
            "pid": proc.pid,
            "log_path": log_path,
            "start_time": time.time(),
        }
        save_pipeline_state(st.session_state.pipeline)
        st.rerun()

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
