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


PIPELINE_STATE_FILE = "/workspace/tmp/pipeline_state.json"
PRESETS_FILE        = "/workspace/tmp/pipeline_presets.json"

# ── プリセット管理 ───────────────────────────────────────────────────────────

def load_presets() -> dict:
    try:
        p = Path(PRESETS_FILE)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_presets(presets: dict):
    Path(PRESETS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(PRESETS_FILE).write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
    "feature_type": "superpoint_aachen",
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
    "eval": False,
    "resolution": None,   # None = VRAM自動判定
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

if "pl_selected_test_iters" not in st.session_state:
    st.session_state["pl_selected_test_iters"] = {1000, 3000, 7000, 15000, 30000}


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
    text = p.read_text(errors="replace")
    # \r を改行として扱い、空行を除いて末尾n行を返す（省略なし）
    lines = [l for l in text.replace("\r", "\n").splitlines() if l.strip()]
    return "\n".join(lines[-n:])


# ════════════════════════════════════════════════════════════════════════════
#  ステップ進行ロジック
# ════════════════════════════════════════════════════════════════════════════

def _get_proc_starttime(pid) -> str | None:
    """/proc/<pid>/stat の starttime フィールドを返す（PID再利用検出用）"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.split()[21]  # 22番目フィールド（0-indexed: 21）
    except Exception:
        return None


def _pid_returncode(pid) -> int | None:
    """PIDのみでプロセスの終了を確認する（Streamlit再起動後の復元用）。
    実行中: None、終了（成功）: 0、終了（エラー）: 1 を返す。"""
    if not pid:
        return None

    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        try:
            # PID再利用チェック: 起動時刻が保存値と一致するか確認
            saved_starttime = pl.get("proc_starttime")
            if saved_starttime:
                current_starttime = _get_proc_starttime(pid)
                if current_starttime and current_starttime != str(saved_starttime):
                    # 別プロセスがPIDを再利用している → 元のプロセスは終了済み
                    pass  # ログ判定へ
                else:
                    # 同じプロセス or 時刻不明 → Stateで判断
                    for line in status_path.read_text().splitlines():
                        if line.startswith("State:"):
                            if "Z" in line:
                                break  # ゾンビ = 終了済み → ログ判定へ
                            return None  # 生きているプロセス
            else:
                # 起動時刻未保存（旧形式の状態ファイル）→ 従来どおり
                for line in status_path.read_text().splitlines():
                    if line.startswith("State:"):
                        if "Z" in line:
                            break
                        return None
        except OSError:
            pass

    # プロセスが消えた or ゾンビ or PID再利用 → ログ末尾でエラー判定
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
    res = pl.get("resolution")
    res_label = f"{res}x 縮小" if res else "自動（VRAM量から判定）"
    lines += [
        "",
        "  [3DGS学習]",
        f"    イテレーション : {pl.get('iterations')}",
        f"    保存           : {save_iters}",
        f"    テスト         : {test_iters}",
        f"    解像度         : {res_label}",
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
    pl["proc_starttime"] = _get_proc_starttime(pl["pid"])
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
    if pl.get("eval"):
        cmd.append("--eval")
    if pl.get("resolution") is not None:
        cmd += ["--resolution", str(pl["resolution"])]
    log_file = open(log_path, "w")
    _write_settings_header(log_file, "3DGS学習")
    pl["proc"] = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    pl["pid"]  = pl["proc"].pid
    pl["proc_starttime"] = _get_proc_starttime(pl["pid"])
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
#  実行ステータスバナー（設定フォームより上に常時表示）
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.get("_pipeline_restored"):
    st.info("🔄 Streamlit 再起動前のパイプライン実行状態を復元しました。")
    del st.session_state["_pipeline_restored"]

if pl["active"]:
    advance_pipeline()
    current_step = pl["step"]

    if current_step == "done":
        exp_dir = pl["experiment_dir"]
        elapsed = time.time() - pl["start_time"]
        st.success(
            f"🎉 パイプライン完了！　実験: `{Path(exp_dir).name}`　"
            f"総時間: {elapsed/60:.1f} 分　→「🖼️ 結果確認」で確認できます"
        )
        if st.button("✕ クリア", key="pl_clear_done"):
            clear_pipeline_state()
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()

    elif current_step == "failed":
        st.error(f"❌ {pl['error_msg']}")
        if st.button("✕ リセット", key="pl_clear_failed"):
            clear_pipeline_state()
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()

    else:
        step_ja = {"extracting": "フレーム抽出", "colmap": "COLMAP", "training": "3DGS学習"}.get(current_step, current_step)
        scene   = Path(pl.get("experiment_dir", "")).name
        elapsed = time.time() - pl.get("start_time", time.time())
        ba, bb  = st.columns([5, 1])
        ba.info(f"🔄 **{step_ja}** 実行中　｜　{scene}　｜　{elapsed/60:.1f} 分経過　→ 詳細は「🗂️ バッチキュー」ページ")
        if bb.button("⏹ 中断", key="pl_stop"):
            proc = pl["proc"]
            pid  = pl.get("pid")
            kill_pid = proc.pid if (proc and proc.poll() is None) else pid
            if kill_pid:
                try: os.kill(kill_pid, signal.SIGTERM)
                except Exception: pass
            clear_pipeline_state()
            st.session_state.pipeline = DEFAULT_PIPELINE.copy()
            st.rerun()
        time.sleep(3)
        st.rerun()

st.divider()

# ════════════════════════════════════════════════════════════════════════════
#  設定ビュー（常に表示）
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
        import sys as _sys2; _sys2.path.insert(0, "/workspace")
        from queue_helper import next_exp_name as _nxt
        st.session_state["exp_name"] = _nxt(scene_name)
    exp_name = st.text_input("実験フォルダ名", key="exp_name")
    experiment_dir = f"/workspace/experiments/{exp_name}"
    st.caption(f"作成先: `{experiment_dir}`")

# note.md エディタ（実験名に連動してデフォルト内容をリセット）
_note_key_marker = f"_note_loaded_for_{st.session_state.get('exp_name','')}"
if st.session_state.get("_note_loaded_marker") != _note_key_marker:
    _existing_note = Path(f"/workspace/experiments/{st.session_state.get('exp_name','')}/note.md")
    st.session_state["pl_note"] = _existing_note.read_text(encoding="utf-8") if _existing_note.exists() else ""
    st.session_state["_note_loaded_marker"] = _note_key_marker

st.text_area(
    "📝 事前メモ（note.md）",
    key="pl_note",
    height=120,
    placeholder="実験の目的・仮説・試したいこと・注意点など、自由に記入できます。",
    help="実験開始前にメモを残せます。実験フォルダ内の note.md に保存されます。",
)

# ── プリセット UI ─────────────────────────────────────────────────────────────
presets = load_presets()
with st.expander("📌 設定プリセット（保存・呼び出し）", expanded=False):
    st.caption("姿勢推定・学習パラメータをまとめて保存できます。"
               "下の設定を決めてから「💾 保存」、次回は「📂 読み込む」で即反映します。")
    if presets:
        pc1, pc2, pc3 = st.columns([4, 1, 1])
        with pc1:
            sel_preset = st.selectbox("保存済みプリセット", list(presets.keys()),
                                      label_visibility="collapsed")
        with pc2:
            if st.button("📂 読み込む", use_container_width=True, key="load_preset_btn"):
                p = presets[sel_preset]
                st.session_state["pl_use_hloc"]        = bool(p.get("use_hloc", False))
                st.session_state["pl_feature"]         = p.get("feature_type", "superpoint_aachen")
                st.session_state["pl_matcher"]         = p.get("matcher_type", "superpoint+lightglue")
                st.session_state["pl_pair_method"]     = p.get("pair_method", "exhaustive")
                st.session_state["pl_retrieval_model"] = p.get("retrieval_model", "netvlad")
                st.session_state["pl_num_matched"]     = int(p.get("num_matched", 20))
                st.session_state["pl_camera_model"]    = p.get("camera_model", "OPENCV")
                st.session_state["pl_use_gpu"]         = bool(p.get("use_gpu", True))
                st.session_state["pl_iterations"]      = int(p.get("iterations", 30000))
                save_iters_loaded = p.get("save_iterations", [7000, 30000])
                test_iters_loaded = p.get("test_iterations", [7000, 30000])
                st.session_state["pl_save_iters_str"]      = ",".join(str(i) for i in save_iters_loaded)
                st.session_state["pl_selected_test_iters"] = set(test_iters_loaded)
                st.session_state["pl_use_eval"]        = bool(p.get("eval", False))
                st.toast(f"プリセット「{sel_preset}」を読み込みました。")
                st.rerun()
        with pc3:
            if st.button("🗑️ 削除", use_container_width=True, key="delete_preset_btn"):
                del presets[sel_preset]
                save_presets(presets)
                st.toast(f"プリセット「{sel_preset}」を削除しました。")
                st.rerun()

        # プリセット内容をサマリー表示
        _p = presets.get(sel_preset, {})
        _sfm = "HLoc" if _p.get("use_hloc") else "COLMAP"
        _feat = _p.get("feature_type", "-") if _p.get("use_hloc") else _p.get("camera_model", "COLMAP内蔵")
        st.caption(
            f"姿勢推定: {_sfm} / {_feat}　｜　"
            f"学習: {_p.get('iterations', '-')} iter　｜　"
            f"保存: {_p.get('save_iterations', '-')}"
        )
        st.divider()
    else:
        st.info("保存済みプリセットがありません。")

    pn_col1, pn_col2 = st.columns([4, 1])
    with pn_col1:
        new_preset_name = st.text_input("プリセット名を入力して保存",
                                        placeholder="例: SuperPoint高精度30k / COLMAP標準",
                                        label_visibility="collapsed")
    with pn_col2:
        if st.button("💾 保存", use_container_width=True, key="save_preset_btn"):
            name = new_preset_name.strip()
            if name:
                _si      = st.session_state.get("pl_save_iters_str", "7000,30000")
                _sel_ti  = st.session_state.get("pl_selected_test_iters", {7000, 30000})
                cur = {
                    "use_hloc":        st.session_state.get("pl_use_hloc", False),
                    "feature_type":    st.session_state.get("pl_feature", "superpoint_aachen"),
                    "matcher_type":    st.session_state.get("pl_matcher", "superpoint+lightglue"),
                    "pair_method":     st.session_state.get("pl_pair_method", "exhaustive"),
                    "retrieval_model": st.session_state.get("pl_retrieval_model", "netvlad"),
                    "num_matched":     int(st.session_state.get("pl_num_matched", 20)),
                    "camera_model":    st.session_state.get("pl_camera_model", "OPENCV"),
                    "use_gpu":         bool(st.session_state.get("pl_use_gpu", True)),
                    "iterations":      int(st.session_state.get("pl_iterations", 30000)),
                    "save_iterations": [int(s.strip()) for s in _si.split(",") if s.strip().isdigit()],
                    "test_iterations": sorted(_sel_ti),
                }
                presets[name] = cur
                save_presets(presets)
                st.toast(f"プリセット「{name}」を保存しました。")
                st.rerun()
            else:
                st.warning("プリセット名を入力してください。")

st.subheader("⚙️ 姿勢推定（Step 2）設定")

# 特徴量プリセット
PRESETS = [
    ("COLMAP\n（標準）",         False, None,             None),
    ("SuperPoint\n+LightGlue",  True,  "superpoint_aachen", "superpoint+lightglue"),
    ("SuperPoint\n+SuperGlue",  True,  "superpoint_aachen", "superglue"),
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
        _cam_opts = ["OPENCV", "PINHOLE", "SIMPLE_RADIAL"]
        _cam_def  = st.session_state.get("pl_camera_model", "OPENCV")
        _cam_idx  = _cam_opts.index(_cam_def) if _cam_def in _cam_opts else 0
        camera_model = st.selectbox("カメラモデル", _cam_opts, index=_cam_idx)
        st.session_state["pl_camera_model"] = camera_model
        use_gpu = st.checkbox("GPU使用", value=st.session_state.get("pl_use_gpu", True))
        st.session_state["pl_use_gpu"] = use_gpu
    else:
        camera_model = "OPENCV"
        use_gpu = True

with col_sfm2:
    if use_hloc:
        feat_default = st.session_state.get("pl_feature", "superpoint_aachen")
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

col6, col7 = st.columns(2)
with col6:
    iterations = st.number_input("学習ステップ数", min_value=1000, max_value=100000,
                                 value=int(st.session_state.get("pl_iterations", 30000)),
                                 step=1000)
    st.session_state["pl_iterations"] = iterations
with col7:
    save_iters_str = st.text_input("保存タイミング（カンマ区切り）",
                                   value=st.session_state.get("pl_save_iters_str", "7000,30000"))
    st.session_state["pl_save_iters_str"] = save_iters_str

st.markdown("**評価タイミング（ボタンで選択 / 解除）**")
st.caption("クリックで切り替え。青 = 選択中。学習ステップ数を超えるボタンは表示されません。")

def _pl_toggle_test_iter(i):
    if i in st.session_state["pl_selected_test_iters"]:
        st.session_state["pl_selected_test_iters"].discard(i)
    else:
        st.session_state["pl_selected_test_iters"].add(i)

_PL_COLS_PER_ROW = 10
_pl_max_iter  = int(iterations)
_pl_all_iters = list(range(1000, _pl_max_iter + 1, 1000))

for _pl_row_start in range(0, len(_pl_all_iters), _PL_COLS_PER_ROW):
    _pl_row_iters = _pl_all_iters[_pl_row_start:_pl_row_start + _PL_COLS_PER_ROW]
    _pl_cols = st.columns(_PL_COLS_PER_ROW)
    for _pl_idx, _pl_i in enumerate(_pl_row_iters):
        _pl_sel = _pl_i in st.session_state["pl_selected_test_iters"]
        _pl_cols[_pl_idx].button(
            f"{_pl_i // 1000}k",
            key=f"pl_test_iter_btn_{_pl_i}",
            type="primary" if _pl_sel else "secondary",
            on_click=_pl_toggle_test_iter,
            args=(_pl_i,),
            use_container_width=True,
        )

_pl_sel_display = sorted(i for i in st.session_state["pl_selected_test_iters"] if i <= _pl_max_iter)
st.caption(f"選択中: {', '.join(str(i) for i in _pl_sel_display) if _pl_sel_display else '（なし）'}")

save_iters = [int(s.strip()) for s in save_iters_str.split(",") if s.strip().isdigit()]
test_iters = sorted(i for i in st.session_state["pl_selected_test_iters"] if i <= int(iterations))

use_eval = st.checkbox(
    "train/test 分割を有効にする（--eval）",
    value=st.session_state.get("pl_use_eval", False),
    help="ONにすると8枚に1枚をtestデータとして学習から除外し、未学習視点でPSNRを評価します。研究・比較目的に推奨。",
)
st.session_state["pl_use_eval"] = use_eval
if use_eval:
    st.caption("📊 testデータ: 入力画像の約12.5%（8枚に1枚）が自動で割り当てられます。学習には残り87.5%が使われます。")
else:
    st.caption("📊 全フレームを学習に使用します。test PSNRは計算されません。")

_res_options = {
    "自動（VRAM量から自動判定）★推奨": None,
    "元解像度のまま（1x）": 1,
    "1/2 に縮小（2x）": 2,
    "1/4 に縮小（4x）": 4,
    "1/8 に縮小（8x）": 8,
}
_res_default = st.session_state.get("pl_resolution", None)
_res_label_default = next((k for k, v in _res_options.items() if v == _res_default),
                          "自動（VRAM量から自動判定）★推奨")
_res_label = st.selectbox(
    "画像解像度（--resolution）",
    list(_res_options.keys()),
    index=list(_res_options.keys()).index(_res_label_default),
    help="「自動」にするとVRAMと画像枚数・サイズからOOMにならない倍率を自動計算します。",
)
resolution = _res_options[_res_label]
if resolution == 1:
    resolution = None  # 1x = 縮小なし = None扱い
st.session_state["pl_resolution"] = resolution

st.divider()

# ── キューに追加 ──────────────────────────────────────────────────────────────
import sys as _sys; _sys.path.insert(0, "/workspace")
from queue_helper import add_to_queue as _add_q, pending_size as _psize

if st.button(
    f"📋 バッチキューに追加（待ち: {_psize()} 件）",
    disabled=not (video_path and exp_name),
    use_container_width=True,
):
    if not os.path.exists(video_path):
        st.error(f"動画ファイルが見つかりません: {video_path}")
    else:
        _add_q(
            job_type="pipeline",
            label=f"パイプライン {fps}fps / {int(iterations):,}iter",
            exp_name=exp_name,
            exp_dir=experiment_dir,
            config={
                "video_path":      video_path,
                "fps":             float(fps),
                "is_360":          is_360,
                "fov":             fov_val,
                "out_w":           out_w_val,
                "out_h":           out_h_val,
                "angles":          sel_angles,
                "use_hloc":        use_hloc,
                "camera_model":    camera_model,
                "feature_type":    feature_type,
                "matcher_type":    matcher_type,
                "pair_method":     pair_method,
                "retrieval_model": retrieval_model,
                "num_matched":     num_matched,
                "iterations":      int(iterations),
                "save_iterations": save_iters or [7000, int(iterations)],
                "test_iterations": test_iters or [1000, 3000, 7000, 15000, int(iterations)],
                "eval":            use_eval,
                "resolution":      resolution,
                "note_md":         st.session_state.get("pl_note", ""),
            },
        )
        st.success(f"「{exp_name}」をバッチキューに追加しました。「🗂️ バッチキュー」ページから実行できます。")

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

        note_content = st.session_state.get("pl_note", "")
        note_path = Path(experiment_dir) / "note.md"
        if note_content or not note_path.exists():
            note_path.write_text(note_content, encoding="utf-8")

        pipeline_cfg = {
            "saved_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_path":      video_path,
            "fps":             float(fps),
            "is_360":          is_360,
            "fov":             fov_val,
            "out_w":           out_w_val,
            "out_h":           out_h_val,
            "angles":          sel_angles,
            "use_hloc":        use_hloc,
            "feature_type":    feature_type,
            "matcher_type":    matcher_type,
            "pair_method":     pair_method,
            "retrieval_model": retrieval_model,
            "num_matched":     num_matched,
            "camera_model":    camera_model,
            "use_gpu":         use_gpu,
            "iterations":      int(iterations),
            "save_iterations": save_iters or [7000, 30000],
            "test_iterations": test_iters or [7000, 30000],
            "eval":            use_eval,
            "resolution":      resolution,
        }
        import json as _json_pl
        (Path(experiment_dir) / "pipeline_config.json").write_text(
            _json_pl.dumps(pipeline_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
            "eval": use_eval,
            "resolution": resolution,
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

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### パイプラインの流れ

```
[Step 1] 動画 / 360度動画 → フレーム抽出（FFmpeg）
[Step 2] フレーム群 → カメラ姿勢推定（COLMAP または HLoc）
[Step 3] 姿勢推定結果 → 3DGS学習
```

全ステップを自動で順番に実行します。途中のステップから再開も可能です。

---

### 入力種別

| 種別 | 配置場所 | 備考 |
|---|---|---|
| 通常動画 | `data/movies/` | mp4, mov など |
| 360度動画 | `data/360movies/` | 等距円筒（Equirectangular）形式 |

---

### 姿勢推定の選択

| 方式 | 特徴 | 推奨シーン |
|---|---|---|
| **COLMAP**（デフォルト） | 汎用・安定 | 手持ち撮影・一般的なシーン |
| **HLoc** | 高精度・SuperPoint使用 | 難しいシーン・繰り返しパターン |

---

### 学習パラメータ

| パラメータ | 説明 | 目安 |
|---|---|---|
| 学習ステップ数 | 多いほど高品質・長時間 | 確認7k、標準30k、高品質100k |
| 保存タイミング | カンマ区切りで指定。そのステップのモデルをディスクに保存 | `7000,30000` |
| 評価タイミング | 1000刻みのボタンで選択。PSNR等のスコアを計算してグラフ化 | 好きなステップを複数選択 |
| train/test 分割（--eval） | 8枚に1枚をtest用に確保、未学習視点でPSNR評価 | 研究・比較目的に推奨 |

---

### 評価タイミング（test_iterations）とは
- PSNR・SSIM・LPIPS のスコアを計算してログに記録するタイミングです
- モデルの保存はしないのでストレージを圧迫しません
- 点を多く選ぶほど学習曲線グラフが滑らかになります
- 学習ステップ数を超えるボタンは自動で非表示になります

---

### プリセット
よく使う設定を名前をつけて保存できます。評価タイミングの選択状態もプリセットに含まれます。
""")