# パイプライン設定ページ
# 実験設定を行い、バッチキューに追加する。実行はデーモンまたはキューページが担当する。

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, "/workspace")

PRESETS_FILE     = "/workspace/tmp/pipeline_presets.json"
DAEMON_PID_FILE  = "/workspace/tmp/batch_daemon.pid"

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

# ── デーモン状態 ──────────────────────────────────────────────────────────────

def _daemon_alive() -> bool:
    try:
        p = Path(DAEMON_PID_FILE)
        if not p.exists():
            return False
        pid = int(p.read_text().strip())
        status = Path(f"/proc/{pid}/status")
        if not status.exists():
            return False
        for line in status.read_text().splitlines():
            if line.startswith("State:"):
                return "Z" not in line
        return False
    except Exception:
        return False

def _start_daemon():
    import subprocess
    subprocess.Popen(
        [sys.executable, "/workspace/scripts/batch_daemon.py"],
        stdout=open("/workspace/tmp/batch_daemon.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

@st.cache_data(ttl=600)
def _video_duration_sec(path: str) -> float:
    """ffprobeで動画の長さ（秒）を取得する。失敗時は0"""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0

# ── セッション状態初期化 ──────────────────────────────────────────────────────

if "pl_selected_test_iters" not in st.session_state:
    st.session_state["pl_selected_test_iters"] = {1000, 3000, 7000, 15000, 30000}

# ════════════════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════════════════
st.title("🚀 パイプライン設定")
st.caption("実験を設定してキューに追加します。実行はバックグラウンドデーモンが自動で行います。")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
#  設定ビュー
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
    if st.session_state.get("_exp_scene") != scene_name or "exp_name" not in st.session_state:
        st.session_state["_exp_scene"] = scene_name
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

PRESETS = [
    ("COLMAP\n（標準）",         False, None,               None),
    ("SuperPoint\n+LightGlue",  True,  "superpoint_aachen", "superpoint+lightglue"),
    ("SuperPoint\n+SuperGlue",  True,  "superpoint_aachen", "superglue"),
    ("DISK\n+LightGlue",        True,  "disk",              "disk+lightglue"),
    ("SIFT\n+NN",               True,  "sift",              "NN-ratio"),
]
st.caption("▼ プリセット（クリックで下の設定に反映）")
preset_cols = st.columns(len(PRESETS))
for col, (label, hloc, feat, matcher) in zip(preset_cols, PRESETS):
    if col.button(label, use_container_width=True, key=f"pl_preset_{label}"):
        st.session_state["pl_use_hloc"] = hloc
        if feat:    st.session_state["pl_feature"] = feat
        if matcher: st.session_state["pl_matcher"] = matcher
        st.rerun()

col_sfm1, col_sfm2, col_sfm3 = st.columns(3)

import sys as _sys_hloc
if "/workspace" not in _sys_hloc.path:
    _sys_hloc.path.insert(0, "/workspace")
from hloc_options import FEATURE_OPTIONS, MATCHER_OPTIONS, FEATURE_DESC, MATCHER_DESC

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
        st.caption(FEATURE_DESC.get(feature_type, ""))
    else:
        st.selectbox("特徴点抽出器", ["SIFT（COLMAP内蔵）"], disabled=True)
        st.caption("SIFT — 古典的な特徴点アルゴリズム。高速だが深層学習ベースより複雑なシーンでは精度が落ちる。")
        feature_type = "superpoint_max"

with col_sfm3:
    if use_hloc:
        match_default = st.session_state.get("pl_matcher", "superpoint+lightglue")
        match_idx = MATCHER_OPTIONS.index(match_default) if match_default in MATCHER_OPTIONS else 0
        matcher_type = st.selectbox("マッチャー", MATCHER_OPTIONS, index=match_idx)
        st.session_state["pl_matcher"] = matcher_type
        st.caption(MATCHER_DESC.get(matcher_type, ""))
    else:
        st.selectbox("マッチャー", ["SIFT最近傍（COLMAP内蔵）"], disabled=True)
        st.caption("比率テスト付き最近傍マッチング — SIFT と組み合わせる古典的手法。高速・軽量。")
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
            # 入力動画の長さ×FPS×方向数から、これから抽出されるフレーム数を見積もる
            _dur = _video_duration_sec(video_path) if video_path and os.path.exists(video_path) else 0.0
            if _dur > 0:
                n_images_est = max(1, int(_dur * fps)) * (len(sel_angles) if is_360 else 1)
                st.caption(f"推定: 約 {n_images_est:,} 枚 → 約 {n_images_est * num_matched:,} ペア"
                           f"（全ペアなら {n_images_est * (n_images_est - 1) // 2:,}）")
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

_ti_label_col, _ti_all_col, _ti_none_col = st.columns([6, 1, 1])
_ti_label_col.markdown("**評価タイミング（ボタンで選択 / 解除）**")
if _ti_all_col.button("全選択", use_container_width=True, key="pl_test_iter_all"):
    st.session_state["pl_selected_test_iters"] = set(range(1000, int(iterations) + 1, 1000))
if _ti_none_col.button("全解除", use_container_width=True, key="pl_test_iter_none"):
    st.session_state["pl_selected_test_iters"] = set()
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
# 1x はそのまま --resolution 1 として渡す（Noneにすると自動判定になり勝手に縮小されうる）
resolution = _res_options[_res_label]
st.session_state["pl_resolution"] = resolution

st.divider()

# ── キューに追加して実行 ──────────────────────────────────────────────────────
from queue_helper import add_to_queue as _add_q, pending_size as _psize

_job_config = {
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
}

_can_submit = bool(video_path and exp_name)

if st.button(
    f"🚀 キューに追加（待ち: {_psize()} 件）",
    disabled=not _can_submit,
    use_container_width=True,
    type="primary",
):
    if not os.path.exists(video_path):
        st.error(f"動画ファイルが見つかりません: {video_path}")
    else:
        _add_q(
            job_type="pipeline",
            label=f"パイプライン {fps}fps / {int(iterations):,}iter",
            exp_name=exp_name,
            exp_dir=experiment_dir,
            config=_job_config,
        )
        # デーモンが止まっていれば自動起動
        if not _daemon_alive():
            _start_daemon()
            st.success(f"「{exp_name}」をキューに追加し、デーモンを起動しました。ブラウザを閉じても実行が続きます。")
        else:
            st.success(f"「{exp_name}」をキューに追加しました。デーモンが自動で実行します。")

st.divider()

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### パイプラインの流れ

```
[Step 1] 動画 / 360度動画 → フレーム抽出（FFmpeg）
[Step 2] フレーム群 → カメラ姿勢推定（COLMAP または HLoc）
[Step 3] 姿勢推定結果 → 3DGS学習
```

設定してキューに追加すると、バックグラウンドデーモンが自動でステップを順番に実行します。

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

### プリセット
よく使う設定を名前をつけて保存できます。評価タイミングの選択状態もプリセットに含まれます。
""")
