# COLMAPまたはHLocでカメラ姿勢推定（Structure from Motion）を実行するページ
# バックエンドをプリセット or 自由組み合わせで選択できる。完了後にカメラ位置の3D可視化も表示する

import os
import re
import signal
import sys
import subprocess
import time
import numpy as np
from pathlib import Path

import streamlit as st

sys.path.insert(0, "/opt/gaussian-splatting")
from scene.colmap_loader import read_extrinsics_binary, read_points3D_binary, qvec2rotmat


# ── セッション状態の初期化 ────────────────────────────────────────────────────
if "colmap_proc"        not in st.session_state: st.session_state.colmap_proc        = None
if "colmap_log_path"    not in st.session_state: st.session_state.colmap_log_path    = None
if "colmap_source_path" not in st.session_state: st.session_state.colmap_source_path = None
if "colmap_use_hloc"    not in st.session_state: st.session_state.colmap_use_hloc    = False
if "sfm_use_hloc"       not in st.session_state: st.session_state.sfm_use_hloc       = False
if "sfm_feature"        not in st.session_state: st.session_state.sfm_feature        = "superpoint_max"
if "sfm_matcher"        not in st.session_state: st.session_state.sfm_matcher        = "superpoint+lightglue"
if "sfm_pair_method"    not in st.session_state: st.session_state.sfm_pair_method    = "exhaustive"
if "sfm_retrieval_model" not in st.session_state: st.session_state.sfm_retrieval_model = "netvlad"
if "sfm_num_matched"    not in st.session_state: st.session_state.sfm_num_matched    = 20


def _colmap_running():
    p = st.session_state.colmap_proc
    return p is not None and p.poll() is None


# ════════════════════════════════════════════════════════════════════════════
#  実行中 / 完了ビュー
# ════════════════════════════════════════════════════════════════════════════
if _colmap_running() or (
    st.session_state.colmap_proc is not None
    and st.session_state.colmap_proc.poll() is not None
):
    proc    = st.session_state.colmap_proc
    running = proc.poll() is None

    st.title("📷 姿勢推定（COLMAP / HLoc）")
    if running:
        st.markdown('<span style="color:#00cc66">● 実行中</span>', unsafe_allow_html=True)
    elif proc.returncode == 0:
        st.success("✅ 姿勢推定完了！")
    else:
        st.error(f"❌ エラーで終了しました（終了コード: {proc.returncode}）")

    log_path = Path(st.session_state.colmap_log_path or "")
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""

    # サブステップ進捗（pipeline_widgetを再利用）
    try:
        from pipeline_widget import _parse_colmap_substeps, _render_substep_bars, _parse_progress
        _state = {
            "step":      "colmap",
            "log_path":  str(log_path),
            "use_hloc":  st.session_state.colmap_use_hloc,
        }
        substeps = _parse_colmap_substeps(_state)
        if substeps:
            _render_substep_bars(substeps)
        else:
            pct, label = _parse_progress(_state)
            if pct is not None:
                st.progress(pct, text=label)
            else:
                st.caption("ログを解析中...")
    except Exception:
        pass

    with st.expander("📋 ログ（末尾）", expanded=False):
        lines = [l for l in log_text.replace("\r", "\n").split("\n") if l.strip()]
        st.code("\n".join(lines[-30:]) or "（まだログがありません）", language=None)

    st.divider()
    c1, _ = st.columns([1, 5])
    with c1:
        if running:
            if st.button("⏹ 中断", type="secondary"):
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except Exception:
                    pass
                st.session_state.colmap_proc = None
                st.session_state.pop("active_task", None)
                st.rerun()
        else:
            if st.button("← 設定に戻る"):
                st.session_state.colmap_proc = None
                st.session_state.pop("active_task", None)
                st.rerun()

    if running:
        time.sleep(3)
        st.rerun()

    try:
        from pipeline_widget import render_sticky_footer
        render_sticky_footer()
    except Exception:
        pass
    st.stop()


st.title("📷 姿勢推定（COLMAP / HLoc）")
st.caption("フレーム画像からカメラ姿勢を推定します（Structure from Motion）")

st.divider()

# ── 入力設定 ──────────────────────────────────────────────────────────────────
st.subheader("入力設定")

experiments_dir = Path("/workspace/experiments")
exp_dirs = []
if experiments_dir.exists():
    for p in sorted(experiments_dir.iterdir()):
        if p.is_dir():
            inp = p / "input"
            if inp.exists() and (list(inp.glob("*.jpg")) or list(inp.glob("*.png"))):
                n = len(list(inp.glob("*.jpg"))) + len(list(inp.glob("*.png")))
                exp_dirs.append((str(p), n))

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

st.divider()

# ── バックエンド設定 ──────────────────────────────────────────────────────────
st.subheader("バックエンド設定")

# プリセット定義
PRESETS = [
    ("COLMAP\n（標準）",           False, None,              None),
    ("SuperPoint\n+ LightGlue",   True,  "superpoint_aachen",  "superpoint+lightglue"),
    ("SuperPoint\n+ SuperGlue",   True,  "superpoint_aachen",  "superglue"),
    ("DISK\n+ LightGlue",         True,  "disk",            "disk+lightglue"),
    ("ALIKED\n+ LightGlue",       True,  "aliked-n16",      "aliked+lightglue"),
    ("SIFT\n+ NN",                True,  "sift",            "NN-ratio"),
]

st.caption("▼ プリセット（クリックで下の設定に反映）")
preset_cols = st.columns(len(PRESETS))
for col, (label, hloc, feat, matcher) in zip(preset_cols, PRESETS):
    if col.button(label, use_container_width=True):
        st.session_state.sfm_use_hloc = hloc
        if feat:
            st.session_state.sfm_feature = feat
        if matcher:
            st.session_state.sfm_matcher = matcher
        st.rerun()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# 個別設定
col_b1, col_b2, col_b3 = st.columns(3)

with col_b1:
    use_hloc = st.checkbox(
        "HLocを使用",
        value=st.session_state.sfm_use_hloc,
        help="チェックなし = COLMAP標準（SIFT）。チェックあり = 深層学習ベースの特徴点（精度向上の可能性あり）。",
    )
    st.session_state.sfm_use_hloc = use_hloc

FEATURE_OPTIONS = ["superpoint_max", "superpoint_aachen", "disk", "aliked-n16",
                   "sift", "r2d2", "d2net-ss"]
MATCHER_OPTIONS = ["superpoint+lightglue", "disk+lightglue", "aliked+lightglue",
                   "superglue", "superglue-fast", "NN-superpoint", "NN-ratio",
                   "NN-mutual", "adalam"]

with col_b2:
    if use_hloc:
        feat_idx = FEATURE_OPTIONS.index(st.session_state.sfm_feature) \
            if st.session_state.sfm_feature in FEATURE_OPTIONS else 0
        feature_type = st.selectbox("特徴点抽出器", FEATURE_OPTIONS, index=feat_idx)
        st.session_state.sfm_feature = feature_type
    else:
        st.selectbox("特徴点抽出器", ["SIFT（COLMAP内蔵）"], disabled=True)
        feature_type = None

with col_b3:
    if use_hloc:
        match_idx = MATCHER_OPTIONS.index(st.session_state.sfm_matcher) \
            if st.session_state.sfm_matcher in MATCHER_OPTIONS else 0
        matcher_type = st.selectbox("マッチャー", MATCHER_OPTIONS, index=match_idx)
        st.session_state.sfm_matcher = matcher_type
    else:
        st.selectbox("マッチャー", ["SIFT最近傍（COLMAP内蔵）"], disabled=True)
        matcher_type = None

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
    pair_col1, pair_col2 = st.columns([1, 1])
    with pair_col1:
        pair_method = st.radio(
            "方式",
            ["exhaustive", "retrieval"],
            index=0 if st.session_state.sfm_pair_method == "exhaustive" else 1,
            format_func=lambda x: "Exhaustive（全ペア・精度優先）" if x == "exhaustive"
                                  else "Retrieval（類似画像のみ・速度優先）",
            label_visibility="collapsed",
        )
        st.session_state.sfm_pair_method = pair_method
    with pair_col2:
        if pair_method == "retrieval":
            ret_idx = RETRIEVAL_MODELS.index(st.session_state.sfm_retrieval_model) \
                if st.session_state.sfm_retrieval_model in RETRIEVAL_MODELS else 0
            retrieval_model = st.selectbox(
                "検索モデル", RETRIEVAL_MODELS, index=ret_idx,
                format_func=lambda x: RETRIEVAL_DESC.get(x, x),
            )
            st.session_state.sfm_retrieval_model = retrieval_model
            num_matched = st.number_input(
                "top-K（1画像あたりのペア数）",
                min_value=5, max_value=100,
                value=st.session_state.sfm_num_matched, step=5,
            )
            st.session_state.sfm_num_matched = num_matched
            if source_path:
                inp = Path(source_path) / "input"
                n = len(list(inp.glob("*.jpg"))) + len(list(inp.glob("*.png"))) if inp.exists() else 0
                if n:
                    st.caption(f"推定ペア数: 約 {n * num_matched:,}（全ペア: {n*(n-1)//2:,}）")
        else:
            retrieval_model = st.session_state.sfm_retrieval_model
            num_matched     = st.session_state.sfm_num_matched
            st.caption("全ペア総当たり。\n画像が多いと非常に時間がかかります。")
else:
    pair_method     = "exhaustive"
    retrieval_model = st.session_state.sfm_retrieval_model
    num_matched     = st.session_state.sfm_num_matched

# COLMAP固有設定
with st.expander("詳細設定（COLMAP）", expanded=False):
    if use_hloc:
        st.info("HLoc使用時はカメラモデルは自動設定されます。")
        camera_model = "OPENCV"
        use_gpu = True
    else:
        camera_model = st.selectbox(
            "カメラモデル",
            ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "SIMPLE_PINHOLE"],
            help="通常はOPENCV推奨。360度変換済みや合成画像はPINHOLE。",
        )
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

if use_hloc:
    cmd_args = [
        "python /workspace/scripts/run_hloc.py",
        f'--source_path "{source_path}"',
        f"--feature_type {feature_type}",
        f"--matcher_type {matcher_type}",
        f"--pair_method {pair_method}",
    ]
    if pair_method == "retrieval":
        cmd_args += [
            f"--retrieval_model {retrieval_model}",
            f"--num_matched {num_matched}",
        ]
else:
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
        st.success("✅ すでに sparse/0/ が存在します。再実行すると上書きされます。")

# ── 実行 ─────────────────────────────────────────────────────────────────────
st.divider()
st.warning("⚠️ GPUを使用します。フレーム数によって数分〜数十分かかります。")

btn_label = "▶ HLocを実行" if use_hloc else "▶ COLMAPを実行"
if st.button(btn_label, type="primary", disabled=not source_path):
    input_dir = Path(source_path) / "input"
    if not input_dir.exists():
        st.error(f"input/ フォルダが見つかりません: {input_dir}")
    else:
        if use_hloc:
            run_args = [sys.executable, "/workspace/scripts/run_hloc.py",
                        "--source_path",    source_path,
                        "--feature_type",   feature_type,
                        "--matcher_type",   matcher_type,
                        "--pair_method",    pair_method,
                        "--retrieval_model", retrieval_model,
                        "--num_matched",    str(num_matched)]
            method_label = f"retrieval/{retrieval_model} top-{num_matched}" \
                           if pair_method == "retrieval" else "exhaustive"
            spinner_msg = f"HLoc実行中（{feature_type} + {matcher_type} / {method_label}）..."
        else:
            run_args = [sys.executable, "/workspace/scripts/run_colmap.py",
                        "--source_path", source_path,
                        "--camera_model", camera_model]
            if not use_gpu:
                run_args.append("--no_gpu")
            spinner_msg = "COLMAP実行中（数分〜数十分かかります）..."

        log_path = str(Path(source_path) / "colmap_log.txt")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(run_args, stdout=log_file, stderr=subprocess.STDOUT)

        scene = Path(source_path).name
        st.session_state.colmap_proc        = proc
        st.session_state.colmap_log_path    = log_path
        st.session_state.colmap_source_path = source_path
        st.session_state.colmap_use_hloc    = use_hloc
        st.session_state.active_task = {
            "step":       "colmap",
            "label":      "HLoc" if use_hloc else "COLMAP",
            "scene":      scene,
            "log_path":   log_path,
            "pid":        proc.pid,
            "start_time": time.time(),
            "use_hloc":   use_hloc,
        }
        st.rerun()

# ── キューに追加 ──────────────────────────────────────────────────────────────
import sys as _sys; _sys.path.insert(0, "/workspace")
from queue_helper import add_to_queue as _add_q, pending_size as _psize

if st.button(
    f"📋 バッチキューに追加（待ち: {_psize()} 件）",
    disabled=not source_path,
    use_container_width=True,
):
    method = "HLoc" if use_hloc else f"COLMAP({camera_model})"
    _add_q(
        job_type="colmap",
        label=f"姿勢推定 {method}",
        exp_name=Path(source_path).name,
        exp_dir=source_path,
        config={
            "use_hloc": use_hloc,
            "camera_model": camera_model,
            "use_gpu": use_gpu,
            "feature_type": feature_type if use_hloc else "sift",
            "matcher_type": matcher_type if use_hloc else "",
            "pair_method": pair_method,
            "retrieval_model": retrieval_model,
            "num_matched": num_matched,
        },
    )
    st.success("バッチキューに追加しました。")

# ── 結果の可視化 ──────────────────────────────────────────────────────────────
if source_path:
    sparse_dir = Path(source_path) / "sparse" / "0"
    images_bin = sparse_dir / "images.bin"
    points_bin = sparse_dir / "points3D.bin"

    if images_bin.exists():
        st.divider()
        st.subheader("📍 推定結果の可視化")
        st.caption("推定されたカメラ位置（青）と疎な点群を3D表示します")

        try:
            import plotly.graph_objects as go

            images_data = read_extrinsics_binary(str(images_bin))
            cam_positions = []
            for img in images_data.values():
                R = qvec2rotmat(img.qvec)
                t = np.array(img.tvec)
                cam_positions.append(-R.T @ t)
            cam_positions = np.array(cam_positions)

            traces = []

            if points_bin.exists():
                pts = read_points3D_binary(str(points_bin))
                if pts:
                    xyz = np.array([p.xyz for p in pts.values()])
                    rgb = np.array([p.rgb for p in pts.values()])
                    colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
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

            traces.append(go.Scatter3d(
                x=cam_positions[:, 0], y=cam_positions[:, 1], z=cam_positions[:, 2],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name=f"カメラ（{len(cam_positions)}台）",
            ))

            fig = go.Figure(data=traces)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(x=0, y=1),
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                           aspectmode="data"),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("推定カメラ数", len(cam_positions))
            if points_bin.exists() and pts:
                c2.metric("疎な点群数", f"{len(pts):,} 点")

        except Exception as e:
            st.warning(f"可視化に失敗しました: {e}")

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### バックエンドの選び方

| バックエンド | 特徴 | 向いているシーン |
|---|---|---|
| **COLMAP（標準）** | SIFTによる古典的手法。軽量・安定。 | 枚数が少ない・動作確認したいとき |
| **HLoc** | 深層学習ベースの特徴点＋マッチング。再構成精度が高い。 | 本番運用・大規模シーン |

---

### 特徴点抽出器（HLoc使用時）

| 名前 | キーポイント数 | 速度 | 特徴 |
|---|---|---|---|
| **superpoint_aachen** ★推奨 | 1024 | 速い | 汎用・屋外向け・速度と精度のバランスが良い |
| **superpoint_max** | 4096 | 遅い | 最高精度。枚数が少ないときや精度最優先のとき |
| **disk** | 可変 | 普通 | 繰り返しパターン（タイル・床など）に強い |
| **aliked-n16** | 可変 | 速い | 軽量・省メモリ。手軽に試したいとき |
| **sift** | 可変 | 速い | 古典的特徴量。HLoc経由でCOLMAP互換マッチャーと組み合わせる |
| **r2d2** | 可変 | 普通 | 信頼度付き特徴点。照明変化や昼夜差に強い |
| **d2net-ss** | 密 | 遅い | 低テクスチャ・繰り返し面に強い。処理が重い |

> 迷ったら **superpoint_aachen**。精度を上げたいなら **superpoint_max**（処理時間に注意）。

---

### マッチャー（HLoc使用時）

| 名前 | 速度 | 精度 | 特徴 |
|---|---|---|---|
| **superpoint+lightglue** ★推奨 | 速い | 高い | SuperPoint特徴との相性が最も良い |
| **disk+lightglue** | 速い | 高い | DISK特徴専用 |
| **aliked+lightglue** | 速い | 普通 | ALIKED特徴専用 |
| **superglue** | 遅い | 非常に高い | Transformer型。精度最優先のとき |
| **superglue-fast** | 普通 | 高い | SuperGlueの軽量版 |
| **NN-ratio** | 非常に速い | 普通 | Lowe比率テスト付き最近傍。汎用 |
| **adalam** | 速い | 高い | 幾何検証付き最近傍。誤対応に強い |

---

### ペアリスト生成方式（HLoc使用時）

| 方式 | 速度 | 説明 |
|---|---|---|
| **Exhaustive** | 遅い（O(n²)） | 全ペア総当たり。〜300枚向け |
| **Retrieval** | 速い | 類似画像のみマッチング。大量枚数でも現実的な時間で完了 |

**top-K の目安**：〜500枚→20〜30、500〜2000枚→15〜20、2000枚以上→10〜15

---

### 処理時間の目安（RTX A6000）

| 枚数 | Exhaustive | Retrieval top-20 |
|---|---|---|
| 200枚 | 数分 | 数分 |
| 1000枚 | 1〜2時間 | 20〜40分 |
| 5000枚 | 数十時間 | 3〜6時間 |
| 12000枚 | 現実的でない | 15〜25時間 |
""")

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
