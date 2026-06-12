# 3D Gaussian Splattingの学習をtrain.pyで実行するページ
# リアルタイムでログ・Loss・PSNRを表示する。学習中断ボタン付き。

import os
import re
import signal
import time
import subprocess
from pathlib import Path

import streamlit as st


# ── セッション状態の初期化 ────────────────────────────────────────────────────
if "train_proc" not in st.session_state:
    st.session_state.train_proc = None
    st.session_state.train_log_path = None
    st.session_state.train_model_path = None
    st.session_state.train_source = None
    st.session_state.train_iterations = 30000

if "selected_test_iters" not in st.session_state:
    st.session_state.selected_test_iters = {1000, 3000, 7000, 15000, 30000}


def is_training():
    proc = st.session_state.train_proc
    return proc is not None and proc.poll() is None


def parse_log(log_text):
    """ログからLoss・PSNR・SSIM・LPIPSを抽出してリストで返す"""
    cleaned = "\n".join(
        line.split("\r")[-1] for line in log_text.split("\n")
    )

    losses = []
    for m in re.finditer(r"Loss:\s*([\d.]+)", cleaned):
        losses.append(float(m.group(1)))

    eval_records = []
    for m in re.finditer(
        r"\[ITER (\d+)\] Evaluating (\w+): L1 ([^\s]+) PSNR ([^\s]+)"
        r"(?:\s+SSIM ([^\s]+))?(?:\s+LPIPS ([^\s\[]+))?",
        cleaned,
    ):
        rec = {
            "iteration": int(m.group(1)),
            "split":     m.group(2),
            "PSNR":      float(m.group(4)),
            "L1":        float(m.group(3)),
        }
        if m.group(5):
            rec["SSIM"]  = float(m.group(5))
        if m.group(6):
            rec["LPIPS"] = float(m.group(6))
        eval_records.append(rec)

    return losses, eval_records, cleaned


# ── 実行ステータスバナー ───────────────────────────────────────────────────────
_tproc = st.session_state.train_proc
if _tproc is not None:
    _trunning = _tproc.poll() is None
    if _trunning:
        _iters = st.session_state.train_iterations
        _ta, _tb = st.columns([5, 1])
        _ta.info(f"🔄 3DGS学習実行中　｜　詳細は「🗂️ バッチキュー」ページで確認できます")
        if _tb.button("⏹ 中断", key="train_stop"):
            try: os.kill(_tproc.pid, signal.SIGTERM)
            except Exception: pass
            st.session_state.train_proc = None
            st.session_state.pop("active_task", None)
            import sys as _s3; _s3.path.insert(0, "/workspace")
            from queue_helper import clear_active_task_file as _clf; _clf()
            st.rerun()
        time.sleep(3)
        st.rerun()
    elif _tproc.returncode == 0:
        st.success("✅ 学習完了！「🖼️ 結果確認」ページで結果を確認できます。")
        st.session_state.pop("active_task", None)
        if st.button("✕ クリア", key="train_clear"):
            st.session_state.train_proc = None
            st.rerun()
    else:
        st.error(f"❌ エラーで終了しました（終了コード: {_tproc.returncode}）")
        st.session_state.pop("active_task", None)
        if st.button("✕ クリア", key="train_clear_err"):
            st.session_state.train_proc = None
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  設定ビュー（常に表示）
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

col1, col2 = st.columns(2)

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

# ── 評価タイミング（ボタン選択） ───────────────────────────────────────────────
st.markdown("**評価タイミング（ボタンで選択 / 解除）**")
st.caption("クリックで切り替え。青 = 選択中。学習ステップ数を超えるボタンは表示されません。")

def _toggle_test_iter(i):
    if i in st.session_state.selected_test_iters:
        st.session_state.selected_test_iters.discard(i)
    else:
        st.session_state.selected_test_iters.add(i)

_COLS_PER_ROW = 10
_max_iter = int(iterations)
_all_iters = list(range(1000, _max_iter + 1, 1000))

for _row_start in range(0, len(_all_iters), _COLS_PER_ROW):
    _row_iters = _all_iters[_row_start:_row_start + _COLS_PER_ROW]
    _cols = st.columns(_COLS_PER_ROW)
    for _idx, _i in enumerate(_row_iters):
        _selected = _i in st.session_state.selected_test_iters
        _cols[_idx].button(
            f"{_i // 1000}k",
            key=f"test_iter_btn_{_i}",
            type="primary" if _selected else "secondary",
            on_click=_toggle_test_iter,
            args=(_i,),
            use_container_width=True,
        )

_selected_display = sorted(i for i in st.session_state.selected_test_iters if i <= _max_iter)
st.caption(f"選択中: {', '.join(str(i) for i in _selected_display) if _selected_display else '（なし）'}")

# ── 解像度設定 ────────────────────────────────────────────────────────────────
st.subheader("解像度設定（OOM対策）")

resolution_options = {
    "自動（VRAM量から自動判定）★推奨": None,
    "元解像度のまま（1x）": 1,
    "1/2 に縮小（2x）": 2,
    "1/4 に縮小（4x）": 4,
    "1/8 に縮小（8x）": 8,
}
resolution_label = st.selectbox(
    "画像縮小倍率",
    list(resolution_options.keys()),
    index=0,
    help="「自動」にするとVRAMと画像枚数・サイズからOOMにならない倍率を自動計算します。",
)
# 1x はそのまま --resolution 1 として渡す（Noneにすると自動判定になり勝手に縮小されうる）
resolution = resolution_options[resolution_label]

# ── 評価設定 ──────────────────────────────────────────────────────────────────
use_eval = st.checkbox(
    "train/test 分割を有効にする（--eval）",
    value=False,
    help="ONにすると8枚に1枚をtestデータとして学習から除外し、未学習視点でPSNRを評価します。研究・比較目的に推奨。",
)
if use_eval:
    st.caption("📊 testデータ: 入力画像の約12.5%（8枚に1枚）が自動で割り当てられます。学習には残り87.5%が使われます。")
else:
    st.caption("📊 全フレームを学習に使用します。test PSNRは計算されません。")

# ── 出力設定 ──────────────────────────────────────────────────────────────────
st.subheader("出力設定")

default_model = str(Path(source_path) / "output") if source_path else ""
model_path = st.text_input("モデル出力フォルダ", value=default_model)

# ── コマンドプレビュー ────────────────────────────────────────────────────────
st.subheader("実行コマンド（プレビュー）")

save_list = [s.strip() for s in save_iterations.split(",") if s.strip()]
test_list = [str(i) for i in sorted(i for i in st.session_state.selected_test_iters if i <= int(iterations))]

cmd_parts = [
    "python3 /workspace/scripts/run_train.py",
    f'--source "{source_path}"',
    f'--model_path "{model_path}"',
    f"--iterations {iterations}",
]
if save_list:
    cmd_parts.append("--save_iterations " + " ".join(save_list))
if test_list:
    cmd_parts.append("--test_iterations " + " ".join(test_list))
if resolution is not None:
    cmd_parts.append(f"--resolution {resolution}")
if use_eval:
    cmd_parts.append("--eval")

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
            "python3", "/workspace/scripts/run_train.py",
            "--source", source_path,
            "--model_path", model_path,
            "--iterations", str(iterations),
        ]
        if save_list:
            run_args += ["--save_iterations"] + save_list
        if test_list:
            run_args += ["--test_iterations"] + test_list
        if resolution is not None:
            run_args += ["--resolution", str(resolution)]
        if use_eval:
            run_args.append("--eval")

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
        st.session_state.active_task = {
            "step":       "training",
            "label":      "3DGS学習",
            "scene":      Path(source_path).name,
            "log_path":   log_path,
            "pid":        proc.pid,
            "start_time": time.time(),
            "iterations": iterations,
        }
        from queue_helper import save_active_task_file as _satf
        _satf(st.session_state.active_task)

        st.rerun()

# ── キューに追加 ──────────────────────────────────────────────────────────────
import sys as _sys; _sys.path.insert(0, "/workspace")
from queue_helper import add_to_queue as _add_q, pending_size as _psize

if st.button(
    f"📋 バッチキューに追加（待ち: {_psize()} 件）",
    disabled=not (source_path and model_path),
    use_container_width=True,
):
    _add_q(
        job_type="train",
        label=f"学習 {iterations:,}iter",
        exp_name=Path(source_path).name,
        exp_dir=source_path,
        config={
            "model_path":      model_path,
            "iterations":      int(iterations),
            "save_iterations": [int(s) for s in save_list] if save_list else [7000, int(iterations)],
            "test_iterations": [int(s) for s in test_list] if test_list else [1000, 7000, 15000, int(iterations)],
            "eval":            use_eval,
            "resolution":      resolution,
        },
    )
    st.success("バッチキューに追加しました。")

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### 学習ステップ数（iterations）

| ステップ数 | 時間目安 | 用途 |
|---|---|---|
| 7,000 | 約5〜10分 | クイック確認・プリセット調整 |
| 30,000 | 約30〜60分 | 標準品質（デフォルト） |
| 100,000 | 約2〜4時間 | 高品質・論文品質 |

### 保存タイミング / 評価タイミング
- **保存タイミング（save_iterations）**：チェックポイントを保存するステップ。カンマ区切りで複数指定可能。ストレージを使うため必要なステップのみ指定推奨。
- **評価タイミング（test_iterations）**：1000刻みのボタンで選択します。青いボタンが選択中。PSNR・SSIM・LPIPS を計算してグラフ化するタイミングで、モデル保存は行いません。学習ステップ数を超えるボタンは自動で非表示になります。

### train/test 分割（--eval）
- ONにすると入力画像の **8枚に1枚**（約12.5%）をtestデータとして自動割り当てします。
- 学習には残り **87.5%** が使われます。
- testデータは学習に使用されないため、未学習視点でのPSNRを客観的に評価できます。
- **研究・比較目的に推奨**。手元確認のみなら不要です。

### モデル出力フォルダ
- デフォルトは実験フォルダ内の `output/` サブフォルダです。
- 変更も可能ですが、特別な理由がない限りデフォルトのままを推奨します。
- 出力フォルダには `point_cloud/`、`cameras.json`、`cfg_args` などが生成されます。

### 実行後
- 学習が完了したら「📷 結果確認」ページで点群・PSNRグラフ・レンダリングを確認できます。
- 実験設定・ログは「🗂️ 実験管理」ページで閲覧・メモ編集できます。
""")