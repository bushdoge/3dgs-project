# SAM2マスク生成ページ
# 画像をクリックして撮影者の位置を指定し、SAM2で全フレームへマスクを伝播・保存する
# 生成したマスクは学習時（run_train.py）に自動で合成され、撮影者領域が学習から除外される
# あわせてSOR（統計的外れ値除去）による点群クリーニングも実行できる

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/scripts")
from generate_masks import group_frames_by_direction  # torchは遅延importなので軽い

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICK_UI = True
except ImportError:
    HAS_CLICK_UI = False

DISP_W = 760   # クリック用画像の表示幅（座標は元解像度に換算して保存する）


# ── セッション状態の初期化 ────────────────────────────────────────────────────
if "mask_proc" not in st.session_state:
    st.session_state.mask_proc     = None
    st.session_state.mask_log_path = None
    st.session_state.mask_exp      = None

# クリック座標: {実験名: {方向: [[x, y, label], ...]}}
if "sam2_clicks" not in st.session_state:
    st.session_state.sam2_clicks = {}

# プロンプトを与えるフレーム番号: {実験名: {方向: フレームidx}}
# （1枚目に撮影者が写っていない場合、写っているフレームを選んでクリックするため）
if "sam2_ann_frames" not in st.session_state:
    st.session_state.sam2_ann_frames = {}


def _mask_running() -> bool:
    p = st.session_state.mask_proc
    return p is not None and p.poll() is None


def _gpu_available() -> bool:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                              timeout=5).returncode == 0
    except Exception:
        return False


def _parse_mask_progress(log_text: str):
    """ログから (現在方向, cur, total, 完了方向数) を返す"""
    prog = re.findall(r"PROGRESS (\S+) (\d+)/(\d+)", log_text)
    done = len(re.findall(r"完了: \d+ 枚保存", log_text))
    if prog:
        d, cur, total = prog[-1]
        return d, int(cur), int(total), done
    return None, 0, 0, done


def _overlay_mask(img_path: Path, mask_path: Path, width: int = 320) -> Image.Image:
    """元画像にマスク（白=撮影者）を赤の半透明で重ねたプレビューを作る"""
    img = Image.open(img_path).convert("RGB")
    scale = width / img.width
    img = img.resize((width, int(img.height * scale)))
    mask = Image.open(mask_path).convert("L").resize(img.size)
    arr  = np.array(img)
    m    = np.array(mask) > 127
    arr[m] = (arr[m] * 0.4 + np.array([255, 40, 40]) * 0.6).astype(np.uint8)
    return Image.fromarray(arr)


# ════════════════════════════════════════════════════════════════════════════
#  実行ステータスバナー
# ════════════════════════════════════════════════════════════════════════════
_proc = st.session_state.mask_proc
if _proc is not None:
    log_path = st.session_state.mask_log_path
    log_text = ""
    if log_path and Path(log_path).exists():
        log_text = Path(log_path).read_text(errors="replace")

    if _proc.poll() is None:
        ba, bb = st.columns([5, 1])
        ba.info(f"🔄 SAM2マスク生成 実行中 — {st.session_state.mask_exp}")
        if bb.button("⏹ 中断", key="mask_stop"):
            try:
                os.kill(_proc.pid, signal.SIGTERM)
            except Exception:
                pass
            st.session_state.mask_proc = None
            st.session_state.pop("active_task", None)
            from queue_helper import clear_active_task_file
            clear_active_task_file()
            st.rerun()

        d, cur, total, done = _parse_mask_progress(log_text)
        if d is not None and total > 0:
            st.progress(min(cur / total, 1.0),
                        text=f"[{d}] {cur}/{total} フレーム　（完了方向: {done}）")
        lines = [l for l in log_text.replace("\r", "\n").splitlines() if l.strip()]
        if lines:
            st.code("\n".join(lines[-5:]), language=None)
        time.sleep(3)
        st.rerun()
    else:
        st.session_state.pop("active_task", None)
        if _proc.returncode == 0:
            st.success("✅ 完了！下の「生成済みマスクのプレビュー」で結果を確認できます。")
        else:
            st.error(f"❌ エラーで終了しました（終了コード: {_proc.returncode}）")
            lines = [l for l in log_text.replace("\r", "\n").splitlines() if l.strip()]
            if lines:
                st.code("\n".join(lines[-10:]), language=None)
        if st.button("✕ クリア", key="mask_clear"):
            st.session_state.mask_proc = None
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  設定ビュー
# ════════════════════════════════════════════════════════════════════════════
st.title("🎭 SAM2 マスク生成")
st.caption("画像クリックで撮影者を指定 → SAM2で全フレームにマスクを伝播 → 学習時に自動で除外されます")

if not _gpu_available():
    st.warning("⚠️ GPUが検出できません（nvidia-smi失敗）。SAM2はCPU実行となり非常に時間がかかります。"
               "コンテナのGPU接続を確認してください。")

st.divider()

# ── 実験選択 ──────────────────────────────────────────────────────────────────
st.subheader("実験フォルダの選択")

experiments_dir = Path("/workspace/experiments")
ready_dirs = []
if experiments_dir.exists():
    for p in sorted(experiments_dir.iterdir(), reverse=True):
        if p.is_dir() and (p / "input").exists():
            ready_dirs.append(str(p))

if not ready_dirs:
    st.warning("input/ フォルダを含む実験が見つかりません。先にフレーム抽出を実行してください。")
    st.stop()

exp_dir  = st.selectbox("実験フォルダ（input/ を含むもの）", ready_dirs,
                        format_func=lambda x: Path(x).name)
exp_name = Path(exp_dir).name
input_groups = group_frames_by_direction(Path(exp_dir) / "input")

if not input_groups:
    st.warning("input/ に画像が見つかりません。")
    st.stop()

n_total = sum(len(v) for v in input_groups.values())
st.caption(f"方向グループ: {len(input_groups)}　／　総フレーム数: {n_total:,} 枚")

# ── モード選択（等距円筒フレームがある実験のみ）─────────────────────────────
eq_dir    = Path(exp_dir) / "equirect"
eq_frames = sorted(eq_dir.glob("*.jpg")) if (eq_dir / "meta.json").exists() else []

equirect_mode = False
if eq_frames:
    mode = st.radio(
        "マスク生成モード", ["🌐 等距円筒（推奨）", "📐 方向別"], horizontal=True,
        help="等距円筒: 360度画像上で1回クリック → SAM2を1系統実行 → 全ピンホール方向にマスクを投影。"
             "撮影者が複数方向に写っていても指定は1回で済みます。　"
             "方向別: ピンホール各方向で個別にクリック・実行（従来方式）",
    )
    equirect_mode = mode.startswith("🌐")

# クリックUIの対象: 等距円筒モードでは equirect/ の1系統、方向別では input/ の方向グループ
groups = {"equirect": eq_frames} if equirect_mode else input_groups

exp_clicks = st.session_state.sam2_clicks.setdefault(exp_name, {})
exp_ann    = st.session_state.sam2_ann_frames.setdefault(exp_name, {})

# ── クリック座標の指定 ────────────────────────────────────────────────────────
st.subheader("撮影者の位置をクリックで指定")

label_mode = st.radio(
    "クリックの種類",
    [1, 0],
    format_func=lambda v: "🔴 撮影者（マスクする）" if v == 1 else "🔵 背景（マスクしない）",
    horizontal=True,
    help="まず撮影者の上を1〜3点クリック。マスクが広がりすぎる場合は背景点を追加して抑制します。",
)

if not HAS_CLICK_UI:
    st.error("streamlit-image-coordinates が未インストールのため、座標は数値入力で指定してください。"
             "（pip install streamlit-image-coordinates で画像クリックが使えます）")

dir_keys = sorted(groups.keys())
tabs = st.tabs([f"{d}（{len(groups[d])}枚）" for d in dir_keys])

for tab, direction in zip(tabs, dir_keys):
    with tab:
        frames   = groups[direction]
        n_frames = len(frames)

        # クリック対象フレームの選択（撮影者が1枚目に写っていないシーン用）
        ann_idx = 0
        if n_frames > 1:
            ann_idx = st.slider(
                "クリックするフレーム（撮影者がはっきり写っているものを選ぶ）",
                0, n_frames - 1,
                key=f"annframe_{exp_name}_{direction}",
                help="マスクはこのフレームから前後両方向に伝播されます。"
                     "撮影者が1枚目に写っていない場合は、写っているフレームを選んでください。",
            )
        exp_ann[direction] = ann_idx

        ann_frame = frames[ann_idx]
        img = Image.open(ann_frame).convert("RGB")
        orig_w, orig_h = img.size
        scale = DISP_W / orig_w
        disp = img.resize((DISP_W, int(orig_h * scale)))

        clicks = exp_clicks.setdefault(direction, [])

        # 現在のクリック点をマーカー描画
        draw = ImageDraw.Draw(disp)
        for x, y, l in clicks:
            dx, dy = x * scale, y * scale
            color = (255, 40, 40) if l == 1 else (40, 120, 255)
            draw.ellipse([dx - 7, dy - 7, dx + 7, dy + 7], outline=color, width=3)
            draw.line([dx - 10, dy, dx + 10, dy], fill=color, width=2)
            draw.line([dx, dy - 10, dx, dy + 10], fill=color, width=2)

        st.caption(f"フレーム {ann_idx}: `{ann_frame.name}`（{orig_w}×{orig_h}px）— 画像をクリックして点を追加")

        if HAS_CLICK_UI:
            res = streamlit_image_coordinates(disp, key=f"imgclick_{exp_name}_{direction}")
            # 同じクリックイベントが再実行のたびに返るため、前回値と比較して重複追加を防ぐ
            last_key = f"_lastclick_{exp_name}_{direction}"
            if res is not None:
                tag = (res["x"], res["y"])
                if st.session_state.get(last_key) != tag:
                    st.session_state[last_key] = tag
                    clicks.append([int(res["x"] / scale), int(res["y"] / scale), int(label_mode)])
                    st.rerun()
        else:
            st.image(disp)
            ic1, ic2, ic3 = st.columns([2, 2, 1])
            mx = ic1.number_input("x", 0, orig_w - 1, orig_w // 2, key=f"mx_{direction}")
            my = ic2.number_input("y", 0, orig_h - 1, orig_h - orig_h // 4, key=f"my_{direction}")
            if ic3.button("追加", key=f"addpt_{direction}", use_container_width=True):
                clicks.append([int(mx), int(my), int(label_mode)])
                st.rerun()

        # クリック一覧と操作ボタン
        if clicks:
            st.caption("　".join(
                f"{'🔴' if l == 1 else '🔵'}({x}, {y})" for x, y, l in clicks
            ))
        bc1, bc2, bc3 = st.columns(3)
        if bc1.button("↩ 最後の点を削除", key=f"undo_{direction}",
                      disabled=not clicks, use_container_width=True):
            clicks.pop()
            st.rerun()
        if bc2.button("🗑 この方向を全削除", key=f"clr_{direction}",
                      disabled=not clicks, use_container_width=True):
            clicks.clear()
            st.rerun()
        if bc3.button("📋 この座標を全方向にコピー", key=f"cp_{direction}",
                      disabled=not clicks or len(dir_keys) == 1, use_container_width=True):
            for d in dir_keys:
                exp_clicks[d] = [list(c) for c in clicks]
                # フレーム番号もコピー（方向ごとの枚数を超えないように丸める）
                d_idx = min(ann_idx, len(groups[d]) - 1)
                exp_ann[d] = d_idx
                if len(groups[d]) > 1:
                    st.session_state[f"annframe_{exp_name}_{d}"] = d_idx
            st.rerun()

# ── 実行設定 ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("実行")

if equirect_mode:
    sel_dirs = ["equirect"]
    mask_dilate = st.number_input(
        "マスク膨張（px）", 0, 50, 7,
        help="投影前に等距円筒マスクを膨張させて境界の取りこぼしを防ぎます（消しすぎ側に倒す）")
else:
    dirs_with_clicks = [d for d in dir_keys if exp_clicks.get(d)]
    sel_dirs = st.multiselect(
        "マスク生成する方向（クリック座標がある方向のみ実行されます）",
        dir_keys, default=dirs_with_clicks,
    )
run_sor = st.checkbox("マスク生成後にSOR（点群クリーニング）も実行する", value=False,
                      help="sparse/0 と dense/sparse/0 の points3D.bin から外れ値を除去します。"
                           "元のモデルは before_sor/ にバックアップされます。")
with st.expander("SOR 詳細設定", expanded=False):
    sor_neighbors = st.number_input("近傍点数（nb_neighbors）", 5, 100, 20)
    sor_std       = st.number_input("標準偏差倍率（std_ratio）", 0.5, 5.0, 2.0, 0.1,
                                    help="小さいほど積極的に除去します")

target_dirs = [d for d in sel_dirs if exp_clicks.get(d)]
if equirect_mode:
    # SAM2は等距円筒1系統、出力は全ピンホールフレーム分
    n_frames_run = n_total if target_dirs else 0
    run_label = f"▶ マスク生成を実行（等距円筒 {len(eq_frames)} フレーム → 全方向 {n_frames_run:,} 枚に投影）"
else:
    n_frames_run = sum(len(groups[d]) for d in target_dirs)
    run_label = f"▶ マスク生成を実行（{len(target_dirs)} 方向 / {n_frames_run:,} フレーム）"

masks_dir = Path(exp_dir) / "masks"
if masks_dir.exists() and any(masks_dir.iterdir()):
    st.info(f"ℹ️ masks/ に既存マスクがあります（{len(list(masks_dir.glob('*.png')))} 枚）。"
            "同名フレームのマスクは上書きされます。")

if st.button(
    run_label,
    type="primary",
    disabled=_mask_running() or not target_dirs,
    use_container_width=True,
):
    # {"frame": N, "points": [...]} 形式：フレームNにプロンプトを与え前後両方向に伝播
    clicks_payload = {d: {"frame": int(exp_ann.get(d, 0)), "points": exp_clicks[d]}
                      for d in target_dirs}
    cmd = [sys.executable, "/workspace/scripts/generate_masks.py", exp_dir,
           "--clicks-json", json.dumps(clicks_payload),
           "--sor-neighbors", str(int(sor_neighbors)),
           "--sor-std-ratio", str(float(sor_std))]
    if equirect_mode:
        cmd += ["--equirect", "--mask-dilate", str(int(mask_dilate))]
    else:
        cmd += ["--directions", ",".join(target_dirs)]
    if not run_sor:
        cmd.append("--sam-only")

    log_path = str(Path(exp_dir) / "masks_log.txt")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    st.session_state.mask_proc     = proc
    st.session_state.mask_log_path = log_path
    st.session_state.mask_exp      = exp_name
    st.session_state.active_task = {
        "step":       "masks",
        "label":      "SAM2マスク生成",
        "scene":      exp_name,
        "log_path":   log_path,
        "pid":        proc.pid,
        "start_time": time.time(),
    }
    from queue_helper import save_active_task_file
    save_active_task_file(st.session_state.active_task)
    st.rerun()

# SORのみ実行
if st.button("⚙️ SORのみ実行（マスク生成をスキップ）",
             disabled=_mask_running(), use_container_width=True):
    cmd = [sys.executable, "/workspace/scripts/generate_masks.py", exp_dir,
           "--sor-only",
           "--sor-neighbors", str(int(sor_neighbors)),
           "--sor-std-ratio", str(float(sor_std))]
    log_path = str(Path(exp_dir) / "masks_log.txt")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    st.session_state.mask_proc     = proc
    st.session_state.mask_log_path = log_path
    st.session_state.mask_exp      = exp_name
    st.rerun()

# ── 生成済みマスクのプレビュー ────────────────────────────────────────────────
if masks_dir.exists() and any(masks_dir.glob("*.png")):
    st.divider()
    st.subheader("生成済みマスクのプレビュー")
    st.caption("🔴 赤い領域 = マスク（学習から除外される部分）。各方向の先頭・中間・末尾フレームを表示します。")

    for direction in sorted(input_groups.keys()):
        frames = input_groups[direction]
        samples = []
        for f in [frames[0], frames[len(frames) // 2], frames[-1]]:
            mp = masks_dir / (f.stem + ".png")
            if mp.exists() and (f, mp) not in samples:
                samples.append((f, mp))
        if not samples:
            continue
        st.markdown(f"**{direction}**")
        cols = st.columns(max(len(samples), 1))
        for col, (f, mp) in zip(cols, samples):
            col.image(_overlay_mask(f, mp), caption=f.name)

# ── 使い方 ────────────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### ワークフロー

1. **実験フォルダを選択**（フレーム抽出済みのもの）
2. **モードを選択**（`equirect/` がある実験のみ表示）
   - **🌐 等距円筒（推奨）**: 360度画像上で撮影者を1回クリックするだけ。SAM2を等距円筒フレーム列に
     1系統実行し、出来たマスクを画像変換と同じremapで**全ピンホール方向に投影**します。
     左右端の継ぎ目はクリック点が中央に来るよう自動でrollして回避します。
   - **📐 方向別（従来方式）**: ピンホール各方向で個別にクリック・実行。`equirect/` がない実験や
     通常動画・画像群はこちらになります。
3. フレームが表示されるので、**撮影者の上をクリック**（1〜3点）
   - **1枚目に撮影者が写っていない場合**は、スライダーで写っているフレームを選んでからクリック
   - マスクが広がりすぎる場合は「背景」モードで撮影者の外側をクリックして抑制
   - 方向別モードでは撮影者の写る位置が方向ごとに違うため、**方向ごとに指定**します
4. **▶ マスク生成を実行** — SAM2が選択フレームのクリック点を前後両方向の全フレームへ時系列伝播します
   （撮影者が写っていないフレームのマスクは自然に空になります）
5. プレビューで赤い領域（除外部分）を確認。ずれていたら点を調整して再実行
6. そのまま **3DGS学習** を実行すると `masks/` が自動検出され、撮影者が学習から除外されます

### SOR（点群クリーニング）
COLMAP点群の外れ値（ノイズ点）を統計的に除去します。`sparse/0/` と `dense/sparse/0/` の
`points3D.bin` を直接書き換え、元モデルは `before_sor/` にバックアップされます。
マスク生成と同時実行（チェックボックス）または単独実行（SORのみボタン）ができます。

### 注意
- SAM2の推論はGPUで1フレームあたり約0.1〜0.3秒です（CPUでは数十倍かかります）
- マスクは `masks/<フレーム名>.png`（白=撮影者、黒=背景）として保存されます
""")
