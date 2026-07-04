# ホーム画面：ナビゲーション・ToDo管理・パイプライン進捗・使用方法

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

TODO_FILE = "/workspace/tmp/todo.json"

_TAG_COLORS = [
    ("#00e5ff", "#002a33"), ("#a855f7", "#1a0033"), ("#22c55e", "#0a2010"),
    ("#f59e0b", "#2d1f00"), ("#ec4899", "#2d0015"), ("#3b82f6", "#0a1a40"),
    ("#ef4444", "#2d0a0a"), ("#14b8a6", "#002a27"), ("#84cc16", "#1a2600"),
    ("#ff6b35", "#331a0d"),
]

def get_tag_color(tag: str):
    return _TAG_COLORS[hash(tag) % len(_TAG_COLORS)]

def extract_tags(text: str) -> list:
    return list(dict.fromkeys(re.findall(r'#([A-Za-z0-9_぀-ヿ一-鿿]+)', text)))

def render_text_with_tags(text: str) -> str:
    def replace_tag(m):
        tag = m.group(1)
        fg, bg = get_tag_color(tag)
        return (f'<span style="background:{bg};color:{fg};border:1px solid {fg}44;'
                f'border-radius:4px;padding:1px 7px;font-size:0.72rem;margin:0 2px;'
                f'letter-spacing:0.05em;">#{tag}</span>')
    return re.sub(r'#([A-Za-z0-9_぀-ヿ一-鿿]+)', replace_tag, text)

def load_todos():
    if not os.path.exists(TODO_FILE):
        return []
    try:
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_todos(todos):
    os.makedirs(os.path.dirname(TODO_FILE), exist_ok=True)
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

# ── ヘッダー（ガウスくん）─────────────────────────────────────────────────────
st.markdown("""
<style>
.gauss-hero { display:flex; align-items:center; gap:18px; margin-bottom:1.0rem; }
.gauss-chan { position:relative; width:72px; height:62px; flex:none;
  background: radial-gradient(circle at 35% 32%, #ffd166, #ff8552 72%);
  border-radius: 58% 42% 55% 45% / 55% 48% 52% 45%;
  animation: gauss-bob 3.2s ease-in-out infinite;
  box-shadow: 0 6px 18px rgba(255,133,82,.32); }
.gauss-chan::before, .gauss-chan::after { content:""; position:absolute; top:25px;
  width:7px; height:11px; background:#26150c; border-radius:50%;
  animation: gauss-blink 4.5s infinite; }
.gauss-chan::before { left:23px; } .gauss-chan::after { left:43px; }
@keyframes gauss-bob { 0%,100%{ transform:translateY(0) rotate(-2deg);} 50%{ transform:translateY(-6px) rotate(2deg);} }
@keyframes gauss-blink { 0%,92%,100%{ transform:scaleY(1);} 95%{ transform:scaleY(.1);} }
.gauss-title { font-size:1.85rem; font-weight:800; letter-spacing:.1em; line-height:1.15;
  background: linear-gradient(90deg, #ff8552, #ffd166 55%, #7ee8b2);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.gauss-sub { font-size:.72rem; color:#8fa3b8; letter-spacing:.2em; margin-top:.1rem; }
</style>
""", unsafe_allow_html=True)

# ── ステータスカード ──────────────────────────────────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def _gpu_status() -> str:
    """GPU名を返す。見えない場合は要対処のサインなので明示する"""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            name, used, total = [x.strip() for x in r.stdout.strip().split(",")]
            return f"{name.replace('NVIDIA ', '')}｜{used} / {total}"
    except Exception:
        pass
    return "未接続（要 docker restart）"

@st.cache_data(ttl=30, show_spinner=False)
def _disk_free() -> str:
    import shutil
    du = shutil.disk_usage("/workspace")
    return f"{du.free / 1e12:.1f} TB 空き"

@st.cache_data(ttl=30, show_spinner=False)
def _exp_count() -> int:
    p = Path("/workspace/experiments")
    return sum(1 for d in p.iterdir() if d.is_dir()) if p.exists() else 0

_gpu = _gpu_status()

st.markdown("""
<div class="gauss-hero">
  <div class="gauss-chan"></div>
  <div>
    <div class="gauss-title">3DGS LAB</div>
    <div class="gauss-sub">3D GAUSSIAN SPLATTING EXPERIMENT DASHBOARD</div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("GPU", _gpu.split("｜")[0], _gpu.split("｜")[1] if "｜" in _gpu else None,
          delta_color="off")
c2.metric("実験数", f"{_exp_count()} 件")
c3.metric("ストレージ (/workspace)", _disk_free())
if _gpu.startswith("未接続"):
    st.error("GPUが見えていません。学習・COLMAP(GPU)は失敗します。"
             "ホスト側で docker restart が必要です（memo/SETUP.md 1.5節）。")

# ── クイックアクセス ──────────────────────────────────────────────────────────
q1, q2, q3, q4, q5 = st.columns(5)
q1.page_link("pages/01_pipeline.py", label="パイプライン", icon="🚀", use_container_width=True)
q2.page_link("pages/00_batch.py",    label="キュー",       icon="🗂️", use_container_width=True)
q3.page_link("pages/08_sam2_masks.py", label="SAM2マスク", icon="🎭", use_container_width=True)
q4.page_link("pages/04_training.py", label="3DGS学習",     icon="🧠", use_container_width=True)
q5.page_link("pages/05_results.py",  label="結果確認",     icon="🖼️", use_container_width=True)

# ── 実行中のタスク ─────────────────────────────────────────────────────────────
st.markdown("### 実行中のタスク")

try:
    import sys as _sys
    if "/workspace" not in _sys.path:
        _sys.path.insert(0, "/workspace")
    from pipeline_widget import _load_state as _pw_load_state, _parse_progress as _pw_parse_progress
    _pl = _pw_load_state()
except Exception:
    _pl = st.session_state.get("pipeline", {})
    if not _pl.get("active"):
        try:
            _state_file = Path("/workspace/tmp/pipeline_state.json")
            if _state_file.exists():
                _pl = json.loads(_state_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    _pw_parse_progress = None

_pipeline_active = (
    _pl.get("active")
    and _pl.get("step") not in ("done", "failed", "setup", None)
)

if not _pipeline_active:
    # 各ページから起動された単品ジョブ（抽出・姿勢推定・マスク・学習）も表示する
    _at = {}
    try:
        from queue_helper import load_active_task_file as _latf
        _at = _latf() or {}
        if _at.get("pid"):
            os.kill(int(_at["pid"]), 0)   # 生存確認（例外=死んでいる）
    except Exception:
        _at = {}
    if _at:
        _el = (time.time() - _at.get("start_time", time.time())) / 60
        st.info(f"▶ **{_at.get('label','ジョブ')}** — `{_at.get('scene','')}`（{_el:.1f} 分経過）")
        if _pw_parse_progress is not None:
            try:
                _p, _l = _pw_parse_progress(_at)
                if _p is not None:
                    st.progress(_p, text=_l)
            except Exception:
                pass
        time.sleep(5)
        st.rerun()
    else:
        st.caption("いまは何も走っていません。キューにジョブを積むと、ここに進捗が出ます 🍵")
else:
    _step        = _pl["step"]
    _exp_dir     = _pl.get("experiment_dir", "")
    _scene       = Path(_exp_dir).name if _exp_dir else "不明"
    _start       = _pl.get("start_time", time.time())
    _elapsed     = time.time() - _start
    _step_times  = _pl.get("step_times", {})
    _step_status = _pl.get("step_status", {})
    _step_starts = {
        "extracting": _start,
        "colmap":     _step_times.get("extracting", _start),
        "training":   _step_times.get("colmap", _step_times.get("extracting", _start)),
    }
    _step_name_ja = {"extracting": "フレーム抽出", "colmap": "COLMAP", "training": "3DGS学習"}

    st.info(f"🚀 **Pipeline** — `{_scene}` | {_step_name_ja.get(_step, _step)} 実行中（{_elapsed/60:.1f} 分経過）")

    STEPS = [("extracting", "① フレーム抽出"), ("colmap", "② COLMAP"), ("training", "③ 3DGS学習")]
    step_cols = st.columns(3)
    for col, (sk, slabel) in zip(step_cols, STEPS):
        st_status = _step_status.get(sk, "waiting")
        if _step == sk and st_status != "done":
            st_status = "running"
        with col:
            if st_status == "done":
                dur = _step_times.get(sk, time.time()) - _step_starts[sk]
                st.success(f"✅ {slabel}（{dur/60:.1f} 分）")
            elif st_status == "running":
                step_elapsed = time.time() - _step_starts.get(sk, _start)
                st.warning(f"🔄 {slabel}（{step_elapsed/60:.1f} 分経過）")
            else:
                st.markdown(f"⏳ {slabel}")

    _log_path = _pl.get("log_path")
    if _log_path and Path(_log_path).exists():
        _content = Path(_log_path).read_text(errors="replace")

        # pipeline_widget の _parse_progress があればそれを使う（全ステップ対応）
        if _pw_parse_progress is not None:
            _pct, _bar_label = _pw_parse_progress(_pl)
        else:
            _pct, _bar_label = None, ""
            if _step == "training":
                _total = _pl.get("iterations", 30000)
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
            st.caption("進捗を解析中...")

        # \r を改行として扱い、空行を除いて末尾5行を表示（省略なし）
        _lines = [l for l in _content.replace("\r", "\n").splitlines() if l.strip()]
        if _lines:
            with st.expander("最新ログ（直近5行）", expanded=False):
                st.code("\n".join(_lines[-5:]), language=None)

    _rc1, _rc2 = st.columns([1, 7])
    with _rc1:
        if st.button("🔄 更新", key="home_refresh"):
            st.rerun()
    with _rc2:
        st.caption("5秒ごとに自動更新されます")

st.divider()

# ── ToDo ─────────────────────────────────────────────────────────────────────
st.markdown("### ToDo リスト")

todos = load_todos()

with st.form("add_todo", clear_on_submit=True):
    add_col, btn_col = st.columns([5, 1])
    with add_col:
        new_task = st.text_input(
            "新しいタスク",
            placeholder="タスクを入力... タグは #tag で指定",
            label_visibility="collapsed",
        )
    with btn_col:
        submitted = st.form_submit_button("追加", use_container_width=True)
    if submitted and new_task.strip():
        todos.append({
            "id": datetime.now().isoformat(),
            "text": new_task.strip(),
            "done": False,
            "tags": extract_tags(new_task.strip()),
        })
        save_todos(todos)
        st.rerun()

all_tags = sorted(set(tag for t in todos for tag in t.get("tags", [])))
selected_tags = []
if all_tags:
    selected_tags = st.multiselect(
        "タグで絞り込み", options=all_tags,
        format_func=lambda t: f"#{t}",
        placeholder="タグを選択（複数可）",
        key="tag_filter_select",
    )

display_items = [
    (i, t) for i, t in enumerate(todos)
    if not selected_tags or any(tag in t.get("tags", []) for tag in selected_tags)
]

if not todos:
    st.caption("タスクはまだありません。上のフォームから追加できます。")
elif not display_items:
    st.caption("選択したタグに一致するタスクはありません。")
else:
    for orig_i, todo in display_items:
        chk_col, t_col, d_col = st.columns([0.4, 8, 0.7])
        with chk_col:
            checked = st.checkbox("done", value=todo["done"],
                                  key=f"todo_{todo['id']}", label_visibility="collapsed")
            if checked != todo["done"]:
                todos[orig_i]["done"] = checked
                save_todos(todos)
                st.rerun()
        with t_col:
            text_html = render_text_with_tags(todo["text"])
            style = "text-decoration:line-through;color:#888;" if todo["done"] else ""
            st.markdown(f'<div style="{style}font-size:0.88rem;line-height:2;">{text_html}</div>',
                        unsafe_allow_html=True)
        with d_col:
            if st.button("🗑️", key=f"del_{todo['id']}", help="削除"):
                todos.pop(orig_i)
                save_todos(todos)
                st.rerun()

    done_count = sum(1 for t in todos if t["done"])
    if done_count > 0:
        if st.button(f"完了済み {done_count} 件を削除"):
            todos = [t for t in todos if not t["done"]]
            save_todos(todos)
            st.rerun()

st.divider()

# ── 使用方法 ──────────────────────────────────────────────────────────────────
with st.expander("使用方法を表示する", expanded=False):
    st.markdown("""
### パイプライン全体の流れ

```
[Step 1] 動画を data/movies/ または data/360movies/ に配置
    ↓
[Step 2] フレーム抽出（FFmpeg / ピンホール変換）
    ↓
[Step 3] カメラ姿勢推定（COLMAP または HLoc）
    ↓
[Step 4] 3DGS学習
    ↓
[Step 5] 結果確認・レンダリング
```

**全ステップ自動実行は 🚀 Pipeline Runner から。**

---

### 各ページの説明

| ページ | 主な機能 |
|---|---|
| 🚀 Pipeline Runner | フレーム抽出→姿勢推定→学習を一括自動実行。設定プリセット保存対応 |
| 🎞️ フレーム抽出 | 動画から連番画像を切り出す。360度動画はピンホール変換対応 |
| 📷 姿勢推定 | COLMAP / HLoc でカメラ姿勢推定。完了後に3D可視化表示 |
| 🧠 3DGS学習 | 学習実行。リアルタイムでLoss・PSNRグラフ表示。中断ボタンあり |
| 🖼️ 結果確認 | COLMAP品質・point_cloud・レンダリング実行・画像確認 |
| 📊 実験比較 | 複数実験のPSNR・L1 Loss学習曲線を重ね比較 |
| 🗂️ 実験管理 | 実験一覧・ログ閲覧・設定確認・メモ編集・削除 |
| ⚡ システムモニター | GPU / CPU / メモリのリアルタイム監視 |
| ⚗️ ミニゲーム | ガウシアンを育てるアイドルゲーム |
| 🐾 ガウスくん | たまごっち風ペット育成ゲーム |
""")
if _pipeline_active:
    time.sleep(5)
    st.rerun()
