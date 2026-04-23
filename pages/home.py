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

# ── ヘッダー ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:2.2rem;font-weight:700;letter-spacing:0.15em;color:#00e5ff;'
    'text-shadow:0 0 12px #00e5ff88;margin-bottom:0;">🔬 3DGS LAB</div>'
    '<div style="font-size:0.75rem;color:#4a90b8;letter-spacing:0.2em;margin-top:0.2rem;'
    'margin-bottom:1.5rem;">3D GAUSSIAN SPLATTING EXPERIMENT DASHBOARD</div>',
    unsafe_allow_html=True,
)

# ── 実行中のタスク ─────────────────────────────────────────────────────────────
st.markdown("### 実行中のタスク")

_pl = st.session_state.get("pipeline", {})
if not _pl.get("active"):
    try:
        _state_file = Path("/workspace/tmp/pipeline_state.json")
        if _state_file.exists():
            _pl = json.loads(_state_file.read_text(encoding="utf-8"))
    except Exception:
        pass

_pipeline_active = (
    _pl.get("active")
    and _pl.get("step") not in ("done", "failed", "setup", None)
)

if not _pipeline_active:
    st.caption("現在実行中のタスクはありません。")
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

        _lines = [l for l in _content.split("\n") if l.strip()]
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

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass

if _pipeline_active:
    time.sleep(5)
    st.rerun()
