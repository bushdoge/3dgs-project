# ホーム画面：ナビゲーション・ToDo管理・使用方法の表示

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

TODO_FILE = "/workspace/tmp/todo.json"

# タグカラーパレット（fg, bg）
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

st.set_page_config(
    page_title="3DGS Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── スタイル ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a;
    color: #e0e6f0;
  }
  .block-container { padding: 1.5rem 2rem; }

  .home-title {
    font-size: 2.2rem; font-weight: 700; letter-spacing: 0.15em;
    color: #00e5ff; text-shadow: 0 0 12px #00e5ff88; margin-bottom: 0;
  }
  .home-sub {
    font-size: 0.75rem; color: #4a90b8; letter-spacing: 0.2em; margin-top: 0.2rem;
    margin-bottom: 1.5rem;
  }

  /* ── ナビカード ── */
  .nav-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #0f2340 100%);
    border: 1px solid #1a3a5c; border-radius: 12px;
    padding: 1rem 1.2rem; text-align: center; cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
    margin-bottom: 0.5rem;
  }
  .nav-card:hover { border-color: #00e5ff; box-shadow: 0 0 12px #00e5ff33; }
  .nav-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
  .nav-label { font-size: 0.7rem; letter-spacing: 0.15em; color: #4a90b8;
               text-transform: uppercase; }

  /* ── セクションヘッダー ── */
  .section-title {
    font-size: 0.7rem; letter-spacing: 0.25em; text-transform: uppercase;
    color: #4a90b8; border-bottom: 1px solid #1a3a5c;
    padding-bottom: 0.4rem; margin: 1.5rem 0 0.8rem 0;
  }

  /* ── タスクカード ── */
  .task-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a1520 100%);
    border: 1px solid #1a3a5c; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
  }
  .task-card.running { border-color: #00aaff; box-shadow: 0 0 10px #00aaff22; }
  .step-done    { color: #00cc66; font-size: 0.82rem; }
  .step-running { color: #00aaff; font-size: 0.82rem; }
  .step-waiting { color: #334455; font-size: 0.82rem; }

  /* ── Streamlit ボタン上書き ── */
  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0d1b2e, #0f2340);
    border: 1px solid #1a3a5c; border-radius: 10px;
    color: #e0e6f0; font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem; letter-spacing: 0.1em;
    padding: 0.7rem 0.5rem; width: 100%;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #00e5ff; box-shadow: 0 0 10px #00e5ff33; color: #00e5ff;
  }
</style>
""", unsafe_allow_html=True)


# ── ToDo ファイル読み書き ──────────────────────────────────────────────────────
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
    '<div class="home-title">🔬 3DGS LAB</div>'
    '<div class="home-sub">3D GAUSSIAN SPLATTING EXPERIMENT DASHBOARD</div>',
    unsafe_allow_html=True,
)

# ── ナビゲーション ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)

pages = [
    ("🚀", "Pipeline\nRunner",   "00_pipeline"),
    ("🎞️", "Frame\nExtraction",  "01_frame_extraction"),
    ("📐", "COLMAP\nEstimation", "02_colmap"),
    ("🧠", "3DGS\nTraining",     "03_training"),
    ("🖼️", "Results\nViewer",    "04_results"),
    ("📊", "Compare\nResults",   "05_compare"),
    ("🗂️", "Experiment\nManager","06_experiment_manager"),
    ("⚡", "System\nMonitor",    "monitor"),
    ("⚗️", "Mini\nGame",         "07_minigame"),
]

nav_cols = st.columns(len(pages))
for col, (icon, label, page_name) in zip(nav_cols, pages):
    with col:
        st.page_link(
            f"pages/{page_name}.py",
            label=f"{icon}\n{label}",
            use_container_width=True,
        )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── 実行中のタスク ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">実行中のタスク</div>', unsafe_allow_html=True)

_pl = st.session_state.get("pipeline", {})
# session_state が空（Streamlit再起動直後など）のときはファイルから読む
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
    st.markdown(
        '<span style="color:#2a6080;font-size:0.8rem;">現在実行中のタスクはありません。</span>',
        unsafe_allow_html=True,
    )
else:
    _step        = _pl["step"]
    _exp_dir     = _pl.get("experiment_dir", "")
    _scene       = Path(_exp_dir).name if _exp_dir else "不明"
    _start       = _pl.get("start_time", time.time())
    _elapsed     = time.time() - _start
    _step_times  = _pl.get("step_times", {})
    _step_status = _pl.get("step_status", {})

    # 各ステップの開始時刻（直前ステップの完了時刻＝次ステップの開始時刻）
    _step_starts = {
        "extracting": _start,
        "colmap":     _step_times.get("extracting", _start),
        "training":   _step_times.get("colmap", _step_times.get("extracting", _start)),
    }
    _step_name_ja = {"extracting": "フレーム抽出", "colmap": "COLMAP", "training": "3DGS学習"}

    st.markdown(
        f'<div class="task-card running">'
        f'<b>🚀 Pipeline</b> &nbsp;—&nbsp; <code>{_scene}</code> &nbsp;|&nbsp; '
        f'<span style="color:#00aaff">{_step_name_ja.get(_step, _step)} 実行中</span>'
        f'&nbsp;&nbsp;<span style="color:#4a90b8;font-size:0.8rem;">({_elapsed/60:.1f} 分経過)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── ステップ進捗 ──
    STEPS = [("extracting", "① フレーム抽出"), ("colmap", "② COLMAP"), ("training", "③ 3DGS学習")]
    step_cols = st.columns(3)
    for col, (sk, slabel) in zip(step_cols, STEPS):
        st_status = _step_status.get(sk, "waiting")
        if _step == sk and st_status != "done":
            st_status = "running"
        with col:
            if st_status == "done":
                dur = _step_times.get(sk, time.time()) - _step_starts[sk]
                st.markdown(
                    f'<div class="step-done">✅ {slabel}<br>'
                    f'<span style="color:#2a6080">{dur/60:.1f} 分</span></div>',
                    unsafe_allow_html=True,
                )
            elif st_status == "running":
                step_elapsed = time.time() - _step_starts.get(sk, _start)
                st.markdown(
                    f'<div class="step-running">🔄 {slabel}<br>'
                    f'<span style="color:#1a5080">{step_elapsed/60:.1f} 分経過</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<div class="step-waiting">⏳ {slabel}</div>', unsafe_allow_html=True)

    # ── ステップ別進捗バー ─────────────────────────────────────────────────────
    _log_path = _pl.get("log_path")
    if _log_path and Path(_log_path).exists():
        _content = Path(_log_path).read_text(errors="replace")
        _pct, _bar_label = None, ""

        if _step == "extracting":
            if _pl.get("is_360"):
                # convert_360.py: "[i/N] 変換中..."
                _m = re.findall(r'\[(\d+)/(\d+)\]', _content)
                if _m:
                    _cur, _tot = int(_m[-1][0]), int(_m[-1][1])
                    _pct = min(_cur / _tot, 1.0)
                    _bar_label = f"フレーム変換: {_cur} / {_tot} 枚 ({_pct*100:.0f}%)"
            else:
                # extract_frames.py: "PROGRESS_TOTAL N" + "PROGRESS cur/total"
                _tm = re.search(r'PROGRESS_TOTAL (\d+)', _content)
                _pm = re.findall(r'PROGRESS (\d+)/(\d+)', _content)
                if _tm and _pm:
                    _tot = int(_tm.group(1))
                    _cur = int(_pm[-1][0])
                    _pct = min(_cur / _tot, 1.0) if _tot > 0 else None
                    if _pct is not None:
                        _bar_label = f"フレーム抽出: {_cur} / {_tot} 枚 ({_pct*100:.0f}%)"

        elif _step == "colmap":
            _colmap_sub = {
                4: {1: "特徴点抽出",    2: "マッチング",          3: "3D再構成",    4: "undistortion"},
                5: {1: "局所特徴点抽出", 2: "グローバル特徴量抽出", 3: "ペアリスト生成", 4: "マッチング", 5: "SfM再構成"},
            }
            if _pl.get("use_hloc"):
                _m5 = re.findall(r'\[(\d+)/5\]', _content)
                _m4 = re.findall(r'\[(\d+)/4\]', _content)
                _m, _ts = (_m5, 5) if _m5 else (_m4, 4)
            else:
                _m, _ts = re.findall(r'\[COLMAP (\d+)/4\]', _content), 4
            if _m:
                _cur = int(_m[-1])
                _pct = min(_cur / _ts, 1.0)
                _bar_label = (f"ステップ {_cur}/{_ts}: {_colmap_sub[_ts].get(_cur, '')} "
                              f"({_pct*100:.0f}%)")

        elif _step == "training":
            _total = _pl.get("iterations", 30000)
            _tm = re.findall(rf'(\d+)/{_total}', _content)
            if not _tm:
                _tm = re.findall(r'\[ITER\s+(\d+)\]', _content)
            if _tm:
                _cur = int(_tm[-1])
                _pct = min(_cur / _total, 1.0)
                _bar_label = (f"学習進捗: {_cur:,} / {_total:,} iter ({_pct*100:.0f}%)")

        if _pct is not None:
            st.markdown(
                f'<span style="color:#4a90b8;font-size:0.82rem;">{_bar_label}</span>',
                unsafe_allow_html=True,
            )
            st.progress(_pct)
        else:
            st.caption("進捗を取得中...")

    # ── 最新ログ ──
    _log_path = _pl.get("log_path")
    if _log_path and Path(_log_path).exists():
        _lines = [l for l in Path(_log_path).read_text(errors="replace").split("\n") if l.strip()]
        if _lines:
            with st.expander("最新ログ（直近5行）", expanded=False):
                st.code("\n".join(_lines[-5:]), language=None)

    # 更新ボタン
    _rc1, _rc2 = st.columns([1, 7])
    with _rc1:
        if st.button("🔄 更新", key="home_refresh"):
            st.rerun()
    with _rc2:
        st.caption("5秒ごとに自動更新されます")

# ── ToDo ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">ToDo リスト</div>', unsafe_allow_html=True)

todos = load_todos()

# 追加フォーム
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

# タグフィルター
all_tags = sorted(set(tag for t in todos for tag in t.get("tags", [])))
selected_tags = []
if all_tags:
    # タグバッジ色をラベルに反映できないため multiselect で対応
    selected_tags = st.multiselect(
        "タグで絞り込み",
        options=all_tags,
        format_func=lambda t: f"#{t}",
        placeholder="タグを選択（複数可）",
        key="tag_filter_select",
    )

# 表示するタスクを決定
display_items = [
    (i, t) for i, t in enumerate(todos)
    if not selected_tags or any(tag in t.get("tags", []) for tag in selected_tags)
]

# ToDo 一覧表示
if not todos:
    st.markdown(
        '<span style="color:#2a6080;font-size:0.8rem;">タスクはまだありません。上のフォームから追加できます。</span>',
        unsafe_allow_html=True,
    )
elif not display_items:
    st.markdown(
        '<span style="color:#2a6080;font-size:0.8rem;">選択したタグに一致するタスクはありません。</span>',
        unsafe_allow_html=True,
    )
else:
    for orig_i, todo in display_items:
        chk_col, t_col, d_col = st.columns([0.4, 8, 0.7])
        with chk_col:
            checked = st.checkbox(
                "done",
                value=todo["done"],
                key=f"todo_{todo['id']}",
                label_visibility="collapsed",
            )
            if checked != todo["done"]:
                todos[orig_i]["done"] = checked
                save_todos(todos)
                st.rerun()
        with t_col:
            text_html = render_text_with_tags(todo["text"])
            style = "text-decoration:line-through;color:#2a6080;" if todo["done"] else "color:#e0e6f0;"
            st.markdown(
                f'<div style="{style}font-size:0.88rem;line-height:2;">{text_html}</div>',
                unsafe_allow_html=True,
            )
        with d_col:
            if st.button("🗑️", key=f"del_{todo['id']}", help="削除"):
                todos.pop(orig_i)
                save_todos(todos)
                st.rerun()

    # 完了済みをまとめて削除
    done_count = sum(1 for t in todos if t["done"])
    if done_count > 0:
        if st.button(f"完了済み {done_count} 件を削除", use_container_width=False):
            todos = [t for t in todos if not t["done"]]
            save_todos(todos)
            st.rerun()

# ── 使用方法（トグル） ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">使用方法</div>', unsafe_allow_html=True)

with st.expander("使用方法を表示する", expanded=False):
    st.markdown("""
### パイプライン全体の流れ

```
[Step 1] 動画を data/movies/ または data/360movies/ に配置
    ↓
[Step 2] Frame Extraction（フレーム切り出し、360度はピンホール変換も）
    ↓
[Step 3] COLMAP Estimation（カメラ姿勢推定）
    ↓
[Step 4] 3DGS Training（Gaussian Splatting 学習）
    ↓
[Step 5] Results Viewer（評価・可視化）
```

> 全ステップ自動実行は **Pipeline Runner** から。

---

### 各ページの説明

| ページ | 内容 |
|---|---|
| 🚀 Pipeline Runner | ステップをまとめて自動実行する（ここから始めるのが最速） |
| 🎞️ Frame Extraction | 動画から連番画像を切り出す（360度変換オプションあり） |
| 📐 COLMAP Estimation | COLMAP でカメラ姿勢を推定する |
| 🧠 3DGS Training | Gaussian Splatting の学習を実行する |
| 🖼️ Results Viewer | 学習結果・レンダリング結果・メモを確認する |
| 📊 Compare Results | 複数の実験結果を比較する |
| 🗂️ Experiment Manager | 実験一覧・ディスク使用量・フォルダ削除 |
| ⚡ System Monitor | GPU / CPU / メモリのリアルタイム監視 |

---

### 実験フォルダの構造

実験結果は `experiments/YYYYMMDD_HHMMSS_<scene_name>/` に保存されます。

```
experiments/
└── 20240420_120000_scene1/
    ├── config.yaml      # 実験設定
    ├── frames/          # 切り出し画像
    ├── colmap/          # COLMAP 出力
    ├── output/          # 3DGS 学習結果
    ├── renders/         # レンダリング結果
    └── logs/            # 学習ログ
```

---

### よくある操作

**動画からフレームを切り出す（CLI）**
```bash
python scripts/extract_frames.py \\
  --input data/scene1/video.mp4 \\
  --output experiments/YYYYMMDD_HHMMSS_scene1/frames/
```

**COLMAP を実行する（CLI）**
```bash
python scripts/run_colmap.py \\
  --image_path experiments/YYYYMMDD_HHMMSS_scene1/frames/
```

**3DGS 学習を実行する（CLI）**
```bash
python scripts/run_train.py \\
  --source experiments/YYYYMMDD_HHMMSS_scene1/
```

---

### 注意事項

- `data/` フォルダ内のファイルは **絶対に削除しない**でください。
- GPU を使う長時間処理は実行前に必ず確認しましょう。
- 360度動画の場合はピンホール変換を先に行います。
""")

# ── パイプライン実行中は5秒ごと自動更新（全UI描画後に実行） ──────────────────────
if _pipeline_active:
    time.sleep(5)
    st.rerun()
