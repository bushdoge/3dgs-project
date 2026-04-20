# ホーム画面：ナビゲーション・ToDo管理・使用方法の表示

import json
import os
import streamlit as st
from datetime import datetime

TODO_FILE = "/workspace/tmp/todo.json"

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
    ("⚡", "System\nMonitor",    "0_monitor"),
    ("🎞️", "Frame\nExtraction",  "1_frame_extraction"),
    ("📐", "COLMAP\nEstimation", "2_colmap"),
    ("🧠", "3DGS\nTraining",     "3_training"),
    ("🖼️", "Results\nViewer",    "4_results"),
    ("⚙️", "Pipeline\nRunner",   "5_pipeline"),
    ("📊", "Compare\nResults",   "6_compare"),
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

# ── ToDo ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">ToDo リスト</div>', unsafe_allow_html=True)

todos = load_todos()

# セッション初期化
if "todo_edit_mode" not in st.session_state:
    st.session_state.todo_edit_mode = False

# 追加フォーム
with st.form("add_todo", clear_on_submit=True):
    add_col, btn_col = st.columns([5, 1])
    with add_col:
        new_task = st.text_input("新しいタスク", placeholder="タスクを入力...",
                                 label_visibility="collapsed")
    with btn_col:
        submitted = st.form_submit_button("追加", use_container_width=True)

    if submitted and new_task.strip():
        todos.append({
            "id": datetime.now().isoformat(),
            "text": new_task.strip(),
            "done": False,
        })
        save_todos(todos)
        st.rerun()

# ToDo 一覧表示（完了トグル + 削除ボタン）
if not todos:
    st.markdown(
        '<span style="color:#2a6080;font-size:0.8rem;">タスクはまだありません。上のフォームから追加できます。</span>',
        unsafe_allow_html=True,
    )
else:
    for i, todo in enumerate(todos):
        t_col, d_col = st.columns([8, 1])
        with t_col:
            checked = st.checkbox(
                todo["text"],
                value=todo["done"],
                key=f"todo_{todo['id']}",
            )
            if checked != todo["done"]:
                todos[i]["done"] = checked
                save_todos(todos)
                st.rerun()
        with d_col:
            if st.button("🗑️", key=f"del_{todo['id']}", help="削除"):
                todos.pop(i)
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
[Step 1] 動画 / 画像を data/<scene_name>/ に配置
    ↓
[Step 2] Frame Extraction（FFmpeg でフレーム切り出し）
    ↓
[Step 3] COLMAP Estimation（カメラ姿勢推定）
    ↓
[Step 4] 3DGS Training（Gaussian Splatting 学習）
    ↓
[Step 5] Results Viewer（評価・可視化）
```

---

### 各ページの説明

| ページ | 内容 |
|---|---|
| ⚡ System Monitor | GPU / CPU / メモリのリアルタイム監視 |
| 🎞️ Frame Extraction | 動画から連番画像を切り出す |
| 📐 COLMAP Estimation | COLMAP でカメラ姿勢を推定する |
| 🧠 3DGS Training | Gaussian Splatting の学習を実行する |
| 🖼️ Results Viewer | 学習結果・レンダリング結果を確認する |
| ⚙️ Pipeline Runner | ステップをまとめて自動実行する |
| 📊 Compare Results | 複数の実験結果を比較する |

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
