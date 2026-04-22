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
    ("🚀", "Pipeline\nRunner",    "00_pipeline"),
    ("🎞️", "Frame\nExtraction",   "01_frame_extraction"),
    ("📐", "COLMAP\nEstimation",  "02_colmap"),
    ("🧠", "3DGS\nTraining",      "03_training"),
    ("🖼️", "Results\nViewer",     "04_results"),
    ("📊", "Compare\nResults",    "05_compare"),
    ("🗂️", "Experiment\nManager", "06_experiment_manager"),
    ("⚡", "System\nMonitor",     "monitor"),
    ("⚗️", "Mini\nGame",          "07_minigame"),
    ("🐾", "Pet\nGaus",           "08_pet"),
]

for row_pages in [pages[:5], pages[5:]]:
    nav_cols = st.columns(5)
    for col, (icon, label, page_name) in zip(nav_cols, row_pages):
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
        _pct, _bar_label, _substep_rendered = None, "", False

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
            try:
                from pipeline_widget import _parse_colmap_substeps, _render_substep_bars
                _substeps = _parse_colmap_substeps(_pl)
                if _substeps:
                    _render_substep_bars(_substeps)
                    _substep_rendered = True
            except Exception:
                pass

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
        elif not _substep_rendered:
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
[Step 2] Frame Extraction
         通常動画 → FFmpeg でフレーム抽出
         360度動画 → 8方向ピンホール変換（方向・FOV・解像度を選択）
    ↓
[Step 3] カメラ姿勢推定
         COLMAP（シンプル・高互換）
         または HLoc（高精度・SuperPoint/LightGlue等）
           └─ ペア生成: Exhaustive（全ペア・精度優先）
                       / Retrieval（NetVLAD等で類似画像を選択・速度優先）
    ↓
[Step 4] 3DGS Training（Gaussian Splatting 学習）
    ↓
[Step 5] Results Viewer（評価・可視化・レンダリング実行）
```

> **全ステップ自動実行は 🚀 Pipeline Runner から。**
> Streamlit を再起動してもパイプラインの状態は自動復元されます。
> 各ステップのログ先頭には実験設定が記録されます。

---

### 各ページの説明

| ページ | 主な機能 |
|---|---|
| 🚀 Pipeline Runner | フレーム抽出→姿勢推定→学習を一括自動実行。**設定プリセット**で姿勢推定・学習パラメータを保存・呼び出せる。進捗はホーム画面にも表示される |
| 🎞️ Frame Extraction | 動画から連番画像を切り出す。360度動画は方向（8×3グリッド）・FOV・解像度を指定してピンホール変換 |
| 📐 COLMAP / HLoc | COLMAP または HLoc でカメラ姿勢推定（SfM）。HLoc はペア生成方式（Exhaustive / Retrieval）と特徴量・マッチャーを詳細設定できる。完了後にカメラ位置の3D可視化あり |
| 🧠 3DGS Training | Gaussian Splatting の学習を実行。リアルタイムでログ・PSNR グラフを表示。学習中断ボタンあり |
| 🖼️ Results Viewer | パイプライン進捗・**COLMAP再構成品質**（登録率・3D点数・再投影誤差）・保存済み point_cloud・PSNR推移・**レンダリング実行**・画像・動画生成・メモを確認 |
| 📊 Compare Results | 複数実験を選択して **PSNR・L1 Loss の学習曲線を重ね比較**。split（test/train）と指標を切り替え可能。最良値サマリーテーブルも表示 |
| 🗂️ Experiment Manager | 実験一覧・ディスク管理。**ログ閲覧**（フレーム抽出/COLMAP/学習/レンダリングを個別タブ）・**設定確認**（config.yaml・cfg_args）・メモ編集・フォルダ削除 |
| ⚡ System Monitor | GPU / CPU / メモリのリアルタイム監視とパイプライン進捗ウィジェット |
| ⚗️ Mini Game | ガウシアンを育てるアイドルゲーム。プレステージで強化パックがもらえる |
| 🐾 Pet Gaus | たまごっち風ペット育成。実時間でステータスが変化。パイプライン完了・ログイン・マイルストーン達成でアイテム獲得 |

---

### 設定プリセットの使い方（🚀 Pipeline Runner）

よく使う姿勢推定・学習パラメータを名前をつけて保存しておくと、次回以降の実験で素早く呼び出せます。

1. **保存**: Pipeline Runner で姿勢推定・学習設定を決めてから「📌 設定プリセット」を開き、名前を入力して「💾 保存」
2. **読み込み**: 「📌 設定プリセット」でプリセット名を選択して「📂 読み込む」→ 設定欄に即反映
3. **削除**: 「🗑️ 削除」ボタンで不要なプリセットを削除

> プリセットには姿勢推定（HLoc/COLMAP・特徴量・マッチャー等）と学習（iter数・保存タイミング等）の設定が保存されます。FPS・360°設定は入力素材ごとに異なるため対象外。

---

### COLMAP 再構成品質の目安（🖼️ Results Viewer）

COLMAP / HLoc が完了した実験では、`sparse/0/` の情報から以下の品質指標が表示されます。

| 指標 | 目安 |
|---|---|
| 登録率（登録カメラ数 / 入力枚数） | 80% 以上で良好。50% 未満なら特徴点不足や類似フレームが多すぎる可能性 |
| 3D 点数 | 多いほど点群が密。少なすぎると学習結果も荒くなる |
| 平均再投影誤差 | 1.0 px 未満で良好。2.0 px 超は要確認（画像ブレ・光量変化が原因のことが多い） |

品質が低い場合は、FPS を下げてフレーム数を減らす / HLoc の特徴量を変える / 動画の撮影条件を見直すことを検討してください。

---

### 実験フォルダの構造

実験結果は `experiments/YYYYMMDD_HHMMSS_<scene_name>/` に保存されます。

```
experiments/
└── 20260421_123456_CenterForest/
    ├── input/                   # 変換・抽出済み画像（COLMAP/HLoc の入力）
    ├── sparse/0/                # SfM 結果（cameras/images/points3D .txt or .bin）
    ├── output/                  # 3DGS 学習結果
    │   ├── point_cloud/         # イテレーション別 point_cloud.ply
    │   ├── train_log.txt        # 学習ログ（PSNR・L1 Lossを含む）
    │   └── cfg_args             # gaussian-splatting の学習引数（自動生成）
    ├── renders/                 # レンダリング結果
    ├── config.yaml              # 実験設定（Pipeline Runner が自動生成）
    ├── extract_log.txt          # フレーム抽出ログ
    ├── colmap_log.txt           # 姿勢推定ログ
    └── note.md                  # 自由メモ（Results Viewer / Experiment Manager から編集可）
```

---

### よくある操作（CLI）

**通常動画からフレームを切り出す**
```bash
python scripts/extract_frames.py \\
  --input data/movies/scene1.mp4 \\
  --output experiments/20260421_scene1/input/ \\
  --fps 1.0
```

**360度動画をピンホール変換する**
```bash
python scripts/convert_360.py \\
  --input data/360movies/scene1.mp4 \\
  --output experiments/20260421_scene1/input/ \\
  --fov 90 --width 1024 --height 1024 --fps 1.0
```

**HLoc でカメラ姿勢推定する**
```bash
python scripts/run_hloc.py \\
  --source_path experiments/20260421_scene1/ \\
  --feature_type superpoint_max \\
  --matcher_type superpoint+lightglue \\
  --pair_method retrieval --retrieval_model netvlad
```

**3DGS 学習を実行する**
```bash
python scripts/run_train.py \\
  --source experiments/20260421_scene1/ \\
  --iterations 30000
```

**レンダリングを実行する**
```bash
python scripts/run_render.py \\
  -m experiments/20260421_scene1/output/ \\
  -s experiments/20260421_scene1/ \\
  --iteration 30000
```

---

### 注意事項

- `data/` フォルダ内のファイルは **絶対に削除しない**でください（元動画・元画像が入っています）。
- 3DGS 学習・レンダリングなど GPU を長時間占有する処理は、実行前に他の処理が動いていないか確認してください。
- パイプライン実行中に `.py` ファイルを編集して Streamlit がリロードされても、サブプロセス（COLMAP・学習）は止まりません。
- プリセットファイルは `tmp/pipeline_presets.json` に保存されます。
""")

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass

# ── パイプライン実行中は5秒ごと自動更新（全UI描画後に実行） ──────────────────────
if _pipeline_active:
    time.sleep(5)
    st.rerun()
