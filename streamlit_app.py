# エントリーポイント：st.navigation でページ管理
# ホーム画面のコンテンツ（ToDo・進捗・使用方法）も含む

import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

st.set_page_config(
    page_title="3DGS Lab",
    page_icon="🔬",
    layout="wide",
)

# ── グローバルスタイル ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,700;1,400&display=swap');

:root {
  --bg:      #050a14;
  --bg-1:    #0c1a28;
  --bg-2:    #122030;
  --border:  #1a3a5c;
  --glow:    rgba(0,229,255,0.18);
  --cyan:    #00e5ff;
  --green:   #00ff9d;
  --orange:  #ff6b35;
  --red:     #ff4455;
  --t1:      #ddf0ff;
  --t2:      #4a90b8;
  --t3:      #243a50;
}

/* ── ベース ─────────────────────────────────── */
.stApp {
  background-color: var(--bg) !important;
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px) !important;
  background-size: 48px 48px !important;
}

html, body, [class*="st-"] {
  font-family: 'JetBrains Mono', monospace !important;
}

.main .block-container {
  background: transparent !important;
  padding-top: 1.5rem !important;
}

/* ── サイドバー ──────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #080f1c 0%, #050a14 100%) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebarNavSectionHeader"] span {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.5rem !important;
  letter-spacing: 0.22em !important;
  color: var(--t3) !important;
}

[data-testid="stSidebarNavLink"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.76rem !important;
  letter-spacing: 0.03em !important;
  border-radius: 2px !important;
  color: var(--t2) !important;
  transition: all 0.15s !important;
}
[data-testid="stSidebarNavLink"]:hover {
  background: rgba(0,229,255,0.07) !important;
  color: var(--cyan) !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: rgba(0,229,255,0.09) !important;
  border-left: 2px solid var(--cyan) !important;
  color: var(--cyan) !important;
}

/* ── 見出し ──────────────────────────────────── */
h1, h2, h3 {
  font-family: 'Orbitron', sans-serif !important;
  color: var(--t1) !important;
  letter-spacing: 0.08em !important;
}
h1 { font-size: 1.5rem !important; }
h2 { font-size: 1.1rem !important; }
h3 { font-size: 0.9rem !important; color: var(--t2) !important; }

/* ── ボタン ──────────────────────────────────── */
.stButton > button {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.06em !important;
  border-radius: 2px !important;
  transition: all 0.15s !important;
  border: 1px solid var(--border) !important;
  background: transparent !important;
  color: var(--t2) !important;
}
.stButton > button:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
  box-shadow: 0 0 12px var(--glow) !important;
  background: rgba(0,229,255,0.05) !important;
}
.stButton > button[kind="primary"] {
  background: var(--cyan) !important;
  border-color: var(--cyan) !important;
  color: #050a14 !important;
  font-weight: 700 !important;
  box-shadow: 0 0 16px var(--glow) !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 28px rgba(0,229,255,0.5) !important;
}
.stButton > button[kind="secondary"] {
  border-color: var(--t3) !important;
  color: var(--t2) !important;
}

/* ── インプット ──────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  color: var(--t1) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
  border-radius: 2px !important;
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 8px var(--glow) !important;
}

/* ── セレクトボックス ────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  color: var(--t1) !important;
}

/* ── プログレスバー ───────────────────────────── */
[data-testid="stProgressBar"] > div {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  height: 6px !important;
}
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--cyan) 0%, var(--green) 100%) !important;
  box-shadow: 0 0 8px rgba(0,229,255,0.5) !important;
  border-radius: 2px !important;
}

/* ── アラート ────────────────────────────────── */
[data-testid="stInfo"] {
  background: rgba(0,229,255,0.05) !important;
  border: 1px solid rgba(0,229,255,0.2) !important;
  border-radius: 2px !important;
}
[data-testid="stSuccess"] {
  background: rgba(0,255,157,0.05) !important;
  border: 1px solid rgba(0,255,157,0.2) !important;
  border-radius: 2px !important;
}
[data-testid="stWarning"] {
  background: rgba(255,107,53,0.06) !important;
  border: 1px solid rgba(255,107,53,0.22) !important;
  border-radius: 2px !important;
}
[data-testid="stError"] {
  background: rgba(255,68,85,0.06) !important;
  border: 1px solid rgba(255,68,85,0.22) !important;
  border-radius: 2px !important;
}

/* ── エクスパンダー ──────────────────────────── */
[data-testid="stExpander"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.8rem !important;
  color: var(--t2) !important;
}

/* ── タブ ────────────────────────────────────── */
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-baseweb="tab"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.05em !important;
  color: var(--t3) !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
  padding: 0.5rem 1.2rem !important;
}
[data-baseweb="tab"]:hover { color: var(--t2) !important; }
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--cyan) !important;
  border-bottom-color: var(--cyan) !important;
}

/* ── コードブロック ───────────────────────────── */
[data-testid="stCode"], [data-testid="stCodeBlock"],
pre, code {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── メトリクス ──────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] p {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--t3) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Orbitron', sans-serif !important;
  color: var(--cyan) !important;
}

/* ── チェックボックス・ラジオ ─────────────────── */
.stCheckbox label, .stRadio label {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
  color: var(--t1) !important;
}

/* ── Divider ─────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  opacity: 1 !important;
  margin: 1.25rem 0 !important;
}

/* ── キャプション ────────────────────────────── */
[data-testid="stCaptionContainer"] p,
small, .caption {
  font-family: 'JetBrains Mono', monospace !important;
  color: var(--t3) !important;
  font-size: 0.72rem !important;
}

/* ── スクロールバー ──────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── フォーム ────────────────────────────────── */
[data-testid="stForm"] {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
}

/* ── スライダー ──────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
  background: var(--cyan) !important;
  box-shadow: 0 0 8px var(--glow) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ナビゲーション定義
# ══════════════════════════════════════════════════════════════════════════════

pg = st.navigation(
    {
        "": [
            st.Page("pages/home.py",    title="ホーム",           icon="🏠", default=True),
            st.Page("pages/monitor.py", title="システムモニター", icon="⚡"),
        ],
        "🧪 パイプライン": [
            st.Page("pages/00_pipeline.py",         title="Pipeline Runner", icon="🚀"),
            st.Page("pages/01_frame_extraction.py", title="フレーム抽出",    icon="🎞️"),
            st.Page("pages/02_colmap.py",           title="姿勢推定",         icon="📷"),
            st.Page("pages/03_training.py",         title="3DGS学習",         icon="🧠"),
        ],
        "📊 結果・管理": [
            st.Page("pages/04_results.py",            title="結果確認",   icon="🖼️"),
            st.Page("pages/05_compare.py",            title="実験比較",   icon="📊"),
            st.Page("pages/06_experiment_manager.py", title="実験管理",   icon="🗂️"),
        ],
        "🎮 ゲーム": [
            st.Page("pages/07_minigame.py", title="ミニゲーム", icon="⚗️"),
            st.Page("pages/08_pet.py",      title="ガウスくん", icon="🐾"),
        ],
    },
    expanded=True,
)

pg.run()
