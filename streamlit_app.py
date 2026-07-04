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

# ── 全ページ共通スタイル ─────────────────────────────────────────────────────
# テーマ本体は .streamlit/config.toml（インクブルー×コーラル）。
# 方針: 角丸+ホバーの「触って気持ちいい」UI。無機質にならないよう温色を差す。
st.markdown("""
<style>
/* コンテンツ幅と余白 */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* 見出し: 左にコーラルのアクセントバーを付けて階層を可視化 */
h1 { font-size: 1.7rem !important; letter-spacing: .01em; }
h2, h3 { position: relative; padding-left: .65rem !important; }
h2::before, h3::before {
  content: ""; position: absolute; left: 0; top: .28em; bottom: .28em;
  width: 4px; border-radius: 2px;
  background: linear-gradient(180deg, #ff8552, #ffd166);
}
h2 { font-size: 1.28rem !important; margin-top: 1.1rem; }
h3 { font-size: 1.06rem !important; }

/* サイドバー: ナビ項目を丸く、ホバーでじわっと */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.07); }
[data-testid="stSidebarNav"] a {
  border-radius: 10px; transition: background .15s ease, transform .1s ease;
}
[data-testid="stSidebarNav"] a:hover { transform: translateX(2px); }

/* カード類: 角丸+薄枠+ホバーで持ち上がる */
[data-testid="stMetric"], [data-testid="stExpander"] {
  background: rgba(255,255,255,.025);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 14px;
  transition: transform .12s ease, border-color .12s ease;
}
[data-testid="stMetric"] { padding: 12px 16px; }
[data-testid="stMetric"]:hover, [data-testid="stExpander"]:hover {
  border-color: rgba(255,133,82,.45);
}

/* ボタン: コーラル系。primaryは濃色文字（白文字だと読みづらい）*/
.stButton > button, .stFormSubmitButton > button {
  border-radius: 12px; transition: transform .1s ease, box-shadow .1s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 3px 12px rgba(255,133,82,.25);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  color: #26150c !important; font-weight: 700;
}

/* ページリンク(クイックアクセス等)をタイル化 */
[data-testid="stPageLink-NavLink"] {
  border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
  padding: .55rem .4rem; justify-content: center;
  background: rgba(255,255,255,.03);
  transition: transform .12s ease, border-color .12s ease, background .12s ease;
}
[data-testid="stPageLink-NavLink"]:hover {
  transform: translateY(-2px); border-color: #ff8552; background: rgba(255,133,82,.08);
}

/* ログ・コード表示は小さめに */
.stCode, pre, code { font-size: .78rem !important; }

/* プログレスバー: 太く、コーラル→ゴールドのグラデ */
[data-testid="stProgress"] > div > div { height: 12px; border-radius: 7px; }
[data-testid="stProgress"] div[role="progressbar"] > div {
  background: linear-gradient(90deg, #ff8552, #ffd166) !important;
  border-radius: 7px;
}

/* テーブル・画像の角丸 */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 10px; }

hr { margin: .8rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ナビゲーション定義
# ══════════════════════════════════════════════════════════════════════════════

pg = st.navigation(
    {
        "": [
            st.Page("pages/90_home.py",    title="ホーム",           icon="🏠", default=True),
            st.Page("pages/91_monitor.py", title="システムモニター", icon="⚡"),
        ],
        "🧪 パイプライン": [
            st.Page("pages/00_batch.py",    title="キュー",         icon="🗂️"),
            st.Page("pages/01_pipeline.py", title="パイプライン",   icon="🚀"),
            st.Page("pages/02_frame_extraction.py",  title="フレーム抽出",    icon="🎞️"),
            st.Page("pages/03_colmap.py",            title="姿勢推定",         icon="📷"),
            st.Page("pages/08_sam2_masks.py",        title="SAM2マスク",       icon="🎭"),
            st.Page("pages/04_training.py",          title="3DGS学習",         icon="🧠"),
        ],
        "📊 結果・管理": [
            st.Page("pages/05_results.py",           title="結果確認",   icon="🖼️"),
            st.Page("pages/06_compare.py",           title="実験比較",   icon="📊"),
            st.Page("pages/07_experiment_manager.py", title="実験管理",  icon="🗂️"),
        ],
        "🎮 ゲーム": [
            st.Page("pages/70_minigame.py", title="ミニゲーム", icon="⚗️"),
            st.Page("pages/71_pet.py",      title="ガウスくん", icon="🐾"),
            st.Page("pages/72_poker.py",    title="ポーカー",   icon="🃏"),
        ],
    },
    expanded=True,
)

pg.run()

# 全ページ共通のフッター進捗バー
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
