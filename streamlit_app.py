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
# テーマ本体は .streamlit/config.toml（ダーク+テールアクセント）。ここは微調整のみ。
st.markdown("""
<style>
/* コンテンツ幅と余白を整える（wideでも読みやすい幅に） */
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }

/* 見出しの階層をはっきりさせつつ大きすぎを抑える */
h1 { font-size: 1.75rem !important; letter-spacing: .01em; padding-bottom: .2rem; }
h2 { font-size: 1.3rem  !important; margin-top: 1.2rem; }
h3 { font-size: 1.08rem !important; }

/* サイドバー: 境界を薄く、ナビ項目を角丸に */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebarNav"] a { border-radius: 8px; }

/* カード類（メトリクス・エクスパンダ）に薄い枠と角丸 */
[data-testid="stMetric"] {
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 12px; padding: 10px 14px;
}
[data-testid="stExpander"] {
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 12px;
}

/* ボタンとログ表示 */
.stButton > button { border-radius: 10px; }
.stCode, pre, code { font-size: .78rem !important; }

/* プログレスバーを少し太く */
[data-testid="stProgress"] > div > div { height: 10px; border-radius: 6px; }

/* テーブル・データフレームの角丸 */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

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
