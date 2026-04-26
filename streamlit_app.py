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
            st.Page("pages/11_batch.py",            title="バッチ実験",       icon="🗂️"),
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
            st.Page("pages/09_poker.py",    title="ポーカー",   icon="🃏"),
        ],
    },
    expanded=True,
)

pg.run()
