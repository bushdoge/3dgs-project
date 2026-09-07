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

# ── 全ページ共通スタイル（Material 3 Expressive）───────────────────────────────
# 色・タイポ・角丸の「値」は .streamlit/config.toml のテーマAPIで指定している。
# ここで書くのはAPIで表現できないもの＝形の変化・動き・状態レイヤ・面の階層。
#
# M3 Expressive として意図的に効かせている点:
#   1. タイポの落差 … 見出しを display 級まで大きく重くする（config の headingFontSizes/Weights）
#   2. 形の変化     … ボタンは pill、押すと角が締まる。カードはホバーで角の形が変わる（シェイプモーフ）
#   3. 動き         … 直線的なeaseではなくバネ曲線で行き過ぎてから戻る
#   4. 色面         … 半透明の白ではなく tonal container を「広い面」で使う
#   5. 面の階層     … surface container を5段持ち、重ねる要素ほど明るくする
# 元の意匠に戻す: bash tmp/ui_restore_20260903_0330.sh
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:opsz,wght@8..144,300..900&family=Roboto+Mono:wght@400;500&display=swap');

:root{
  /* ── M3 カラーロール（ダークスキーム / seed = コーラル）── */
  --m3-primary:#FFB59B;             --m3-on-primary:#5B1B00;
  --m3-primary-container:#8E3A17;   --m3-on-primary-container:#FFDBCF;
  --m3-secondary:#E7BDAC;           --m3-on-secondary:#442A1E;
  --m3-secondary-container:#5D4033; --m3-on-secondary-container:#FFDBCF;
  --m3-tertiary:#EFC77A;            --m3-on-tertiary:#402D00;
  --m3-tertiary-container:#5C4300;  --m3-on-tertiary-container:#FFDFA6;
  --m3-error:#FFB4AB;               --m3-error-container:#93000A;
  --m3-success:#8ED9A8;             --m3-success-container:#22503A;

  --m3-surface:#14181F;
  --m3-surface-lowest:#0D1117;
  --m3-surface-low:#1A1F26;
  --m3-surface-c:#1E232B;
  --m3-surface-high:#282D35;
  --m3-surface-highest:#333840;
  --m3-on-surface:#EAE6DC;
  --m3-on-surface-var:#CFC7BC;
  --m3-outline:#998F84;
  --m3-outline-var:#4B443C;

  /* ── Expressive シェイプスケール ── */
  --m3-r-xs:8px; --m3-r-s:12px; --m3-r-m:16px;
  --m3-r-l:20px; --m3-r-xl:28px; --m3-r-full:999px;

  /* ── Expressive モーション（バネ＝行き過ぎてから戻る）── */
  --m3-spring:cubic-bezier(.34,1.56,.64,1);
  --m3-spring-soft:cubic-bezier(.22,1.2,.36,1);
  --m3-emph:cubic-bezier(.2,0,0,1);

  --m3-e1:0 1px 3px rgba(0,0,0,.35), 0 1px 2px rgba(0,0,0,.25);
  --m3-e2:0 2px 8px 2px rgba(0,0,0,.26), 0 1px 2px rgba(0,0,0,.35);
  --m3-e3:0 6px 16px 4px rgba(0,0,0,.28), 0 2px 4px rgba(0,0,0,.35);
}

.block-container{ padding-top:2.2rem; padding-bottom:3.4rem; max-width:1180px; }

/* ══ 見出し ══════════════════════════════════════════════════════════════════
   大きさ・太さは config の headingFontSizes / headingFontWeights が持つ。
   ここでは Expressive の「形で強調する」＝pill形のアクセントと字間だけ足す。         */
h1{ letter-spacing:-.025em !important; line-height:1.08 !important; margin-bottom:.2rem; }
h2{ letter-spacing:-.015em !important; margin-top:1.8rem !important; }
h3{ letter-spacing:-.008em !important; }
h2, h3{ position:relative; padding-left:.9rem !important; }
h2::before, h3::before{
  content:""; position:absolute; left:0; top:.22em; bottom:.22em;
  width:6px; border-radius:var(--m3-r-full);
  background:linear-gradient(180deg, var(--m3-primary), var(--m3-tertiary));
}
/* h1 は display 扱い。下に太いtertiaryの下線を敷いて「一番大きい情報」を明示する */
h1::after{
  content:""; display:block; width:72px; height:6px; margin-top:.55rem;
  border-radius:var(--m3-r-full);
  background:linear-gradient(90deg, var(--m3-primary), var(--m3-tertiary));
}

/* ══ サイドバー = M3 ナビゲーションドロワー ═══════════════════════════════════
   Expressive のドロワーは項目が大きく、選択中は塗りつぶしの pill になる。          */
[data-testid="stSidebarNav"] a{
  border-radius:var(--m3-r-full) !important;
  margin:3px 10px; padding:.62rem 1.1rem !important; min-height:46px;
  color:var(--m3-on-surface-var) !important;
  transition:background .28s var(--m3-emph), color .28s var(--m3-emph),
             padding-left .4s var(--m3-spring);
}
[data-testid="stSidebarNav"] a:hover{
  background:var(--m3-surface-high) !important; padding-left:1.45rem !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNav"] li > div[aria-selected="true"] a{
  background:var(--m3-secondary-container) !important;
  color:var(--m3-on-secondary-container) !important;
  font-variation-settings:'wght' 700;
}

/* ══ ボタン = M3 filled / tonal（pill + シェイプモーフ）══════════════════════
   角丸そのものは config の buttonRadius="full"。ここは押したときの形の変化と動き。 */
.stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] button{
  min-height:44px; padding:.5rem 1.6rem !important; border:none !important;
  background:var(--m3-secondary-container) !important;
  color:var(--m3-on-secondary-container) !important;
  font-variation-settings:'wght' 650; letter-spacing:.01em;
  transition:border-radius .34s var(--m3-spring), transform .34s var(--m3-spring),
             box-shadow .25s var(--m3-emph), filter .2s var(--m3-emph);
}
.stButton > button:hover, .stFormSubmitButton > button:hover,
[data-testid="stDownloadButton"] button:hover{
  filter:brightness(1.14); box-shadow:var(--m3-e2); transform:translateY(-2px);
}
/* 押すと角が締まって縮む ＝ Expressive のシェイプモーフ */
.stButton > button:active, .stFormSubmitButton > button:active,
[data-testid="stDownloadButton"] button:active{
  border-radius:var(--m3-r-s) !important; transform:scale(.95); filter:brightness(1.22);
}
/* primary は「一番大きい操作」として他より一回り大きく塗る（Expressiveの強弱） */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--m3-primary) !important; color:var(--m3-on-primary) !important;
  font-variation-settings:'wght' 750;
  min-height:52px; padding:.6rem 2.1rem !important; font-size:1rem !important;
}
.stButton > button:focus-visible{ outline:3px solid var(--m3-tertiary) !important; outline-offset:3px; }

/* ══ カード = surface container（枠線ではなく面で区切る）══════════════════════
   ホバーで角の形が非対称に変わる＝M3 Expressive のシェイプモーフ。                */
[data-testid="stMetric"]{
  background:var(--m3-surface-c); border:none;
  border-radius:var(--m3-r-xl); padding:1.1rem 1.4rem;
  transition:background .28s var(--m3-emph), transform .38s var(--m3-spring),
             box-shadow .28s var(--m3-emph), border-radius .38s var(--m3-spring);
}
[data-testid="stMetric"]:hover{
  background:var(--m3-surface-high); transform:translateY(-3px); box-shadow:var(--m3-e2);
  border-radius:var(--m3-r-xl) var(--m3-r-xl) var(--m3-r-xl) 10px;
}
[data-testid="stMetricLabel"]{
  color:var(--m3-on-surface-var) !important; letter-spacing:.06em;
  text-transform:uppercase; font-size:.7rem !important;
}
[data-testid="stMetricValue"]{ letter-spacing:-.03em; line-height:1.05; }

[data-testid="stExpander"]{
  background:var(--m3-surface-c); border:none !important;
  border-radius:var(--m3-r-l) !important; overflow:hidden;
  transition:background .28s var(--m3-emph);
}
[data-testid="stExpander"]:hover{ background:var(--m3-surface-high); }
[data-testid="stExpander"] summary{ padding:.85rem 1.2rem; font-variation-settings:'wght' 600; }
/* 開いている間は secondaryContainer で塗る＝「いまここ」を色面で示す */
[data-testid="stExpander"] details[open] > summary{
  background:var(--m3-secondary-container); color:var(--m3-on-secondary-container);
}

/* ══ タブ = M3 セグメンテッドボタン ═══════════════════════════════════════════ */
[data-baseweb="tab-list"]{
  gap:4px; background:var(--m3-surface-c); padding:6px;
  border-radius:var(--m3-r-full); border-bottom:none !important; width:fit-content;
}
[data-baseweb="tab"]{
  border-radius:var(--m3-r-full) !important; padding:.45rem 1.25rem !important;
  color:var(--m3-on-surface-var) !important; border:none !important;
  transition:background .28s var(--m3-emph), color .28s var(--m3-emph),
             transform .34s var(--m3-spring);
}
[data-baseweb="tab"]:hover{ background:var(--m3-surface-high); }
[data-baseweb="tab"][aria-selected="true"]{
  background:var(--m3-secondary-container) !important;
  color:var(--m3-on-secondary-container) !important;
  font-variation-settings:'wght' 700; transform:scale(1.04);
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none !important; }

/* ══ 入力 = M3 filled テキストフィールド ══════════════════════════════════════ */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"]{
  background:var(--m3-surface-high) !important;
  border:2px solid transparent !important;
  transition:border-color .2s var(--m3-emph), background .2s var(--m3-emph);
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"]:focus-within{
  border-color:var(--m3-primary) !important; background:var(--m3-surface-highest) !important;
}

/* ══ ページリンク = M3 tonal カード ═══════════════════════════════════════════ */
[data-testid="stPageLink-NavLink"]{
  border:none; border-radius:var(--m3-r-l); padding:.8rem .6rem; justify-content:center;
  background:var(--m3-surface-c); color:var(--m3-on-surface-var);
  transition:background .28s var(--m3-emph), transform .38s var(--m3-spring),
             box-shadow .28s var(--m3-emph), border-radius .38s var(--m3-spring);
}
[data-testid="stPageLink-NavLink"]:hover{
  background:var(--m3-secondary-container); color:var(--m3-on-secondary-container);
  transform:translateY(-4px); box-shadow:var(--m3-e2);
  border-radius:var(--m3-r-xl) 10px var(--m3-r-xl) 10px;   /* シェイプモーフ */
}
[data-testid="stPageLink-NavLink"]:active{ transform:scale(.96); border-radius:var(--m3-r-s); }

/* ══ 進捗 = M3 リニアインジケータ ═════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div{
  height:14px; border-radius:var(--m3-r-full); background:var(--m3-surface-high);
}
[data-testid="stProgress"] div[role="progressbar"] > div{
  background:linear-gradient(90deg, var(--m3-primary), var(--m3-tertiary)) !important;
  border-radius:var(--m3-r-full);
  transition:width .6s var(--m3-spring-soft);
}
/* スピナー: 角丸が回りながら変形する＝Expressive のローディング表現 */
[data-testid="stSpinner"] > div{ color:var(--m3-primary); }

/* ══ 表・画像・コード ═════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"]{ border-radius:var(--m3-r-m); overflow:hidden; }
[data-testid="stImage"] img{ border-radius:var(--m3-r-m); }
.stCode, pre{ background:var(--m3-surface-lowest) !important; border-radius:var(--m3-r-s) !important; }
.stCode, pre, code{ font-size:.78rem !important; }

hr{ margin:1.2rem 0; border-color:var(--m3-outline-var); }

::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{
  background:var(--m3-surface-highest); border-radius:var(--m3-r-full);
  border:2px solid var(--m3-surface); }
::-webkit-scrollbar-thumb:hover{ background:var(--m3-outline); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ナビゲーション定義
# ══════════════════════════════════════════════════════════════════════════════

pg = st.navigation(
    {
        "": [
            st.Page("pages/90_home.py",    title="ホーム",           icon="🏠", default=True),
            st.Page("pages/92_jobs.py",    title="作業モニター",     icon="📋"),
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
            st.Page("pages/73_cauldron.py", title="魔女の大釜", icon="🧪"),
            st.Page("pages/74_typing.py",   title="ことのは撃墜", icon="⌨️"),
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
