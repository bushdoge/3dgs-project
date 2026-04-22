# ミニゲーム：3DGS Laboratory インクリメンタルゲーム
# ガウシアンを生産して研究施設を拡大するアイドルゲーム
# セーブデータは /workspace/tmp/minigame_save.json に保存される

import sys
sys.path.insert(0, "/workspace")

import json
import math
import random
import time
from pathlib import Path

import streamlit as st

from pipeline_widget import render_pipeline_status

st.set_page_config(page_title="3DGS Laboratory", page_icon="⚗️", layout="wide")

SAVE_FILE = "/workspace/tmp/minigame_save.json"
COST_SCALE = 1.15   # 施設購入ごとのコスト増加率
TICK_CAP   = 3600   # オフライン進捗の上限（秒）
GOLDEN_CHANCE      = 0.003   # 毎秒のゴールデン発生確率
GOLDEN_DURATION    = 30      # ゴールデン持続時間（秒）
GOLDEN_MULTIPLIER  = 10      # ゴールデン中のクリック倍率

# ── ゲームデータ定義 ──────────────────────────────────────────────────────────

GENERATORS = [
    {"id": "gpu",      "name": "Single GPU",      "icon": "🎮",
     "base_cost": 15,       "base_gps": 0.1,
     "flavor": ["古いGTX 1080が唸りを上げる", "ファンが全速力で回っている", "GPUメモリが輝いている"]},
    {"id": "cluster",  "name": "GPU Cluster",     "icon": "🖥️",
     "base_cost": 150,      "base_gps": 0.8,
     "flavor": ["8枚のGPUが同期する", "NVLinkが光る", "消費電力が跳ね上がる"]},
    {"id": "colmap",   "name": "COLMAP Server",   "icon": "📐",
     "base_cost": 1100,     "base_gps": 6.0,
     "flavor": ["特徴点が飛び交う", "カメラ姿勢が収束した！", "SfMが加速している"]},
    {"id": "rig",      "name": "Training Rig",    "icon": "🧠",
     "base_cost": 8000,     "base_gps": 47.0,
     "flavor": ["損失が下がり続ける", "Gaussianが爆発的に増える", "学習が収束しつつある"]},
    {"id": "dc",       "name": "Data Center",     "icon": "🏢",
     "base_cost": 60000,    "base_gps": 350.0,
     "flavor": ["冷却水が滝のように流れる", "サーバーラックが無限に続く", "電力消費が一棟分を超えた"]},
    {"id": "cloud",    "name": "Cloud Farm",      "icon": "☁️",
     "base_cost": 450000,   "base_gps": 2600.0,
     "flavor": ["世界中のクラウドが唸る", "グローバル分散処理が起動", "ネットワーク帯域が飽和した"]},
    {"id": "quantum",  "name": "Quantum Renderer","icon": "⚛️",
     "base_cost": 3_500_000, "base_gps": 21000.0,
     "flavor": ["量子もつれがレンダリングを加速", "観測するたびにGaussianが確定する", "宇宙の情報量をGaussianに変換中"]},
]

UPGRADES = [
    # ── クリック強化 ──
    {"id": "click_1", "name": "高精度マウス",        "icon": "🖱️", "cost": 100,
     "desc": "クリックG ×2", "type": "click", "mult": 2},
    {"id": "click_2", "name": "連打デバイス",         "icon": "⚡",  "cost": 2_000,
     "desc": "クリックG ×3", "type": "click", "mult": 3},
    {"id": "click_3", "name": "AI自動クリッカー",     "icon": "🤖",  "cost": 60_000,
     "desc": "クリックG ×5", "type": "click", "mult": 5},
    {"id": "click_4", "name": "神経接続インターフェース","icon": "🧬","cost": 3_000_000,
     "desc": "クリックG ×10", "type": "click", "mult": 10},
    # ── GPU ──
    {"id": "gpu_1",   "name": "CUDA最適化",           "icon": "🔧",  "cost": 200,
     "desc": "GPU ×2",  "type": "gen", "target": "gpu", "mult": 2},
    {"id": "gpu_2",   "name": "Mixed Precision",      "icon": "🔬",  "cost": 4_000,
     "desc": "GPU ×5",  "type": "gen", "target": "gpu", "mult": 5},
    {"id": "gpu_3",   "name": "カスタムシリコン",      "icon": "💎",  "cost": 120_000,
     "desc": "GPU ×10", "type": "gen", "target": "gpu", "mult": 10},
    # ── Cluster ──
    {"id": "cluster_1","name": "InfiniBand接続",      "icon": "🔗",  "cost": 2_000,
     "desc": "Cluster ×2",  "type": "gen", "target": "cluster", "mult": 2},
    {"id": "cluster_2","name": "NVLink Bridge",       "icon": "🌉",  "cost": 40_000,
     "desc": "Cluster ×5",  "type": "gen", "target": "cluster", "mult": 5},
    {"id": "cluster_3","name": "独自クラスタOS",       "icon": "🛡️",  "cost": 600_000,
     "desc": "Cluster ×10", "type": "gen", "target": "cluster", "mult": 10},
    # ── COLMAP ──
    {"id": "colmap_1", "name": "SuperPoint採用",      "icon": "✨",  "cost": 12_000,
     "desc": "COLMAP ×2",  "type": "gen", "target": "colmap", "mult": 2},
    {"id": "colmap_2", "name": "LightGlue統合",       "icon": "💡",  "cost": 200_000,
     "desc": "COLMAP ×5",  "type": "gen", "target": "colmap", "mult": 5},
    {"id": "colmap_3", "name": "Neural SLAM",         "icon": "🧩",  "cost": 5_000_000,
     "desc": "COLMAP ×10", "type": "gen", "target": "colmap", "mult": 10},
    # ── Rig ──
    {"id": "rig_1",    "name": "積算誤差削減",         "icon": "📉",  "cost": 90_000,
     "desc": "Rig ×2",  "type": "gen", "target": "rig", "mult": 2},
    {"id": "rig_2",    "name": "勾配チェックポイント",  "icon": "💾",  "cost": 1_500_000,
     "desc": "Rig ×5",  "type": "gen", "target": "rig", "mult": 5},
    {"id": "rig_3",    "name": "分散学習",             "icon": "🌐",  "cost": 30_000_000,
     "desc": "Rig ×10", "type": "gen", "target": "rig", "mult": 10},
    # ── DC ──
    {"id": "dc_1",     "name": "液冷システム",         "icon": "❄️",  "cost": 700_000,
     "desc": "DataCenter ×2",  "type": "gen", "target": "dc", "mult": 2},
    {"id": "dc_2",     "name": "再生可能エネルギー",   "icon": "🌱",  "cost": 15_000_000,
     "desc": "DataCenter ×5",  "type": "gen", "target": "dc", "mult": 5},
    # ── Cloud ──
    {"id": "cloud_1",  "name": "グローバルCDN",        "icon": "🌍",  "cost": 6_000_000,
     "desc": "Cloud ×2",  "type": "gen", "target": "cloud", "mult": 2},
    {"id": "cloud_2",  "name": "マルチリージョン展開",  "icon": "🗺️",  "cost": 100_000_000,
     "desc": "Cloud ×5",  "type": "gen", "target": "cloud", "mult": 5},
    # ── Quantum ──
    {"id": "quantum_1","name": "量子誤り訂正",         "icon": "🔮",  "cost": 50_000_000,
     "desc": "Quantum ×2",  "type": "gen", "target": "quantum", "mult": 2},
    {"id": "quantum_2","name": "位相反転キャンセル",   "icon": "🌀",  "cost": 800_000_000,
     "desc": "Quantum ×5",  "type": "gen", "target": "quantum", "mult": 5},
]

MILESTONES = [
    (100,         "🏅 100 Gaussians！　研究が始まった"),
    (1_000,       "🏅 1K Gaussians！　最初のクラスタが動いた"),
    (10_000,      "🥈 10K Gaussians！　研究室が賑やかになってきた"),
    (100_000,     "🥈 100K Gaussians！　論文が書けそう"),
    (1_000_000,   "🥇 1M Gaussians！　伝説の研究者への道が開けた"),
    (100_000_000, "👑 100M Gaussians！　プレステージ解禁！"),
    (10_000_000,  "🥇 10M Gaussians！　伝説の研究者"),
    (1_000_000_000, "💠 1B Gaussians！　時空を超えた"),
    (1_000_000_000_000, "🌌 1T Gaussians！　宇宙を満たした"),
]

NEWS = [
    "研究室の気温が上昇している…GPUのせいだろうか",
    "隣の研究室がNeRFを使っているらしい。時代遅れだ",
    "電気代の請求書が届いた。見なかったことにしよう",
    "3DGSの論文がまた引用された！",
    "スポンサーから問い合わせが来ている",
    "新しいGPUアーキテクチャの噂が出回っている",
    "ガウシアンが多すぎてビューワーがクラッシュした",
    "世界中の研究者がこの施設を羨ましがっている",
    "量子コンピュータの冷却剤が切れそうだ",
    "点群が銀河を超えた",
]

# ── セーブ / ロード ─────────────────────────────────────────────────────────

def default_save() -> dict:
    return {
        "gaussians":       0.0,
        "total_gaussians": 0.0,
        "prestige_points": 0,
        "generators":      {g["id"]: 0 for g in GENERATORS},
        "upgrades":        [],
        "total_clicks":    0,
        "last_tick":       time.time(),
        "golden_until":    0.0,
        "seen_milestones": [],
    }

def load_save() -> dict:
    try:
        if Path(SAVE_FILE).exists():
            data = json.loads(Path(SAVE_FILE).read_text(encoding="utf-8"))
            d = default_save()
            for k, v in d.items():
                data.setdefault(k, v)
            for gid in d["generators"]:
                data["generators"].setdefault(gid, 0)
            return data
    except Exception:
        pass
    return default_save()

def save_game(g: dict):
    Path(SAVE_FILE).parent.mkdir(parents=True, exist_ok=True)
    out = dict(g)
    out["last_tick"] = time.time()
    Path(SAVE_FILE).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

# ── ゲームロジック ──────────────────────────────────────────────────────────

def gen_cost(gen_id: str, count: int) -> float:
    base = next(g["base_cost"] for g in GENERATORS if g["id"] == gen_id)
    return math.ceil(base * (COST_SCALE ** count))

def gen_cost_bulk(gen_id: str, count: int, amount: int) -> float:
    return sum(gen_cost(gen_id, count + i) for i in range(amount))

def gen_multiplier(gen_id: str, owned: list) -> float:
    mult = 1.0
    for u in UPGRADES:
        if u["type"] == "gen" and u.get("target") == gen_id and u["id"] in owned:
            mult *= u["mult"]
    return mult

def calc_gps(g: dict) -> float:
    prestige = 1.0 + g["prestige_points"] * 0.05
    total = 0.0
    for gen in GENERATORS:
        cnt = g["generators"][gen["id"]]
        if cnt > 0:
            total += gen["base_gps"] * cnt * gen_multiplier(gen["id"], g["upgrades"])
    return total * prestige

def calc_click_value(g: dict) -> float:
    prestige = 1.0 + g["prestige_points"] * 0.05
    mult = 1.0
    for u in UPGRADES:
        if u["type"] == "click" and u["id"] in g["upgrades"]:
            mult *= u["mult"]
    base = max(1.0, calc_gps(g) * 0.01)
    return base * mult * prestige

def do_tick(g: dict) -> float:
    now = time.time()
    elapsed = min(now - g["last_tick"], TICK_CAP)
    earned = calc_gps(g) * elapsed
    g["gaussians"]       += earned
    g["total_gaussians"] += earned
    g["last_tick"]        = now
    return earned

PRESTIGE_BASE = 100_000_000  # 100M Gで初回プレステージ解禁

def prestige_pts_available(g: dict) -> int:
    if g["total_gaussians"] < PRESTIGE_BASE:
        return 0
    return int(math.log10(g["total_gaussians"] / PRESTIGE_BASE)) + 1

def do_prestige(g: dict):
    new_pp = prestige_pts_available(g)
    g["prestige_points"] = new_pp
    g["gaussians"]       = 0.0
    g["generators"]      = {gen["id"]: 0 for gen in GENERATORS}
    g["upgrades"]        = []
    g["golden_until"]    = 0.0
    g["last_tick"]       = time.time()

def fmt(n: float) -> str:
    if n < 1_000:          return f"{n:.1f}"
    if n < 1_000_000:      return f"{n/1e3:.2f}K"
    if n < 1_000_000_000:  return f"{n/1e6:.2f}M"
    if n < 1e12:           return f"{n/1e9:.2f}B"
    if n < 1e15:           return f"{n/1e12:.2f}T"
    return f"{n/1e15:.2f}Qa"

def check_milestones(g: dict) -> list:
    new_ones = []
    for threshold, text in MILESTONES:
        key = str(threshold)
        if g["total_gaussians"] >= threshold and key not in g["seen_milestones"]:
            g["seen_milestones"].append(key)
            new_ones.append(text)
    return new_ones

# ── CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a; color: #e0e6f0;
  }
  .block-container { padding: 1rem 1.5rem; }

  .game-title {
    font-size: 1.8rem; font-weight: 700; letter-spacing: 0.15em;
    color: #00e5ff; text-shadow: 0 0 14px #00e5ff88; margin-bottom: 0;
  }
  .game-sub { font-size: 0.7rem; color: #4a90b8; letter-spacing: 0.2em; }

  .stat-box {
    background: linear-gradient(135deg,#0d1b2e,#0a1520);
    border: 1px solid #1a3a5c; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem; text-align: center;
  }
  .stat-label { font-size: 0.65rem; color: #4a90b8; letter-spacing: 0.15em; text-transform: uppercase; }
  .stat-value { font-size: 1.3rem; color: #00e5ff; }

  .click-btn-wrap > div > button {
    background: linear-gradient(135deg,#003344,#004455) !important;
    border: 2px solid #00e5ff !important; border-radius: 50% !important;
    width: 130px !important; height: 130px !important;
    font-size: 2.5rem !important; cursor: pointer;
    box-shadow: 0 0 20px #00e5ff44;
    transition: box-shadow 0.1s;
  }
  .click-btn-wrap > div > button:hover {
    box-shadow: 0 0 35px #00e5ff88 !important;
  }
  .golden-btn-wrap > div > button {
    background: linear-gradient(135deg,#332200,#443300) !important;
    border: 2px solid #ffcc00 !important; border-radius: 50% !important;
    width: 130px !important; height: 130px !important;
    font-size: 2.5rem !important;
    box-shadow: 0 0 25px #ffcc0066 !important;
    animation: pulse 1s infinite;
  }
  @keyframes pulse { 0%,100%{box-shadow:0 0 25px #ffcc0066} 50%{box-shadow:0 0 45px #ffcc00cc} }

  .gen-card {
    background: linear-gradient(135deg,#0d1b2e,#0a1520);
    border: 1px solid #1a3a5c; border-radius: 8px;
    padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .gen-card.can-afford { border-color: #00aa66; }
  .gen-card.maxed { border-color: #4a90b8; }
  .gen-name { font-size: 0.8rem; color: #e0e6f0; flex: 1; }
  .gen-count { font-size: 1.1rem; color: #00e5ff; min-width: 2.5rem; text-align: right; }
  .gen-gps { font-size: 0.65rem; color: #4a90b8; }
  .gen-cost { font-size: 0.7rem; color: #cc8800; }
  .gen-cost.green { color: #00cc66; }

  .upg-card {
    background: linear-gradient(135deg,#0d1b2e,#0a1520);
    border: 1px solid #1a3a5c; border-radius: 6px;
    padding: 0.4rem 0.6rem; margin-bottom: 0.3rem;
    font-size: 0.75rem;
  }
  .upg-card.affordable { border-color: #00aa66; }
  .upg-card.locked { opacity: 0.4; }

  .milestone-banner {
    background: linear-gradient(135deg,#001a00,#002a00);
    border: 1px solid #00cc66; border-radius: 8px;
    padding: 0.5rem 1rem; margin-bottom: 0.4rem;
    color: #00cc66; font-size: 0.85rem; text-align: center;
  }
  .golden-banner {
    background: linear-gradient(135deg,#221100,#332200);
    border: 1px solid #ffcc00; border-radius: 8px;
    padding: 0.4rem 1rem; text-align: center;
    color: #ffcc00; font-size: 0.8rem; margin-bottom: 0.5rem;
  }
  .news-ticker {
    font-size: 0.7rem; color: #2a6080; font-style: italic;
    margin-top: 0.3rem;
  }
  .section-title {
    font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase;
    color: #4a90b8; border-bottom: 1px solid #1a3a5c;
    padding-bottom: 0.3rem; margin: 0.8rem 0 0.5rem 0;
  }
  .prestige-box {
    background: linear-gradient(135deg,#1a0033,#0d001a);
    border: 1px solid #a855f7; border-radius: 10px;
    padding: 0.8rem 1rem; text-align: center;
  }

  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg,#0d1b2e,#0f2340);
    border: 1px solid #1a3a5c; border-radius: 8px;
    color: #e0e6f0; font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem; letter-spacing: 0.08em;
    padding: 0.4rem 0.5rem; width: 100%;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #00e5ff; box-shadow: 0 0 8px #00e5ff33; color: #00e5ff;
  }
</style>
""", unsafe_allow_html=True)

# ── 状態初期化 ──────────────────────────────────────────────────────────────

if "game" not in st.session_state:
    st.session_state.game   = load_save()
    st.session_state._news  = random.choice(NEWS)
    st.session_state._ticks = 0
    st.session_state._notifs = []
    st.session_state._last_save = time.time()

g = st.session_state.game

# ── ティック処理 ─────────────────────────────────────────────────────────────

offline_earned = do_tick(g)
st.session_state._ticks += 1

# ゴールデンガウシアン自然発生（GPS > 0 の場合のみ）
if (calc_gps(g) > 0
        and time.time() > g["golden_until"]
        and random.random() < GOLDEN_CHANCE):
    g["golden_until"] = time.time() + GOLDEN_DURATION

# マイルストーンチェック
new_ms = check_milestones(g)
for ms in new_ms:
    st.session_state._notifs.append(("milestone", ms))

# 10秒ごとにニュースを更新
if st.session_state._ticks % 10 == 0:
    st.session_state._news = random.choice(NEWS)

# 自動セーブ（30秒ごと）
if time.time() - st.session_state._last_save > 30:
    save_game(g)
    st.session_state._last_save = time.time()

# ── 計算値のキャッシュ ───────────────────────────────────────────────────────

gps      = calc_gps(g)
cpv      = calc_click_value(g)
is_gold  = time.time() < g["golden_until"]
gold_rem = max(0.0, g["golden_until"] - time.time())

# ── ヘッダー ────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="game-title">⚗️ 3DGS LABORATORY</div>'
    '<div class="game-sub">INCREMENTAL GAUSSIAN RESEARCH SIMULATOR</div>',
    unsafe_allow_html=True,
)

# 通知バナー
notifs = st.session_state._notifs.copy()
st.session_state._notifs.clear()
for kind, msg in notifs:
    if kind == "milestone":
        st.markdown(f'<div class="milestone-banner">{msg}</div>', unsafe_allow_html=True)

# オフライン進捗（初回ロード時のみ大きかったら表示）
if "shown_offline" not in st.session_state and offline_earned > 100:
    st.info(f"⏰ オフライン中に **{fmt(offline_earned)} G** 生産されました！")
st.session_state.shown_offline = True

# ゴールデン中バナー
if is_gold:
    st.markdown(
        f'<div class="golden-banner">✨ ゴールデンガウシアン発動中！'
        f' クリック×{GOLDEN_MULTIPLIER} ── あと {gold_rem:.0f}秒</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── メインレイアウト ─────────────────────────────────────────────────────────

left, mid, right = st.columns([1.3, 2, 1.7])

# ════════════════════════════════════════════════════════════════════════════
#  左カラム：クリックパネル + ステータス
# ════════════════════════════════════════════════════════════════════════════
with left:
    # クリックボタン
    btn_class = "golden-btn-wrap" if is_gold else "click-btn-wrap"
    st.markdown(f'<div class="{btn_class}" style="display:flex;justify-content:center;margin-bottom:0.5rem;">', unsafe_allow_html=True)
    clicked = st.button("🔬" if not is_gold else "✨", key="main_click", help="クリックしてGaussianを生産！")
    st.markdown("</div>", unsafe_allow_html=True)

    if clicked:
        earned = cpv * (GOLDEN_MULTIPLIER if is_gold else 1)
        g["gaussians"]       += earned
        g["total_gaussians"] += earned
        g["total_clicks"]    += 1
        check_milestones(g)

    # ステータスボックス
    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-label">Gaussians</div>'
        f'<div class="stat-value">{fmt(g["gaussians"])} G</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-label">生産速度</div>'
        f'<div class="stat-value">{fmt(gps)} <span style="font-size:0.8rem;color:#4a90b8">G/s</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-label">クリック値</div>'
        f'<div class="stat-value">{fmt(cpv * (GOLDEN_MULTIPLIER if is_gold else 1))} G</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # プレステージ情報
    pp_avail = prestige_pts_available(g)
    st.markdown('<div class="section-title">Prestige</div>', unsafe_allow_html=True)
    if pp_avail > g["prestige_points"]:
        new_pts = pp_avail - g["prestige_points"]
        st.markdown(
            f'<div class="prestige-box">'
            f'<span style="color:#a855f7;font-size:0.85rem;">✦ プレステージ可能！<br>'
            f'<b style="color:#e0e6f0">+{new_pts}pt</b> 獲得<br>'
            f'<span style="font-size:0.7rem;color:#6b21a8">（全施設をリセット）</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("✦ プレステージ実行", key="prestige_btn"):
            do_prestige(g)
            save_game(g)
            st.rerun()
    else:
        if g["prestige_points"] > 0:
            st.markdown(
                f'<div style="color:#a855f7;font-size:0.8rem;">'
                f'✦ {g["prestige_points"]}pt（生産 ×{1+g["prestige_points"]*0.05:.2f}）<br>'
                f'<span style="color:#4a90b8;font-size:0.7rem;">'
                f'次: {fmt(10**(g["prestige_points"]) * PRESTIGE_BASE)} G</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#2a6080;font-size:0.75rem;">'
                '100M G 達成で解禁<br>全リセット＋永続ボーナス</span>',
                unsafe_allow_html=True,
            )

    # 統計 & ニュース
    st.markdown('<div class="section-title">Stats</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span style="color:#4a90b8;font-size:0.72rem;">'
        f'総生産: {fmt(g["total_gaussians"])} G<br>'
        f'総クリック: {g["total_clicks"]:,} 回<br>'
        f'Prestige: {g["prestige_points"]} pt'
        f'</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="news-ticker">📡 {st.session_state._news}</div>', unsafe_allow_html=True)

    # セーブボタン
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 セーブ", key="save_btn"):
        save_game(g)
        st.toast("セーブしました", icon="💾")

# ════════════════════════════════════════════════════════════════════════════
#  中央カラム：施設（ジェネレーター）
# ════════════════════════════════════════════════════════════════════════════
with mid:
    st.markdown('<div class="section-title">施設</div>', unsafe_allow_html=True)

    # ×1 / ×10 購入モード切替
    if "buy_mode" not in st.session_state:
        st.session_state.buy_mode = 1
    bm_col1, bm_col2 = st.columns(2)
    with bm_col1:
        if st.button("×1 購入", key="bm1",
                     help="1つずつ購入"):
            st.session_state.buy_mode = 1
    with bm_col2:
        if st.button("×10 購入", key="bm10",
                     help="10個まとめて購入"):
            st.session_state.buy_mode = 10
    buy_mode = st.session_state.buy_mode
    st.markdown(
        f'<span style="color:#4a90b8;font-size:0.7rem;">購入モード: ×{buy_mode}</span>',
        unsafe_allow_html=True,
    )

    for gen in GENERATORS:
        gid   = gen["id"]
        count = g["generators"][gid]
        cost  = gen_cost_bulk(gid, count, buy_mode)
        can   = g["gaussians"] >= cost
        mult  = gen_multiplier(gid, g["upgrades"])
        contribution = gen["base_gps"] * count * mult * (1 + g["prestige_points"] * 0.05)

        # フレーバーテキスト（保有数によって変化）
        flavor_idx = min(count // 10, len(gen["flavor"]) - 1)
        flavor = gen["flavor"][flavor_idx] if count > 0 else "まだ稼働していない"

        card_class = "gen-card can-afford" if can else "gen-card"
        cost_class = "gen-cost green" if can else "gen-cost"

        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(
                f'<div class="{card_class}">'
                f'<span style="font-size:1.4rem">{gen["icon"]}</span>'
                f'<div style="flex:1;">'
                f'  <div class="gen-name">{gen["name"]}</div>'
                f'  <div class="gen-gps">{fmt(contribution)} G/s ｜ {flavor}</div>'
                f'</div>'
                f'<div>'
                f'  <div class="gen-count">{count}</div>'
                f'  <div class="{cost_class}">{fmt(cost)} G</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(f"購入", key=f"buy_{gid}",
                         disabled=not can,
                         help=f"{buy_mode}個購入: {fmt(cost)} G"):
                actual_buy = 0
                for _ in range(buy_mode):
                    c = gen_cost(gid, g["generators"][gid])
                    if g["gaussians"] >= c:
                        g["gaussians"] -= c
                        g["generators"][gid] += 1
                        actual_buy += 1
                    else:
                        break
                if actual_buy > 0:
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  右カラム：アップグレード
# ════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="section-title">アップグレード</div>', unsafe_allow_html=True)

    # 購入済み / 未購入 に分類
    available = [u for u in UPGRADES if u["id"] not in g["upgrades"]]
    purchased_count = len(g["upgrades"])

    # 購入可能 → 上に、高すぎるもの → 下に
    can_buy   = [u for u in available if g["gaussians"] >= u["cost"]]
    coming_up = [u for u in available if u not in can_buy][:8]  # 先頭8件

    if purchased_count > 0:
        st.markdown(
            f'<span style="color:#00cc66;font-size:0.72rem;">✓ {purchased_count} 件購入済み</span>',
            unsafe_allow_html=True,
        )

    if not available:
        st.markdown(
            '<span style="color:#00cc66;font-size:0.8rem;">🎉 全アップグレード購入済み！</span>',
            unsafe_allow_html=True,
        )
    else:
        # 購入可能なアップグレード
        if can_buy:
            st.markdown(
                '<span style="color:#00aa66;font-size:0.7rem;">── 購入可能 ──</span>',
                unsafe_allow_html=True,
            )
        for upg in can_buy:
            col_info2, col_btn2 = st.columns([3, 1])
            with col_info2:
                st.markdown(
                    f'<div class="upg-card affordable">'
                    f'{upg["icon"]} <b>{upg["name"]}</b><br>'
                    f'<span style="color:#00cc66">{upg["desc"]}</span>'
                    f' ── <span style="color:#cc8800">{fmt(upg["cost"])} G</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_btn2:
                if st.button("購入", key=f"upg_{upg['id']}"):
                    g["gaussians"] -= upg["cost"]
                    g["upgrades"].append(upg["id"])
                    st.rerun()

        # まもなく購入可能なアップグレード
        if coming_up:
            st.markdown(
                '<span style="color:#2a6080;font-size:0.7rem;">── 今後解禁 ──</span>',
                unsafe_allow_html=True,
            )
        for upg in coming_up:
            affordable_pct = min(g["gaussians"] / upg["cost"], 1.0) * 100
            st.markdown(
                f'<div class="upg-card locked">'
                f'{upg["icon"]} {upg["name"]}<br>'
                f'<span style="color:#2a6080">{upg["desc"]}</span>'
                f' ── <span style="color:#664400">{fmt(upg["cost"])} G</span>'
                f'<br><span style="color:#1a3a5c;font-size:0.65rem;">'
                f'進捗: {affordable_pct:.0f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # プレステージ後の永続ボーナス表示
    if g["prestige_points"] > 0:
        st.markdown('<div class="section-title">Prestige ボーナス</div>', unsafe_allow_html=True)
        for pp in range(g["prestige_points"]):
            labels = ["Ω プレステージレベル1 +5%", "Ω プレステージレベル2 +5%",
                      "Ω プレステージレベル3 +5%", "Ω プレステージレベル4 +5%",
                      "Ω プレステージレベル5 +5%"]
            label = labels[pp] if pp < len(labels) else f"Ω プレステージレベル{pp+1} +5%"
            st.markdown(
                f'<div style="color:#a855f7;font-size:0.72rem;margin-bottom:2px;">✦ {label}</div>',
                unsafe_allow_html=True,
            )

# ── パイプライン進捗 ──────────────────────────────────────────────────────────
with st.expander("🚀 パイプライン進捗", expanded=False):
    render_pipeline_status(compact=True)

# ── フッター：自動リロード ────────────────────────────────────────────────────
st.divider()
fc1, fc2, fc3 = st.columns([1, 1, 1])
with fc1:
    st.markdown(
        f'<span style="color:#2a6080;font-size:0.7rem;">⏱ 自動更新: 1秒ごと</span>',
        unsafe_allow_html=True,
    )
with fc2:
    if st.button("🗑️ データリセット", key="reset_btn",
                 help="全データを削除して最初から始めます"):
        if "confirm_reset" not in st.session_state:
            st.session_state.confirm_reset = True
        else:
            del st.session_state["game"]
            if Path(SAVE_FILE).exists():
                Path(SAVE_FILE).unlink()
            for k in list(st.session_state.keys()):
                if k.startswith("_") or k in ("buy_mode", "confirm_reset", "shown_offline"):
                    del st.session_state[k]
            st.rerun()
    if st.session_state.get("confirm_reset"):
        st.warning("もう一度押すと全データが消えます！")

with fc3:
    pass

# ページロード完了後に自動セーブ & 1秒後に再実行
save_game(g)

try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass

time.sleep(1)
st.rerun()
