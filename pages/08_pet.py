# たまごっち風育成シミュレーション MVP
# st.session_state で状態管理、time.sleep + st.rerun で自動更新（1秒ごと）

import random
import time
from datetime import datetime

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  定数
# ─────────────────────────────────────────────────────────────────────────────
PHASE_ORDER    = ["egg", "infant", "child", "teen", "adult"]
PHASE_DURATION = {"egg": 30, "infant": 60, "child": 120, "teen": 180}  # 秒（テスト用短縮）

PHASE_LABEL = {
    "egg":    "🥚 たまご",
    "infant": "🐣 幼少期",
    "child":  "🐥 子ども期",
    "teen":   "🐤 思春期",
    "adult":  "成体",
}

CHAR_EMOJI = {
    "egg":    "🥚",
    "infant": "🐣",
    "child":  "🐥",
    "teen":   "🐤",
    "adult": {
        "やんちゃ": "😈",
        "エリート": "🦉",
        "アイドル": "⭐",
        "ふつう":   "🐔",
    },
}

DECAY_HUNGER   = 0.12   # 毎秒の減少量
DECAY_CLEAN    = 0.07
POOP_RANGE     = (25, 50)   # うんち発生間隔（秒）
CARE_WARN_SECS = 20         # 警告放置でケアミス扱いになる秒数


# ─────────────────────────────────────────────────────────────────────────────
#  ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────
def now() -> float:
    return time.time()


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


def add_log(msg: str, cd_key: str = "", cd_secs: float = 8.0) -> None:
    """スパム防止クールダウン付きログ追加"""
    if cd_key:
        if now() - st.session_state.log_cd.get(cd_key, 0.0) < cd_secs:
            return
        st.session_state.log_cd[cd_key] = now()
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"`{ts}` {msg}")
    st.session_state.logs = st.session_state.logs[:6]


# ─────────────────────────────────────────────────────────────────────────────
#  初期化
# ─────────────────────────────────────────────────────────────────────────────
def init() -> None:
    n = now()
    st.session_state.update({
        "initialized":    True,
        "phase":          "egg",
        "phase_start":    n,
        "last_update":    n,
        "evolution_type": None,
        # 表示パラメータ (0–100)
        "hunger":      80.0,
        "intellect":   20.0,
        "cleanliness": 80.0,
        "friendship":  20.0,
        # 隠しパラメータ
        "care_mistakes": 0,
        "stress":        0.0,
        # うんち管理
        "poop":      False,
        "next_poop": n + random.uniform(*POOP_RANGE),
        # 警告タイマー（ケアミス判定用）
        "hunger_low_since": None,
        "clean_low_since":  None,
        # ログ
        "logs":   ["🥚 たまごが産まれました！大切に育ててね！"],
        "log_cd": {},
    })


# ─────────────────────────────────────────────────────────────────────────────
#  進化ロジック
# ─────────────────────────────────────────────────────────────────────────────
def determine_evolution() -> str:
    cm     = st.session_state.care_mistakes
    intel  = st.session_state.intellect
    friend = st.session_state.friendship
    if cm > 3:
        return "やんちゃ"
    if intel > 70 and cm == 0:
        return "エリート"
    if friend > 70:
        return "アイドル"
    return "ふつう"


def get_emoji() -> str:
    p = st.session_state.phase
    if p == "adult":
        evo = st.session_state.evolution_type or "ふつう"
        return CHAR_EMOJI["adult"].get(evo, "🐔")
    return CHAR_EMOJI.get(p, "❓")


def get_label() -> str:
    p = st.session_state.phase
    if p == "adult":
        evo = st.session_state.evolution_type or "ふつう"
        return {
            "やんちゃ": "😈 やんちゃ系",
            "エリート": "🦉 エリート系",
            "アイドル": "⭐ アイドル系",
            "ふつう":   "🐔 ふつう系",
        }.get(evo, "成体")
    return PHASE_LABEL.get(p, p)


def advance_phase() -> None:
    p   = st.session_state.phase
    idx = PHASE_ORDER.index(p)
    if idx + 1 >= len(PHASE_ORDER):
        return
    next_p = PHASE_ORDER[idx + 1]
    st.session_state.phase       = next_p
    st.session_state.phase_start = now()
    if next_p == "adult":
        evo = determine_evolution()
        st.session_state.evolution_type = evo
        add_log(f"🎉 ついに成体になった！【{evo}系】だよ！")
    else:
        add_log(f"✨ {PHASE_LABEL[next_p]} に成長した！")


# ─────────────────────────────────────────────────────────────────────────────
#  時間経過更新（バックグラウンド進行対応）
# ─────────────────────────────────────────────────────────────────────────────
def update() -> None:
    n    = now()
    last = st.session_state.last_update
    dt   = min(n - last, 60.0)   # 最大60秒分まで一括適用
    if dt <= 0:
        return
    st.session_state.last_update = n
    p = st.session_state.phase

    # 卵フェーズ：フェーズ進行チェックのみ
    if p == "egg":
        if n - st.session_state.phase_start >= PHASE_DURATION["egg"]:
            advance_phase()
        return

    # ── パラメータ自然減少 ──────────────────────────────────────────────────
    st.session_state.hunger      = clamp(st.session_state.hunger      - DECAY_HUNGER * dt)
    st.session_state.cleanliness = clamp(st.session_state.cleanliness - DECAY_CLEAN  * dt)

    # 汚い環境だとストレス上昇
    if st.session_state.cleanliness < 30:
        st.session_state.stress = clamp(st.session_state.stress + 0.05 * dt)

    # ── うんち発生 ──────────────────────────────────────────────────────────
    if not st.session_state.poop and n >= st.session_state.next_poop:
        st.session_state.poop        = True
        st.session_state.cleanliness = clamp(st.session_state.cleanliness - 20)
        add_log("💩 うんちが出た！お掃除してあげて！", "poop", 30)
        st.session_state.next_poop   = n + random.uniform(*POOP_RANGE)

    # ── お腹警告 & ケアミス ─────────────────────────────────────────────────
    if st.session_state.hunger < 20:
        add_log("🍽️ お腹ペコペコ！ごはんをあげて！", "hunger_warn")
        if st.session_state.hunger_low_since is None:
            st.session_state.hunger_low_since = n
        elif n - st.session_state.hunger_low_since >= CARE_WARN_SECS:
            st.session_state.care_mistakes   += 1
            st.session_state.hunger_low_since = n
            add_log(f"⚠️ ケアミス発生！（計 {st.session_state.care_mistakes} 回）")
    else:
        st.session_state.hunger_low_since = None

    # ── きれいさ警告 & ケアミス ─────────────────────────────────────────────
    if st.session_state.cleanliness < 15:
        add_log("🚿 とても汚れてる！お掃除して！", "clean_warn")
        if st.session_state.clean_low_since is None:
            st.session_state.clean_low_since = n
        elif n - st.session_state.clean_low_since >= CARE_WARN_SECS:
            st.session_state.care_mistakes  += 1
            st.session_state.clean_low_since = n
            add_log(f"⚠️ ケアミス発生！（計 {st.session_state.care_mistakes} 回）")
    else:
        st.session_state.clean_low_since = None

    # ── フェーズ進行 ────────────────────────────────────────────────────────
    dur = PHASE_DURATION.get(p)
    if dur and n - st.session_state.phase_start >= dur:
        advance_phase()


# ─────────────────────────────────────────────────────────────────────────────
#  アクション
# ─────────────────────────────────────────────────────────────────────────────
def do_feed():
    st.session_state.hunger = clamp(st.session_state.hunger + 30)
    st.session_state.stress = clamp(st.session_state.stress - 5)
    add_log("🍙 ごはんを食べた。おいしそう！")


def do_clean():
    st.session_state.cleanliness = clamp(st.session_state.cleanliness + 40)
    st.session_state.poop        = False
    st.session_state.stress      = clamp(st.session_state.stress - 10)
    add_log("🛁 きれいになった！さっぱり～！")


def do_study():
    st.session_state.intellect = clamp(st.session_state.intellect + 10)
    st.session_state.hunger    = clamp(st.session_state.hunger    - 5)
    st.session_state.stress    = clamp(st.session_state.stress    + 8)
    add_log("📚 べんきょうした！かしこくなれるかな？")


def do_play():
    st.session_state.friendship = clamp(st.session_state.friendship + 15)
    st.session_state.hunger     = clamp(st.session_state.hunger     - 8)
    st.session_state.stress     = clamp(st.session_state.stress     - 15)
    add_log("🎮 一緒に遊んだ！たのしかった！")


def do_scold():
    st.session_state.stress     = clamp(st.session_state.stress     + 20)
    st.session_state.friendship = clamp(st.session_state.friendship - 10)
    add_log("😤 叱った。しょんぼりしてる…")


# ─────────────────────────────────────────────────────────────────────────────
#  初期化チェック → 更新
# ─────────────────────────────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    init()
else:
    update()

p = st.session_state.phase

# ═════════════════════════════════════════════════════════════════════════════
#  サイドバー
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 ステータス")
    st.caption(f"**{get_label()}**")

    # フェーズ進行バー
    if p in PHASE_DURATION:
        dur     = PHASE_DURATION[p]
        elapsed = now() - st.session_state.phase_start
        remain  = max(0.0, dur - elapsed)
        st.progress(min(elapsed / dur, 1.0), text=f"次の成長まで {remain:.0f} 秒")

    st.divider()

    for label, key in [
        ("🍙 おなか",     "hunger"),
        ("🧠 かしこさ",   "intellect"),
        ("🛁 きれいさ",   "cleanliness"),
        ("💕 なかよし度", "friendship"),
    ]:
        val = st.session_state[key]
        lc, rc = st.columns([3, 1])
        lc.caption(label)
        rc.caption(f"**{val:.0f}**")
        st.progress(val / 100)

    st.divider()

    if st.session_state.hunger < 25:
        st.warning("🍽️ お腹が空いています！")
    if st.session_state.cleanliness < 25 or st.session_state.poop:
        st.warning("🛁 きれいにしてあげて！")
    if st.session_state.stress > 70:
        st.warning("😰 ストレスが溜まってる！")

    st.divider()

    if st.button("🔄 リセット", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  メイン画面
# ═════════════════════════════════════════════════════════════════════════════
st.title("🥚 ガウスくん育成ゲーム")

# ── キャラクター表示 ──────────────────────────────────────────────────────────
_, char_col, _ = st.columns([1, 2, 1])
with char_col:
    poop_str = " 💩" if st.session_state.poop else ""
    st.markdown(
        f'<div style="text-align:center;font-size:7rem;line-height:1.2;'
        f'padding:1.5rem 0.5rem;border:2px solid #1e3a5c;border-radius:16px;'
        f'background:rgba(0,229,255,0.03);">'
        f'{get_emoji()}{poop_str}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="text-align:center;color:#4a90b8;margin-top:0.4rem;">'
        f'{get_label()}</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ── アクションボタン ──────────────────────────────────────────────────────────
if p == "egg":
    dur    = PHASE_DURATION["egg"]
    remain = max(0.0, dur - (now() - st.session_state.phase_start))
    st.info(f"🥚 たまごが孵るのを待っています… あと約 **{remain:.0f}** 秒")
else:
    is_infant = (p == "infant")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("🍙\nごはん", use_container_width=True):
            do_feed(); st.rerun()
    with c2:
        if st.button("🛁\nお掃除", use_container_width=True):
            do_clean(); st.rerun()
    with c3:
        if st.button("📚\nお勉強", use_container_width=True, disabled=is_infant):
            do_study(); st.rerun()
    with c4:
        if st.button("🎮\n遊ぶ", use_container_width=True):
            do_play(); st.rerun()
    with c5:
        if st.button("😤\n叱る", use_container_width=True, disabled=is_infant):
            do_scold(); st.rerun()

# ── ログ ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 📋 最近のできごと")
for log in (st.session_state.logs or ["まだ何もありません…"])[:5]:
    st.caption(log)

# ── 隠しパラメータ確認（デバッグ用） ─────────────────────────────────────────
with st.expander("🔍 詳細情報（隠しパラメータ）", expanded=False):
    ma, mb, mc = st.columns(3)
    ma.metric("ケアミス",  st.session_state.care_mistakes)
    mb.metric("ストレス",  f"{st.session_state.stress:.1f}")
    mc.metric("うんち",    "あり" if st.session_state.poop else "なし")

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass

# ── 自動更新（1秒ごと）────────────────────────────────────────────────────────
time.sleep(1)
st.rerun()
