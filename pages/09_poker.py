# ビデオポーカー（Jacks or Better）ゲームページ
# ジャック以上のペアから配当あり。5枚のカードをもらい、キープするカードを選んで引き直す。

import random
import streamlit as st

# ─── 定数 ────────────────────────────────────────────────────────────────────
SUITS  = ["♠", "♥", "♦", "♣"]
RANKS  = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
RANK_V = {r: i + 2 for i, r in enumerate(RANKS)}  # "2"→2 ... "A"→14

PAYTABLE = [
    ("ロイヤルフラッシュ",   800),
    ("ストレートフラッシュ",  50),
    ("フォーカード",          25),
    ("フルハウス",             9),
    ("フラッシュ",             6),
    ("ストレート",             4),
    ("スリーカード",           3),
    ("ツーペア",               2),
    ("ジャックスオアベター",   1),
]

# ─── セッション状態の初期化 ──────────────────────────────────────────────────
_DEFAULTS = {
    "poker_chips":  1000,
    "poker_phase":  "bet",   # "bet" | "draw" | "result"
    "poker_hand":   [],      # [(rank, suit), ...]
    "poker_deck":   [],
    "poker_held":   [False] * 5,
    "poker_bet":    10,
    "poker_result": None,    # (name, mult, winnings) or None
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── ゲームロジック ──────────────────────────────────────────────────────────

def _new_deck():
    deck = [(r, s) for s in SUITS for r in RANKS]
    random.shuffle(deck)
    return deck


def _evaluate(hand):
    ranks  = sorted([RANK_V[r] for r, s in hand], reverse=True)
    suits  = [s for r, s in hand]
    cnt    = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    counts = sorted(cnt.values(), reverse=True)

    is_flush    = len(set(suits)) == 1
    r_set       = set(ranks)
    is_straight = (len(r_set) == 5 and max(ranks) - min(ranks) == 4) or \
                  r_set == {14, 2, 3, 4, 5}  # A-2-3-4-5（ホイール）

    if is_flush and is_straight:
        return ("ロイヤルフラッシュ",  800) if min(ranks) == 10 else ("ストレートフラッシュ", 50)
    if counts[0] == 4:                         return ("フォーカード",          25)
    if counts == [3, 2]:                       return ("フルハウス",              9)
    if is_flush:                               return ("フラッシュ",              6)
    if is_straight:                            return ("ストレート",              4)
    if counts[0] == 3:                         return ("スリーカード",            3)
    if counts[:2] == [2, 2]:                   return ("ツーペア",                2)
    if counts[0] == 2:
        pair_r = [r for r, c in cnt.items() if c == 2][0]
        if pair_r >= 11:                       return ("ジャックスオアベター",    1)
    return ("ハズレ", 0)


def _deal():
    deck = _new_deck()
    st.session_state.poker_hand  = deck[:5]
    st.session_state.poker_deck  = deck[5:]
    st.session_state.poker_held  = [False] * 5
    st.session_state.poker_chips -= st.session_state.poker_bet
    st.session_state.poker_result = None
    st.session_state.poker_phase  = "draw"


def _draw():
    hand = list(st.session_state.poker_hand)
    deck = list(st.session_state.poker_deck)
    for i in range(5):
        if not st.session_state.poker_held[i]:
            hand[i] = deck.pop(0)
    st.session_state.poker_hand = hand
    st.session_state.poker_deck = deck
    name, mult = _evaluate(hand)
    winnings = st.session_state.poker_bet * mult
    st.session_state.poker_chips  += winnings
    st.session_state.poker_result  = (name, mult, winnings)
    st.session_state.poker_phase   = "result"


def _toggle_hold(i):
    st.session_state.poker_held[i] = not st.session_state.poker_held[i]


def _reset():
    st.session_state.poker_phase  = "bet"
    st.session_state.poker_result = None


# ─── カード描画ヘルパー ────────────────────────────────────────────────────────

def _card_html(rank, suit, held=False, face_down=False):
    is_red = suit in ("♥", "♦")
    color  = "#ff6b6b" if is_red else "#dce8f8"
    bg     = "#0d2540" if held else "#0a1520"
    border = "2px solid #00e5ff" if held else "1px solid #1e3a54"
    shadow = "0 0 14px #00e5ff55" if held else "none"
    badge  = (
        '<div style="font-size:0.55rem;color:#00e5ff;letter-spacing:0.12em;'
        'margin-bottom:2px;font-weight:bold;">▼ HELD</div>'
        if held else
        '<div style="font-size:0.55rem;color:transparent;margin-bottom:2px;">▼ HELD</div>'
    )
    if face_down:
        return (
            '<div style="background:#071020;border:1px solid #1e3a54;border-radius:10px;'
            'padding:14px 10px;text-align:center;height:108px;'
            'display:flex;align-items:center;justify-content:center;">'
            '<span style="font-size:2.4rem;filter:grayscale(0.3);">🂠</span></div>'
        )
    return (
        f'<div style="background:{bg};border:{border};border-radius:10px;'
        f'padding:10px 10px 6px;text-align:center;height:108px;'
        f'font-family:monospace;box-shadow:{shadow};">'
        f'{badge}'
        f'<div style="font-size:1.8rem;font-weight:bold;color:{color};line-height:1.15;">{rank}</div>'
        f'<div style="font-size:1.5rem;color:{color};line-height:1.15;">{suit}</div>'
        f'</div>'
    )


# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🃏 ビデオポーカー")
st.caption("Jacks or Better — ジャック以上のペアから配当あり")

phase  = st.session_state.poker_phase
hand   = st.session_state.poker_hand
held   = st.session_state.poker_held
chips  = st.session_state.poker_chips
result = st.session_state.poker_result

# ── ヘッダー行（チップ・ベット・配当表） ──────────────────────────────────────
hc1, hc2, hc3 = st.columns([2, 2, 2])
with hc1:
    st.metric("💰 チップ", f"{chips:,}")
with hc2:
    if phase == "bet":
        max_bet = max(1, min(chips, 500))
        bet = st.number_input(
            "ベット額", min_value=1, max_value=max_bet,
            value=min(st.session_state.poker_bet, max_bet),
            step=1, key="bet_input",
        )
        st.session_state.poker_bet = int(bet)
    else:
        st.metric("ベット", f"{st.session_state.poker_bet:,}")
with hc3:
    with st.expander("📊 配当表"):
        for name, mult in PAYTABLE:
            pa, pb = st.columns([4, 1])
            pa.caption(name)
            pb.caption(f"×{mult}")

st.divider()

# ── カード表示エリア ──────────────────────────────────────────────────────────
card_cols = st.columns(5)
if hand:
    for i, (col, (r, s)) in enumerate(zip(card_cols, hand)):
        with col:
            st.markdown(_card_html(r, s, held[i]), unsafe_allow_html=True)
            if phase == "draw":
                label    = "✅ HELD" if held[i] else "HOLD"
                btn_type = "primary" if held[i] else "secondary"
                st.button(label, key=f"hold_{i}", on_click=_toggle_hold,
                          args=(i,), use_container_width=True, type=btn_type)
else:
    for col in card_cols:
        with col:
            st.markdown(_card_html("", "♠", face_down=True), unsafe_allow_html=True)

# ── 結果バナー ────────────────────────────────────────────────────────────────
if result:
    name, mult, winnings = result
    if winnings > 0:
        st.success(f"🎉 **{name}** ＋{winnings:,} チップ獲得！（×{mult}）")
    else:
        st.error(f"😔 **{name}** — 残念、外れです")

# ── アクションボタン ──────────────────────────────────────────────────────────
st.divider()

if phase == "bet":
    if chips <= 0:
        st.error("チップがなくなりました！")
        if st.button("🔄 チップをリセット（1,000）", use_container_width=True):
            st.session_state.poker_chips = 1000
            st.rerun()
    else:
        st.button(
            f"🃏  DEAL  （{st.session_state.poker_bet} チップ消費）",
            type="primary", on_click=_deal,
            use_container_width=True,
            disabled=(st.session_state.poker_bet > chips),
        )

elif phase == "draw":
    st.button(
        "🎴  DRAW  （選んだカードを引き直す）",
        type="primary", on_click=_draw, use_container_width=True,
    )
    st.caption("キープしたいカードの **HOLD** を押してから **DRAW** してください。何も選ばなければ5枚全部引き直します。")

elif phase == "result":
    rc1, rc2 = st.columns(2)
    with rc1:
        st.button("▶ もう一回", type="primary", on_click=_reset,
                  use_container_width=True)
    with rc2:
        if st.button("🔄 チップをリセット（1,000）", use_container_width=True):
            st.session_state.poker_chips = 1000
            _reset()
            st.rerun()

# ─── 固定フッター ─────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
