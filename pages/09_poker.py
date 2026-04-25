# テキサスホールデム 1v1 ポーカー（プレイヤー vs CPU ディーラー）
# ブラインド制・フロップ/ターン/リバーあり。CPUは手の強さに応じてレイズ/コール/フォールドを判断。

import random
from itertools import combinations
import streamlit as st

# ─── 定数 ────────────────────────────────────────────────────────────────────
SUITS   = ["♠", "♥", "♦", "♣"]
RANKS   = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
RANK_V  = {r: i + 2 for i, r in enumerate(RANKS)}
SB, BB  = 10, 20
HAND_NAMES = [
    "ハイカード","ワンペア","ツーペア","スリーカード",
    "ストレート","フラッシュ","フルハウス","フォーカード",
    "ストレートフラッシュ","ロイヤルフラッシュ",
]

# ─── 役の評価 ────────────────────────────────────────────────────────────────

def _score5(hand):
    rv = sorted([RANK_V[r] for r, s in hand], reverse=True)
    sv = [s for r, s in hand]
    c = {}
    for r in rv: c[r] = c.get(r, 0) + 1
    counts  = sorted(c.values(), reverse=True)
    by_cnt  = sorted(c, key=lambda r: (c[r], r), reverse=True)
    flush   = len(set(sv)) == 1
    uniq    = len(set(rv)) == 5
    straight = uniq and rv[0] - rv[4] == 4
    wheel    = set(rv) == {14, 2, 3, 4, 5}
    if wheel: straight, rv = True, [5, 4, 3, 2, 1]
    if flush and straight: return (9 if rv[0] == 14 else 8, rv)
    if counts[0] == 4:        return (7, by_cnt)
    if counts == [3, 2]:      return (6, by_cnt)
    if flush:                 return (5, rv)
    if straight:              return (4, rv)
    if counts[0] == 3:        return (3, by_cnt)
    if counts[:2] == [2, 2]:  return (2, by_cnt)
    if counts[0] == 2:        return (1, by_cnt)
    return (0, rv)

def best_score(cards):
    if len(cards) < 5: return (0, [])
    return max(_score5(list(c)) for c in combinations(cards, 5))

def hand_name(cards):
    return HAND_NAMES[best_score(cards)[0]] if len(cards) >= 5 else ""

# ─── CPU AI ──────────────────────────────────────────────────────────────────

def _preflop_strength(hole):
    rv = sorted([RANK_V[r] for r, s in hole], reverse=True)
    suited = hole[0][1] == hole[1][1]
    pair   = rv[0] == rv[1]
    s = (rv[0] + rv[1]) / 28.0
    if pair:   s += 0.25
    if suited: s += 0.06
    if rv[0] - rv[1] <= 1 and not pair: s += 0.04
    return min(s, 1.0)

def _postflop_strength(hole, community):
    if not community: return _preflop_strength(hole)
    return best_score(hole + community)[0] / 9.0

def cpu_decide(hole, community, to_call, pot, chips, phase):
    strength  = _postflop_strength(hole, community)
    bluffing  = random.random() < 0.10
    eff_str   = min(1.0, strength + (0.3 if bluffing else 0.0))
    if chips <= 0:
        return ("check" if to_call == 0 else "call"), 0
    if to_call == 0:
        if eff_str > 0.65:
            raise_amt = min(max(BB, int(pot * 0.75)), chips)
            return "raise", raise_amt
        return "check", 0
    else:
        pot_odds = to_call / max(pot + to_call, 1)
        if eff_str > 0.78:
            raise_amt = min(max(to_call, int(pot * 0.8)), chips)
            return "raise", raise_amt
        if eff_str > pot_odds + 0.08:
            return "call", to_call
        return "fold", 0

# ─── ユーティリティ ───────────────────────────────────────────────────────────

def _log(msg):
    st.session_state.tx_log.insert(0, msg)
    st.session_state.tx_log = st.session_state.tx_log[:20]

def _new_deck():
    d = [(r, s) for s in SUITS for r in RANKS]
    random.shuffle(d)
    return d

def _pay(who, amount):
    amount = max(0, int(amount))
    if who == "player":
        amount = min(amount, st.session_state.tx_pchips)
        st.session_state.tx_pchips -= amount
    else:
        amount = min(amount, st.session_state.tx_cchips)
        st.session_state.tx_cchips -= amount
    st.session_state.tx_pot += amount
    return amount

def _award_pot(winner):
    pot = st.session_state.tx_pot
    if winner == "player":
        st.session_state.tx_pchips += pot
    elif winner == "cpu":
        st.session_state.tx_cchips += pot
    else:
        half = pot // 2
        st.session_state.tx_pchips += half
        st.session_state.tx_cchips += pot - half
    st.session_state.tx_pot = 0

def _start_betting_round():
    st.session_state.tx_p_bet_r = 0
    st.session_state.tx_c_bet_r = 0
    st.session_state.tx_p_acted = False
    st.session_state.tx_c_acted = False

def _round_over():
    if not (st.session_state.tx_p_acted and st.session_state.tx_c_acted):
        return False
    if st.session_state.tx_p_bet_r == st.session_state.tx_c_bet_r:
        return True
    return st.session_state.tx_pchips == 0 or st.session_state.tx_cchips == 0

# ─── フェーズ進行 ──────────────────────────────────────────────────────────────

def _advance_game():
    phase = st.session_state.tx_phase
    if phase == "pre_flop":
        deck = st.session_state.tx_deck
        st.session_state.tx_community = deck[:3]
        st.session_state.tx_deck = deck[3:]
        st.session_state.tx_phase = "flop"
        _log("── フロップ ──")
        _start_betting_round()
    elif phase == "flop":
        deck = st.session_state.tx_deck
        st.session_state.tx_community.append(deck[0])
        st.session_state.tx_deck = deck[1:]
        st.session_state.tx_phase = "turn"
        _log("── ターン ──")
        _start_betting_round()
    elif phase == "turn":
        deck = st.session_state.tx_deck
        st.session_state.tx_community.append(deck[0])
        st.session_state.tx_deck = deck[1:]
        st.session_state.tx_phase = "river"
        _log("── リバー ──")
        _start_betting_round()
    elif phase == "river":
        _showdown()

def _showdown():
    st.session_state.tx_phase = "showdown"
    p_cards = st.session_state.tx_phand + st.session_state.tx_community
    c_cards = st.session_state.tx_chand + st.session_state.tx_community
    ps = best_score(p_cards)
    cs = best_score(c_cards)
    if ps > cs:
        _award_pot("player")
        st.session_state.tx_result = f"🎉 プレイヤーの勝ち！  {hand_name(p_cards)}"
    elif cs > ps:
        _award_pot("cpu")
        st.session_state.tx_result = f"😔 CPUの勝ち  {hand_name(c_cards)}"
    else:
        _award_pot("tie")
        st.session_state.tx_result = f"🤝 引き分け  {hand_name(p_cards)}"
    _log(f"【ショーダウン】{st.session_state.tx_result}")
    _log(f"  CPU手札: {' '.join(r+s for r,s in st.session_state.tx_chand)}")

# ─── CPU アクション ────────────────────────────────────────────────────────────

def _cpu_act():
    to_call = max(0, st.session_state.tx_p_bet_r - st.session_state.tx_c_bet_r)
    action, extra = cpu_decide(
        st.session_state.tx_chand, st.session_state.tx_community,
        to_call, st.session_state.tx_pot,
        st.session_state.tx_cchips, st.session_state.tx_phase,
    )
    if action == "fold":
        _log("CPU: フォールド")
        _award_pot("player")
        st.session_state.tx_result = "🎉 CPUがフォールド！プレイヤーの勝ち！"
        st.session_state.tx_phase  = "showdown"
        return
    if action == "check":
        st.session_state.tx_c_acted = True
        _log("CPU: チェック")
    elif action == "call":
        paid = _pay("cpu", min(to_call, st.session_state.tx_cchips))
        st.session_state.tx_c_bet_r += paid
        st.session_state.tx_c_acted  = True
        _log(f"CPU: コール（{paid}）")
    elif action == "raise":
        total = to_call + extra
        paid  = _pay("cpu", min(total, st.session_state.tx_cchips))
        st.session_state.tx_c_bet_r += paid
        st.session_state.tx_c_acted  = True
        st.session_state.tx_p_acted  = False
        _log(f"CPU: レイズ（{paid}）← 応答してください")
        return
    if _round_over():
        _advance_game()

# ─── プレイヤーアクション ──────────────────────────────────────────────────────

def player_action(action, raise_extra=0):
    to_call = st.session_state.tx_c_bet_r - st.session_state.tx_p_bet_r
    if action == "fold":
        _log("プレイヤー: フォールド")
        _award_pot("cpu")
        st.session_state.tx_result = "😔 フォールド。CPUの勝ち"
        st.session_state.tx_phase  = "showdown"
        return
    if action == "check":
        st.session_state.tx_p_acted = True
        _log("プレイヤー: チェック")
    elif action == "call":
        paid = _pay("player", min(to_call, st.session_state.tx_pchips))
        st.session_state.tx_p_bet_r += paid
        st.session_state.tx_p_acted  = True
        _log(f"プレイヤー: コール（{paid}）")
    elif action == "raise":
        total = to_call + raise_extra
        paid  = _pay("player", min(total, st.session_state.tx_pchips))
        st.session_state.tx_p_bet_r += paid
        st.session_state.tx_p_acted  = True
        st.session_state.tx_c_acted  = False
        _log(f"プレイヤー: レイズ（{paid}）")
    elif action == "allin":
        paid = _pay("player", st.session_state.tx_pchips)
        st.session_state.tx_p_bet_r += paid
        st.session_state.tx_p_acted  = True
        st.session_state.tx_c_acted  = False
        _log(f"プレイヤー: オールイン（{paid}）")
    if _round_over():
        _advance_game()
    else:
        _cpu_act()

# ─── 新しいハンドの開始 ────────────────────────────────────────────────────────

def start_hand():
    if st.session_state.tx_pchips <= 0 or st.session_state.tx_cchips <= 0:
        return
    deck = _new_deck()
    st.session_state.tx_phand     = deck[:2]
    st.session_state.tx_chand     = deck[2:4]
    st.session_state.tx_deck      = deck[4:]
    st.session_state.tx_community = []
    st.session_state.tx_pot       = 0
    st.session_state.tx_result    = None
    st.session_state.tx_phase     = "pre_flop"
    st.session_state.tx_hand_num  = st.session_state.get("tx_hand_num", 0) + 1
    _log(f"═══ ハンド #{st.session_state.tx_hand_num} ═══")
    dealer = st.session_state.tx_dealer
    if dealer == "player":
        sb_paid = _pay("player", min(SB, st.session_state.tx_pchips))
        bb_paid = _pay("cpu",    min(BB, st.session_state.tx_cchips))
        _log(f"プレイヤー SB:{sb_paid}  CPU BB:{bb_paid}")
    else:
        sb_paid = _pay("cpu",    min(SB, st.session_state.tx_cchips))
        bb_paid = _pay("player", min(BB, st.session_state.tx_pchips))
        _log(f"CPU SB:{sb_paid}  プレイヤー BB:{bb_paid}")
    _start_betting_round()
    if dealer == "player":
        st.session_state.tx_p_bet_r = SB
        st.session_state.tx_c_bet_r = BB
    else:
        st.session_state.tx_p_bet_r = BB
        st.session_state.tx_c_bet_r = SB
    if dealer == "cpu":
        _cpu_act()

# ─── セッション初期化 ──────────────────────────────────────────────────────────
if "tx_pchips" not in st.session_state:
    st.session_state.tx_pchips    = 1000
    st.session_state.tx_cchips    = 1000
    st.session_state.tx_dealer    = "player"
    st.session_state.tx_phase     = "idle"
    st.session_state.tx_phand     = []
    st.session_state.tx_chand     = []
    st.session_state.tx_community = []
    st.session_state.tx_pot       = 0
    st.session_state.tx_p_bet_r   = 0
    st.session_state.tx_c_bet_r   = 0
    st.session_state.tx_p_acted   = False
    st.session_state.tx_c_acted   = False
    st.session_state.tx_result    = None
    st.session_state.tx_log       = []
    st.session_state.tx_hand_num  = 0

# ─── カード描画ヘルパー ────────────────────────────────────────────────────────

def _card_html(rank, suit, highlight=False, facedown=False):
    if facedown:
        return (
            '<div style="background:#0c1520;border:1px solid #1e3348;border-radius:8px;'
            'width:58px;height:84px;display:inline-flex;align-items:center;'
            'justify-content:center;flex-shrink:0;">'
            '<span style="font-size:2rem;opacity:0.6;">🂠</span></div>'
        )
    red   = suit in ("♥", "♦")
    color = "#ff6060" if red else "#d8eaf8"
    bg    = "#132840" if highlight else "#0d1e30"
    bdr   = "2px solid #00ccff" if highlight else "1px solid #1e3348"
    shd   = "0 0 10px #00ccff55" if highlight else "none"
    return (
        f'<div style="background:{bg};border:{bdr};border-radius:8px;'
        f'width:58px;height:84px;padding:6px 5px;box-sizing:border-box;'
        f'display:inline-flex;flex-direction:column;align-items:center;'
        f'justify-content:center;flex-shrink:0;box-shadow:{shd};">'
        f'<div style="font-size:1.5rem;font-weight:bold;color:{color};line-height:1;">{rank}</div>'
        f'<div style="font-size:1.3rem;color:{color};line-height:1;margin-top:2px;">{suit}</div>'
        f'</div>'
    )

def _cards_row(cards, facedown=False, highlight=False, n_slots=None):
    parts = []
    if facedown:
        parts = [_card_html("", "", facedown=True) for _ in (cards or range(n_slots or 2))]
    elif cards:
        parts = [_card_html(r, s, highlight=highlight) for r, s in cards]
    else:
        parts = [_card_html("", "", facedown=True) for _ in range(n_slots or 2)]
    return '<div style="display:flex;gap:7px;align-items:center;">' + "".join(parts) + "</div>"

def _community_row(comm, phase):
    # 常に5枚表示。フェーズに応じて公開枚数を変える
    n_reveal = {"flop": 3, "turn": 4, "river": 5, "showdown": 5}.get(phase, 0)
    parts = []
    for i in range(5):
        if i < len(comm) and i < n_reveal:
            r, s = comm[i]
            parts.append(_card_html(r, s))
        else:
            parts.append(_card_html("", "", facedown=True))
    return '<div style="display:flex;gap:7px;justify-content:center;">' + "".join(parts) + "</div>"

def _badge(text, fg="#4a90b8", bg="#0d1e2e"):
    return (
        f'<span style="border:1px solid {fg};border-radius:4px;padding:1px 6px;'
        f'font-size:0.62rem;color:{fg};background:{bg};font-weight:bold;'
        f'font-family:monospace;">{text}</span>'
    )

def _player_panel(label, chips, hand, is_dealer, blind_tag,
                  facedown=False, highlight=False, role_label=""):
    name_col = "#00ccff" if highlight else "#a0bcd4"
    chip_col = "#00dd77" if highlight else "#60a878"
    bg       = "linear-gradient(135deg,#0d2035,#0a1828)" if highlight else "linear-gradient(135deg,#0a1520,#080f18)"
    bdr      = "1px solid #00ccff55" if highlight else "1px solid #1a2e40"
    d_badge  = _badge("D", "#f0c030", "#1a1000") if is_dealer else ""
    b_badge  = _badge(blind_tag, "#4a90b8")
    cards_html = _cards_row(hand, facedown=facedown, highlight=highlight)
    role_html  = (f'<span style="font-size:0.72rem;color:#3a8888;margin-left:10px;">'
                  f'{role_label}</span>') if role_label else ""
    return f"""
<div style="background:{bg};border:{bdr};border-radius:10px;
            padding:12px 16px;margin:5px 0;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
    <span style="font-size:0.9rem;font-weight:bold;color:{name_col};">{label}</span>
    {d_badge}
    {b_badge}
    <span style="margin-left:auto;font-size:0.85rem;color:{chip_col};font-weight:bold;">💰 {chips:,} chips</span>
  </div>
  <div style="display:flex;align-items:center;">
    {cards_html}
    {role_html}
  </div>
</div>"""

# ─── メイン UI ──────────────────────────────────────────────────────────────

st.title("🃏 テキサスホールデム")
st.caption("1v1  プレイヤー vs CPU  ｜  SB:10 / BB:20  ｜  D = ディーラー（スモールブラインド）")

phase   = st.session_state.tx_phase
pchips  = st.session_state.tx_pchips
cchips  = st.session_state.tx_cchips
pot     = st.session_state.tx_pot
result  = st.session_state.tx_result
phand   = st.session_state.tx_phand
chand   = st.session_state.tx_chand
comm    = st.session_state.tx_community
dealer  = st.session_state.tx_dealer
to_call = max(0, st.session_state.get("tx_c_bet_r", 0) - st.session_state.get("tx_p_bet_r", 0))

phase_label = {
    "idle": "—", "pre_flop": "Pre-Flop", "flop": "Flop",
    "turn": "Turn", "river": "River", "showdown": "Showdown",
}.get(phase, "")

# ── CPU パネル ────────────────────────────────────────────────────────────────
c_dealer   = (dealer == "cpu")
c_hand_lbl = hand_name(chand + comm) if (phase == "showdown" and chand and comm) else ""
st.markdown(_player_panel(
    "🤖  CPU", cchips, chand,
    is_dealer=c_dealer, blind_tag="SB" if c_dealer else "BB",
    facedown=(phase != "showdown"),
    role_label=c_hand_lbl,
), unsafe_allow_html=True)

# ── コミュニティカード ────────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;padding:6px 0 4px;">'
    f'  <span style="font-size:0.7rem;letter-spacing:0.18em;color:#2a6080;">'
    f'  {phase_label}</span>'
    f'  <span style="font-size:0.85rem;font-weight:bold;color:#4a90b8;margin-left:16px;">'
    f'  POT &nbsp; {pot:,}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(_community_row(comm, phase), unsafe_allow_html=True)

# ── プレイヤーパネル ──────────────────────────────────────────────────────────
p_dealer   = (dealer == "player")
p_hand_lbl = hand_name(phand + comm) if (phand and comm) else ""
st.markdown(_player_panel(
    "👤  あなた", pchips, phand,
    is_dealer=p_dealer, blind_tag="SB" if p_dealer else "BB",
    highlight=True,
    role_label=p_hand_lbl,
), unsafe_allow_html=True)

st.divider()

# ── 結果バナー ────────────────────────────────────────────────────────────────
if result:
    if "🎉" in result:
        st.success(f"**{result}**")
    elif "🤝" in result:
        st.info(f"**{result}**")
    else:
        st.error(f"**{result}**")

# ── アクション ────────────────────────────────────────────────────────────────
if phase in ("idle", "showdown"):
    if pchips <= 0:
        st.error("チップがなくなりました！")
        if st.button("🔄 リセット（各 1,000 チップ）", use_container_width=True):
            for k, v in [("tx_pchips",1000),("tx_cchips",1000),
                         ("tx_phase","idle"),("tx_result",None)]:
                setattr(st.session_state, k, v)
            st.rerun()
    elif cchips <= 0:
        st.success("🏆 CPUのチップがなくなりました！完全勝利！")
        if st.button("🔄 リセット（各 1,000 チップ）", use_container_width=True):
            for k, v in [("tx_pchips",1000),("tx_cchips",1000),
                         ("tx_phase","idle"),("tx_result",None)]:
                setattr(st.session_state, k, v)
            st.rerun()
    else:
        next_d  = "cpu" if dealer == "player" else "player"
        next_sb = "あなた" if next_d == "player" else "CPU"
        if st.button(
            f"🃏  次のハンドをディール  ─  次の SB: {next_sb}",
            type="primary", use_container_width=True,
        ):
            st.session_state.tx_dealer = next_d
            start_hand()
            st.rerun()

else:
    if not st.session_state.tx_p_acted:
        # コール金額の案内
        if to_call > 0:
            st.markdown(
                f'<div style="font-size:0.8rem;color:#f0a040;margin-bottom:4px;">'
                f'⚠️ コールに必要: <b>{to_call} chips</b></div>',
                unsafe_allow_html=True,
            )

        a1, a2, a3, a4, a5 = st.columns([2, 2, 2, 2, 1])
        half_pot = max(BB, pot // 2)
        full_pot = max(BB, pot)
        can_raise = pchips > to_call

        with a1:
            if to_call == 0:
                st.button("✅ チェック", use_container_width=True, type="primary",
                          on_click=player_action, args=("check",))
            else:
                st.button(f"📞 コール  {min(to_call, pchips)}", use_container_width=True,
                          type="primary", on_click=player_action, args=("call",))
        with a2:
            st.button(f"⬆ ½ポット  +{half_pot}", use_container_width=True,
                      on_click=player_action, args=("raise", half_pot),
                      disabled=not can_raise)
        with a3:
            st.button(f"🚀 ポット  +{full_pot}", use_container_width=True,
                      on_click=player_action, args=("raise", full_pot),
                      disabled=not can_raise)
        with a4:
            st.button(f"💥 オールイン  {pchips}", use_container_width=True,
                      on_click=player_action, args=("allin",),
                      disabled=(pchips <= 0))
        with a5:
            st.button("🏳", use_container_width=True,
                      on_click=player_action, args=("fold",),
                      help="フォールド")
    else:
        st.caption("🤖 CPU が考えています...")

# ── アクションログ ────────────────────────────────────────────────────────────
with st.expander("📋 アクションログ", expanded=True):
    for line in st.session_state.tx_log[:12]:
        st.caption(line)

# ─── 固定フッター ─────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
