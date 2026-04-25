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
    """(action, extra_amount) を返す。action は fold/check/call/raise"""
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
    else:  # tie
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
    return (st.session_state.tx_p_bet_r == st.session_state.tx_c_bet_r
            and st.session_state.tx_p_acted
            and st.session_state.tx_c_acted)

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
    _log(f"  CPU手札: {_hand_str(st.session_state.tx_chand)}")

def _hand_str(hand):
    return " ".join(f"{r}{s}" for r, s in hand)

# ─── CPU アクション ────────────────────────────────────────────────────────────

def _cpu_act():
    to_call = st.session_state.tx_p_bet_r - st.session_state.tx_c_bet_r
    action, extra = cpu_decide(
        st.session_state.tx_chand,
        st.session_state.tx_community,
        to_call,
        st.session_state.tx_pot,
        st.session_state.tx_cchips,
        st.session_state.tx_phase,
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
        st.session_state.tx_p_acted  = False   # プレイヤーの応答が必要
        _log(f"CPU: レイズ（{paid}）← 応答してください")
        return  # プレイヤーに返す

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

    # ブラインド（dealer=SBとして交互）
    dealer = st.session_state.tx_dealer
    if dealer == "player":
        sb_paid = _pay("player", min(SB, st.session_state.tx_pchips))
        bb_paid = _pay("cpu",    min(BB, st.session_state.tx_cchips))
        _log(f"プレイヤー: SB {sb_paid}  CPU: BB {bb_paid}")
    else:
        sb_paid = _pay("cpu",    min(SB, st.session_state.tx_cchips))
        bb_paid = _pay("player", min(BB, st.session_state.tx_pchips))
        _log(f"CPU: SB {sb_paid}  プレイヤー: BB {bb_paid}")

    _start_betting_round()
    if dealer == "player":
        st.session_state.tx_p_bet_r = SB
        st.session_state.tx_c_bet_r = BB
    else:
        st.session_state.tx_p_bet_r = BB
        st.session_state.tx_c_bet_r = SB

    # プリフロップ：SBが先にアクション → SBがプレイヤーならプレイヤー先行
    if dealer == "cpu":
        # CPUがSBなのでCPUが先にアクション
        _cpu_act()

# ─── セッション初期化 ──────────────────────────────────────────────────────────
if "tx_pchips" not in st.session_state:
    st.session_state.tx_pchips    = 1000
    st.session_state.tx_cchips    = 1000
    st.session_state.tx_dealer    = "player"  # 交互に切り替え
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

# ─── カード描画 ───────────────────────────────────────────────────────────────

def _card(rank, suit, highlight=False):
    red   = suit in ("♥", "♦")
    color = "#ff6b6b" if red else "#dce8f8"
    bg    = "#0d2540" if highlight else "#0a1520"
    bdr   = "2px solid #00e5ff" if highlight else "1px solid #1e3a54"
    shd   = "0 0 12px #00e5ff55" if highlight else "none"
    return (
        f'<div style="background:{bg};border:{bdr};border-radius:9px;'
        f'padding:9px 8px 6px;min-width:62px;text-align:center;'
        f'font-family:monospace;box-shadow:{shd};display:inline-block;">'
        f'<div style="font-size:1.7rem;font-weight:bold;color:{color};line-height:1.1;">{rank}</div>'
        f'<div style="font-size:1.4rem;color:{color};line-height:1.1;">{suit}</div>'
        f'</div>'
    )

def _back():
    return (
        '<div style="background:#071020;border:1px solid #1e3a54;border-radius:9px;'
        'padding:9px 8px 6px;min-width:62px;text-align:center;display:inline-block;">'
        '<div style="font-size:1.7rem;line-height:1.1;">🂠</div>'
        '<div style="font-size:0.5rem;color:#1e3a54;line-height:1.1;">　</div>'
        '</div>'
    )

def _show_cards(cards, face_down=False, highlight=False):
    if not cards:
        html = " ".join(_back() for _ in range(2))
    elif face_down:
        html = " ".join(_back() for _ in cards)
    else:
        html = " ".join(_card(r, s, highlight) for r, s in cards)
    st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{html}</div>',
                unsafe_allow_html=True)

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🃏 テキサスホールデム")
st.caption("1v1 — プレイヤー vs CPU ディーラー　｜　ブラインド SB:10 / BB:20")

phase   = st.session_state.tx_phase
pchips  = st.session_state.tx_pchips
cchips  = st.session_state.tx_cchips
pot     = st.session_state.tx_pot
result  = st.session_state.tx_result
phand   = st.session_state.tx_phand
chand   = st.session_state.tx_chand
comm    = st.session_state.tx_community
to_call = max(0, st.session_state.get("tx_c_bet_r", 0) - st.session_state.get("tx_p_bet_r", 0))

# ── チップ・ポット情報 ─────────────────────────────────────────────────────────
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("👤 あなた",    f"{pchips:,}")
mc2.metric("🤖 CPU",      f"{cchips:,}")
mc3.metric("🏆 ポット",    f"{pot:,}")
dealer_label = "あなた" if st.session_state.tx_dealer == "player" else "CPU"
mc4.metric("🎯 ディーラー（SB）", dealer_label)

st.divider()

# ── テーブル ──────────────────────────────────────────────────────────────────
# CPU の手札
st.markdown("**🤖 CPU の手札**")
if phase == "showdown":
    _show_cards(chand)
    if chand and comm:
        st.caption(f"役: {hand_name(chand + comm)}")
else:
    _show_cards(chand, face_down=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# コミュニティカード
phase_labels = {
    "idle": "　", "pre_flop": "プリフロップ",
    "flop": "フロップ", "turn": "ターン",
    "river": "リバー", "showdown": "ショーダウン",
}
st.markdown(f"**🂡 コミュニティカード　— {phase_labels.get(phase, '')}**")
if comm:
    _show_cards(comm)
else:
    cols_c = st.columns(5)
    for col in cols_c:
        col.markdown(_back(), unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# プレイヤーの手札
st.markdown("**👤 あなたの手札**")
_show_cards(phand, highlight=True)
if phand and comm:
    st.caption(f"現在の役: **{hand_name(phand + comm)}**")

st.divider()

# ── 結果バナー ────────────────────────────────────────────────────────────────
if result:
    if "勝ち" in result or "🎉" in result:
        st.success(f"**{result}**")
    elif "引き分け" in result:
        st.info(f"**{result}**")
    else:
        st.error(f"**{result}**")

# ── アクションボタン ──────────────────────────────────────────────────────────
if phase == "idle" or phase == "showdown":
    # 次のハンドへ
    if pchips <= 0:
        st.error("チップがなくなりました！")
        if st.button("🔄 リセット（各1,000チップ）", use_container_width=True):
            st.session_state.tx_pchips = 1000
            st.session_state.tx_cchips = 1000
            st.session_state.tx_phase  = "idle"
            st.session_state.tx_result = None
            st.rerun()
    elif cchips <= 0:
        st.success("🏆 CPUのチップがなくなりました！あなたの完全勝利！")
        if st.button("🔄 リセット（各1,000チップ）", use_container_width=True):
            st.session_state.tx_pchips = 1000
            st.session_state.tx_cchips = 1000
            st.session_state.tx_phase  = "idle"
            st.session_state.tx_result = None
            st.rerun()
    else:
        if st.button("🃏  次のハンドをディール", type="primary", use_container_width=True):
            # ディーラーを交替
            st.session_state.tx_dealer = "cpu" if st.session_state.tx_dealer == "player" else "player"
            start_hand()
            st.rerun()

else:
    # ベッティングアクション
    p_acted   = st.session_state.tx_p_acted
    cpu_acted = st.session_state.tx_c_acted

    # CPUがレイズしてプレイヤーの応答待ち、またはプレイヤーが未アクション
    player_needs_to_act = not p_acted or (cpu_acted and not p_acted)

    if player_needs_to_act:
        if to_call > 0:
            st.caption(f"コールに必要: **{to_call}** チップ")

        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            if to_call == 0:
                st.button("✅ チェック", use_container_width=True,
                          on_click=player_action, args=("check",))
            else:
                call_amt = min(to_call, pchips)
                st.button(f"📞 コール（{call_amt}）", use_container_width=True,
                          on_click=player_action, args=("call",))
        with ac2:
            half_pot = max(BB, pot // 2)
            disabled = pchips <= to_call
            st.button(f"⬆ ハーフポット（+{half_pot}）", use_container_width=True,
                      on_click=player_action, args=("raise", half_pot),
                      disabled=disabled)
        with ac3:
            full_pot = max(BB, pot)
            st.button(f"🚀 ポット（+{full_pot}）", use_container_width=True,
                      on_click=player_action, args=("raise", full_pot),
                      disabled=disabled)
        with ac4:
            st.button("💥 オールイン", use_container_width=True,
                      on_click=player_action, args=("allin",),
                      type="primary" if pchips <= to_call else "secondary")

        st.button("🏳 フォールド", use_container_width=False,
                  on_click=player_action, args=("fold",))
    else:
        st.info("CPU がアクション中...")

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
