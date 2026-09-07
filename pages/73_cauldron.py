# ミニゲーム：魔女の大釜（デッキ構築ローグライク）
#
# 既存3本（70=アイドル・71=育成・72=ポーカー）は「毎回同じことをする」構造なので飽きる。
# 本作は毎回デッキとレリックが変わり、その組み合わせで戦い方が別物になる設計にしてある。
#
# 独自システム「ボイル値」:
#   薬草を刻んで大釜のボイル値を上げるほど攻撃が伸びるが、
#   ターン終了時に ボイル値 > 大釜の容量 だと超過分だけ「吹きこぼれ」ダメージを受ける。
#   ボイルを盛る／容量を広げる／溜めた瞬間にぶちまける、の3方向のビルドが噛み合う。
#   さらに敵「火の玉」はボイルが高いほど強く殴ってくるので、盤面によって最適解が変わる。
#
# セーブ（メタ進行）は /workspace/tmp/cauldron_save.json。
# 大魔女を倒すと「魔女ランク」が上がり、敵が強くなる＝長く遊べる仕掛け。

import json
import random
from pathlib import Path

import streamlit as st

SAVE_FILE = Path("/workspace/tmp/cauldron_save.json")
MAX_FLOOR = 12          # 12層目がボス
HAND_MAX  = 10

# ═══════════════════════════════════════════════════════════════════════════════
#  ダメージ計算などの基本ヘルパー
# ═══════════════════════════════════════════════════════════════════════════════

def atk_value(B, base):
    """プレイヤーの攻撃力（魔力を加算し、衰弱なら25%減）"""
    v = base + B["pst"].get("str", 0)
    if B["pst"].get("weak", 0) > 0:
        v = int(v * 0.75)
    return max(0, v)


def hit_enemy(S, B, e, v):
    """敵に v ダメージ。呪縛中なら1.5倍。敵の身構えを先に削る。棘があれば反射"""
    if v <= 0:
        return 0
    if e["st"].get("vuln", 0) > 0:
        v = int(v * 1.5)
    if e.get("blk", 0) > 0:
        used = min(e["blk"], v)
        e["blk"] -= used
        v -= used
    e["hp"] -= v
    if e.get("thorn", 0) > 0 and v > 0:
        hit_player(S, B, e["thorn"], pierce=True)
        B["log"].append(f"　└ {e['name']}の棘が {e['thorn']} 跳ね返した")
    return v


def hit_player(S, B, v, pierce=False):
    """プレイヤーに v ダメージ。pierce=True は防御を無視（吹きこぼれ・毒など）"""
    if v <= 0:
        return 0
    if B["pst"].get("vuln", 0) > 0:
        v = int(v * 1.5)
    v = max(0, v - B.get("dr", 0))
    if not pierce:
        used = min(B["block"], v)
        B["block"] -= used
        v -= used
    S["hp"] -= v
    return v


def gain_block(B, n):
    B["block"] += max(0, n)


def gain_g(B, n):
    B["G"] = max(0, B["G"] + n)


def draw_cards(B, n):
    for _ in range(n):
        if len(B["hand"]) >= HAND_MAX:
            break
        if not B["draw"]:
            if not B["disc"]:
                break
            B["draw"] = B["disc"]
            B["disc"] = []
            random.shuffle(B["draw"])
        B["hand"].append(B["draw"].pop())


def add_card_to_hand(B, cid, k=1):
    for _ in range(k):
        if len(B["hand"]) < HAND_MAX:
            B["hand"].append(cid)
        else:
            B["disc"].append(cid)


def alive(B):
    return [e for e in B["enemies"] if e["hp"] > 0]


# ═══════════════════════════════════════════════════════════════════════════════
#  カード定義
#   fx(S, B, tgt) -> ログ文字列のリスト   tgt は対象の敵dict
# ═══════════════════════════════════════════════════════════════════════════════

def _f_strike(S, B, t):
    d = hit_enemy(S, B, t, atk_value(B, 6));  return [f"炎の一撃 → {t['name']} に {d}"]

def _f_guard(S, B, t):
    gain_block(B, 5);  return ["守りの霧 → 防御+5"]

def _f_double(S, B, t):
    a = hit_enemy(S, B, t, atk_value(B, 4)); b = hit_enemy(S, B, t, atk_value(B, 4))
    return [f"二連打 → {t['name']} に {a}+{b}"]

def _f_curse(S, B, t):
    d = hit_enemy(S, B, t, atk_value(B, 7)); t["st"]["vuln"] = t["st"].get("vuln", 0) + 2
    return [f"呪いの一滴 → {t['name']} に {d}・呪縛2"]

def _f_pour(S, B, t):
    d = hit_enemy(S, B, t, atk_value(B, B["G"] // 2))
    return [f"煮汁を注ぐ → ボイル{B['G']}の半分 = {d}"]

def _f_blast(S, B, t):
    d = hit_enemy(S, B, t, atk_value(B, 14)); return [f"大火力 → {t['name']} に {d}"]

def _f_dump(S, B, t):
    g = B["G"]; d = hit_enemy(S, B, t, atk_value(B, g)); B["G"] = 0
    return [f"ぶちまける → ボイル{g}を全部 {d}ダメージ"]

def _f_ladle(S, B, t):
    d = hit_enemy(S, B, t, atk_value(B, 3)); draw_cards(B, 1)
    return [f"木べらで一閃 → {d}・1ドロー"]

def _f_herb(S, B, t):
    gain_g(B, 6); return ["薬草を刻む → ボイル+6"]

def _f_toss(S, B, t):
    gain_g(B, 3); draw_cards(B, 1); return ["ひとつまみ → ボイル+3・1ドロー"]

def _f_skim(S, B, t):
    gain_g(B, -4); gain_block(B, 6); return ["灰汁取り → ボイル-4・防御+6"]

def _f_lid(S, B, t):
    gain_g(B, -6); gain_block(B, 10); B["no_oom"] = True
    return ["蓋をする → ボイル-6・防御+10・今ターン吹きこぼれ無効"]

def _f_moon(S, B, t):
    S["hp"] = min(S["maxhp"], S["hp"] + 6); return ["満月の雫 → 体力+6"]

def _f_early(S, B, t):
    gain_block(B, 10); B["_endturn"] = True; return ["早仕舞い → 防御+10・ターン終了"]

def _f_amp(S, B, t):
    B["pst"]["str"] = B["pst"].get("str", 0) + 2; return ["魔力増幅 → 魔力+2"]

def _f_breath(S, B, t):
    draw_cards(B, 2); return ["深呼吸 → 2ドロー"]

def _f_forbidden(S, B, t):
    gain_g(B, 10); B["disc"].append("c_ash")
    return ["禁書のページ → ボイル+10（捨札に灰1枚）"]

def _f_circle(S, B, t):
    B["pst"]["str"] = B["pst"].get("str", 0) + 3; B["mip"] = True
    return ["【魔法陣】魔力+3・以後の刻む系が+2"]

def _f_ironpot(S, B, t):
    B["vram"] += 8; return ["【鉄の大釜】容量+8"]

def _f_simmer(S, B, t):
    B["pow_dense"] = B.get("pow_dense", 0) + 3
    return ["【とろ火】ターン開始時 ボイル+3"]

def _f_ward(S, B, t):
    B["dr"] = B.get("dr", 0) + 3; return ["【守りの結界】受けるダメージ-3"]

def _f_heat(S, B, t):
    B["pow_str"] = B.get("pow_str", 0) + 1; return ["【高まる熱】ターン開始時 魔力+1"]

def _f_dragon(S, B, t):
    d = hit_enemy(S, B, t, atk_value(B, 20)); return [f"竜の血 → {d}（使い捨て）"]

def _f_overturn(S, B, t):
    g = B["G"]; d = hit_enemy(S, B, t, atk_value(B, g * 2)); B["G"] = 0
    return [f"大釜をひっくり返す → ボイル{g}×2 = {d}（使い捨て）"]

def _f_stir(S, B, t):
    for e in alive(B):
        e["st"]["str"] = 0; e["blk"] = 0
    gain_block(B, 12)
    return ["撹拌 → 敵の強化と身構えを消して防御+12"]

def _f_powder(S, B, t):
    B["pst"]["str"] = B["pst"].get("str", 0) + 6; B["ks"] = True
    return ["【火薬を混ぜる】魔力+6・ただし毎ターン2ダメージ"]

def _f_flee(S, B, t):
    n = len(B["hand"])
    B["disc"] += B["hand"]; B["hand"] = []
    draw_cards(B, 5)
    return [f"鍋を置いて逃げる → 手札{n}枚を捨てて5ドロー（使い捨て）"]

def _f_widen(S, B, t):
    B["vram"] += 4; gain_g(B, 8); return ["大釜を広げる → 容量+4・ボイル+8"]

def _f_numb(S, B, t):
    for e in alive(B):
        e["st"]["weak"] = e["st"].get("weak", 0) + 2
    return ["しびれ粉 → 全ての敵に衰弱2"]

def _f_salt(S, B, t):
    d = hit_enemy(S, B, t, atk_value(B, 9))
    B["pst"]["vuln"] = 0; B["pst"]["weak"] = 0
    return [f"清めの塩 → {d}・自分の呪いを解く"]

def _f_whirl(S, B, t):
    tot = 0
    for e in alive(B):
        tot += hit_enemy(S, B, e, atk_value(B, 7))
    return [f"渦を巻く → 全体に計{tot}"]


CARDS = {
    # ── 基本カード ──
    "c_strike": dict(name="炎の一撃", cost=1, type="attack", rarity="basic", fx=_f_strike,
                     desc="6ダメージ"),
    "c_guard":  dict(name="守りの霧", cost=1, type="skill",  rarity="basic", fx=_f_guard,
                     desc="防御+5"),
    # ── コモン ──
    "c_double": dict(name="二連打",   cost=1, type="attack", rarity="common", fx=_f_double,
                     desc="4ダメージ×2"),
    "c_curse":  dict(name="呪いの一滴", cost=1, type="attack", rarity="common", fx=_f_curse,
                     desc="7ダメージ・呪縛2"),
    "c_pour":   dict(name="煮汁を注ぐ", cost=1, type="attack", rarity="common", fx=_f_pour,
                     desc="ボイルの半分ダメージ"),
    "c_ladle":  dict(name="木べらで一閃", cost=0, type="attack", rarity="common", fx=_f_ladle,
                     desc="3ダメージ・1ドロー"),
    "c_herb":   dict(name="薬草を刻む", cost=1, type="skill",  rarity="common", fx=_f_herb,
                     desc="ボイル+6"),
    "c_toss":   dict(name="ひとつまみ", cost=1, type="skill",  rarity="common", fx=_f_toss,
                     desc="ボイル+3・1ドロー"),
    "c_skim":   dict(name="灰汁取り",  cost=0, type="skill",  rarity="common", fx=_f_skim,
                     desc="ボイル-4・防御+6"),
    "c_moon":   dict(name="満月の雫",  cost=1, type="skill",  rarity="common", fx=_f_moon,
                     desc="体力+6"),
    "c_breath": dict(name="深呼吸",    cost=1, type="skill",  rarity="common", fx=_f_breath,
                     desc="2ドロー"),
    # ── アンコモン ──
    "c_blast":  dict(name="大火力",    cost=2, type="attack", rarity="uncommon", fx=_f_blast,
                     desc="14ダメージ"),
    "c_dump":   dict(name="ぶちまける", cost=2, type="attack", rarity="uncommon", fx=_f_dump,
                     desc="ボイルと同じダメージ・ボイル→0"),
    "c_lid":    dict(name="蓋をする",  cost=1, type="skill",  rarity="uncommon", fx=_f_lid,
                     desc="ボイル-6・防御+10・今ターン吹きこぼれ無効"),
    "c_early":  dict(name="早仕舞い",  cost=0, type="skill",  rarity="uncommon", fx=_f_early,
                     desc="防御+10・ターン終了"),
    "c_amp":    dict(name="魔力増幅",  cost=1, type="skill",  rarity="uncommon", fx=_f_amp,
                     desc="魔力+2"),
    "c_forbid": dict(name="禁書のページ", cost=1, type="skill", rarity="uncommon", fx=_f_forbidden,
                     desc="ボイル+10・捨札に灰1枚"),
    "c_numb":   dict(name="しびれ粉",  cost=1, type="skill",  rarity="uncommon", fx=_f_numb,
                     desc="全ての敵に衰弱2"),
    "c_salt":   dict(name="清めの塩",  cost=1, type="attack", rarity="uncommon", fx=_f_salt,
                     desc="9ダメージ・自分の呪いを解く"),
    "c_whirl":  dict(name="渦を巻く",  cost=2, type="attack", rarity="uncommon", fx=_f_whirl,
                     desc="全ての敵に7ダメージ"),
    "c_widen":  dict(name="大釜を広げる", cost=1, type="skill", rarity="uncommon", fx=_f_widen,
                     desc="容量+4・ボイル+8"),
    "c_circle": dict(name="魔法陣",    cost=1, type="power",  rarity="uncommon", fx=_f_circle,
                     desc="魔力+3・以後の刻む系が+2"),
    "c_simmer": dict(name="とろ火",    cost=1, type="power",  rarity="uncommon", fx=_f_simmer,
                     desc="ターン開始時 ボイル+3"),
    # ── レア ──
    "c_iron":   dict(name="鉄の大釜",  cost=2, type="power",  rarity="rare", fx=_f_ironpot,
                     desc="大釜の容量+8"),
    "c_ward":   dict(name="守りの結界", cost=2, type="power", rarity="rare", fx=_f_ward,
                     desc="受けるダメージ-3"),
    "c_heat":   dict(name="高まる熱",  cost=2, type="power",  rarity="rare", fx=_f_heat,
                     desc="ターン開始時 魔力+1"),
    "c_dragon": dict(name="竜の血",    cost=2, type="attack", rarity="rare", fx=_f_dragon,
                     desc="20ダメージ", exhaust=True),
    "c_stir":   dict(name="撹拌",      cost=1, type="skill",  rarity="rare", fx=_f_stir,
                     desc="敵の強化と身構えを消去・防御+12"),
    "c_flee":   dict(name="鍋を置いて逃げる", cost=0, type="skill", rarity="rare", fx=_f_flee,
                     desc="手札を全部捨てて5ドロー", exhaust=True),
    # ── レア（初回クリアで解禁）──
    "c_over":   dict(name="大釜をひっくり返す", cost=3, type="attack", rarity="rare", fx=_f_overturn,
                     desc="ボイルの2倍ダメージ・ボイル→0", exhaust=True, locked=True),
    "c_powder": dict(name="火薬を混ぜる", cost=1, type="power", rarity="rare", fx=_f_powder,
                     desc="魔力+6・毎ターン2ダメージ", locked=True),
    # ── 妨害カード（プレイ不可）──
    "c_ash":  dict(name="灰", cost=0, type="status", rarity="status", fx=None,
                   desc="プレイ不可。ただの邪魔"),
    "c_mud":  dict(name="泥", cost=0, type="status", rarity="status", fx=None,
                   desc="プレイ不可。ターン終了時、手札にあると2ダメージ"),
    "c_damp": dict(name="湿った薪", cost=0, type="status", rarity="status", fx=None,
                   desc="プレイ不可。ターン終了時、手札にあると防御-3"),
}

TYPE_COLOR = {"attack": "#FFB59B", "skill": "#A8C7FA", "power": "#CFBCFF", "status": "#9E948A"}
RARITY_JA  = {"basic": "基本", "common": "コモン", "uncommon": "アンコモン",
              "rare": "レア", "status": "妨害"}

# ═══════════════════════════════════════════════════════════════════════════════
#  レリック（戦い方そのものを変えるものを優先）
# ═══════════════════════════════════════════════════════════════════════════════

RELICS = {
    "r_copper": dict(name="銅の大釜",     emoji="🥘", desc="大釜の容量 +10"),
    "r_sala":   dict(name="火トカゲ",     emoji="🦎", desc="ターン開始時 体力+2"),
    "r_mitt":   dict(name="鍋つかみ",     emoji="🧤", desc="吹きこぼれのダメージを受けない"),
    "r_cat":    dict(name="黒猫の使い魔", emoji="🐈‍⬛", desc="戦闘開始時に1ドロー"),
    "r_hat":    dict(name="魔女の帽子",   emoji="🎩", desc="戦闘開始時 魔力+2"),
    "r_boots":  dict(name="風の靴",       emoji="👢", desc="戦闘開始時 マナ+2"),
    "r_vent":   dict(name="通気口",       emoji="🌬️", desc="吹きこぼれのダメージ半減"),
    "r_bundle": dict(name="干し薬草の束", emoji="🌿", desc="戦闘開始時 ボイル+10"),
    "r_map":    dict(name="行商人の地図", emoji="🗺️", desc="カード報酬の選択肢が4つになる"),
    "r_pack":   dict(name="大きな背嚢",   emoji="🎒", desc="最大体力 +20（取得時に全回復）"),
    "r_wing":   dict(name="妖精の羽",     emoji="🧚", desc="毎ターンのマナ +1"),
    "r_spoon":  dict(name="お気に入りの木べら", emoji="🥄", desc="戦闘開始時、手札に「薬草を刻む」"),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  敵（moves を順番に繰り返す＝先が読めるので戦略が立つ）
# ═══════════════════════════════════════════════════════════════════════════════

ENEMIES = {
    "e_slime": dict(name="スライム", emoji="🟢", hp=(20, 26), tier="normal", moves=[
        dict(n="体当たり", t="atk", v=6),
        dict(n="分裂", t="card", card="c_mud", k=1),
        dict(n="体当たり", t="atk", v=8)]),
    "e_flame": dict(name="火の玉", emoji="🔥", hp=(26, 32), tier="normal", moves=[
        dict(n="熱を吸う", t="atk_g", v=5),
        dict(n="ちらつく", t="blk", v=8),
        dict(n="熱を吸う", t="atk_g", v=6)]),
    "e_shroom": dict(name="キノコの精", emoji="🍄", hp=(22, 28), tier="normal", moves=[
        dict(n="胞子", t="debuff", st="weak", v=2),
        dict(n="突進", t="atk", v=9),
        dict(n="毒の霧", t="debuff", st="vuln", v=2)]),
    "e_vine": dict(name="蔓の魔物", emoji="🌿", hp=(28, 34), tier="normal", moves=[
        dict(n="根を張る", t="buff", v=3),
        dict(n="締め上げ", t="atk", v=7),
        dict(n="締め上げ", t="atk", v=7)]),
    "e_crow": dict(name="鴉の群れ", emoji="🐦‍⬛", hp=(24, 30), tier="normal", moves=[
        dict(n="つつく", t="atk2", v=4),
        dict(n="目つぶし", t="card", card="c_damp", k=1),
        dict(n="つつく", t="atk2", v=5)]),
    # ── 強敵 ──
    "e_thicket": dict(name="藪の主", emoji="🌳", hp=(52, 58), tier="elite", thorn=4, moves=[
        dict(n="薙ぎ払い", t="atk", v=13),
        dict(n="棘を伸ばす", t="buff", v=4),
        dict(n="薙ぎ払い", t="atk", v=13),
        dict(n="樹皮を固める", t="blk", v=14)]),
    "e_wraith": dict(name="沼の亡霊", emoji="👻", hp=(48, 54), tier="elite", moves=[
        dict(n="呪詛", t="debuff", st="vuln", v=2),
        dict(n="沈める", t="atk", v=11),
        dict(n="泥を撒く", t="card", card="c_mud", k=2),
        dict(n="沈める", t="atk", v=14)]),
    # ── ボス ──
    "e_witch": dict(name="大魔女ヴェルナ", emoji="🧙‍♀️", hp=(116, 116), tier="boss", moves=[
        dict(n="二重の呪い", t="debuff2", st=("weak", "vuln"), v=2),
        dict(n="灰を撒く", t="card", card="c_ash", k=1),
        dict(n="魔弾", t="atk", v=15),
        dict(n="大呪文", t="atk", v=24)]),
}

NORMALS = ["e_slime", "e_flame", "e_shroom", "e_vine", "e_crow"]
ELITES  = ["e_thicket", "e_wraith"]

EVENTS = [
    dict(title="森の泉", text="澄んだ水が湧いている。飲むと力が湧きそうだ。",
         opts=[("たっぷり飲む（体力+15）", dict(hp=15)),
               ("水筒に汲む（ゴールド+30）", dict(gold=30))]),
    dict(title="行商人の荷車", text="怪しい行商人が荷を広げている。",
         opts=[("値切って買う（レリック獲得・体力-10）", dict(hp=-10, relic=1)),
               ("素通りする（ゴールド+25）", dict(gold=25))]),
    dict(title="迷子の妖精", text="羽を痛めた妖精が震えている。",
         opts=[("薬を分ける（体力-8・カード獲得）", dict(hp=-8, card=1)),
               ("巣まで送る（ゴールド+45）", dict(gold=45))]),
    dict(title="嵐の夜", text="雷が鳴っている。大釜の火が消えそうだ。",
         opts=[("火を守る（体力-12・カード獲得）", dict(hp=-12, card=1)),
               ("荷物を捨てて逃げる（カード1枚を失う）", dict(remove=1))]),
    dict(title="古い書架", text="埃をかぶった魔導書。読むと知識が増えるが目が疲れる。",
         opts=[("読みふける（体力-6・ゴールド+50）", dict(hp=-6, gold=50)),
               ("要らないページを破り捨てる（カード1枚除去）", dict(remove=1))]),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  セーブ（メタ進行）
# ═══════════════════════════════════════════════════════════════════════════════

def load_meta():
    try:
        return json.loads(SAVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": 0, "wins": 0, "best_floor": 0, "asc": 0, "cleared_asc": -1}


def save_meta(m):
    try:
        SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SAVE_FILE.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def pool_cards(meta, rarities):
    out = []
    for cid, c in CARDS.items():
        if c["rarity"] not in rarities:
            continue
        if c.get("locked") and meta.get("wins", 0) < 1:
            continue
        out.append(cid)
    return out


def roll_reward_cards(meta, n):
    picks = []
    pool_c = pool_cards(meta, ["common"])
    pool_u = pool_cards(meta, ["uncommon"])
    pool_r = pool_cards(meta, ["rare"])
    guard = 0
    while len(picks) < n and guard < 200:
        guard += 1
        r = random.random()
        src = pool_r if r < 0.12 else (pool_u if r < 0.52 else pool_c)
        cid = random.choice(src)
        if cid not in picks:
            picks.append(cid)
    return picks


# ═══════════════════════════════════════════════════════════════════════════════
#  ラン管理
# ═══════════════════════════════════════════════════════════════════════════════

def gen_map():
    rows = []
    for f in range(MAX_FLOOR):
        if f == MAX_FLOOR - 1:
            rows.append(["boss"]); continue
        if f == 0:
            rows.append(["battle", "battle"]); continue
        pool = ["battle", "battle", "battle", "elite", "shop", "rest", "event"]
        if f < 3:
            pool = ["battle", "battle", "event", "rest"]
        k = 3 if f % 4 == 2 else 2
        opts = []
        guard = 0
        while len(opts) < k and guard < 100:
            guard += 1
            c = random.choice(pool)
            if c in opts:
                continue
            opts.append(c)
        while len(opts) < k:
            opts.append("battle")
        rows.append(opts)
    return rows


def new_run(meta):
    deck = ["c_strike"] * 4 + ["c_guard"] * 4 + ["c_herb", "c_pour"]
    return {
        "scene": "map", "hp": 78, "maxhp": 78, "gold": 60,
        "floor": 0, "deck": deck, "relics": [], "map": gen_map(),
        "battle": None, "pending": None, "asc": meta.get("asc", 0),
        "msg": "森の奥へ。大魔女ヴェルナの棲む館を目指す。",
    }


def has(S, rid):
    return rid in S["relics"]


def start_battle(S, kind):
    asc = S["asc"]
    hp_mul = 1.0 + 0.10 * asc
    dmg_add = asc
    if kind == "boss":
        ids = ["e_witch"]
    elif kind == "elite":
        ids = [random.choice(ELITES)]
    else:
        n = 1 if S["floor"] < 3 else random.choice([1, 1, 2])
        ids = [random.choice(NORMALS) for _ in range(n)]

    enemies = []
    for i, eid in enumerate(ids):
        d = ENEMIES[eid]
        lo, hi = d["hp"]
        hp = int(random.randint(lo, hi) * (1 + 0.05 * S["floor"]) * hp_mul)
        enemies.append(dict(id=eid, name=d["name"], emoji=d["emoji"], hp=hp, maxhp=hp,
                            blk=0, st={}, thorn=d.get("thorn", 0), mi=0, uid=i, intent=None))

    draw = S["deck"][:]
    random.shuffle(draw)
    B = dict(enemies=enemies, hand=[], draw=draw, disc=[], exh=[], powers=[],
             energy=3, maxenergy=3, block=0, G=0, vram=12,
             pst={"str": 0, "weak": 0, "vuln": 0}, turn=1, log=[], kind=kind,
             dmg_add=dmg_add, no_oom=False, dr=0, mip=False, ks=False,
             pow_dense=0, pow_str=0, target=0)

    if has(S, "r_copper"): B["vram"] += 10
    if has(S, "r_wing"):   B["maxenergy"] += 1; B["energy"] += 1
    if has(S, "r_boots"):  B["energy"] += 2
    if has(S, "r_hat"):    B["pst"]["str"] += 2
    if has(S, "r_bundle"): B["G"] += 10
    if has(S, "r_spoon"):  B["hand"].append("c_herb")

    draw_cards(B, 5 + (1 if has(S, "r_cat") else 0))
    roll_intents(B)
    S["battle"] = B
    S["scene"] = "battle"


def roll_intents(B):
    for e in alive(B):
        d = ENEMIES[e["id"]]
        e["intent"] = d["moves"][e["mi"] % len(d["moves"])]


INTENT_TIP = {
    "atk":  "次のターン、この数値のダメージであなたを攻撃してくる。防御で減らせる。",
    "atk2": "次のターン、この数値の攻撃を2回してくる。防御は1回ぶんずつ削られる。",
    "atk_g": "次のターン攻撃してくる。ボイル値が4上がるごとにダメージが1増える。",
    "blk":  "次のターン、この敵は身構える（受けるダメージを肩代わりする値を得る）。",
    "buff": "次のターン、この敵は自分の魔力を上げて攻撃力を永続的に強化する。",
    "debuff": "次のターン、あなたに弱体化をかけてくる。",
    "debuff2": "次のターン、あなたに衰弱と呪縛の両方をかけてくる。",
    "card": "次のターン、あなたの手札に使えない妨害カードを送り込んでくる。",
}


def intent_tip(e):
    mv = e["intent"]
    return INTENT_TIP.get(mv["t"], "") if mv else ""


def intent_text(B, e):
    mv = e["intent"]
    if mv is None:
        return "…"
    base = mv.get("v", 0) + (B["dmg_add"] if mv["t"] in ("atk", "atk2", "atk_g") else 0)
    v = base + e["st"].get("str", 0)
    if e["st"].get("weak", 0) > 0:
        v = int(v * 0.75)
    if mv["t"] == "atk":
        return f"⚔️ {mv['n']} {v}"
    if mv["t"] == "atk2":
        return f"⚔️ {mv['n']} {v}×2"
    if mv["t"] == "atk_g":
        return f"⚔️ {mv['n']} {v + B['G'] // 4}（ボイルに反応）"
    if mv["t"] == "blk":
        return f"🛡️ {mv['n']} +{base}"
    if mv["t"] == "buff":
        return f"⬆️ {mv['n']} 魔力+{base}"
    if mv["t"] in ("debuff", "debuff2"):
        return f"⬇️ {mv['n']}"
    if mv["t"] == "card":
        return f"🃏 {mv['n']} ×{mv['k']}"
    return mv["n"]


# ═══════════════════════════════════════════════════════════════════════════════
#  ターン処理
# ═══════════════════════════════════════════════════════════════════════════════

def play_card(S, B, idx):
    if idx >= len(B["hand"]):
        return
    cid = B["hand"][idx]
    c = CARDS[cid]
    if c["type"] == "status" or B["energy"] < c["cost"]:
        return
    tgts = alive(B)
    if not tgts:
        return
    tgt = tgts[min(B["target"], len(tgts) - 1)]

    B["energy"] -= c["cost"]
    B["hand"].pop(idx)
    bonus = 2 if (B["mip"] and cid in ("c_herb", "c_toss", "c_forbid")) else 0
    logs = c["fx"](S, B, tgt)
    if bonus:
        gain_g(B, bonus); logs.append(f"　└ 魔法陣で ボイル+{bonus}")
    B["log"] += logs

    if c.get("exhaust") or c["type"] == "power":
        B["exh"].append(cid)
        if c["type"] == "power":
            B["powers"].append(cid)
    else:
        B["disc"].append(cid)

    for e in B["enemies"]:
        if e["hp"] <= 0 and not e.get("dead_logged"):
            e["dead_logged"] = True
            B["log"].append(f"💥 {e['name']} を倒した！")
    if not alive(B):
        finish_battle(S, B)
        return
    if B.get("_endturn"):
        B["_endturn"] = False
        end_turn(S, B)


def end_turn(S, B):
    # 1) 手札に残った妨害カードの罰
    for cid in B["hand"]:
        if cid == "c_mud":
            hit_player(S, B, 2, pierce=True); B["log"].append("泥が跳ねた（2ダメージ）")
        elif cid == "c_damp":
            B["block"] = max(0, B["block"] - 3); B["log"].append("湿った薪で防御-3")
    B["disc"] += B["hand"]; B["hand"] = []

    # 2) 吹きこぼれ判定（本作の肝）
    if B["G"] > B["vram"]:
        over = B["G"] - B["vram"]
        if B["no_oom"] or has(S, "r_mitt"):
            B["log"].append(f"蓋のおかげで超過{over}を捨てた（吹きこぼれ回避）")
        else:
            dm = over // 2 if has(S, "r_vent") else over
            hit_player(S, B, dm, pierce=True)
            B["log"].append(f"⚠️ 吹きこぼれ！ 容量{B['vram']}にボイル{B['G']} → {dm}ダメージ")
        B["G"] = B["vram"]
    B["no_oom"] = False

    if B["ks"]:
        hit_player(S, B, 2, pierce=True); B["log"].append("火薬の煙にむせた（2ダメージ）")

    if S["hp"] <= 0:
        S["scene"] = "lose"; return

    # 3) 敵の行動
    for e in alive(B):
        e["blk"] = 0
        mv = e["intent"]
        base = mv.get("v", 0) + (B["dmg_add"] if mv["t"] in ("atk", "atk2", "atk_g") else 0)
        ev = base + e["st"].get("str", 0)
        if e["st"].get("weak", 0) > 0:
            ev = int(ev * 0.75)
        if mv["t"] == "atk":
            d = hit_player(S, B, ev); B["log"].append(f"{e['name']}の{mv['n']} → {d}ダメージ")
        elif mv["t"] == "atk2":
            d1 = hit_player(S, B, ev); d2 = hit_player(S, B, ev)
            B["log"].append(f"{e['name']}の{mv['n']} → {d1}+{d2}ダメージ")
        elif mv["t"] == "atk_g":
            d = hit_player(S, B, ev + B["G"] // 4)
            B["log"].append(f"{e['name']}の{mv['n']} → {d}ダメージ（ボイル{B['G']}に反応）")
        elif mv["t"] == "blk":
            e["blk"] += base; B["log"].append(f"{e['name']}が{mv['n']}（身構え+{base}）")
        elif mv["t"] == "buff":
            e["st"]["str"] = e["st"].get("str", 0) + base
            B["log"].append(f"{e['name']}の{mv['n']}（魔力+{base}）")
        elif mv["t"] == "debuff":
            B["pst"][mv["st"]] = B["pst"].get(mv["st"], 0) + base
            nm = {"weak": "衰弱", "vuln": "呪縛"}[mv["st"]]
            B["log"].append(f"{e['name']}の{mv['n']} → 自分に{nm}{base}")
        elif mv["t"] == "debuff2":
            for s in mv["st"]:
                B["pst"][s] = B["pst"].get(s, 0) + base
            B["log"].append(f"{e['name']}の{mv['n']} → 衰弱・呪縛")
        elif mv["t"] == "card":
            add_card_to_hand(B, mv["card"], mv["k"])
            B["log"].append(f"{e['name']}の{mv['n']} → 手札に{CARDS[mv['card']]['name']}×{mv['k']}")
        e["mi"] += 1
        for s in ("vuln", "weak"):
            if e["st"].get(s, 0) > 0:
                e["st"][s] -= 1

    if S["hp"] <= 0:
        S["scene"] = "lose"; return

    # 4) 次のターンへ
    for s in ("vuln", "weak"):
        if B["pst"].get(s, 0) > 0:
            B["pst"][s] -= 1
    B["turn"] += 1
    B["block"] = 0
    B["energy"] = B["maxenergy"]
    if B["pow_dense"]:
        gain_g(B, B["pow_dense"])
    if B["pow_str"]:
        B["pst"]["str"] += B["pow_str"]
    if has(S, "r_sala"):
        S["hp"] = min(S["maxhp"], S["hp"] + 2)
    draw_cards(B, 5)
    roll_intents(B)
    B["log"] = B["log"][-14:]


def finish_battle(S, B):
    meta = st.session_state.cd_meta
    kind = B["kind"]
    if kind == "boss":
        meta["wins"] = meta.get("wins", 0) + 1
        meta["asc"] = max(meta.get("asc", 0), S["asc"] + 1)
        meta["cleared_asc"] = max(meta.get("cleared_asc", -1), S["asc"])
        save_meta(meta)
        S["scene"] = "win"
        S["battle"] = None
        return
    gold = random.randint(22, 34) + (26 if kind == "elite" else 0)
    S["gold"] += gold
    n = 4 if has(S, "r_map") else 3
    S["pending"] = dict(kind="reward", cards=roll_reward_cards(meta, n), gold=gold,
                        relic=(pick_relic(S) if kind == "elite" else None))
    S["scene"] = "reward"
    S["battle"] = None


def pick_relic(S):
    avail = [r for r in RELICS if r not in S["relics"]]
    return random.choice(avail) if avail else None


def grant_relic(S, rid):
    if rid and rid not in S["relics"]:
        S["relics"].append(rid)
        if rid == "r_pack":
            S["maxhp"] += 20; S["hp"] = S["maxhp"]


def advance(S):
    S["floor"] += 1
    S["scene"] = "map"
    S["pending"] = None
    if S["floor"] >= MAX_FLOOR:
        S["scene"] = "win"


# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.cd-card{ border-radius:18px; padding:.7rem .8rem; min-height:120px;
  background:var(--m3-surface-c,#1E232B); border-top:5px solid var(--c);
  transition:transform .3s cubic-bezier(.34,1.56,.64,1); }
.cd-card:hover{ transform:translateY(-3px); }
.cd-cname{ font-weight:700; font-size:.88rem; color:var(--m3-on-surface,#EAE6DC); }
.cd-cdesc{ font-size:.72rem; color:var(--m3-on-surface-var,#CFC7BC); margin-top:.25rem; line-height:1.35; }
.cd-cost{ float:right; font-weight:800; color:var(--c); }
.cd-enemy{ background:var(--m3-surface-c,#1E232B); border-radius:20px; padding:.9rem 1.1rem; }
.cd-intent{ display:inline-block; background:var(--m3-tertiary-container,#5C4300);
  color:var(--m3-on-tertiary-container,#FFDFA6); border-radius:999px;
  padding:2px 12px; font-size:.78rem; font-weight:700; }
.cd-hpbar{ height:12px; border-radius:999px; background:#282D35; overflow:hidden; margin-top:.35rem; }
.cd-hpbar i{ display:block; height:100%; border-radius:999px;
  background:linear-gradient(90deg,#FFB4AB,#FFB59B); }
.cd-chip{ display:inline-block; border-radius:999px; padding:2px 11px; font-size:.74rem;
  font-weight:700; margin-right:.35rem; }
.cd-log{ font-size:.76rem; color:var(--m3-on-surface-var,#CFC7BC); line-height:1.7;
  background:var(--m3-surface-lowest,#0D1117); border-radius:12px; padding:.6rem .8rem;
  max-height:230px; overflow-y:auto; }
</style>
""", unsafe_allow_html=True)

if "cd_meta" not in st.session_state:
    st.session_state.cd_meta = load_meta()
meta = st.session_state.cd_meta

st.title("魔女の大釜")
st.caption("薬草を刻んでボイル値を上げ、溢れる前に敵へぶちまける。デッキ構築ローグライク。")

if "cd" not in st.session_state:
    st.session_state.cd = None
S = st.session_state.cd


def chip(label, color, bg, tip=""):
    """ステータスの丸バッジ。tip をカーソルを合わせたときの説明として出す"""
    tt = f' title="{tip}"' if tip else ""
    return (f'<span class="cd-chip" style="color:{color};background:{bg};"{tt}>'
            f'{label}</span>')


# ステータスの説明（自分用／敵用で主語が変わるので2種類持つ）
TIP_SELF = {
    "str":  "魔力：あなたの与えるダメージが数値ぶん増える。戦闘が終わるまで続く。",
    "weak": "衰弱：あなたの与えるダメージが25%減る。ターン終了ごとに残り1減る。",
    "vuln": "呪縛：あなたの受けるダメージが50%増える。ターン終了ごとに残り1減る。",
}
TIP_ENEMY = {
    "str":  "魔力：この敵の与えるダメージが数値ぶん増える。戦闘が終わるまで続く。",
    "weak": "衰弱：この敵の与えるダメージが25%減る。ターン終了ごとに残り1減る。",
    "vuln": "呪縛：この敵の受けるダメージが50%増える。ターン終了ごとに残り1減る。",
    "blk":  "身構え：この敵が受けるダメージを数値ぶん肩代わりする。敵の行動時に0に戻る。",
    "thorn":"棘：この敵を攻撃するたび、あなたが数値ぶんダメージを受ける（防御では防げない）。",
}


def card_label(cid):
    """選択肢に出す1行表記。名前だけだと中身が分からないので効果も添える"""
    c = CARDS[cid]
    kind = {"attack": "攻撃", "skill": "技", "power": "常時", "status": "妨害"}[c["type"]]
    return f"{c['name']}（{c['cost']}マナ・{kind}）— {c['desc']}"


def card_html(cid, playable=True):
    c = CARDS[cid]
    col = TYPE_COLOR[c["type"]]
    op = "1" if playable else ".45"
    return (f'<div class="cd-card" style="--c:{col};opacity:{op};">'
            f'<span class="cd-cost">{c["cost"]}</span>'
            f'<div class="cd-cname">{c["name"]}</div>'
            f'<div class="cd-cdesc">{c["desc"]}</div></div>')


# ── タイトル ──────────────────────────────────────────────────────────────────
if S is None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("挑戦回数", meta.get("runs", 0))
    c2.metric("クリア", meta.get("wins", 0))
    c3.metric("最高到達", f"{meta.get('best_floor',0)} 層")
    c4.metric("魔女ランク", meta.get("asc", 0))

    st.markdown("### 遊び方")
    st.markdown("""
- 手札のカードを **マナ** を払って使い、敵を倒しながら森を12層進む。
- **ボイル値** を上げるほど `煮汁を注ぐ` `ぶちまける` の火力が伸びる。
- ただしターン終了時に **ボイル値が大釜の容量を超えていると、超過分だけ吹きこぼれ**て
  防御を無視したダメージを受ける。容量を広げる／溜めた瞬間に吐き出す／灰汁取りで抑える、
  の3方向がある。
- 敵 **火の玉** はボイル値が高いほど強く殴ってくるので、盛るだけでは勝てない。
- 敵の行動は**次の1手が見えている**ので、防御と攻撃の配分は読んで決められる。
- 12層目の **大魔女ヴェルナ** を倒せばクリア。クリアするたび魔女ランクが上がり、敵が強くなる。
""")
    if meta.get("wins", 0) < 1:
        st.info("初回クリアでレアカードが2種解禁される。")
    if st.button("森へ入る", type="primary", use_container_width=True):
        meta["runs"] = meta.get("runs", 0) + 1
        save_meta(meta)
        st.session_state.cd = new_run(meta)
        st.rerun()
    st.stop()

# ── 共通ヘッダー ──────────────────────────────────────────────────────────────
h1, h2, h3, h4, h5 = st.columns([2, 1, 1, 1, 2])
h1.progress(max(0.0, min(1.0, S["hp"] / S["maxhp"])), text=f"体力 {S['hp']} / {S['maxhp']}")
h2.metric("ゴールド", S["gold"], help="行商人でカード・レリックを買ったり、カードを手放すのに使う。")
h3.metric("階層", f"{S['floor']+1} / {MAX_FLOOR}", help="12層目が大魔女ヴェルナ。")
h4.metric("ランク", S["asc"], help="魔女ランク。クリアするたび上がり、敵の体力と攻撃力が増す。")
with h5:
    if S["relics"]:
        st.markdown(" ".join(
            f'<span title="{RELICS[r]["name"]}：{RELICS[r]["desc"]}" style="font-size:1.5rem">'
            f'{RELICS[r]["emoji"]}</span>' for r in S["relics"]), unsafe_allow_html=True)
    else:
        st.caption("レリックなし")

meta["best_floor"] = max(meta.get("best_floor", 0), S["floor"] + 1)

# ── マップ ────────────────────────────────────────────────────────────────────
if S["scene"] == "map":
    st.markdown(f"### 第 {S['floor']+1} 層")
    if S.get("msg"):
        st.caption(S["msg"])
    NODE = {"battle": ("魔物", "⚔️", "敵1〜2体"), "elite": ("強敵", "💀", "レリック確定・危険"),
            "shop": ("行商人", "🛒", "カード・レリックを買う"),
            "rest": ("焚き火", "🔥", "回復かデッキ整理"),
            "event": ("なにか", "❓", "行ってみないと分からない"),
            "boss": ("大魔女ヴェルナ", "🧙‍♀️", "最終戦")}
    opts = S["map"][S["floor"]]
    cols = st.columns(len(opts))
    for i, (col, nd) in enumerate(zip(cols, opts)):
        label, emo, sub = NODE[nd]
        with col:
            st.markdown(f"<div style='text-align:center;font-size:2.6rem'>{emo}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"**{label}**")
            st.caption(sub)
            if st.button("進む", key=f"go_{S['floor']}_{i}", use_container_width=True,
                         type="primary" if nd == "boss" else "secondary"):
                S["msg"] = ""
                if nd in ("battle", "elite", "boss"):
                    start_battle(S, nd)
                elif nd == "shop":
                    S["pending"] = dict(kind="shop", cards=roll_reward_cards(meta, 3),
                                        relic=pick_relic(S), bought=[])
                    S["scene"] = "shop"
                elif nd == "rest":
                    S["scene"] = "rest"
                else:
                    S["pending"] = dict(kind="event", ev=random.choice(EVENTS))
                    S["scene"] = "event"
                st.rerun()

    with st.expander(f"手持ちのデッキ（{len(S['deck'])}枚）"):
        cnt = {}
        for cid in S["deck"]:
            cnt[cid] = cnt.get(cid, 0) + 1
        cs = st.columns(4)
        for i, (cid, k) in enumerate(sorted(cnt.items(), key=lambda x: -x[1])):
            with cs[i % 4]:
                st.markdown(card_html(cid), unsafe_allow_html=True)
                st.caption(f"×{k}")

# ── 戦闘 ──────────────────────────────────────────────────────────────────────
elif S["scene"] == "battle":
    B = S["battle"]
    st.markdown(f"### ターン {B['turn']}")

    living = alive(B)
    ecols = st.columns(len(living) if living else 1)
    for i, (col, e) in enumerate(zip(ecols, living)):
        with col:
            sts = ""
            if e["st"].get("vuln", 0):
                sts += chip(f"呪縛{e['st']['vuln']}", "#FFDAD6", "#93000A", TIP_ENEMY["vuln"])
            if e["st"].get("weak", 0):
                sts += chip(f"衰弱{e['st']['weak']}", "#CDE5FF", "#2E4A63", TIP_ENEMY["weak"])
            if e["st"].get("str", 0):
                sts += chip(f"魔力{e['st']['str']}", "#FFDFA6", "#5C4300", TIP_ENEMY["str"])
            if e.get("blk", 0):
                sts += chip(f"🛡{e['blk']}", "#B8F2CE", "#22503A", TIP_ENEMY["blk"])
            if e.get("thorn", 0):
                sts += chip(f"棘{e['thorn']}", "#E9DDFF", "#3F3153", TIP_ENEMY["thorn"])
            pct = max(0, e["hp"]) / e["maxhp"] * 100
            st.markdown(
                f'<div class="cd-enemy"><div style="font-size:2.2rem">{e["emoji"]}</div>'
                f'<div style="font-weight:700">{e["name"]}</div>'
                f'<div class="cd-hpbar"><i style="width:{pct:.0f}%"></i></div>'
                f'<div style="font-size:.78rem;margin-top:.2rem">{max(0,e["hp"])} / {e["maxhp"]}</div>'
                f'<div style="margin-top:.5rem">{sts}</div>'
                f'<div style="margin-top:.5rem"><span class="cd-intent" title="{intent_tip(e)}">'
                f'{intent_text(B,e)}</span></div>'
                f'</div>', unsafe_allow_html=True)

    if len(living) > 1:
        names = [f"{e['emoji']} {e['name']}" for e in living]
        B["target"] = st.radio("狙う相手", range(len(living)),
                               format_func=lambda i: names[i], horizontal=True,
                               index=min(B["target"], len(living) - 1))

    st.divider()

    s1, s2, s3, s4 = st.columns([1, 1, 1, 2])
    s1.metric("マナ", f"{B['energy']} / {B['maxenergy']}",
              help="カードを使うために払う。毎ターン最大値まで回復する。")
    s2.metric("防御", B["block"], help="受けるダメージを肩代わりする。あなたのターン開始時に0に戻る。")
    over = " 🔥" if B["G"] > B["vram"] else ""
    s3.metric("ボイル値", f"{B['G']} / {B['vram']}{over}",
              help="大釜の煮え具合。高いほど「煮汁を注ぐ」「ぶちまける」の威力が上がる。"
                   "ターン終了時に容量（右の数字）を超えていると、超過分が防御を無視した"
                   "ダメージになる（吹きこぼれ）。")
    with s4:
        ps = ""
        if B["pst"].get("str", 0):
            ps += chip(f"魔力{B['pst']['str']}", "#FFDFA6", "#5C4300", TIP_SELF["str"])
        if B["pst"].get("weak", 0):
            ps += chip(f"衰弱{B['pst']['weak']}", "#CDE5FF", "#2E4A63", TIP_SELF["weak"])
        if B["pst"].get("vuln", 0):
            ps += chip(f"呪縛{B['pst']['vuln']}", "#FFDAD6", "#93000A", TIP_SELF["vuln"])
        for pid in B["powers"]:
            ps += chip(CARDS[pid]["name"], "#E9DDFF", "#3F3153",
                       f"常時効果：{CARDS[pid]['desc']}（戦闘が終わるまで続く）")
        st.markdown(ps or "<span style='opacity:.5'>状態なし</span>", unsafe_allow_html=True)
        if B["G"] > B["vram"]:
            st.markdown(f"<span style='color:#FFB4AB;font-weight:700'>"
                        f"このままだとターン終了時に {B['G']-B['vram']} の吹きこぼれ</span>",
                        unsafe_allow_html=True)

    st.markdown("#### 手札")
    if B["hand"]:
        ncol = min(len(B["hand"]), 6)
        hcols = st.columns(ncol)
        for i, cid in enumerate(B["hand"]):
            c = CARDS[cid]
            ok = c["type"] != "status" and B["energy"] >= c["cost"]
            with hcols[i % ncol]:
                st.markdown(card_html(cid, ok), unsafe_allow_html=True)
                if st.button("使う", key=f"pl_{B['turn']}_{i}_{cid}",
                             disabled=not ok, use_container_width=True):
                    play_card(S, B, i)
                    st.rerun()
    else:
        st.caption("手札なし")

    b1, b2 = st.columns([1, 3])
    if b1.button("ターン終了", type="primary", use_container_width=True):
        end_turn(S, B)
        st.rerun()
    b2.caption(f"山札 {len(B['draw'])} / 捨札 {len(B['disc'])} / 使い捨て {len(B['exh'])}")

    if B["log"]:
        st.markdown('<div class="cd-log">' + "<br>".join(B["log"][-14:]) + "</div>",
                    unsafe_allow_html=True)

    with st.expander("用語の意味"):
        st.markdown("""
| 表示 | 意味 |
|---|---|
| **魔力** | 与えるダメージが数値ぶん増える。戦闘が終わるまで続く |
| **衰弱** | 与えるダメージが25%減る。ターン終了ごとに残り1減る |
| **呪縛** | 受けるダメージが50%増える。ターン終了ごとに残り1減る |
| **防御** | 受けるダメージを肩代わりする。自分のターン開始時に0に戻る |
| **身構え（🛡）** | 敵版の防御。その敵が行動すると0に戻る |
| **棘** | その敵を攻撃するたび、こちらが数値ぶんダメージを受ける（防御では防げない） |
| **ボイル値** | 高いほど「煮汁を注ぐ」「ぶちまける」が強い。ターン終了時に容量を超えた分は吹きこぼれて防御を無視したダメージになる |
| **マナ** | カードを使うために払う。毎ターン最大値まで回復する |

バッジや数値にカーソルを合わせても同じ説明が出る。
""")

# ── 報酬 ──────────────────────────────────────────────────────────────────────
elif S["scene"] == "reward":
    p = S["pending"]
    st.markdown("### 勝利")
    st.success(f"ゴールド +{p['gold']}")
    if p.get("relic"):
        r = RELICS[p["relic"]]
        st.markdown(f"**レリック獲得：{r['emoji']} {r['name']}** — {r['desc']}")
        if not p.get("relic_taken"):
            grant_relic(S, p["relic"]); p["relic_taken"] = True
    st.markdown("#### カードを1枚選ぶ")
    cols = st.columns(len(p["cards"]))
    for i, (col, cid) in enumerate(zip(cols, p["cards"])):
        with col:
            st.markdown(card_html(cid), unsafe_allow_html=True)
            st.caption(RARITY_JA[CARDS[cid]["rarity"]])
            if st.button("取る", key=f"rw_{i}", use_container_width=True, type="primary"):
                S["deck"].append(cid); advance(S); st.rerun()
    if st.button("受け取らずに進む", use_container_width=True):
        advance(S); st.rerun()

# ── 行商人 ────────────────────────────────────────────────────────────────────
elif S["scene"] == "shop":
    p = S["pending"]
    st.markdown("### 行商人")
    cols = st.columns(len(p["cards"]))
    for i, (col, cid) in enumerate(zip(cols, p["cards"])):
        price = {"common": 45, "uncommon": 75, "rare": 120}[CARDS[cid]["rarity"]]
        with col:
            st.markdown(card_html(cid), unsafe_allow_html=True)
            bought = f"card{i}" in p["bought"]
            if st.button(f"{price}G で買う" if not bought else "購入済み",
                         key=f"sh_{i}", use_container_width=True,
                         disabled=bought or S["gold"] < price):
                S["gold"] -= price; S["deck"].append(cid); p["bought"].append(f"card{i}")
                st.rerun()
    if p.get("relic"):
        r = RELICS[p["relic"]]
        st.markdown(f"**{r['emoji']} {r['name']}** — {r['desc']}")
        bought = "relic" in p["bought"]
        if st.button("150G で買う" if not bought else "購入済み", key="shrel",
                     disabled=bought or S["gold"] < 150):
            S["gold"] -= 150; grant_relic(S, p["relic"]); p["bought"].append("relic")
            st.rerun()
    st.divider()
    st.markdown("#### カードを1枚手放す（80G）")
    order = sorted(range(len(S["deck"])), key=lambda i: CARDS[S["deck"][i]]["name"])
    rm = st.selectbox("手放すカード", order,
                      format_func=lambda i: card_label(S["deck"][i]), key="shoprm")
    rc1, rc2 = st.columns([1, 3])
    with rc1:
        st.markdown(card_html(S["deck"][rm]), unsafe_allow_html=True)
    with rc2:
        st.caption("デッキが薄いほど強いカードを引きやすくなる。基本カードから抜くのが定石。")
        if st.button("このカードを手放す", key="shoprmbtn",
                     disabled=S["gold"] < 80 or "remove" in p["bought"] or len(S["deck"]) <= 4):
            S["gold"] -= 80; S["deck"].pop(rm); p["bought"].append("remove"); st.rerun()
    if st.button("先へ進む", type="primary", use_container_width=True):
        advance(S); st.rerun()

# ── 焚き火 ────────────────────────────────────────────────────────────────────
elif S["scene"] == "rest":
    st.markdown("### 焚き火")
    st.caption("休むか、荷物を軽くするか。どちらか一方だけ。")
    c1, c2 = st.columns(2)
    with c1:
        heal = int(S["maxhp"] * 0.35)
        st.markdown(f"**🔥 眠る** — 体力 +{heal}")
        if st.button("眠る", use_container_width=True, type="primary"):
            S["hp"] = min(S["maxhp"], S["hp"] + heal); advance(S); st.rerun()
    with c2:
        st.markdown("**🗑️ デッキを整理する** — カード1枚を手放す")
        order = sorted(range(len(S["deck"])), key=lambda i: CARDS[S["deck"][i]]["name"])
        rm = st.selectbox("手放すカード", order,
                          format_func=lambda i: card_label(S["deck"][i]), key="restrm")
        st.markdown(card_html(S["deck"][rm]), unsafe_allow_html=True)
        if st.button("手放して進む", use_container_width=True, disabled=len(S["deck"]) <= 4):
            S["deck"].pop(rm); advance(S); st.rerun()

# ── イベント ──────────────────────────────────────────────────────────────────
elif S["scene"] == "event":
    ev = S["pending"]["ev"]
    st.markdown(f"### {ev['title']}")
    st.write(ev["text"])
    for i, (label, eff) in enumerate(ev["opts"]):
        if st.button(label, key=f"ev_{i}", use_container_width=True):
            if "hp" in eff:
                S["hp"] = max(1, min(S["maxhp"], S["hp"] + eff["hp"]))
            if "gold" in eff:
                S["gold"] += eff["gold"]
            if "relic" in eff:
                grant_relic(S, pick_relic(S))
            if "card" in eff:
                S["deck"].append(random.choice(roll_reward_cards(meta, 1)))
            if "remove" in eff and len(S["deck"]) > 4:
                S["deck"].pop(random.randrange(len(S["deck"])))
            advance(S); st.rerun()

# ── 決着 ──────────────────────────────────────────────────────────────────────
elif S["scene"] == "win":
    st.balloons()
    st.markdown("## 大魔女を倒した")
    st.success(f"クリア！ 次から魔女ランク {meta.get('asc',0)} に挑戦できる（敵の体力と攻撃力が上がる）。")
    if meta.get("wins", 0) == 1:
        st.info("レアカード「大釜をひっくり返す」「火薬を混ぜる」が解禁された。")
    if st.button("タイトルへ", type="primary", use_container_width=True):
        st.session_state.cd = None; st.rerun()

elif S["scene"] == "lose":
    meta["best_floor"] = max(meta.get("best_floor", 0), S["floor"] + 1)
    save_meta(meta)
    st.markdown("## 力尽きた")
    st.error(f"第 {S['floor']+1} 層で倒れた。")
    with st.expander("最後のデッキ"):
        cnt = {}
        for cid in S["deck"]:
            cnt[cid] = cnt.get(cid, 0) + 1
        st.write("　".join(f"{CARDS[c]['name']}×{k}" for c, k in cnt.items()))
    if st.button("もう一度", type="primary", use_container_width=True):
        st.session_state.cd = None; st.rerun()

save_meta(meta)
