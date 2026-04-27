# たまごっち風育成シミュレーション v2
# 隠しパラメータ・冒険・バフデバフ・飽きシステム・14種進化分岐対応

import random
import time
from datetime import datetime

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  定数
# ─────────────────────────────────────────────────────────────────────────────
PHASE_ORDER    = ["egg", "infant", "child", "teen", "adult"]
PHASE_DURATION = {"egg": 30, "infant": 60, "child": 120, "teen": 180}

PHASE_LABEL = {
    "egg":    "🥚 たまご",
    "infant": "🐣 幼少期",
    "child":  "🐥 子ども期",
    "teen":   "🐤 思春期",
    "adult":  "成体",
}

CHAR_EMOJI = {"egg": "🥚", "infant": "🐣", "child": "🐥", "teen": "🐤"}

EVO_DEFS = {
    "魔物":        {"emoji": "👹", "label": "👹 魔物系",       "desc": "野生が高く、ケアミスだらけの荒ぶる存在"},
    "筋肉バカ":    {"emoji": "💪", "label": "💪 筋肉バカ系",   "desc": "筋肉だけが取り柄の脳筋"},
    "野生の王":    {"emoji": "🦁", "label": "🦁 野生の王系",   "desc": "野生と筋力を兼ね備えた獣の王"},
    "格闘家":      {"emoji": "🥊", "label": "🥊 格闘家系",     "desc": "鍛え抜かれた肉体の持ち主"},
    "アスリート":  {"emoji": "🏃", "label": "🏃 アスリート系", "desc": "野心と筋肉で頂点を目指す"},
    "賢者":        {"emoji": "🧙", "label": "🧙 賢者系",       "desc": "知性とストレスフリーな環境で開花"},
    "エリート学者":{"emoji": "🦉", "label": "🦉 エリート学者系","desc": "高い知性と野心、ケアミスなしで到達"},
    "サイボーグ":  {"emoji": "🤖", "label": "🤖 サイボーグ系", "desc": "野生が低く知性が高い人工知能系"},
    "アイドル":    {"emoji": "⭐", "label": "⭐ アイドル系",    "desc": "友好度と野心で輝くスター"},
    "幸運児":      {"emoji": "🍀", "label": "🍀 幸運児系",     "desc": "運だけで生き抜いてきた強運の持ち主"},
    "探検家":      {"emoji": "🗺️","label": "🗺️ 探検家系",    "desc": "運と野生で世界を渡り歩く"},
    "やんちゃ":    {"emoji": "😈", "label": "😈 やんちゃ系",   "desc": "ケアミスと野心が生んだ問題児"},
    "ニート":      {"emoji": "🛋️","label": "🛋️ ニート系",    "desc": "野心ゼロ、ケアミスだらけの引きこもり"},
    "ふつう":      {"emoji": "🐔", "label": "🐔 ふつう系",     "desc": "特に特徴のない平凡な成体"},
}

EVO_CONDITIONS = {
    "魔物":        "野生 > 70 かつ ケアミス > 5",
    "筋肉バカ":    "筋肉量 > 80 かつ かしこさ < 30",
    "野生の王":    "野生 > 70 かつ 筋肉量 > 50",
    "格闘家":      "筋肉量 > 70 かつ 野生 > 40",
    "アスリート":  "筋肉量 > 60 かつ 野心 > 60",
    "賢者":        "かしこさ > 80 かつ ストレス < 30",
    "エリート学者":"かしこさ > 70 かつ ケアミス = 0 かつ 野心 > 60",
    "サイボーグ":  "野生 < 20 かつ かしこさ > 60 かつ 筋肉量 < 30",
    "アイドル":    "なかよし度 > 70 かつ 野心 > 60 かつ 野生 < 40",
    "幸運児":      "運 > 80",
    "探検家":      "運 > 60 かつ 野生 > 50",
    "やんちゃ":    "ケアミス > 3 かつ 野心 > 50",
    "ニート":      "野心 < 30 かつ ケアミス > 3",
    "ふつう":      "上記のいずれにも当てはまらない場合",
}

DECAY_HUNGER      = 0.12
DECAY_CLEAN       = 0.07
POOP_RANGE        = (25, 50)
CARE_WARN_SECS    = 20
HISTORY_INTERVAL  = 8.0
MAX_HISTORY       = 120
ADVENTURE_DUR     = 10.0
ZONE_DUR          = 60.0
REBOUND_DUR       = 30.0
INJURY_DUR        = 30.0
SICK_DUR          = 45.0


# ─────────────────────────────────────────────────────────────────────────────
#  ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────
def now() -> float:
    return time.time()


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(v)))


def add_log(msg: str, cd_key: str = "", cd_secs: float = 8.0) -> None:
    if cd_key:
        if now() - st.session_state.log_cd.get(cd_key, 0.0) < cd_secs:
            return
        st.session_state.log_cd[cd_key] = now()
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"`{ts}` {msg}")
    st.session_state.logs = st.session_state.logs[:10]


def set_speech(text: str, duration: float = 6.0) -> None:
    st.session_state.speech       = text
    st.session_state.speech_until = now() + duration


def check_boredom(action: str) -> float:
    """飽き判定：連続3回以上同じアクションで効果0.5倍＋ストレス上昇。戻り値は効果倍率。"""
    s = st.session_state
    if s.last_action == action:
        s.action_streak = min(s.action_streak + 1, 5)
    else:
        s.last_action   = action
        s.action_streak = 1
    if s.action_streak >= 3:
        s.stress = clamp(s.stress + 5)
        add_log(f"😑 また{action}？飽きてきた…", f"bored_{action}", 10)
        return 0.5
    return 1.0


def save_history() -> None:
    s = st.session_state
    n = now()
    if n - s.last_history_save < HISTORY_INTERVAL:
        return
    s.last_history_save = n
    s.history.append({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "おなか":     round(s.hunger,      1),
        "かしこさ":   round(s.intellect,   1),
        "きれいさ":   round(s.cleanliness, 1),
        "なかよし度": round(s.friendship,  1),
        "ストレス":   round(s.stress,      1),
        "野心":       round(s.ambition,    1),
        "野生":       round(s.wildness,    1),
        "筋肉量":     round(s.muscle,      1),
        "運":         round(s.luck,        1),
    })
    if len(s.history) > MAX_HISTORY:
        s.history = s.history[-MAX_HISTORY:]


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
        "ambition":      50.0,
        "wildness":      50.0,
        "muscle":        20.0,
        "luck":          50.0,
        "weight":        50.0,
        # 欲求（セリフとボーナスに使用）
        "pet_desire":    None,
        # うんち管理
        "poop":      False,
        "next_poop": n + random.uniform(*POOP_RANGE),
        # 警告タイマー
        "hunger_low_since": None,
        "clean_low_since":  None,
        # 飽きシステム
        "last_action":   None,
        "action_streak": 0,
        # バフ・デバフ
        "zoned_in":        False,
        "zone_study_count": 0,
        "zone_until":      0.0,
        "rebound":         False,
        "rebound_until":   0.0,
        "snack_times":     [],
        "injured":         False,
        "injured_until":   0.0,
        "sick":            False,
        "sick_until":      0.0,
        # 冒険
        "adventuring":   False,
        "adventure_end": 0.0,
        # セリフ
        "speech":       "🥚 大切にそだててね！",
        "speech_until": n + 10.0,
        # 履歴
        "history":            [],
        "last_history_save":  0.0,
        # ログ
        "logs":   ["🥚 たまごが産まれました！大切に育ててね！"],
        "log_cd": {},
    })


# ─────────────────────────────────────────────────────────────────────────────
#  進化ロジック
# ─────────────────────────────────────────────────────────────────────────────
def determine_evolution() -> str:
    s = st.session_state
    cm    = s.care_mistakes
    intel = s.intellect
    frnd  = s.friendship
    ms    = s.muscle
    wld   = s.wildness
    amb   = s.ambition
    lk    = s.luck
    st_v  = s.stress

    if wld > 70 and cm > 5:               return "魔物"
    if ms > 80 and intel < 30:            return "筋肉バカ"
    if wld > 70 and ms > 50:              return "野生の王"
    if ms > 70 and wld > 40:              return "格闘家"
    if ms > 60 and amb > 60:              return "アスリート"
    if intel > 80 and st_v < 30:          return "賢者"
    if intel > 70 and cm == 0 and amb > 60: return "エリート学者"
    if wld < 20 and intel > 60 and ms < 30: return "サイボーグ"
    if frnd > 70 and amb > 60 and wld < 40: return "アイドル"
    if lk > 80:                           return "幸運児"
    if lk > 60 and wld > 50:              return "探検家"
    if cm > 3 and amb > 50:               return "やんちゃ"
    if amb < 30 and cm > 3:               return "ニート"
    return "ふつう"


def get_emoji() -> str:
    s = st.session_state
    if s.phase == "adult":
        evo = s.evolution_type or "ふつう"
        return EVO_DEFS.get(evo, {}).get("emoji", "🐔")
    return CHAR_EMOJI.get(s.phase, "❓")


def get_label() -> str:
    s = st.session_state
    if s.phase == "adult":
        evo = s.evolution_type or "ふつう"
        return EVO_DEFS.get(evo, {}).get("label", "成体")
    return PHASE_LABEL.get(s.phase, s.phase)


def advance_phase() -> None:
    s   = st.session_state
    idx = PHASE_ORDER.index(s.phase)
    if idx + 1 >= len(PHASE_ORDER):
        return
    next_p = PHASE_ORDER[idx + 1]
    s.phase       = next_p
    s.phase_start = now()
    if next_p == "adult":
        evo   = determine_evolution()
        s.evolution_type = evo
        label = EVO_DEFS.get(evo, {}).get("label", evo)
        add_log(f"🎉 ついに成体になった！【{label}】だよ！")
        set_speech(f"🎊 {label}に進化したよ！")
    else:
        add_log(f"✨ {PHASE_LABEL[next_p]} に成長した！")
        set_speech("✨ 成長したよ！")


# ─────────────────────────────────────────────────────────────────────────────
#  冒険完了処理
# ─────────────────────────────────────────────────────────────────────────────
def resolve_adventure() -> None:
    s  = st.session_state
    lk = s.luck
    injury_chance = max(0.05, 0.35 - lk * 0.003)

    if random.random() < injury_chance:
        s.injured       = True
        s.injured_until = now() + INJURY_DUR
        s.hunger        = clamp(s.hunger - 25)
        add_log("💥 冒険から帰還…でも怪我をした！")
        set_speech("いたたた…怪我しちゃった…")
    else:
        lk_gain  = random.uniform(2, 8) * (1 + lk / 200)
        amb_gain = random.uniform(1, 5)
        s.hunger   = clamp(s.hunger - 15)
        s.luck     = clamp(s.luck     + lk_gain)
        s.ambition = clamp(s.ambition + amb_gain)
        s.wildness = clamp(s.wildness + random.uniform(1, 4))

        rare_chance = 0.1 + lk * 0.003
        if random.random() < rare_chance and s.phase in PHASE_DURATION:
            s.phase_start -= 10
            add_log(f"⚡【レア】秘薬発見！進化が10秒早まった！運+{lk_gain:.1f}")
            set_speech("わあ！すごいもの見つけた！！")
        else:
            add_log(f"🗺️ 無事帰還！運+{lk_gain:.1f}、野心+{amb_gain:.1f}")
            set_speech("ただいまー！楽しかったよ！")

    s.adventuring = False


# ─────────────────────────────────────────────────────────────────────────────
#  欲求システム
# ─────────────────────────────────────────────────────────────────────────────
DESIRE_INFO = {
    "food":    ("🍙", "ごはんが食べたそう"),
    "clean":   ("🛁", "きれいにしてほしそう"),
    "play":    ("🎮", "遊びたそう"),
    "workout": ("🏋️", "体を動かしたそう"),
    "study":   ("📚", "勉強したそう"),
    "snack":   ("🍬", "おやつが食べたそう"),
}


def _update_desire() -> None:
    """現在のパラメータからペットの欲求を更新する（優先度順）"""
    s = st.session_state
    if s.hunger < 20:
        s.pet_desire = "food"
    elif s.cleanliness < 30 or s.poop:
        s.pet_desire = "clean"
    elif s.stress > 55:
        s.pet_desire = "play"
    elif s.friendship < 30:
        s.pet_desire = "play"
    elif s.hunger < 40:
        s.pet_desire = "snack"
    elif s.muscle < 25 and not s.injured:
        s.pet_desire = "workout"
    elif s.intellect < 35 and s.stress < 45:
        s.pet_desire = "study"
    else:
        s.pet_desire = None


def get_desire_bonus(action: str) -> float:
    """欲求と一致するアクションに1.5倍ボーナスを返す"""
    s = st.session_state
    mapping = {
        "food":    {"ごはん"},
        "snack":   {"おやつ"},
        "clean":   {"お掃除"},
        "play":    {"遊ぶ"},
        "workout": {"筋トレ"},
        "study":   {"お勉強"},
    }
    desire = s.get("pet_desire", None)
    for d_key, actions in mapping.items():
        if desire == d_key and action in actions:
            return 1.5
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  ランダムセリフ
# ─────────────────────────────────────────────────────────────────────────────
def _random_speech() -> None:
    s = st.session_state
    _update_desire()

    # 特殊状態が最優先
    if s.zoned_in:
        set_speech(random.choice(["今集中してる！邪魔しないで！", "ゾーンに入った！！"]), 8.0)
        return
    if s.adventuring:
        set_speech(random.choice(["冒険中…どこまで行けるかな", "たのしー！！"]), 8.0)
        return
    if s.injured:
        set_speech(random.choice(["痛い…動けない…", "怪我がいたい…"]), 8.0)
        return
    if s.sick:
        set_speech(random.choice(["気分が悪い…", "病気でつらい…"]), 8.0)
        return

    # 欲求に対応したセリフ
    desire_speech = {
        "food":    ["おなかすいた～…", "ごはん…食べたい…", "なにかたべたい…"],
        "snack":   ["おやつ食べたいな～", "あまいものがほしい！"],
        "clean":   ["くさい…おふろ入りたい…", "きれいにして…", "うんちが…はずかしい…"],
        "play":    ["いっしょにあそぼ！", "たいくつだよ～…", "あそびたい！！"],
        "workout": ["からだ動かしたいな！", "きんにく、きたえてみたい！"],
        "study":   ["べんきょうしたいな～", "かしこくなりたい！"],
    }
    d = s.get("pet_desire", None)
    if d and d in desire_speech:
        set_speech(random.choice(desire_speech[d]), 8.0)
        return

    set_speech(random.choice([
        "きょうもげんき！", "なにかしたいな～", "ねむい…",
        "おそとにいきたい！", "ひまだな～", "いっしょにいてね！",
    ]), 8.0)


# ─────────────────────────────────────────────────────────────────────────────
#  時間経過更新
# ─────────────────────────────────────────────────────────────────────────────
def update() -> None:
    s    = st.session_state
    n    = now()
    dt   = min(n - s.last_update, 60.0)
    if dt <= 0:
        return
    s.last_update = n
    p = s.phase

    if p == "egg":
        if n - s.phase_start >= PHASE_DURATION["egg"]:
            advance_phase()
        save_history()
        return

    # ── バフ/デバフ期限チェック ─────────────────────────────────────────────
    if s.zoned_in  and n >= s.zone_until:
        s.zoned_in = False
        add_log("💫 ゾーンが終わった…", "zone_end", 5)
    if s.rebound   and n >= s.rebound_until:
        s.rebound  = False
        add_log("✅ リバウンドが解消された", "rebound_end", 5)
    if s.injured   and n >= s.injured_until:
        s.injured  = False
        add_log("💪 怪我が治った！", "injury_end", 5)
    if s.sick      and n >= s.sick_until:
        s.sick     = False
        add_log("😊 病気が治った！", "sick_end", 5)

    # ── 冒険完了チェック ────────────────────────────────────────────────────
    if s.adventuring and n >= s.adventure_end:
        resolve_adventure()

    # ── パラメータ自然減少 ──────────────────────────────────────────────────
    decay_mult = 1.5 if s.sick else 1.0
    s.hunger      = clamp(s.hunger      - DECAY_HUNGER * dt * decay_mult)
    s.cleanliness = clamp(s.cleanliness - DECAY_CLEAN  * dt)
    if s.injured:
        s.muscle  = clamp(s.muscle - 0.02 * dt)

    # 環境が汚いとストレス上昇、自然状態では少しずつ低下
    if s.cleanliness < 30:
        s.stress  = clamp(s.stress + 0.05 * dt)
    else:
        s.stress  = clamp(s.stress - 0.04 * dt)

    # ── うんち発生 ──────────────────────────────────────────────────────────
    if not s.poop and n >= s.next_poop:
        s.poop        = True
        s.cleanliness = clamp(s.cleanliness - 20)
        add_log("💩 うんちが出た！お掃除してあげて！", "poop", 30)
        s.next_poop   = n + random.uniform(*POOP_RANGE)

    # ── お腹警告 & ケアミス ─────────────────────────────────────────────────
    if s.hunger < 20:
        add_log("🍽️ お腹ペコペコ！ごはんをあげて！", "hunger_warn")
        if s.hunger_low_since is None:
            s.hunger_low_since = n
        elif n - s.hunger_low_since >= CARE_WARN_SECS:
            s.care_mistakes   += 1
            s.hunger_low_since = n
            add_log(f"⚠️ ケアミス発生！（計 {s.care_mistakes} 回）")
    else:
        s.hunger_low_since = None

    # ── きれいさ警告 & ケアミス ─────────────────────────────────────────────
    if s.cleanliness < 15:
        add_log("🚿 とても汚れてる！お掃除して！", "clean_warn")
        if s.clean_low_since is None:
            s.clean_low_since = n
        elif n - s.clean_low_since >= CARE_WARN_SECS:
            s.care_mistakes  += 1
            s.clean_low_since = n
            add_log(f"⚠️ ケアミス発生！（計 {s.care_mistakes} 回）")
    else:
        s.clean_low_since = None

    # ── ランダムセリフ ───────────────────────────────────────────────────────
    if n > s.speech_until:
        _random_speech()

    # ── フェーズ進行 ────────────────────────────────────────────────────────
    dur = PHASE_DURATION.get(p)
    if dur and n - s.phase_start >= dur:
        advance_phase()

    save_history()


# ─────────────────────────────────────────────────────────────────────────────
#  アクション
# ─────────────────────────────────────────────────────────────────────────────
def do_feed():
    s    = st.session_state
    mult = check_boredom("ごはん") * get_desire_bonus("ごはん")
    s.hunger = clamp(s.hunger + 30 * mult)
    s.stress = clamp(s.stress - 5)
    set_speech("もぐもぐ…おいしい！ありがとう！" if s.pet_desire == "food" else "もぐもぐ…おいしい！")
    add_log(f"🍙 ごはんを食べた（効果 x{mult:.1f}）")


def do_snack():
    s    = st.session_state
    boredom_m = check_boredom("おやつ")
    desire_m  = get_desire_bonus("おやつ")
    mult      = boredom_m * desire_m
    n    = now()
    s.snack_times = [t for t in s.snack_times if n - t < 30]
    s.snack_times.append(n)

    weight_gain = 8.0 * boredom_m  # 体重増加は欲求ボーナス対象外
    if s.rebound:
        weight_gain *= 2
        add_log("😱 リバウンド中！体重が2倍増！", "rebound_warn", 5)

    if len(s.snack_times) >= 3 and not s.rebound:
        s.rebound       = True
        s.rebound_until = n + REBOUND_DUR
        add_log("🔁 リバウンド発生！しばらく体重が増えやすい…")
        set_speech("食べ過ぎた…リバウンドしそう…")

    s.hunger = clamp(s.hunger + 20 * mult)
    s.weight = clamp(s.weight + weight_gain)
    s.muscle = clamp(s.muscle - 2)
    if not s.rebound:
        set_speech("やった！おやつだ！ありがとう！" if s.pet_desire == "snack" else "やった！おやつだ！")
    add_log(f"🍬 おやつを食べた（効果 x{mult:.1f}、体重+{weight_gain:.0f}）")


def do_clean():
    s = st.session_state
    had_poop = s.poop
    # うんちは cleanliness に関わらず常に片付ける
    if had_poop:
        s.poop      = False
        s.next_poop = now() + random.uniform(*POOP_RANGE)

    # うんちがなく既にきれいな状態での掃除は過干渉
    if s.cleanliness >= 90 and not had_poop:
        s.friendship = clamp(s.friendship - 5)
        s.stress     = clamp(s.stress + 10)
        add_log("🚫 もうきれいなのに！過干渉（なかよし度↓）", "overclean", 15)
        set_speech("もうきれいだよ！やめて！")
        return
    mult = check_boredom("お掃除") * get_desire_bonus("お掃除")
    s.cleanliness = clamp(s.cleanliness + 40 * mult)
    s.stress      = clamp(s.stress - 10)
    if had_poop:
        set_speech("うんちも片付けたよ！さっぱり！")
    elif s.pet_desire == "clean":
        set_speech("きれいにしてくれた！ありがとう！")
    else:
        set_speech("さっぱりした～！")
    add_log(f"🛁 きれいになった（効果 x{mult:.1f}）")


def do_study():
    s = st.session_state
    if s.stress > 80:
        if random.random() < 0.4:
            s.sick       = True
            s.sick_until = now() + SICK_DUR
            add_log("🤒 ストレス過多で病気になった！")
            set_speech("もうむり…病気になっちゃった…")
        else:
            add_log("😤 ストレスが多すぎて勉強を拒否した！", "study_refuse", 8)
            set_speech("勉強なんてしたくない！！")
        return

    boredom_m  = check_boredom("お勉強")
    desire_m   = get_desire_bonus("お勉強")
    zone_mult  = 2.0 if s.zoned_in else 1.0
    intel_gain = 10 * boredom_m * desire_m * zone_mult
    s.intellect = clamp(s.intellect + intel_gain)
    s.hunger    = clamp(s.hunger    - 5)
    s.stress    = clamp(s.stress    + 8)
    s.ambition  = clamp(s.ambition  + 2)
    s.wildness  = clamp(s.wildness  - 1)

    if boredom_m >= 1.0 and not s.zoned_in:
        s.zone_study_count += 1
        if s.zone_study_count >= 3:
            s.zoned_in         = True
            s.zone_until       = now() + ZONE_DUR
            s.zone_study_count = 0
            add_log("🔥 ゾーンに突入！かしこさ上昇2倍！")
            set_speech("集中できてる！ゾーンに入った！！")
    elif boredom_m < 1.0:
        s.zone_study_count = 0

    suffix = " ⚡ゾーン中" if s.zoned_in else ""
    add_log(f"📚 べんきょうした！（かしこさ+{intel_gain:.0f}）{suffix}")
    if s.zoned_in:
        set_speech("もっとわかる！もっと！")
    elif s.pet_desire == "study":
        set_speech("べんきょう、たのしい！ありがとう！")
    else:
        set_speech("なるほど～！")


def do_play():
    s    = st.session_state
    mult = check_boredom("遊ぶ") * get_desire_bonus("遊ぶ")
    s.friendship = clamp(s.friendship + 15 * mult)
    s.hunger     = clamp(s.hunger     - 8)
    s.stress     = clamp(s.stress     - 15)
    s.wildness   = clamp(s.wildness   + 2)
    s.zone_study_count = 0
    set_speech("たのしー！！ありがとう！" if s.pet_desire == "play" else "たのしー！！")
    add_log(f"🎮 一緒に遊んだ！（効果 x{mult:.1f}）")


def do_workout():
    s = st.session_state
    if s.injured:
        add_log("🚫 怪我中は筋トレできない！", "workout_injured", 10)
        set_speech("怪我してるから無理だよ！")
        return
    mult        = check_boredom("筋トレ") * get_desire_bonus("筋トレ")
    muscle_gain = 12 * mult
    weight_loss = 5  * mult
    s.muscle   = clamp(s.muscle   + muscle_gain)
    s.weight   = clamp(s.weight   - weight_loss)
    s.hunger   = clamp(s.hunger   - 20)
    s.stress   = clamp(s.stress   + 5)
    s.wildness = clamp(s.wildness + 1)
    set_speech("ふんぬ！！限界突破！")
    add_log(f"🏋️ 筋トレした！（筋肉+{muscle_gain:.0f}、体重-{weight_loss:.0f}）")


def do_scold():
    s = st.session_state
    s.stress     = clamp(s.stress     + 20)
    s.friendship = clamp(s.friendship - 10)
    s.ambition   = clamp(s.ambition   - 5)
    set_speech("…なんで怒るの…")
    add_log("😤 叱った。しょんぼりしてる…")


def do_adventure():
    s = st.session_state
    if s.adventuring:
        return
    if s.injured:
        add_log("🚫 怪我中は冒険できない！", "adv_injured", 10)
        set_speech("怪我してるから冒険は無理！")
        return
    if s.phase in ("egg", "infant"):
        add_log("まだ冒険には早い！", "adv_young", 10)
        return
    s.adventuring   = True
    s.adventure_end = now() + ADVENTURE_DUR
    s.hunger        = clamp(s.hunger - 5)
    set_speech("いってきます！！")
    add_log("🗺️ 冒険に出発した！10秒後に帰ってくる…")


# ─────────────────────────────────────────────────────────────────────────────
#  初期化チェック → 更新
# ─────────────────────────────────────────────────────────────────────────────
if "initialized" not in st.session_state:
    init()
else:
    update()

s = st.session_state
p = s.phase

# ═════════════════════════════════════════════════════════════════════════════
#  サイドバー（リセットのみ）
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if st.button("🔄 リセット", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  メイン画面
# ═════════════════════════════════════════════════════════════════════════════
st.title("🥚 ガウスくん育成ゲーム")

tab_main, tab_analysis, tab_pokedex = st.tabs(["🎮 メイン", "📊 詳細分析", "📖 図鑑"])

# ─────────────────────────────────────────────────────────────────────────────
#  メインタブ
# ─────────────────────────────────────────────────────────────────────────────
with tab_main:
    char_col, stat_col = st.columns([1, 1])

    with char_col:
        status_icons = (
            (" 💩" if s.poop       else "") +
            (" 🗺️" if s.adventuring else "") +
            (" 🤕" if s.injured    else "") +
            (" 🤒" if s.sick       else "")
        )
        st.markdown(
            f'<div style="text-align:center;font-size:7rem;line-height:1.2;'
            f'padding:1.5rem 0.5rem;border:2px solid #1e3a5c;border-radius:16px;'
            f'background:rgba(0,229,255,0.03);">'
            f'{get_emoji()}{status_icons}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="text-align:center;color:#4a90b8;margin-top:0.4rem;">'
            f'{get_label()}</p>',
            unsafe_allow_html=True,
        )

        buffs = []
        if s.zoned_in: buffs.append("🔥 ゾーン")
        if s.rebound:  buffs.append("🔁 リバウンド")
        if s.injured:  buffs.append("🤕 怪我")
        if s.sick:     buffs.append("🤒 病気")
        if buffs:
            st.markdown(
                f'<div style="text-align:center;font-size:0.78rem;color:#f59e0b;">'
                f'{" / ".join(buffs)}</div>',
                unsafe_allow_html=True,
            )

        if s.speech and now() < s.speech_until:
            with st.chat_message("assistant", avatar=get_emoji()):
                st.markdown(s.speech)

    with stat_col:
        st.markdown("**🧬 ステータス**")

        if p in PHASE_DURATION:
            dur     = PHASE_DURATION[p]
            elapsed = now() - s.phase_start
            remain  = max(0.0, dur - elapsed)
            st.progress(min(elapsed / dur, 1.0), text=f"次の成長まで {remain:.0f} 秒")

        st.divider()

        for label, key in [
            ("🍙 おなか",     "hunger"),
            ("🧠 かしこさ",   "intellect"),
            ("🛁 きれいさ",   "cleanliness"),
            ("💕 なかよし度", "friendship"),
        ]:
            val = s[key]
            lc, rc = st.columns([3, 1])
            lc.caption(label)
            rc.caption(f"**{val:.0f}**")
            st.progress(val / 100)

        # 欲求インジケーター（ボタンの選択ヒント）
        desire = s.get("pet_desire", None)
        if desire and desire in DESIRE_INFO:
            d_icon, d_text = DESIRE_INFO[desire]
            st.markdown(
                f'<div style="font-size:0.8rem;color:#00e5ff;margin-top:0.3rem;">'
                f'{d_icon} <b>{d_text}</b>（そのアクションは効果1.5倍！）</div>',
                unsafe_allow_html=True,
            )
        if s.adventuring:
            remain_adv = max(0.0, s.adventure_end - now())
            st.info(f"🗺️ 冒険中… あと **{remain_adv:.0f}** 秒")

    st.divider()

    # ── アクションボタン ──────────────────────────────────────────────────────
    if p == "egg":
        dur    = PHASE_DURATION["egg"]
        remain = max(0.0, dur - (now() - s.phase_start))
        st.info(f"🥚 たまごが孵るのを待っています… あと約 **{remain:.0f}** 秒")
    else:
        is_infant     = (p == "infant")
        can_adventure = p not in ("egg", "infant")
        adv_dis       = not can_adventure or s.adventuring or s.injured
        act_dis       = s.adventuring  # 冒険中は殆ど操作不可

        row1 = st.columns(4)
        row2 = st.columns(4)

        with row1[0]:
            if st.button("🍙\nごはん", use_container_width=True, disabled=act_dis):
                do_feed(); st.rerun()
        with row1[1]:
            if st.button("🍬\nおやつ", use_container_width=True, disabled=act_dis):
                do_snack(); st.rerun()
        with row1[2]:
            if st.button("🛁\nお掃除", use_container_width=True, disabled=act_dis):
                do_clean(); st.rerun()
        with row1[3]:
            if st.button("📚\nお勉強", use_container_width=True,
                         disabled=(is_infant or act_dis)):
                do_study(); st.rerun()

        with row2[0]:
            if st.button("🎮\n遊ぶ", use_container_width=True, disabled=act_dis):
                do_play(); st.rerun()
        with row2[1]:
            if st.button("🏋️\n筋トレ", use_container_width=True,
                         disabled=(is_infant or act_dis or s.injured)):
                do_workout(); st.rerun()
        with row2[2]:
            adv_label = "🗺️\n冒険中…" if s.adventuring else "🗺️\n冒険"
            if st.button(adv_label, use_container_width=True, disabled=adv_dis):
                do_adventure(); st.rerun()
        with row2[3]:
            if st.button("😤\n叱る", use_container_width=True,
                         disabled=(is_infant or act_dis)):
                do_scold(); st.rerun()

        if s.action_streak >= 3:
            st.caption(f"😑 **{s.last_action}** を連続{s.action_streak}回…効果半減中")

    # ── 警告（ボタンの後に描画することでStreamlitの実行順序問題を回避）────────
    if s.hunger < 25:
        st.warning("🍽️ お腹が空いています！")
    if s.cleanliness < 25 or s.poop:
        st.warning("🛁 きれいにしてあげて！")
    if s.stress > 70:
        st.warning("😰 ストレスが溜まってる！")

    st.divider()
    st.markdown("#### 📋 最近のできごと")
    for log in (s.logs or ["まだ何もありません…"])[:6]:
        st.caption(log)


# ─────────────────────────────────────────────────────────────────────────────
#  詳細分析タブ
# ─────────────────────────────────────────────────────────────────────────────
with tab_analysis:
    st.markdown("### 📊 詳細分析")

    st.markdown("#### 🔍 隠しパラメータ")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("ケアミス",  s.care_mistakes)
    c2.metric("ストレス",  f"{s.stress:.0f}")
    c3.metric("野心",      f"{s.ambition:.0f}")
    c4.metric("野生",      f"{s.wildness:.0f}")
    c5.metric("筋肉量",    f"{s.muscle:.0f}")
    c6.metric("運",        f"{s.luck:.0f}")
    c7.metric("体重",      f"{s.weight:.0f}")

    st.markdown("#### ✨ バフ / デバフ")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("ゾーン",      "🔥 ON" if s.zoned_in else "OFF",
              delta=f"残り{max(0,s.zone_until-now()):.0f}秒" if s.zoned_in else None)
    b2.metric("リバウンド",  "🔁 ON" if s.rebound  else "OFF",
              delta=f"残り{max(0,s.rebound_until-now()):.0f}秒" if s.rebound else None)
    b3.metric("怪我",        "🤕 あり" if s.injured else "なし",
              delta=f"回復{max(0,s.injured_until-now()):.0f}秒" if s.injured else None)
    b4.metric("病気",        "🤒 あり" if s.sick    else "なし",
              delta=f"回復{max(0,s.sick_until-now()):.0f}秒" if s.sick else None)

    st.markdown("#### 😑 飽き状態")
    fa, fb = st.columns(2)
    fa.metric("最後のアクション", s.last_action or "なし")
    fb.metric("連続回数",         s.action_streak)
    if s.action_streak >= 3:
        st.warning("同じアクションを連打中！効果が半減しています。別のアクションを試してみて。")

    st.divider()
    st.markdown("#### 📈 成長曲線（パラメータ推移）")

    if len(s.history) < 2:
        st.caption("データが溜まるとここにグラフが表示されます（約16秒後～）")
    else:
        import pandas as pd
        df = pd.DataFrame(s.history).set_index("time")

        st.markdown("**表示パラメータ**")
        st.line_chart(df[["おなか", "かしこさ", "きれいさ", "なかよし度"]])

        st.markdown("**隠しパラメータ**")
        st.line_chart(df[["ストレス", "野心", "野生", "筋肉量", "運"]])


# ─────────────────────────────────────────────────────────────────────────────
#  図鑑タブ
# ─────────────────────────────────────────────────────────────────────────────
with tab_pokedex:
    st.markdown("### 📖 進化図鑑")
    st.caption("成体になると14種類の中からいずれかに進化します。条件は判定優先順です。")

    current_evo = s.evolution_type

    for evo_name, evo_info in EVO_DEFS.items():
        is_current = (current_evo == evo_name)
        border_css = "border:2px solid #00e5ff;" if is_current else "border:1px solid #1e3a5c;"
        bg_css     = "background:rgba(0,229,255,0.08);" if is_current else ""

        ec1, ec2 = st.columns([1, 6])
        with ec1:
            st.markdown(
                f'<div style="font-size:2.5rem;text-align:center;{border_css}{bg_css}'
                f'border-radius:8px;padding:0.3rem;">{evo_info["emoji"]}</div>',
                unsafe_allow_html=True,
            )
        with ec2:
            name_str = f"**{evo_info['label']}**" + (" ← 現在" if is_current else "")
            st.markdown(name_str)
            st.caption(evo_info["desc"])
            st.caption(f"条件: {EVO_CONDITIONS.get(evo_name, '?')}")
# ── 自動更新（1秒ごと）────────────────────────────────────────────────────────
time.sleep(1)
st.rerun()
