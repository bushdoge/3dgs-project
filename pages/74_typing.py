# ミニゲーム：ことのは撃墜（リアルタイム落下タイピング）
#
# Streamlit は本来「操作したときだけ再実行」なのでリアルタイムゲームが作りにくい。
# ここでは st.fragment(run_every=...) で盤面だけを自動再描画し、
# 入力欄はフラグメントの外に置くことで、自動更新のたびに入力フォーカスが外れないようにしている。
#   - 盤面（落下・ダメージ判定・出現）＝ 0.35秒ごとに自動で回るフラグメント
#   - 入力欄 ＝ Enter で on_change → 全体再実行して判定
# 落下位置は「経過時間 × 速度」で毎回計算し直すので、再描画の間隔がぶれても位置がずれない。
#
# セーブ（ハイスコア）は /workspace/tmp/typing_save.json。

import json
import random
import time
from pathlib import Path

import streamlit as st

SAVE_FILE = Path("/workspace/tmp/typing_save.json")
TICK      = "0.35s"     # 盤面の再描画間隔
FIELD_H   = 420         # 盤面の高さ(px)
BURST     = "arasi"     # ゲージ満タン時に打つと全消しできる合言葉

# ═══════════════════════════════════════════════════════════════════════════════
#  お題（かな表記とローマ字。打つのは画面に出ているローマ字そのもの）
# ═══════════════════════════════════════════════════════════════════════════════

WORDS_S = [
    ("ねこ", "neko"), ("いぬ", "inu"), ("とり", "tori"), ("くま", "kuma"),
    ("うみ", "umi"), ("そら", "sora"), ("ほし", "hosi"), ("つき", "tuki"),
    ("かぜ", "kaze"), ("ゆき", "yuki"), ("あめ", "ame"), ("もり", "mori"),
    ("やま", "yama"), ("かわ", "kawa"), ("はな", "hana"), ("みず", "mizu"),
    ("くも", "kumo"), ("いす", "isu"), ("まど", "mado"), ("かさ", "kasa"),
    ("くつ", "kutu"), ("すし", "susi"), ("たまご", "tamago"), ("なし", "nasi"),
    ("もも", "momo"), ("ゆげ", "yuge"), ("うた", "uta"), ("ゆめ", "yume"),
    ("さかな", "sakana"), ("うさぎ", "usagi"), ("きつね", "kitune"),
    ("たぬき", "tanuki"), ("ぱんだ", "panda"), ("らくだ", "rakuda"),
    ("ひかり", "hikari"), ("かがみ", "kagami"), ("とけい", "tokei"),
    ("つくえ", "tukue"), ("とびら", "tobira"), ("かばん", "kaban"),
    ("りんご", "ringo"), ("みかん", "mikan"), ("いちご", "itigo"),
    ("すいか", "suika"), ("ぶどう", "budou"), ("めろん", "meron"),
    ("ばなな", "banana"), ("ちょこ", "tyoko"), ("だんご", "dango"),
    ("おちゃ", "otya"), ("こおり", "koori"), ("みるく", "miruku"),
]

WORDS_M = [
    ("ぼうし", "bousi"), ("ゆびわ", "yubiwa"), ("おにぎり", "onigiri"),
    ("みそしる", "misosiru"), ("やさい", "yasai"), ("くだもの", "kudamono"),
    ("せんべい", "senbei"), ("こうちゃ", "koutya"), ("たいよう", "taiyou"),
    ("にじいろ", "nijiiro"), ("ほしぞら", "hosizora"), ("はるかぜ", "harukaze"),
    ("おんせん", "onsen"), ("おまつり", "omaturi"), ("でんしゃ", "densya"),
    ("ひこうき", "hikouki"), ("ふうせん", "huusen"), ("おんがく", "ongaku"),
    ("うたごえ", "utagoe"), ("ぼうけん", "bouken"), ("ようせい", "yousei"),
    ("まちかど", "matikado"), ("こうえん", "kouen"), ("かいだん", "kaidan"),
    ("やねうら", "yaneura"), ("おふろ", "ohuro"), ("ふとん", "huton"),
    ("まくら", "makura"), ("えんぴつ", "enpitu"), ("けしごむ", "kesigomu"),
    ("てぶくろ", "tebukuro"), ("あきまつり", "akimaturi"),
    ("なつやすみ", "natuyasumi"), ("としょかん", "tosyokan"),
    ("えいがかん", "eigakan"), ("ゆうえんち", "yuuenti"),
    ("じてんしゃ", "jitensya"), ("じどうしゃ", "jidousya"),
    ("たからばこ", "takarabako"), ("ものがたり", "monogatari"),
    ("だいどころ", "daidokoro"), ("ざぶとん", "zabuton"),
    ("あまやどり", "amayadori"), ("よあけまえ", "yoakemae"),
    ("ゆきだるま", "yukidaruma"), ("まんじゅう", "manjuu"),
]

WORDS_L = [
    ("びじゅつかん", "bijutukan"), ("すいぞくかん", "suizokukan"),
    ("どうぶつえん", "doubutuen"), ("しんかんせん", "sinkansen"),
    ("まほうつかい", "mahoutukai"), ("かみなりぐも", "kaminarigumo"),
    ("たそがれどき", "tasogaredoki"), ("ほしのかけら", "hosinokakera"),
    ("にわとりごや", "niwatorigoya"), ("おかしのいえ", "okasinoie"),
    ("かぜのたより", "kazenotayori"), ("たいふういっか", "taihuuikka"),
    ("はなびたいかい", "hanabitaikai"), ("ひみつのとびら", "himitunotobira"),
    ("つきよのばんに", "tukiyonobanni"), ("うたたねのゆめ", "utatanenoyume"),
    ("まほうのじゅうたん", "mahounojuutan"), ("にちようびのあさ", "nitiyoubinoasa"),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  ローマ字の揺れを受理する照合
#   「し」は si / shi / ci、「つ」は tu / tsu、「じゃ」は ja / jya / zya … のように
#   打ち方が複数ある。表示は代表表記ひとつだが、判定はかな側から
#   「その打ち方で書けるか」を再帰的に照合するので、どの流儀でも通る。
# ═══════════════════════════════════════════════════════════════════════════════

ROMA = {
    "あ": ["a"], "い": ["i", "yi"], "う": ["u", "wu"], "え": ["e"], "お": ["o"],
    "か": ["ka", "ca"], "き": ["ki"], "く": ["ku", "cu", "qu"], "け": ["ke"], "こ": ["ko", "co"],
    "が": ["ga"], "ぎ": ["gi"], "ぐ": ["gu"], "げ": ["ge"], "ご": ["go"],
    "さ": ["sa"], "し": ["si", "shi", "ci"], "す": ["su"], "せ": ["se", "ce"], "そ": ["so"],
    "ざ": ["za"], "じ": ["ji", "zi"], "ず": ["zu"], "ぜ": ["ze"], "ぞ": ["zo"],
    "た": ["ta"], "ち": ["ti", "chi"], "つ": ["tu", "tsu"], "て": ["te"], "と": ["to"],
    "だ": ["da"], "ぢ": ["di"], "づ": ["du", "dzu"], "で": ["de"], "ど": ["do"],
    "な": ["na"], "に": ["ni"], "ぬ": ["nu"], "ね": ["ne"], "の": ["no"],
    "は": ["ha"], "ひ": ["hi"], "ふ": ["hu", "fu"], "へ": ["he"], "ほ": ["ho"],
    "ば": ["ba"], "び": ["bi"], "ぶ": ["bu"], "べ": ["be"], "ぼ": ["bo"],
    "ぱ": ["pa"], "ぴ": ["pi"], "ぷ": ["pu"], "ぺ": ["pe"], "ぽ": ["po"],
    "ま": ["ma"], "み": ["mi"], "む": ["mu"], "め": ["me"], "も": ["mo"],
    "や": ["ya"], "ゆ": ["yu"], "よ": ["yo"],
    "ら": ["ra"], "り": ["ri"], "る": ["ru"], "れ": ["re"], "ろ": ["ro"],
    "わ": ["wa"], "を": ["wo", "o"], "ん": ["n", "nn", "xn"],
    "きゃ": ["kya"], "きゅ": ["kyu"], "きょ": ["kyo"],
    "ぎゃ": ["gya"], "ぎゅ": ["gyu"], "ぎょ": ["gyo"],
    "しゃ": ["sya", "sha"], "しゅ": ["syu", "shu"], "しょ": ["syo", "sho"],
    "じゃ": ["ja", "jya", "zya"], "じゅ": ["ju", "jyu", "zyu"], "じょ": ["jo", "jyo", "zyo"],
    "ちゃ": ["tya", "cha"], "ちゅ": ["tyu", "chu"], "ちょ": ["tyo", "cho"],
    "にゃ": ["nya"], "にゅ": ["nyu"], "にょ": ["nyo"],
    "ひゃ": ["hya"], "ひゅ": ["hyu"], "ひょ": ["hyo"],
    "びゃ": ["bya"], "びゅ": ["byu"], "びょ": ["byo"],
    "ぴゃ": ["pya"], "ぴゅ": ["pyu"], "ぴょ": ["pyo"],
    "みゃ": ["mya"], "みゅ": ["myu"], "みょ": ["myo"],
    "りゃ": ["rya"], "りゅ": ["ryu"], "りょ": ["ryo"],
    "ふぁ": ["fa"], "ふぃ": ["fi"], "ふぇ": ["fe"], "ふぉ": ["fo"],
    "うぁ": ["wha"], "てぃ": ["thi"], "でぃ": ["dhi"],
}
SOKUON = ("xtu", "ltu", "xtsu", "ltsu")


def kana_ok(kana: str, typed: str) -> bool:
    """かな kana を typed のローマ字で打てるか（表記の揺れを全部許す）"""
    n, m = len(kana), len(typed)
    memo = {}

    def rec(i, j):
        if i == n:
            return j == m
        key = (i, j)
        if key in memo:
            return memo[key]
        ok = False
        for L in (2, 1):
            if ok or i + L > n:
                continue
            k = kana[i:i + L]
            if L == 1 and k == "っ":
                # 促音：次のかなの頭の子音を重ねる打ち方
                for nl in (2, 1):
                    if i + 1 + nl > n:
                        continue
                    for alt in ROMA.get(kana[i + 1:i + 1 + nl], []):
                        c = alt[0]
                        if c in "aiueo":
                            continue
                        if j < m and typed[j] == c and rec(i + 1, j + 1):
                            ok = True; break
                    if ok: break
                if not ok:
                    for alt in SOKUON:
                        if typed.startswith(alt, j) and rec(i + 1, j + len(alt)):
                            ok = True; break
                continue
            for alt in ROMA.get(k, []):
                if typed.startswith(alt, j) and rec(i + L, j + len(alt)):
                    ok = True; break
        memo[key] = ok
        return ok

    return rec(0, 0)


def canon(kana: str) -> str:
    """かなから代表のローマ字表記を作る（画面に出すお手本）"""
    out, i, n = [], 0, len(kana)
    while i < n:
        if kana[i] == "っ" and i + 1 < n:
            for L in (2, 1):
                if i + 1 + L <= n and kana[i + 1:i + 1 + L] in ROMA:
                    out.append(ROMA[kana[i + 1:i + 1 + L]][0][0]); break
            i += 1
            continue
        for L in (2, 1):
            if i + L <= n and kana[i:i + L] in ROMA:
                out.append(ROMA[kana[i:i + L]][0]); i += L; break
        else:
            i += 1
    return "".join(out)


# 種別: 色・記号・効果
KINDS = {
    "normal": dict(col="#EAE6DC", bg="#282D35", mark="", w=68),
    "bomb":   dict(col="#FFDAD6", bg="#93000A", mark="💣", w=12),
    "heal":   dict(col="#B8F2CE", bg="#22503A", mark="💚", w=8),
    "freeze": dict(col="#CDE5FF", bg="#2E4A63", mark="🧊", w=6),
    "chain":  dict(col="#FFDFA6", bg="#5C4300", mark="⭐", w=6),
}
KIND_IDS = list(KINDS)
KIND_W   = [KINDS[k]["w"] for k in KIND_IDS]


def load_hs():
    try:
        return json.loads(SAVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"score": 0, "combo": 0, "kpm": 0, "plays": 0, "words": 0}


def save_hs(d):
    try:
        SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SAVE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  ゲームロジック
# ═══════════════════════════════════════════════════════════════════════════════

def new_game():
    now = time.time()
    return dict(phase="play", hp=100, maxhp=100, score=0, combo=0, best_combo=0,
                chars=0, hits=0, misses=0, level=1, start=now, last_level=now,
                words=[], next_spawn=now + 0.6, gauge=0, uid=0,
                flash="", flash_t=0.0, end=None)


def level_params(lv):
    """レベルごとの落下速度と出現間隔"""
    speed = 0.062 + 0.011 * (lv - 1)          # 1秒あたりの落下量（1.0で着弾）
    interval = max(0.70, 2.2 - 0.13 * (lv - 1))
    return speed, interval


def spawn(gm, now):
    lv = gm["level"]
    # レベルが上がるほど長い語が増える
    r = random.random()
    if lv <= 2:
        pool = WORDS_S if r < 0.8 else WORDS_M
    elif lv <= 5:
        pool = WORDS_S if r < 0.45 else (WORDS_M if r < 0.9 else WORDS_L)
    else:
        pool = WORDS_S if r < 0.25 else (WORDS_M if r < 0.72 else WORDS_L)
    kana, romaji = random.choice(pool)
    kind = random.choices(KIND_IDS, weights=KIND_W)[0]
    speed, _ = level_params(lv)
    if kind == "bomb":
        speed *= 1.45
    gm["uid"] += 1
    gm["words"].append(dict(id=gm["uid"], kana=kana, r=romaji, kind=kind,
                            t0=now, speed=speed, x=random.uniform(2, 74)))


def word_y(w, now):
    return (now - w["t0"]) * w["speed"]


def tick(gm):
    """時間経過ぶんの処理（落下・着弾・出現・レベルアップ）"""
    if gm["phase"] != "play":
        return
    now = time.time()

    # 着弾判定
    landed = [w for w in gm["words"] if word_y(w, now) >= 1.0]
    for w in landed:
        if w["kind"] == "bomb":
            gm["hp"] -= 22; gm["flash"] = f"💥 {w['kana']} が爆発した！"
        elif w["kind"] in ("heal", "chain", "freeze"):
            gm["hp"] -= 4;  gm["flash"] = f"{w['kana']} を取り逃した"
        else:
            gm["hp"] -= 10; gm["flash"] = f"{w['kana']} が落ちた"
        gm["flash_t"] = now
        gm["combo"] = 0
    if landed:
        ids = {w["id"] for w in landed}
        gm["words"] = [w for w in gm["words"] if w["id"] not in ids]

    if gm["hp"] <= 0:
        gm["hp"] = 0
        gm["phase"] = "over"
        gm["end"] = now
        return

    # 出現
    _, interval = level_params(gm["level"])
    while now >= gm["next_spawn"]:
        spawn(gm, gm["next_spawn"])
        gm["next_spawn"] += interval

    # レベルアップ（18秒ごと）
    if now - gm["last_level"] >= 15:
        gm["level"] += 1
        gm["last_level"] = now
        gm["flash"] = f"レベル {gm['level']}！ 落下が速くなる"
        gm["flash_t"] = now


def resolve(gm, typed):
    """入力の判定。合言葉→全消し、一致→撃墜、不一致→ミス"""
    if gm["phase"] != "play" or not typed:
        return
    now = time.time()
    tick(gm)
    if gm["phase"] != "play":
        return

    if typed == BURST and gm["gauge"] >= 100:
        n = len(gm["words"])
        gain = sum(len(w["r"]) for w in gm["words"]) * 2
        for w in gm["words"]:
            if w["kind"] == "heal":
                gm["hp"] = min(gm["maxhp"], gm["hp"] + 10)
        gm["words"] = []
        gm["score"] += gain
        gm["gauge"] = 0
        gm["chars"] += len(typed)
        gm["flash"] = f"🌪️ あらしが {n} 語を吹き飛ばした（+{gain}）"
        gm["flash_t"] = now
        return

    # 一致する語のうち、いちばん下にあるものを撃墜
    cands = [w for w in gm["words"] if kana_ok(w["kana"], typed)]
    if not cands:
        gm["combo"] = 0
        gm["misses"] += 1
        gm["gauge"] = max(0, gm["gauge"] - 8)
        gm["flash"] = f"「{typed}」は無い"
        gm["flash_t"] = now
        return

    w = max(cands, key=lambda w: word_y(w, now))
    gm["words"] = [x for x in gm["words"] if x["id"] != w["id"]]
    gm["combo"] += 1
    gm["best_combo"] = max(gm["best_combo"], gm["combo"])
    gm["hits"] += 1
    gm["chars"] += len(typed)
    base = len(w["r"]) * 10
    mult = 1.0 + gm["combo"] * 0.08
    gain = int(base * mult)
    gm["score"] += gain
    gm["gauge"] = min(100, gm["gauge"] + len(w["r"]) * 2)

    if w["kind"] == "heal":
        gm["hp"] = min(gm["maxhp"], gm["hp"] + 14)
        gm["flash"] = f"💚 {w['kana']} で回復（+{gain}）"
    elif w["kind"] == "freeze":
        for x in gm["words"]:
            x["t0"] += 2.5                      # 全体を2.5秒ぶん押し戻す
        gm["flash"] = f"🧊 {w['kana']} で全体が減速（+{gain}）"
    elif w["kind"] == "chain":
        gm["combo"] += 3
        gm["gauge"] = min(100, gm["gauge"] + 25)
        gm["flash"] = f"⭐ {w['kana']} でコンボ+3（+{gain}）"
    elif w["kind"] == "bomb":
        gm["flash"] = f"💣 {w['kana']} を解除（+{gain}）"
    else:
        gm["flash"] = f"{w['kana']} 撃墜（+{gain}）"
    gm["flash_t"] = now


# ═══════════════════════════════════════════════════════════════════════════════
#  画面
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.tw-field{ position:relative; height:__FH__px; border-radius:22px; overflow:hidden;
  background:linear-gradient(180deg,#0D1117 0%,#161B22 78%,#2A1418 100%);
  border:1px solid #4B443C; }
.tw-line{ position:absolute; left:0; right:0; bottom:34px; height:2px;
  background:repeating-linear-gradient(90deg,#FFB4AB 0 10px,transparent 10px 20px); }
.tw-w{ position:absolute; text-align:center; padding:.3rem .7rem; border-radius:14px;
  white-space:nowrap; box-shadow:0 2px 8px rgba(0,0,0,.4); }
.tw-kana{ font-size:.78rem; opacity:.85; line-height:1.1; }
.tw-r{ font-size:1.02rem; font-weight:800; letter-spacing:.06em; line-height:1.2;
  font-family:'Roboto Mono',ui-monospace,monospace; }
.tw-hud{ display:flex; gap:1.4rem; align-items:center; flex-wrap:wrap; margin:.3rem 0 .6rem; }
.tw-stat b{ font-size:1.35rem; font-variant-numeric:tabular-nums; }
.tw-stat span{ font-size:.68rem; letter-spacing:.12em; color:#CFC7BC; display:block; }
.tw-bar{ height:12px; border-radius:999px; background:#282D35; overflow:hidden; }
.tw-bar i{ display:block; height:100%; border-radius:999px; }
.tw-flash{ font-size:.86rem; font-weight:700; min-height:1.4em; }
.tw-burst{ background:#5C4300; color:#FFDFA6; border-radius:999px;
  padding:3px 14px; font-weight:800; font-size:.82rem; }
</style>
""".replace("__FH__", str(FIELD_H)), unsafe_allow_html=True)

if "tw_hs" not in st.session_state:
    st.session_state.tw_hs = load_hs()
hs = st.session_state.tw_hs
if "tw" not in st.session_state:
    st.session_state.tw = None

st.title("ことのは撃墜")
st.caption("落ちてくる言葉のローマ字を打ってEnter。下の線に届く前に全部撃ち落とす。")

gm = st.session_state.tw

# ── タイトル ──────────────────────────────────────────────────────────────────
if gm is None or gm["phase"] == "menu":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ハイスコア", f"{hs.get('score',0):,}")
    c2.metric("最長コンボ", hs.get("combo", 0))
    c3.metric("最高打鍵", f"{hs.get('kpm',0)} 打/分")
    c4.metric("総プレイ", hs.get("plays", 0))
    st.markdown("""
### 遊び方
- 落ちてくる言葉の**下段のローマ字**を打って **Enter**。いちばん下にあるものから消える。
- ローマ字の**打ち方の流儀はどれでもよい**。「なし」は `nasi` でも `nashi` でも、
  「ちょこ」は `tyoko` でも `choko` でも通る（画面の表記は代表例）。
- 下の破線まで落ちるとダメージ。体力が尽きたら終わり。
- **15秒ごとにレベルが上がり**、落下が速く・出現が多く・言葉が長くなる。

### 言葉の種類
| | 種類 | 効果 |
|---|---|---|
| 　 | ふつう | 撃墜でスコア |
| 💣 | 爆弾 | 落下が速い。落とすと大ダメージ |
| 💚 | 回復 | 撃墜で体力が戻る |
| 🧊 | 氷結 | 撃墜で画面全体の落下が2.5秒ぶん巻き戻る |
| ⭐ | 連鎖 | 撃墜でコンボ+3・ゲージ大幅増 |

### あらしゲージ
撃墜するたび溜まる。満タンで **`arasi`** と打つと**画面の全部を吹き飛ばす**。
連続撃墜でコンボ倍率が上がるので、ミス入力（存在しない語を打つ）はコンボが切れて損。
""")
    if st.button("はじめる", type="primary", use_container_width=True):
        st.session_state.tw = new_game()
        st.session_state.tw_in = ""
        st.rerun()
    st.stop()

# ── 入力欄（フラグメントの外に置くのが肝。自動更新でフォーカスを奪われない）──────
def _submit():
    s = st.session_state.get("tw_in", "").strip().lower().replace(" ", "")
    st.session_state.tw_in = ""
    if st.session_state.tw:
        resolve(st.session_state.tw, s)


# ── 盤面（0.35秒ごとに自動再描画）────────────────────────────────────────────
@st.fragment(run_every=TICK)
def board():
    g = st.session_state.tw
    if g is None:
        return
    tick(g)
    now = time.time()
    el = max(0.001, now - g["start"])
    kpm = int(g["chars"] / el * 60)
    acc = int(g["hits"] / max(1, g["hits"] + g["misses"]) * 100)

    hpp = g["hp"] / g["maxhp"] * 100
    hpcol = "#8ED9A8" if hpp > 55 else ("#EFC77A" if hpp > 25 else "#FFB4AB")
    burst = ('<span class="tw-burst">あらし発動可：arasi と打つ</span>'
             if g["gauge"] >= 100 else
             f'<span style="font-size:.78rem;color:#CFC7BC">あらしゲージ {g["gauge"]}%</span>')

    st.markdown(
        f'<div class="tw-hud">'
        f'<div class="tw-stat"><span>LEVEL</span><b>{g["level"]}</b></div>'
        f'<div class="tw-stat"><span>打鍵/分</span><b>{kpm}</b></div>'
        f'<div class="tw-stat"><span>正確さ</span><b>{acc}%</b></div>'
        f'<div class="tw-stat"><span>残り語数</span><b>{len(g["words"])}</b></div>'
        f'<div style="flex:1"></div>{burst}</div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="tw-bar"><i style="width:{hpp:.0f}%;background:{hpcol}"></i></div>'
        f'<div style="font-size:.72rem;color:#CFC7BC;margin:.2rem 0 .5rem">体力 {g["hp"]} / {g["maxhp"]}</div>'
        f'<div class="tw-bar" style="height:7px"><i style="width:{g["gauge"]}%;'
        f'background:linear-gradient(90deg,#EFC77A,#FFB59B)"></i></div>',
        unsafe_allow_html=True)

    parts = []
    for w in sorted(g["words"], key=lambda w: word_y(w, now)):
        y = min(1.0, max(0.0, word_y(w, now)))
        k = KINDS[w["kind"]]
        top = 8 + y * (FIELD_H - 78)
        danger = y > 0.78
        border = "2px solid #FFB4AB" if danger else "1px solid rgba(255,255,255,.10)"
        parts.append(
            f'<div class="tw-w" style="top:{top:.0f}px;left:{w["x"]:.1f}%;'
            f'background:{k["bg"]};color:{k["col"]};border:{border};">'
            f'<div class="tw-kana">{k["mark"]}{w["kana"]}</div>'
            f'<div class="tw-r">{w["r"]}</div></div>')
    st.markdown(f'<div class="tw-field">{"".join(parts)}<div class="tw-line"></div></div>',
                unsafe_allow_html=True)

    if g["phase"] == "over":
        st.rerun()          # 打鍵が止まったまま被弾した場合。全体を再実行して結果画面へ


@st.fragment
def typing_input():
    """入力欄。これ自体をフラグメントにしてあるので、Enterを押しても
    再実行されるのはこのブロックだけ。盤面フラグメントの自動更新と
    全体再実行がぶつかって入力欄が描画されないまま消える事故を防いでいる。"""
    g = st.session_state.tw
    if g is None:
        return
    st.text_input("入力", key="tw_in", on_change=_submit,
                  placeholder="ローマ字を打って Enter", label_visibility="collapsed")
    # 再描画のたびに入力欄へフォーカスを戻す
    st.html("""
<script>
  const t = setInterval(() => {
    const el = document.querySelector('input[aria-label="入力"]');
    if (el) { el.focus(); clearInterval(t); }
  }, 50);
  setTimeout(() => clearInterval(t), 3000);
</script>""", unsafe_allow_javascript=True)

    now = time.time()
    fl = g["flash"] if now - g["flash_t"] < 1.8 else ""
    st.markdown(
        f'<div class="tw-hud" style="margin-top:.2rem">'
        f'<div class="tw-stat"><span>SCORE</span><b>{g["score"]:,}</b></div>'
        f'<div class="tw-stat"><span>COMBO</span><b style="color:#FFB59B">{g["combo"]}</b></div>'
        f'<div class="tw-stat"><span>あらし</span><b>{g["gauge"]}%</b></div>'
        f'<div style="flex:1"></div></div>'
        f'<div class="tw-flash">{fl}</div>', unsafe_allow_html=True)

    if st.button("やめる", key="tw_quit"):
        g["phase"] = "over"; g["end"] = time.time()
    if g["phase"] != "play":
        st.rerun()          # 結果画面へ（全体を再実行）


if gm["phase"] == "play":
    board()
    typing_input()

# ── 結果 ──────────────────────────────────────────────────────────────────────
elif gm["phase"] == "over":
    el = max(0.001, (gm["end"] or time.time()) - gm["start"])
    kpm = int(gm["chars"] / el * 60)
    acc = int(gm["hits"] / max(1, gm["hits"] + gm["misses"]) * 100)
    rec = []
    if gm["score"] > hs.get("score", 0):      hs["score"] = gm["score"]; rec.append("スコア")
    if gm["best_combo"] > hs.get("combo", 0): hs["combo"] = gm["best_combo"]; rec.append("コンボ")
    if kpm > hs.get("kpm", 0):                hs["kpm"] = kpm; rec.append("打鍵速度")
    hs["plays"] = hs.get("plays", 0) + 1
    hs["words"] = hs.get("words", 0) + gm["hits"]
    save_hs(hs)

    st.markdown("## 撃墜終了")
    if rec:
        st.success("自己ベスト更新：" + "・".join(rec))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("スコア", f"{gm['score']:,}")
    c2.metric("最長コンボ", gm["best_combo"])
    c3.metric("打鍵/分", kpm)
    c4.metric("正確さ", f"{acc}%")
    c5.metric("到達レベル", gm["level"])
    st.caption(f"撃墜 {gm['hits']} 語／ミス入力 {gm['misses']} 回／プレイ時間 {el:.0f} 秒")

    b1, b2 = st.columns(2)
    if b1.button("もう一度", type="primary", use_container_width=True):
        st.session_state.tw = new_game(); st.session_state.tw_in = ""; st.rerun()
    if b2.button("タイトルへ", use_container_width=True):
        st.session_state.tw = None; st.rerun()
