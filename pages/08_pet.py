# ペット育成ゲーム「ガウスくん」
# えさ・おもちゃ・部屋の組み合わせで進化の見た目・性格が変わる。コレクション要素あり。

import json
import random
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="ガウスくん", page_icon="🐾", layout="wide")

SAVE_FILE           = Path("/workspace/tmp/pet_save.json")
PIPELINE_STATE_FILE = Path("/workspace/tmp/pipeline_state.json")
MINIGAME_SAVE       = Path("/workspace/tmp/minigame_save.json")

# ══════════════════════════════════════════════════════════════════════════════
#  アイテム定義
# ══════════════════════════════════════════════════════════════════════════════

FOOD_ITEMS = {
    "normal_food":  {"name": "ふつうのごはん",  "icon": "🍙", "cat": "food",
                     "hunger": 25, "happiness":  0, "health":  0, "cleanliness":  0, "care":  4},
    "fancy_food":   {"name": "豪華なごはん",    "icon": "🍱", "cat": "food",
                     "hunger": 50, "happiness": 15, "health":  5, "cleanliness":  0, "care": 12},
    "vegetable":    {"name": "やさい",           "icon": "🥦", "cat": "food",
                     "hunger": 20, "happiness": -5, "health": 20, "cleanliness":  0, "care":  6},
    "fish":         {"name": "おさかな",         "icon": "🐟", "cat": "food",
                     "hunger": 30, "happiness": 10, "health":  8, "cleanliness":  0, "care":  7},
    "candy":        {"name": "あめ",             "icon": "🍬", "cat": "food",
                     "hunger":  5, "happiness": 30, "health": -8, "cleanliness":  0, "care":  3},
    "special_cake": {"name": "スペシャルケーキ", "icon": "🎂", "cat": "food",
                     "hunger": 45, "happiness": 45, "health": 15, "cleanliness":  0, "care": 20},
}

TOY_ITEMS = {
    "toy":       {"name": "ぬいぐるみ",        "icon": "🧸", "cat": "toy",
                  "happiness": 20, "hunger": -5,  "cleanliness": -5,  "health":  0, "care":  5},
    "ball":      {"name": "ボール",            "icon": "⚽", "cat": "toy",
                  "happiness": 25, "hunger": -12, "cleanliness": -10, "health":  0, "care":  6},
    "puzzle":    {"name": "パズル",            "icon": "🧩", "cat": "toy",
                  "happiness": 15, "hunger": -3,  "cleanliness":  0,  "health":  0, "care":  8},
    "music_box": {"name": "オルゴール",        "icon": "🎵", "cat": "toy",
                  "happiness": 18, "hunger":  0,  "cleanliness":  0,  "health":  8, "care":  5},
    "robot_toy": {"name": "ロボットおもちゃ",  "icon": "🤖", "cat": "toy",
                  "happiness": 30, "hunger": -8,  "cleanliness": -5,  "health":  0, "care":  9},
}

CARE_ITEMS = {
    "medicine": {"name": "くすり",     "icon": "💊", "cat": "care",
                 "hunger": 0, "happiness": 0, "cleanliness": 0, "health": 25, "care": 5,
                 "cure_sick": True},
    "shampoo":  {"name": "シャンプー", "icon": "🛁", "cat": "care",
                 "hunger": 0, "happiness": 5, "cleanliness": 35, "health":  0, "care": 4},
    "vitamins": {"name": "ビタミン剤", "icon": "🌟", "cat": "care",
                 "hunger": 0, "happiness": 0, "cleanliness":  0, "health": 40, "care": 6},
}

ITEMS = {**FOOD_ITEMS, **TOY_ITEMS, **CARE_ITEMS}

# ══════════════════════════════════════════════════════════════════════════════
#  部屋定義
# ══════════════════════════════════════════════════════════════════════════════

WALLPAPERS = {
    "lab":    {"name": "研究室",     "icon": "🔬", "bg": (10,14,26),  "card": (15,25,40),  "ol": (30, 60,100), "cost":  0},
    "forest": {"name": "もりのなか", "icon": "🌲", "bg": ( 8,20,12),  "card": (12,28,16),  "ol": (40,100, 50), "cost": 40},
    "space":  {"name": "うちゅう",   "icon": "🌌", "bg": ( 5, 5,18),  "card": (10, 8,28),  "ol": (80, 30,140), "cost": 60},
    "flower": {"name": "はなばたけ", "icon": "🌸", "bg": (28,12,22),  "card": (38,18,30),  "ol": (200,70,120), "cost": 50},
    "city":   {"name": "としのよる", "icon": "🌃", "bg": (14,14,24),  "card": (20,20,34),  "ol": (220,160,  0), "cost": 55},
    "ocean":  {"name": "かいてい",   "icon": "🌊", "bg": ( 5,18,38),  "card": ( 8,24,50),  "ol": (  0,120,200), "cost": 70},
}

FURNITURES = {
    "none":      {"name": "なし",            "icon": "　",  "cost":  0},
    "computer":  {"name": "コンピューター",  "icon": "💻",  "cost":  0},
    "bookshelf": {"name": "本だな",          "icon": "📚",  "cost": 25},
    "plant":     {"name": "観葉植物",        "icon": "🪴",  "cost": 20},
    "aquarium":  {"name": "すいそう",        "icon": "🐠",  "cost": 45},
    "telescope": {"name": "ぼうえんきょう",  "icon": "🔭",  "cost": 35},
    "sofa":      {"name": "ソファー",        "icon": "🛋️",  "cost": 30},
}

# ══════════════════════════════════════════════════════════════════════════════
#  性格定義
# ══════════════════════════════════════════════════════════════════════════════

PERSONALITIES = {
    "researcher": {
        "name": "研究者タイプ", "icon": "🔬",
        "desc": "知識欲旺盛な探求者。パズルとやさいが好き。",
        "color": ((100,150,255),(60,100,200)),
        "happy_msg":  "今日もいい発見があった！研究は終わらない！",
        "normal_msg": "データを分析中…なかなか面白い。",
    },
    "athlete": {
        "name": "アスリートタイプ", "icon": "🏃",
        "desc": "元気いっぱいの体力自慢。ボールとおさかなが好き。",
        "color": ((80,210,100),(50,160,70)),
        "happy_msg":  "最高のコンディションだ！もっと走れる！！",
        "normal_msg": "体を動かしたい気分。なんか遊ぼ！",
    },
    "gourmet": {
        "name": "グルメタイプ", "icon": "👨‍🍳",
        "desc": "食の喜びを知るグルメ。豪華なごはんに目がない。",
        "color": ((255,160,50),(200,110,20)),
        "happy_msg":  "ふむ…これは最上級の味わいだ…！！",
        "normal_msg": "今日のごはんは何かな〜楽しみ〜",
    },
    "sweet": {
        "name": "あまえんぼタイプ", "icon": "🍭",
        "desc": "甘えん坊でかわいい。あめとぬいぐるみが大好き。",
        "color": ((255,140,190),(200,90,150)),
        "happy_msg":  "もっとあまやかしてよ〜！大好き〜！！",
        "normal_msg": "ねえねえ、遊んで？さびしいよ〜",
    },
    "mystic": {
        "name": "神秘家タイプ", "icon": "🌙",
        "desc": "宇宙の謎を追う神秘家。オルゴールとロボットが好き。",
        "color": ((180,80,255),(120,40,200)),
        "happy_msg":  "宇宙のリズムと同調している…これが調和だ…",
        "normal_msg": "何か大きな力を感じる…思索の時間だ。",
    },
}

STAGE_NAMES = ["たまご", "ちびガウス", "こガウス", "ガウス博士", "✨スーパーガウス✨"]

# ══════════════════════════════════════════════════════════════════════════════
#  報酬定義
# ══════════════════════════════════════════════════════════════════════════════

STEP_REWARDS = {
    "extracting": {"toy": 1, "vegetable": 2},
    "colmap":     {"shampoo": 1, "medicine": 1, "puzzle": 1},
    "training":   {"fancy_food": 2, "fish": 2, "vitamins": 1},
}

MILESTONES = [
    ( 30, {"normal_food": 3},                                "スコア30達成！ごはん×3をプレゼント🎁"),
    ( 60, {"toy": 1, "shampoo": 1, "vegetable": 2},          "60達成！ぬいぐるみ＆シャンプー＆やさい×2🎁"),
    (100, {"medicine": 1, "fancy_food": 1, "fish": 2},       "100達成！くすり＆豪華ごはん＆おさかな×2🎁"),
    (160, {"ball": 1, "puzzle": 1, "normal_food": 3},        "160達成！ボール＆パズル＆ごはん×3🎁"),
    (250, {"music_box": 1, "fancy_food": 2, "vitamins": 1},  "250達成！オルゴール＆豪華ごはん×2＆ビタミン🎁"),
    (400, {"robot_toy": 1, "special_cake": 1},               "400達成！ロボットおもちゃ＆スペシャルケーキ🎁"),
    (600, {"special_cake": 2, "robot_toy": 1, "vitamins": 2},"600達成！超豪華スペシャルパック🎁"),
]

PRESTIGE_REWARD = {
    "fancy_food": 2, "medicine": 1, "shampoo": 2,
    "toy": 1, "fish": 2, "puzzle": 1,
}

# ══════════════════════════════════════════════════════════════════════════════
#  デフォルト状態
# ══════════════════════════════════════════════════════════════════════════════

_DEF_ITEMS = {k: 0 for k in ITEMS}
_DEF_ITEMS.update({"normal_food": 5, "medicine": 1, "shampoo": 3, "toy": 2})

_DEF_COLLECTION = {
    "foods": [], "toys": [],
    "wallpapers": ["lab"], "furniture": ["none", "computer"],
    "personalities": [],
}
_DEF_ROOM = {"wallpaper": "lab", "furniture1": "computer", "furniture2": "none"}

DEFAULT_PET = {
    "name": "ガウスくん", "stage": 0,
    "hunger": 100.0, "happiness": 80.0, "cleanliness": 100.0, "health": 100.0,
    "age_hours": 0.0, "last_update": 0.0,
    "is_sick": False, "is_dead": False,
    "care_score": 0.0, "sick_hours": 0.0,
    "stars": 0,
    "personality": None,
    "food_history": {}, "toy_history": {}, "room_time": {},
    "room": _DEF_ROOM.copy(),
    "collection": _DEF_COLLECTION.copy(),
    "items": _DEF_ITEMS.copy(),
    "last_login_bonus": 0.0,
    "step_reward_claimed": {"extracting": 0.0, "colmap": 0.0, "training": 0.0},
    "milestones_claimed": [],
    "prestige_reward_claimed": 0,
    "message": "はじめまして！よろしくね！",
    "last_action_time": 0.0,
}

# ══════════════════════════════════════════════════════════════════════════════
#  セーブ / ロード
# ══════════════════════════════════════════════════════════════════════════════

def load_pet() -> dict:
    if SAVE_FILE.exists():
        try:
            data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            for k, v in DEFAULT_PET.items():
                if k not in data:
                    data[k] = v.copy() if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)
            # Items: add new item keys
            for k in _DEF_ITEMS:
                data["items"].setdefault(k, 0)
            # Room
            if not isinstance(data.get("room"), dict):
                data["room"] = _DEF_ROOM.copy()
            for k, v in _DEF_ROOM.items():
                data["room"].setdefault(k, v)
            # Collection
            if not isinstance(data.get("collection"), dict):
                data["collection"] = _DEF_COLLECTION.copy()
            for k, v in _DEF_COLLECTION.items():
                if k not in data["collection"]:
                    data["collection"][k] = list(v) if isinstance(v, list) else v
            # Ensure current wallpaper is in collection
            wp = data["room"].get("wallpaper", "lab")
            if wp not in data["collection"]["wallpapers"]:
                data["collection"]["wallpapers"].append(wp)
            return data
        except Exception:
            pass
    pet = {k: (v.copy() if isinstance(v, dict) else (list(v) if isinstance(v, list) else v))
           for k, v in DEFAULT_PET.items()}
    pet["last_update"] = time.time()
    return pet

def save_pet(pet: dict):
    SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAVE_FILE.write_text(json.dumps(pet, ensure_ascii=False, indent=2), encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
#  ステータス更新
# ══════════════════════════════════════════════════════════════════════════════

def update_pet(pet: dict) -> dict:
    if pet.get("is_dead"):
        return pet
    now = time.time()
    if pet["last_update"] == 0.0:
        pet["last_update"] = now
        return pet
    elapsed = min((now - pet["last_update"]) / 3600.0, 168.0)
    if elapsed < 0.001:
        return pet

    pet["hunger"]      = max(0.0, pet["hunger"]      - 2.5 * elapsed)
    pet["happiness"]   = max(0.0, pet["happiness"]   - 1.2 * elapsed)
    pet["cleanliness"] = max(0.0, pet["cleanliness"] - 0.8 * elapsed)

    if not pet["is_sick"]:
        risk = 0.0
        if pet["hunger"]      < 25: risk += 0.25
        if pet["cleanliness"] < 20: risk += 0.20
        if risk > 0 and random.random() < (1 - (1 - risk) ** elapsed):
            pet["is_sick"] = True
            pet["message"] = "なんか体がだるい…くすりほしいな…"

    if pet["is_sick"]:
        pet["sick_hours"] = pet.get("sick_hours", 0.0) + elapsed

    decay = 0.10 * elapsed
    if pet["hunger"] < 30:  decay += 1.20 * elapsed
    if pet["hunger"] < 5:   decay += 2.50 * elapsed
    if pet["is_sick"]:      decay += 1.20 * elapsed
    if pet["happiness"] < 20: decay += 0.60 * elapsed
    pet["health"] = max(0.0, pet["health"] - decay)

    if pet["health"] <= 0.0:
        pet["is_dead"] = True
        pet["health"]  = 0.0
        pet["message"] = "ガウスくんは…天国へ旅立ちました…"
        pet["last_update"] = now
        return pet

    pet["age_hours"] = pet.get("age_hours", 0.0) + elapsed

    # 部屋の滞在時間を記録（性格決定に使用）
    wp = pet.get("room", _DEF_ROOM).get("wallpaper", "lab")
    rt = pet.setdefault("room_time", {})
    rt[wp] = rt.get(wp, 0.0) + elapsed

    _check_evolution(pet)
    pet["last_update"] = now
    return pet


def _determine_personality(pet: dict) -> str:
    fh = pet.get("food_history", {})
    th = pet.get("toy_history", {})
    rt = pet.get("room_time", {})
    scores = {
        "researcher": fh.get("vegetable",0)*3 + th.get("puzzle",0)*5 + th.get("music_box",0) + rt.get("lab",0)*2 + rt.get("space",0),
        "athlete":    fh.get("fish",0)*3     + th.get("ball",0)*5   + th.get("robot_toy",0)*2 + rt.get("forest",0)*2,
        "gourmet":    fh.get("fancy_food",0)*3 + fh.get("special_cake",0)*8 + fh.get("fish",0)*2 + rt.get("city",0)*2,
        "sweet":      fh.get("candy",0)*4    + th.get("toy",0)*4    + th.get("music_box",0)*2 + rt.get("flower",0)*3,
        "mystic":     th.get("music_box",0)*4 + th.get("robot_toy",0)*4 + rt.get("space",0)*5 + rt.get("ocean",0)*3 + th.get("puzzle",0)*2,
    }
    return max(scores, key=scores.get)


def _check_evolution(pet: dict):
    evolutions = [
        (0,  24,   0, 1, "たまごが孵化したよ！ちびガウス誕生！🐣"),
        (1,  72,  50, 2, "ちびガウスが成長した！こガウスになったよ！"),
        (2, 168, 160, 3, "立派に育った！ガウス博士になったよ！🧑‍🔬"),
        (3, 336, 450, 4, None),
    ]
    for from_s, min_age, min_score, to_s, msg in evolutions:
        if pet["stage"] == from_s and pet["age_hours"] >= min_age and pet["care_score"] >= min_score:
            pet["stage"] = to_s
            if to_s == 4:
                p = _determine_personality(pet)
                pet["personality"] = p
                pname = PERSONALITIES[p]["name"]
                msg = f"ついに最終進化！✨ {pname}のスーパーガウスになったよ！！✨"
                col = pet.setdefault("collection", _DEF_COLLECTION.copy())
                if p not in col["personalities"]:
                    col["personalities"].append(p)
            pet["message"] = msg
            pet["last_action_time"] = time.time()
            break

# ══════════════════════════════════════════════════════════════════════════════
#  気分・メッセージ
# ══════════════════════════════════════════════════════════════════════════════

def get_mood(pet: dict) -> str:
    if pet.get("is_dead"):            return "dead"
    if pet["stage"] == 0:             return "egg"
    if pet.get("is_sick") or pet["health"] < 20: return "sick"
    import datetime as _dt
    if 0 <= _dt.datetime.now().hour < 7: return "sleeping"
    if pet["hunger"]    < 25:         return "hungry"
    if pet["happiness"] < 25:         return "unhappy"
    if pet["hunger"] > 65 and pet["happiness"] > 65 and pet["cleanliness"] > 40:
                                      return "happy"
    return "normal"

def get_mood_message(pet: dict, mood: str) -> str:
    p = PERSONALITIES.get(pet.get("personality", ""), {})
    defaults = {
        "dead":     "……",
        "egg":      "あたたかく見守ってね…",
        "sick":     "うぅ、気持ち悪い…くすりが欲しいな",
        "sleeping": "すやすや…（おやすみ中）",
        "hungry":   "おなかすいた！ごはんちょうだい！！",
        "unhappy":  "かまってよ〜…つまんない",
        "happy":    p.get("happy_msg",  "しあわせ〜！今日もいい日だね！"),
        "normal":   p.get("normal_msg", "ふつうにすごしてるよ。"),
    }
    if (time.time() - pet.get("last_action_time", 0)) < 5:
        return pet.get("message", defaults.get(mood, "…"))
    return defaults.get(mood, "…")

# ══════════════════════════════════════════════════════════════════════════════
#  画像生成（Pillow）
# ══════════════════════════════════════════════════════════════════════════════

def _circle(draw, cx, cy, r, fill, outline=None, width=2):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=width)

def _oval(draw, cx, cy, rx, ry, fill, outline=None, width=2):
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=fill, outline=outline, width=width)

def _body_color(stage: int, mood: str, personality=None):
    if mood == "dead": return (90,90,90),   (60,60,60)
    if mood == "sick": return (130,175,100),(80,130,60)
    if stage == 4 and personality and personality in PERSONALITIES:
        return PERSONALITIES[personality]["color"]
    palette = [
        ((245,240,220),(200,185,150)),
        ((90,185,225),(50,130,180)),
        ((130,80,210),(85,45,160)),
        ((55,165,100),(30,110,65)),
        ((255,210,40),(200,155,0)),
    ]
    return palette[min(stage, 4)]

def _draw_eyes(draw, lx, ly, rx, ry, mood, r=11):
    W, BK = (255,255,255), (20,20,20)
    if mood == "sleeping":
        for ex, ey in [(lx,ly),(rx,ry)]:
            draw.arc([ex-r,ey-r//3,ex+r,ey+r//3], 0, 180, fill=BK, width=3)
        return
    if mood == "dead":
        for ex, ey in [(lx,ly),(rx,ry)]:
            draw.line([ex-r,ey-r,ex+r,ey+r], fill=BK, width=3)
            draw.line([ex+r,ey-r,ex-r,ey+r], fill=BK, width=3)
        return
    if mood == "sick":
        for ex, ey in [(lx,ly),(rx,ry)]:
            _circle(draw, ex, ey, r, W, BK, 2)
            draw.arc([ex-r//2,ey-r//2,ex+r//2,ey+r//2], 0, 270, fill=BK, width=2)
        return
    if mood == "happy":
        for ex, ey in [(lx,ly),(rx,ry)]:
            draw.arc([ex-r,ey-r//2,ex+r,ey+r//2], 200, 340, fill=BK, width=3)
        return
    if mood in ("hungry","unhappy"):
        for ex, ey in [(lx,ly),(rx,ry)]:
            _circle(draw, ex, ey, r, W, BK, 2)
            _circle(draw, ex+2, ey+3, r//2, BK)
        return
    for ex, ey in [(lx,ly),(rx,ry)]:
        _circle(draw, ex, ey, r, W, BK, 2)
        _circle(draw, ex+3, ey, r//2, BK)
        _circle(draw, ex+r//2, ey-r//2+2, 2, W)

def _draw_mouth(draw, cx, cy, mood, s=1.0):
    r, BK = int(14*s), (20,20,20)
    if mood == "happy":
        draw.arc([cx-r,cy-r//2,cx+r,cy+r//2], 0, 180, fill=BK, width=3)
        for chx in [cx-int(28*s), cx+int(28*s)]:
            draw.ellipse([chx-9,cy-4,chx+9,cy+6], fill=(255,140,140,120))
    elif mood in ("hungry","unhappy","sick","dead"):
        draw.arc([cx-r,cy,cx+r,cy+r], 0, 180, fill=BK, width=3)
    elif mood == "sleeping":
        draw.ellipse([cx-7,cy-3,cx+7,cy+5], fill=BK)
    else:
        draw.line([cx-int(r*0.6),cy,cx+int(r*0.6),cy], fill=BK, width=3)

def _sparkles(draw, SIZE, color=(255,215,0)):
    rng = random.Random(42)
    for _ in range(10):
        sx, sy = rng.randint(20,SIZE-20), rng.randint(20,SIZE-20)
        r = rng.randint(3,8)
        for dx, dy in [(r,0),(-r,0),(0,r),(0,-r)]:
            draw.line([sx,sy,sx+dx,sy+dy], fill=color, width=2)

def _draw_room_bg(draw, SIZE, wp_key):
    rng = random.Random(hash(wp_key) % 10000)
    if wp_key == "space":
        for _ in range(35):
            sx, sy = rng.randint(8,SIZE-8), rng.randint(8,SIZE-8)
            r = rng.choice([1,1,1,2])
            draw.ellipse([sx-r,sy-r,sx+r,sy+r], fill=(200,200,255))
    elif wp_key == "forest":
        for cx2, cy2 in [(22,22),(SIZE-22,22),(22,SIZE-22),(SIZE-22,SIZE-22)]:
            for _ in range(4):
                lx = cx2+rng.randint(-18,18); ly = cy2+rng.randint(-18,18)
                draw.ellipse([lx-10,ly-6,lx+10,ly+6], fill=(30,110,40))
    elif wp_key == "flower":
        for _ in range(18):
            fx, fy = rng.randint(12,SIZE-12), rng.randint(12,SIZE-12)
            draw.ellipse([fx-5,fy-5,fx+5,fy+5], fill=(210,90,150))
            draw.ellipse([fx-2,fy-2,fx+2,fy+2], fill=(255,210,0))
    elif wp_key == "city":
        for _ in range(22):
            wx, wy = rng.randint(8,SIZE-8), rng.randint(8,SIZE-8)
            ww, wh = rng.randint(4,8), rng.randint(4,8)
            draw.rectangle([wx,wy,wx+ww,wy+wh], fill=(255,220,80))
    elif wp_key == "ocean":
        for _ in range(22):
            bx, by = rng.randint(8,SIZE-8), rng.randint(8,SIZE-8)
            br = rng.randint(3,9)
            draw.ellipse([bx-br,by-br,bx+br,by+br], outline=(60,160,240), width=1)
    elif wp_key == "lab":
        for _ in range(12):
            cx2, cy2 = rng.randint(8,SIZE-8), rng.randint(8,SIZE-8)
            draw.ellipse([cx2-3,cy2-3,cx2+3,cy2+3], fill=(25,60,130))

def _draw_furniture(draw, key, px, py, size=38):
    if key == "none":
        return
    if key == "bookshelf":
        colors = [(200,50,50),(50,80,200),(50,180,80)]
        for i, c in enumerate(colors):
            draw.rectangle([px,py+i*11,px+size,py+i*11+9], fill=c, outline=(15,15,15), width=1)
        draw.rectangle([px-2,py-2,px+size+2,py+34], outline=(100,70,40), width=2)
    elif key == "plant":
        draw.polygon([(px+size//2-8,py+size-2),(px+size//2+8,py+size-2),(px+size//2+6,py+size//2+5),(px+size//2-6,py+size//2+5)], fill=(150,85,45))
        draw.ellipse([px+2,py+5,px+size//2+4,py+size//2+4], fill=(35,150,55))
        draw.ellipse([px+size//2-4,py+2,px+size-2,py+size//2+2], fill=(45,165,60))
    elif key == "aquarium":
        draw.rectangle([px,py+3,px+size,py+size], fill=(8,28,80), outline=(0,140,200), width=2)
        draw.line([px+3,py+8,px+size-3,py+8], fill=(30,130,255), width=2)
        draw.ellipse([px+size//2-8,py+size//2-4,px+size//2+8,py+size//2+4], fill=(255,140,50))
    elif key == "telescope":
        draw.rectangle([px+size//2-5,py+5,px+size-3,py+size-8], fill=(140,120,90), outline=(90,70,50), width=1)
        draw.ellipse([px+size//2-12,py,px+size//2+2,py+16], fill=(60,60,80), outline=(90,90,110), width=2)
        draw.polygon([(px+size//2-1,py+size-8),(px+size//2-12,py+size),(px+size//2+10,py+size)], fill=(100,80,60))
    elif key == "sofa":
        draw.rectangle([px+3,py+size//2+2,px+size-3,py+size], fill=(100,60,180), outline=(70,40,130), width=1)
        draw.rectangle([px+3,py+size//3,px+size-3,py+size//2+4], fill=(120,75,200), outline=(80,50,150), width=1)
        for ax in [px, px+size-10]:
            draw.rectangle([ax,py+size//3-3,ax+10,py+size], fill=(90,55,160), outline=(65,35,120), width=1)
    elif key == "computer":
        draw.rectangle([px+2,py,px+size-2,py+size-12], fill=(20,25,40), outline=(0,180,255), width=2)
        draw.rectangle([px+6,py+4,px+size-6,py+size-16], fill=(10,80,160))
        draw.rectangle([px+size//2-5,py+size-12,px+size//2+5,py+size], fill=(30,40,60))
        draw.rectangle([px+6,py+size-3,px+size-6,py+size], fill=(30,40,60))

def _stage_egg(draw, cx, cy, mood):
    fill, out = _body_color(0, mood)
    _oval(draw, cx, cy+5, 62, 78, fill, out, 3)
    draw.ellipse([cx-30,cy-40,cx-8,cy-20], fill=tuple(min(v+15,255) for v in fill))
    for sx, sy, sr in [(cx+22,cy-15,9),(cx-28,cy+12,7),(cx+12,cy+30,6)]:
        _circle(draw, sx, sy, sr, tuple(max(v-20,0) for v in fill))
    age = st.session_state.get("_pet_age", 0)
    if age > 20:
        draw.line([cx+15,cy-30,cx+30,cy-10,cx+20,cy+5], fill=(160,140,110), width=2)

def _stage_baby(draw, cx, cy, mood):
    fill, out = _body_color(1, mood)
    _circle(draw, cx, cy+10, 68, fill, out, 3)
    for ax, ay in [(cx-78,cy+18),(cx+78,cy+18)]:
        _circle(draw, ax, ay, 22, fill, out, 2)
    _draw_eyes(draw, cx-24, cy-2, cx+24, cy-2, mood, 12)
    _draw_mouth(draw, cx, cy+26, mood, 0.85)

def _stage_child(draw, cx, cy, mood):
    fill, out = _body_color(2, mood)
    _oval(draw, cx, cy+5, 72, 78, fill, out, 3)
    for ax, ay in [(cx-84,cy+8),(cx+84,cy+8)]:
        _oval(draw, ax, ay, 24, 20, fill, out, 2)
    for fx, fy in [(cx-30,cy+80),(cx+30,cy+80)]:
        _oval(draw, fx, fy, 22, 14, fill, out, 2)
    for gx in [cx-24, cx+24]:
        draw.rectangle([gx-15,cy-22,gx+15,cy-5], outline=(180,220,255), width=2)
    draw.line([cx-9,cy-13,cx+9,cy-13], fill=(180,220,255), width=2)
    _draw_eyes(draw, cx-24, cy-8, cx+24, cy-8, mood, 10)
    _draw_mouth(draw, cx, cy+24, mood, 0.9)

def _stage_adult(draw, cx, cy, mood):
    fill, out = _body_color(3, mood)
    _oval(draw, cx, cy+8, 75, 80, (235,242,255),(190,205,225), 2)
    _oval(draw, cx, cy+5, 65, 72, fill, out, 3)
    for ax, ay in [(cx-82,cy+5),(cx+82,cy+5)]:
        _oval(draw, ax, ay, 24, 22, fill, out, 2)
    for fx, fy in [(cx-30,cy+80),(cx+30,cy+80)]:
        _oval(draw, fx, fy, 22, 15, (235,242,255),(190,205,225), 2)
    draw.rectangle([cx-42,cy-98,cx+42,cy-62], fill=(20,20,35), outline=(10,10,20), width=2)
    draw.rectangle([cx-54,cy-70,cx+54,cy-62], fill=(20,20,35), outline=(10,10,20), width=2)
    draw.line([cx+42,cy-82,cx+58,cy-55], fill=(255,215,0), width=2)
    _circle(draw, cx+58, cy-52, 6, (255,215,0))
    draw.polygon([(cx,cy+35),(cx-9,cy+56),(cx+9,cy+56)], fill=(210,45,45))
    _draw_eyes(draw, cx-24, cy-10, cx+24, cy-10, mood, 11)
    _draw_mouth(draw, cx, cy+20, mood, 1.0)

def _stage_special(draw, cx, cy, mood, SIZE, personality=None):
    fill, out = _body_color(4, mood, personality)
    for gr in range(90, 68, -5):
        a = int(255*(90-gr)/22)
        try:
            tmp = Image.new("RGBA", (SIZE,SIZE), (0,0,0,0))
            ImageDraw.Draw(tmp).ellipse([cx-gr,cy-gr+10,cx+gr,cy+gr+10], fill=(*fill, a))
            draw._image.alpha_composite(tmp)
        except Exception:
            pass
    _circle(draw, cx, cy+10, 68, fill, out, 3)
    for ax, ay in [(cx-82,cy+10),(cx+82,cy+10)]:
        _circle(draw, ax, ay, 24, fill, out, 2)

    # 性格別の帽子/アクセサリ
    if personality == "researcher":
        draw.rectangle([cx-42,cy-98,cx+42,cy-62], fill=(20,20,55), outline=(60,60,160), width=2)
        draw.rectangle([cx-54,cy-70,cx+54,cy-62], fill=(20,20,55), outline=(60,60,160), width=2)
        draw.line([cx+42,cy-82,cx+58,cy-55], fill=(120,160,255), width=2)
        _circle(draw, cx+58, cy-52, 6, (120,160,255))
    elif personality == "athlete":
        draw.arc([cx-42,cy-80,cx+42,cy-40], 180, 0, fill=(80,210,100), width=10)
        _circle(draw, cx, cy-78, 9, (80,210,100))
    elif personality == "gourmet":
        draw.rectangle([cx-28,cy-90,cx+28,cy-62], fill=(240,240,240), outline=(180,180,180), width=1)
        draw.ellipse([cx-34,cy-102,cx+34,cy-80], fill=(240,240,240), outline=(180,180,180), width=1)
        for px2 in [cx-16, cx, cx+16]:
            draw.ellipse([px2-4,cy-104,px2+4,cy-92], fill=(240,240,240))
    elif personality == "sweet":
        for pts in [[(cx-42,cy-95),(cx-10,cy-80),(cx-30,cy-68),(cx-42,cy-95)],
                    [(cx+42,cy-95),(cx+10,cy-80),(cx+30,cy-68),(cx+42,cy-95)]]:
            draw.polygon(pts, fill=(255,100,160), outline=(200,60,120), width=2)
        _circle(draw, cx, cy-80, 9, (255,200,220))
    elif personality == "mystic":
        draw.polygon([(cx,cy-116),(cx-30,cy-70),(cx+30,cy-70)], fill=(80,20,150), outline=(130,50,230), width=2)
        draw.ellipse([cx-36,cy-73,cx+36,cy-61], fill=(100,30,170), outline=(150,55,245), width=2)
        for sx, sy in [(cx-8,cy-102),(cx+12,cy-90)]:
            draw.ellipse([sx-5,sy-5,sx+5,sy+5], fill=(255,220,0))
    else:
        pts = [(cx-42,cy-77),(cx-42,cy-98),(cx-20,cy-88),(cx,cy-105),(cx+20,cy-88),(cx+42,cy-98),(cx+42,cy-77)]
        draw.polygon(pts, fill=(255,180,0), outline=(200,130,0))
        for jx,jy,jc in [(cx-30,cy-84,(255,80,80)),(cx,cy-96,(80,200,255)),(cx+30,cy-84,(80,255,120))]:
            _circle(draw, jx, jy, 6, jc)

    _draw_eyes(draw, cx-24, cy+5, cx+24, cy+5, mood, 13)
    _draw_mouth(draw, cx, cy+30, mood, 1.1)
    sc = {"researcher":(150,200,255),"athlete":(100,255,150),"gourmet":(255,200,100),
          "sweet":(255,150,200),"mystic":(200,100,255)}.get(personality,(255,215,0))
    _sparkles(draw, SIZE, sc)


def generate_pet_image(pet: dict) -> bytes:
    SIZE  = 280
    stage = pet["stage"]
    mood  = get_mood(pet)
    pers  = pet.get("personality")
    wp_key = pet.get("room", _DEF_ROOM).get("wallpaper", "lab")
    f1_key = pet.get("room", _DEF_ROOM).get("furniture1", "computer")
    f2_key = pet.get("room", _DEF_ROOM).get("furniture2", "none")
    st.session_state["_pet_age"] = pet.get("age_hours", 0)

    wp = WALLPAPERS.get(wp_key, WALLPAPERS["lab"])
    img  = Image.new("RGBA", (SIZE,SIZE), tuple(wp["bg"]))
    draw = ImageDraw.Draw(img)

    try:
        draw.rounded_rectangle([4,4,SIZE-4,SIZE-4], radius=22,
                               fill=tuple(wp["card"]), outline=tuple(wp["ol"]), width=2)
    except AttributeError:
        draw.rectangle([4,4,SIZE-4,SIZE-4], fill=tuple(wp["card"]), outline=tuple(wp["ol"]), width=2)

    _draw_room_bg(draw, SIZE, wp_key)
    _draw_furniture(draw, f1_key,  6,   SIZE-52, 40)
    _draw_furniture(draw, f2_key,  SIZE-50, SIZE-52, 40)

    cx, cy = SIZE//2, SIZE//2 + 10
    if   stage == 0: _stage_egg(draw, cx, cy, mood)
    elif stage == 1: _stage_baby(draw, cx, cy, mood)
    elif stage == 2: _stage_child(draw, cx, cy, mood)
    elif stage == 3: _stage_adult(draw, cx, cy, mood)
    else:            _stage_special(draw, cx, cy, mood, SIZE, pers)

    if mood == "happy" and stage > 0:
        for hx, hy in [(cx-65,cy-75),(cx+58,cy-70)]:
            draw.text((hx,hy), "♥", fill=(255,100,150))
    if mood == "hungry":
        draw.text((cx+55,cy-40), "💧", fill=(100,180,255))
    if mood == "sleeping" and stage > 0:
        for zx, zy in [(cx+55,cy-55),(cx+68,cy-70),(cx+80,cy-85)]:
            draw.text((zx,zy), "z", fill=(160,210,255))

    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
#  報酬関数
# ══════════════════════════════════════════════════════════════════════════════

def check_login_bonus(pet: dict) -> bool:
    return (time.time() - pet.get("last_login_bonus", 0.0)) >= 86400

def claim_login_bonus(pet: dict):
    given = {"normal_food": 2}
    roll = random.random()
    if   roll < 0.05: given["special_cake"] = 1
    elif roll < 0.15: given["robot_toy"]    = 1
    elif roll < 0.25: given["medicine"]     = 1
    elif roll < 0.40: given["fish"]         = 2
    elif roll < 0.55: given["shampoo"]      = 1
    elif roll < 0.70: given["vegetable"]    = 2
    else:             given["candy"]        = 2
    for k, v in given.items():
        pet["items"][k] = pet["items"].get(k, 0) + v
    pet["last_login_bonus"] = time.time()
    pet["message"] = "ログインボーナス受け取り！" + "、".join(f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k,v in given.items())
    pet["last_action_time"] = time.time()
    return pet, given

def check_step_rewards(pet: dict) -> dict:
    if not PIPELINE_STATE_FILE.exists():
        return {}
    try:
        state = json.loads(PIPELINE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    step_times = state.get("step_times", {})
    claimed    = pet.get("step_reward_claimed", {})
    return {step: reward for step, reward in STEP_REWARDS.items()
            if step_times.get(step) and step_times[step] > claimed.get(step, 0.0)}

def check_milestones(pet: dict) -> list:
    score = pet.get("care_score", 0.0)
    claimed = pet.get("milestones_claimed", [])
    return [(t, items, msg) for t, items, msg in MILESTONES
            if score >= t and t not in claimed]

def check_prestige_reward(pet: dict) -> int:
    try:
        if not MINIGAME_SAVE.exists():
            return 0
        data = json.loads(MINIGAME_SAVE.read_text(encoding="utf-8"))
        return max(0, int(data.get("prestige_points", 0)) - int(pet.get("prestige_reward_claimed", 0)))
    except Exception:
        return 0

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
  html,body,[class*="css"]{font-family:'Share Tech Mono',monospace;background:#0a0e1a;color:#e0e6f0;}
  .block-container{padding:1.5rem 2rem;}
  .speech-bubble{background:#0d1b2e;border:1px solid #1a3a5c;border-radius:12px;
    padding:10px 16px;font-size:.9rem;color:#a0d0f0;margin-top:8px;}
  .action-title{font-size:.7rem;letter-spacing:.2em;color:#4a90b8;
    text-transform:uppercase;margin:12px 0 6px 0;}
  .item-badge{display:inline-block;background:#0d1b2e;border:1px solid #1a3a5c;
    border-radius:8px;padding:3px 10px;margin:3px;font-size:.78rem;}
  .reward-banner{background:linear-gradient(135deg,#1a2a10,#2a3a15);
    border:1px solid #4a8a30;border-radius:12px;padding:12px 18px;margin:10px 0;color:#a0e060;}
  .dead-banner{background:linear-gradient(135deg,#2a0a0a,#1a0505);
    border:1px solid #8a2020;border-radius:12px;padding:16px 20px;margin:10px 0;
    color:#e06060;text-align:center;}
  .room-card{background:#0d1b2e;border:1px solid #1a3a5c;border-radius:10px;
    padding:10px;margin:4px;text-align:center;font-size:.8rem;}
  .room-card.owned{border-color:#2a5a8c;color:#a0d0f0;}
  .room-card.active{border-color:#00e5ff;box-shadow:0 0 8px #00e5ff44;color:#00e5ff;}
  .coll-item{display:inline-block;width:70px;height:70px;margin:6px;border-radius:10px;
    border:1px solid #1a3a5c;background:#0a1520;text-align:center;line-height:70px;font-size:1.8rem;}
  .coll-item.got{border-color:#2a7a8c;background:#0d2030;}
  .coll-item.not{filter:grayscale(90%) opacity(30%);}
  div[data-testid="stButton"]>button{
    background:#0d1b2e;border:1px solid #1a3a5c;border-radius:10px;
    color:#e0e6f0;font-family:'Share Tech Mono',monospace;font-size:.78rem;
    letter-spacing:.05em;padding:.55rem .4rem;width:100%;
    transition:border-color .2s,box-shadow .2s;}
  div[data-testid="stButton"]>button:hover{
    border-color:#00e5ff;box-shadow:0 0 10px #00e5ff33;color:#00e5ff;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  データ読み込み・更新
# ══════════════════════════════════════════════════════════════════════════════

if "pet" not in st.session_state:
    st.session_state.pet = load_pet()

pet = st.session_state.pet
pet = update_pet(pet)
st.session_state.pet = pet
save_pet(pet)
mood = get_mood(pet)

# ══════════════════════════════════════════════════════════════════════════════
#  アクション共通処理
# ══════════════════════════════════════════════════════════════════════════════

def do_action(effect: dict, msg: str, cost_item: str | None = None):
    items = pet["items"]
    if cost_item and items.get(cost_item, 0) <= 0:
        st.toast(f"{ITEMS[cost_item]['name']}が足りない！")
        return
    for stat in ("hunger","happiness","cleanliness","health"):
        if stat in effect:
            pet[stat] = max(0.0, min(100.0, pet[stat] + effect[stat]))
    if effect.get("cure_sick"):
        pet["is_sick"] = False; pet["sick_hours"] = 0.0
    if cost_item:
        pet["items"][cost_item] -= 1
        # 履歴・コレクション記録
        if cost_item in FOOD_ITEMS:
            pet.setdefault("food_history",{})[cost_item] = pet["food_history"].get(cost_item,0) + 1
            col = pet.setdefault("collection", _DEF_COLLECTION.copy())
            if cost_item not in col["foods"]:
                col["foods"].append(cost_item)
        elif cost_item in TOY_ITEMS:
            pet.setdefault("toy_history",{})[cost_item] = pet["toy_history"].get(cost_item,0) + 1
            col = pet.setdefault("collection", _DEF_COLLECTION.copy())
            if cost_item not in col["toys"]:
                col["toys"].append(cost_item)
    care_earned = effect.get("care", 0)
    pet["care_score"] = pet.get("care_score", 0.0) + care_earned
    pet["stars"]      = pet.get("stars", 0) + care_earned
    pet["message"]    = msg
    pet["last_action_time"] = time.time()
    _check_evolution(pet)
    st.session_state.pet = pet
    save_pet(pet)
    st.session_state.pop("_feeding", None)
    st.session_state.pop("_playing", None)
    st.session_state.pop("_caring",  None)
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  メイン UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 🐾 ガウスくんを育てよう")
st.divider()

col_img, col_info = st.columns([1, 1.6], gap="large")

# ── 左: ペット画像 ───────────────────────────────────────────────────────────
with col_img:
    st.image(generate_pet_image(pet), use_container_width=True)
    st.markdown(f'<div class="speech-bubble">💬 {get_mood_message(pet, mood)}</div>',
                unsafe_allow_html=True)
    age_d = int(pet["age_hours"]//24); age_h = int(pet["age_hours"]%24)
    pers_txt = PERSONALITIES[pet["personality"]]["name"] if pet.get("personality") else STAGE_NAMES[pet["stage"]]
    st.markdown(
        f'<div style="font-size:.75rem;color:#4a90b8;margin-top:8px;text-align:center;">'
        f'{pers_txt}　|　年齢 {age_d}日{age_h}時間　|　'
        f'世話スコア {int(pet["care_score"])}　|　⭐ {pet.get("stars",0)}'
        f'</div>', unsafe_allow_html=True)

# ── 右: ステータス + アクション ───────────────────────────────────────────────
with col_info:
    st.markdown('<div class="action-title">Status</div>', unsafe_allow_html=True)

    def stat_bar(label, icon, value, color="#00aaff", warn=30, danger=15):
        pct = max(0.0, min(100.0, value))
        col = color if pct > warn else ("#ffaa00" if pct > danger else "#ff4444")
        filled = int(pct/5)
        bar = "█"*filled + "░"*(20-filled)
        st.markdown(
            f'<div>{icon} <span style="font-size:.78rem;color:#8ab0c8;">{label}</span>'
            f'<span style="font-size:.75rem;color:{col};margin-left:8px;">{bar} {pct:.0f}%</span></div>',
            unsafe_allow_html=True)

    stat_bar("体力","❤️", pet["health"],      "#ff6688", warn=30, danger=20)
    stat_bar("お腹","🍎", pet["hunger"],      "#00ccaa", warn=30, danger=15)
    stat_bar("機嫌","😊", pet["happiness"],   "#ffcc00", warn=25, danger=10)
    stat_bar("清潔","🛁", pet["cleanliness"], "#44aaff", warn=25, danger=10)

    if pet.get("is_sick"):
        st.markdown('<span style="color:#ffaa44;font-size:.85rem;">⚠️ 病気中 — くすりをあげてください</span>',
                    unsafe_allow_html=True)

    if pet.get("is_dead"):
        st.markdown(
            '<div class="dead-banner"><div style="font-size:1.5rem;margin-bottom:8px;">💀</div>'
            'ガウスくんは天国へ旅立ちました…<br>'
            '<span style="font-size:.8rem;color:#b08080;">もっとたくさん遊んであげてね</span></div>',
            unsafe_allow_html=True)
        if st.button("🥚 あたらしいたまごからやり直す", use_container_width=True):
            old_col = pet.get("collection", _DEF_COLLECTION.copy())
            old_stars = pet.get("stars", 0) // 3
            new_pet = {k: (v.copy() if isinstance(v,dict) else (list(v) if isinstance(v,list) else v))
                       for k,v in DEFAULT_PET.items()}
            new_pet["last_update"]  = time.time()
            new_pet["collection"]   = old_col
            new_pet["stars"]        = old_stars
            for k in new_pet["items"]:
                new_pet["items"][k] = pet["items"].get(k,0) // 2
            st.session_state.pet = new_pet
            save_pet(new_pet)
            st.rerun()

    else:
        # ── アクションボタン 4つ ──
        st.markdown('<div class="action-title">Actions</div>', unsafe_allow_html=True)
        items = pet["items"]
        can_act = not pet.get("is_sick")

        ac1, ac2, ac3, ac4 = st.columns(4)
        total_food = sum(items.get(k,0) for k in FOOD_ITEMS)
        total_toy  = sum(items.get(k,0) for k in TOY_ITEMS)
        has_shampoo = items.get("shampoo",0)
        total_care  = items.get("medicine",0) + items.get("vitamins",0)

        with ac1:
            if st.button(f"🍽️ ごはん\n（{total_food}個）", disabled=(total_food==0), key="btn_food"):
                st.session_state._feeding = not st.session_state.get("_feeding", False)
                st.session_state.pop("_playing", None); st.session_state.pop("_caring", None)
                st.rerun()
        with ac2:
            lbl = f"🎮 あそぶ\n（おもちゃ{total_toy}個）" if total_toy > 0 else "🎮 あそぶ\n（手ぶら）"
            if st.button(lbl, disabled=not can_act, key="btn_play"):
                st.session_state._playing = not st.session_state.get("_playing", False)
                st.session_state.pop("_feeding", None); st.session_state.pop("_caring", None)
                st.rerun()
        with ac3:
            if st.button(f"🛁 おそうじ\n（シャン{has_shampoo}）", key="btn_clean"):
                if has_shampoo > 0:
                    do_action({**CARE_ITEMS["shampoo"]}, "きれいになったよ！", cost_item="shampoo")
                else:
                    do_action({"cleanliness":15,"care":2}, "てでごしごし…きれいになったよ！")
        with ac4:
            if st.button(f"💊 お世話\n（{total_care}個）", disabled=(total_care==0), key="btn_care"):
                st.session_state._caring = not st.session_state.get("_caring", False)
                st.session_state.pop("_feeding", None); st.session_state.pop("_playing", None)
                st.rerun()

        # ── ごはん選択 ──
        if st.session_state.get("_feeding"):
            st.markdown('<div class="action-title">🍽️ どのごはんをあげますか？</div>', unsafe_allow_html=True)
            food_keys = list(FOOD_ITEMS.keys())
            rows = [food_keys[:3], food_keys[3:]]
            for row in rows:
                cols = st.columns(len(row)+1)
                for col, fk in zip(cols, row):
                    fd = FOOD_ITEMS[fk]; cnt = items.get(fk,0)
                    with col:
                        if st.button(f"{fd['icon']} {fd['name']}\n×{cnt}", disabled=(cnt==0), key=f"eat_{fk}"):
                            do_action({**fd}, f"{fd['name']}を食べた！{'おいしい！！' if fk in ('fancy_food','special_cake') else ''}", cost_item=fk)
            if st.button("❌ キャンセル", key="cancel_food"):
                st.session_state.pop("_feeding", None); st.rerun()

        # ── おもちゃ選択 ──
        if st.session_state.get("_playing"):
            st.markdown('<div class="action-title">🎮 どのおもちゃで遊びますか？</div>', unsafe_allow_html=True)
            toy_keys = list(TOY_ITEMS.keys())
            row1, row2 = toy_keys[:3], toy_keys[3:]
            for row in [row1, row2]:
                cols = st.columns(len(row)+1)
                for col, tk in zip(cols, row):
                    td = TOY_ITEMS[tk]; cnt = items.get(tk,0)
                    with col:
                        if st.button(f"{td['icon']} {td['name']}\n×{cnt}", disabled=(cnt==0), key=f"play_{tk}"):
                            do_action({**td}, f"{td['name']}で遊んだ！たのし〜！！", cost_item=tk)
            # 手ぶら
            bare_cols = st.columns(3)
            with bare_cols[0]:
                if st.button("🤲 手ぶらで遊ぶ", key="play_bare"):
                    do_action({"happiness":8,"hunger":-3,"cleanliness":-2,"care":2}, "手ぶらで遊んだよ。まあ楽しかった…")
            with bare_cols[1]:
                if st.button("❌ キャンセル", key="cancel_play"):
                    st.session_state.pop("_playing", None); st.rerun()

        # ── お世話選択 ──
        if st.session_state.get("_caring"):
            st.markdown('<div class="action-title">💊 お世話アイテムを選んでください</div>', unsafe_allow_html=True)
            care_cols = st.columns(3)
            for col, ck in zip(care_cols, ["medicine","vitamins"]):
                cd = CARE_ITEMS[ck]; cnt = items.get(ck,0)
                with col:
                    label_extra = "（病気を治す）" if ck == "medicine" else "（体力を増やす）"
                    if st.button(f"{cd['icon']} {cd['name']}\n×{cnt} {label_extra}", disabled=(cnt==0), key=f"care_{ck}"):
                        msg = "くすりのんだ！元気になってきたよ！" if ck=="medicine" else "ビタミン剤を飲んだ！元気いっぱい！"
                        do_action({**cd}, msg, cost_item=ck)
            with care_cols[2]:
                if st.button("❌ キャンセル", key="cancel_care"):
                    st.session_state.pop("_caring", None); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  報酬バナー
# ══════════════════════════════════════════════════════════════════════════════

if not pet.get("is_dead"):
    # ログインボーナス
    if check_login_bonus(pet):
        st.markdown('<div class="reward-banner">🌅 <b>ログインボーナスが届いています！</b><br>'
                    '<span style="font-size:.85rem;">ごはん×2＋ランダムアイテム</span></div>', unsafe_allow_html=True)
        if st.button("🌅 ログインボーナスを受け取る"):
            pet, _ = claim_login_bonus(pet)
            st.session_state.pet = pet; save_pet(pet); st.rerun()

    # 研究ステップ報酬
    _SJA = {"extracting":"フレーム抽出","colmap":"COLMAP/HLoc","training":"3DGS学習"}
    for _step, _reward in check_step_rewards(pet).items():
        _rs = "、".join(f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k,v in _reward.items())
        st.markdown(f'<div class="reward-banner">🔬 <b>{_SJA[_step]}完了報酬！</b><br>'
                    f'<span style="font-size:.85rem;">{_rs}</span></div>', unsafe_allow_html=True)
        if st.button(f"🎁 受け取る（{_rs}）", key=f"claim_step_{_step}"):
            try:
                _t = json.loads(PIPELINE_STATE_FILE.read_text(encoding="utf-8")).get("step_times",{}).get(_step,time.time())
            except Exception:
                _t = time.time()
            pet.setdefault("step_reward_claimed",{})[_step] = _t
            for k,v in _reward.items(): pet["items"][k] = pet["items"].get(k,0)+v
            pet["message"] = f"研究報酬受け取り！{_rs}"
            pet["last_action_time"] = time.time()
            st.session_state.pet = pet; save_pet(pet); st.rerun()

    # マイルストーン
    for _thr, _mi, _mm in check_milestones(pet):
        _is = "、".join(f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k,v in _mi.items())
        st.markdown(f'<div class="reward-banner">⭐ <b>マイルストーン達成！（スコア{_thr}）</b><br>'
                    f'<span style="font-size:.85rem;">{_is}</span></div>', unsafe_allow_html=True)
        if st.button(f"⭐ 受け取る（スコア{_thr}報酬）", key=f"claim_ms_{_thr}"):
            pet.setdefault("milestones_claimed",[]).append(_thr)
            for k,v in _mi.items(): pet["items"][k] = pet["items"].get(k,0)+v
            pet["message"] = _mm; pet["last_action_time"] = time.time()
            st.session_state.pet = pet; save_pet(pet); st.rerun()

    # プレステージ報酬
    _np = check_prestige_reward(pet)
    if _np > 0:
        _ti = {k:v*_np for k,v in PRESTIGE_REWARD.items()}
        _ps = "、".join(f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k,v in _ti.items())
        st.markdown(f'<div class="reward-banner">✦ <b>プレステージ報酬（×{_np}回分）</b><br>'
                    f'<span style="font-size:.85rem;">{_ps}</span></div>', unsafe_allow_html=True)
        if st.button(f"✦ 受け取る（{_ps}）", key="claim_prestige"):
            try:
                _pp = int(json.loads(MINIGAME_SAVE.read_text(encoding="utf-8")).get("prestige_points",0))
            except Exception:
                _pp = pet.get("prestige_reward_claimed",0) + _np
            pet["prestige_reward_claimed"] = _pp
            for k,v in _ti.items(): pet["items"][k] = pet["items"].get(k,0)+v
            pet["message"] = "プレステージ報酬受け取り！研究お疲れ様！"
            pet["last_action_time"] = time.time()
            st.session_state.pet = pet; save_pet(pet); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  タブ: アイテム / お部屋 / コレクション / ガイド
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
tab_items, tab_room, tab_coll, tab_guide = st.tabs(["🎒 アイテム", "🏠 お部屋", "🏆 コレクション", "📖 ガイド"])

# ── アイテムタブ ──────────────────────────────────────────────────────────────
with tab_items:
    for cat_label, cat_dict in [("🍽️ 食べ物", FOOD_ITEMS), ("🎮 おもちゃ", TOY_ITEMS), ("💊 お世話", CARE_ITEMS)]:
        st.markdown(f'<div class="action-title">{cat_label}</div>', unsafe_allow_html=True)
        badges = ""
        for key, item in cat_dict.items():
            cnt = pet["items"].get(key, 0)
            active = cnt > 0
            badges += (f'<span class="item-badge" style="border-color:{"#2a5a8c" if active else "#1a2a3c"};'
                       f'color:{"#a0d0f0" if active else "#334455"};">'
                       f'{item["icon"]} {item["name"]} × {cnt}</span>')
        st.markdown(badges, unsafe_allow_html=True)

# ── お部屋タブ ────────────────────────────────────────────────────────────────
with tab_room:
    current_wp = pet.get("room", _DEF_ROOM)["wallpaper"]
    current_f1 = pet.get("room", _DEF_ROOM).get("furniture1", "computer")
    current_f2 = pet.get("room", _DEF_ROOM).get("furniture2", "none")
    owned_wp   = pet.get("collection", _DEF_COLLECTION)["wallpapers"]
    owned_furn = pet.get("collection", _DEF_COLLECTION)["furniture"]
    stars      = pet.get("stars", 0)

    st.markdown(f'<div class="action-title">⭐ 所持ほしポイント: {stars}　（アクションで貯まる）</div>',
                unsafe_allow_html=True)
    st.caption("壁紙・家具を購入するとコレクションに追加され、いつでも変更できます。")

    st.markdown("**🎨 壁紙**")
    wp_cols = st.columns(3)
    for i, (wk, wd) in enumerate(WALLPAPERS.items()):
        is_owned   = wk in owned_wp
        is_current = wk == current_wp
        with wp_cols[i % 3]:
            status = "✅ 使用中" if is_current else ("所持済み" if is_owned else f"⭐{wd['cost']}")
            btn_label = f"{wd['icon']} {wd['name']}\n{status}"
            if st.button(btn_label, key=f"wp_{wk}", disabled=is_current, use_container_width=True):
                if is_owned:
                    pet["room"]["wallpaper"] = wk
                    save_pet(pet); st.rerun()
                elif stars >= wd["cost"]:
                    pet["stars"] -= wd["cost"]
                    pet["room"]["wallpaper"] = wk
                    pet.setdefault("collection", _DEF_COLLECTION.copy())["wallpapers"].append(wk)
                    save_pet(pet); st.rerun()
                else:
                    st.toast("ほしポイントが足りません！")

    st.markdown("**🪑 家具スロット 1**")
    f1_cols = st.columns(4)
    for i, (fk, fd) in enumerate(FURNITURES.items()):
        is_owned   = fk in owned_furn
        is_current = fk == current_f1
        with f1_cols[i % 4]:
            status = "✅ 使用中" if is_current else ("所持済み" if is_owned else f"⭐{fd['cost']}")
            if st.button(f"{fd['icon']} {fd['name']}\n{status}", key=f"f1_{fk}",
                         disabled=is_current, use_container_width=True):
                if is_owned:
                    pet["room"]["furniture1"] = fk
                    save_pet(pet); st.rerun()
                elif stars >= fd["cost"]:
                    pet["stars"] -= fd["cost"]
                    pet["room"]["furniture1"] = fk
                    pet.setdefault("collection", _DEF_COLLECTION.copy())["furniture"].append(fk)
                    save_pet(pet); st.rerun()
                else:
                    st.toast("ほしポイントが足りません！")

    st.markdown("**🪑 家具スロット 2**")
    f2_cols = st.columns(4)
    for i, (fk, fd) in enumerate(FURNITURES.items()):
        is_owned   = fk in owned_furn
        is_current = fk == current_f2
        with f2_cols[i % 4]:
            status = "✅ 使用中" if is_current else ("所持済み" if is_owned else f"⭐{fd['cost']}")
            if st.button(f"{fd['icon']} {fd['name']}\n{status}", key=f"f2_{fk}",
                         disabled=is_current, use_container_width=True):
                if is_owned:
                    pet["room"]["furniture2"] = fk
                    save_pet(pet); st.rerun()
                elif stars >= fd["cost"]:
                    pet["stars"] -= fd["cost"]
                    pet["room"]["furniture2"] = fk
                    pet.setdefault("collection", _DEF_COLLECTION.copy())["furniture"].append(fk)
                    save_pet(pet); st.rerun()
                else:
                    st.toast("ほしポイントが足りません！")

# ── コレクションタブ ──────────────────────────────────────────────────────────
with tab_coll:
    col = pet.get("collection", _DEF_COLLECTION)

    st.markdown("**🍽️ 食べ物コレクション**")
    html = ""
    for fk, fd in FOOD_ITEMS.items():
        got = fk in col.get("foods", [])
        cls = "got" if got else "not"
        title = fd["name"] if got else "???"
        html += f'<span class="coll-item {cls}" title="{title}">{fd["icon"] if got else "❓"}</span>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"{len([k for k in FOOD_ITEMS if k in col.get('foods',[])])} / {len(FOOD_ITEMS)} 種類")

    st.markdown("**🎮 おもちゃコレクション**")
    html = ""
    for tk, td in TOY_ITEMS.items():
        got = tk in col.get("toys", [])
        cls = "got" if got else "not"
        html += f'<span class="coll-item {cls}" title="{td["name"] if got else "???"}">{td["icon"] if got else "❓"}</span>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"{len([k for k in TOY_ITEMS if k in col.get('toys',[])])} / {len(TOY_ITEMS)} 種類")

    st.markdown("**🏠 お部屋コレクション**")
    html = ""
    for wk, wd in WALLPAPERS.items():
        got = wk in col.get("wallpapers", ["lab"])
        cls = "got" if got else "not"
        html += f'<span class="coll-item {cls}" title="{wd["name"] if got else "???"}">{wd["icon"] if got else "❓"}</span>'
    for fk, fd in FURNITURES.items():
        got = fk in col.get("furniture", ["none","computer"])
        cls = "got" if got else "not"
        html += f'<span class="coll-item {cls}" title="{fd["name"] if got else "???"}">{fd["icon"] if got else "❓"}</span>'
    st.markdown(html, unsafe_allow_html=True)
    owned_room_cnt = len([k for k in WALLPAPERS if k in col.get("wallpapers",[])]) + len([k for k in FURNITURES if k in col.get("furniture",[])])
    st.caption(f"{owned_room_cnt} / {len(WALLPAPERS)+len(FURNITURES)} 種類")

    st.markdown("**✨ 性格コレクション**")
    html = ""
    for pk, pd in PERSONALITIES.items():
        got = pk in col.get("personalities", [])
        cls = "got" if got else "not"
        desc = f'{pd["name"]}: {pd["desc"]}' if got else "???"
        html += f'<span class="coll-item {cls}" title="{desc}">{pd["icon"] if got else "❓"}</span>'
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"{len(col.get('personalities',[]))} / {len(PERSONALITIES)} 種類　—　最終進化（スーパーガウス）で解放")

    with st.expander("性格の詳細"):
        for pk, pd in PERSONALITIES.items():
            got = pk in col.get("personalities",[])
            icon = pd["icon"] if got else "❓"
            name = pd["name"] if got else "???"
            desc = pd["desc"] if got else "最終進化で解放されます"
            st.markdown(f"**{icon} {name}** — {desc}")

# ── ガイドタブ ────────────────────────────────────────────────────────────────
with tab_guide:
    st.markdown("""
**進化の条件**

| 段階 | 必要な年齢 | 必要スコア |
|---|---|---|
| 🥚 たまご → ちびガウス | 24時間 | なし |
| ちびガウス → こガウス | 3日 | 50以上 |
| こガウス → ガウス博士 | 7日 | 160以上 |
| ガウス博士 → スーパーガウス | 14日 | 450以上 |

**性格の決まり方（最終進化時）**

| 性格 | 好きな食べ物 | 好きなおもちゃ | 好きな部屋 |
|---|---|---|---|
| 🔬 研究者 | やさい | パズル・オルゴール | 研究室・うちゅう |
| 🏃 アスリート | おさかな | ボール・ロボット | もりのなか |
| 👨‍🍳 グルメ | 豪華なごはん・ケーキ | — | としのよる |
| 🍭 あまえんぼ | あめ | ぬいぐるみ・オルゴール | はなばたけ |
| 🌙 神秘家 | — | オルゴール・ロボット | うちゅう・かいてい |

**アイテム効果一覧**

| アイテム | お腹 | 機嫌 | 体力 | 清潔 | 世話スコア |
|---|---|---|---|---|---|
| 🍙 ふつうのごはん | +25 | — | — | — | +4 |
| 🍱 豪華なごはん | +50 | +15 | +5 | — | +12 |
| 🥦 やさい | +20 | -5 | +20 | — | +6 |
| 🐟 おさかな | +30 | +10 | +8 | — | +7 |
| 🍬 あめ | +5 | +30 | -8 | — | +3 |
| 🎂 スペシャルケーキ | +45 | +45 | +15 | — | +20 |
| 🧸 ぬいぐるみ | -5 | +20 | — | -5 | +5 |
| ⚽ ボール | -12 | +25 | — | -10 | +6 |
| 🧩 パズル | -3 | +15 | — | — | +8 |
| 🎵 オルゴール | — | +18 | +8 | — | +5 |
| 🤖 ロボットおもちゃ | -8 | +30 | — | -5 | +9 |
| 💊 くすり | — | — | +25 | — | +5 |
| 🛁 シャンプー | — | +5 | — | +35 | +4 |
| 🌟 ビタミン剤 | — | — | +40 | — | +6 |

**⭐ほしポイントの貯め方**: 各アクションで世話スコアと同量が貯まります。

**アイテム入手方法**
- 🌅 **ログインボーナス**: 24時間ごとにごはん×2＋ランダムアイテム
- 🔬 **研究報酬**: 抽出→おもちゃ+やさい、COLMAP→シャンプー+くすり+パズル、学習→豪華ごはん+おさかな+ビタミン
- ⭐ **マイルストーン**: スコア30/60/100/160/250/400/600達成で報酬
- ✦ **プレステージ報酬**: ミニゲームでプレステージするたびに豪華ごはん・くすり・おさかな等
""")

    with st.expander("✏️ 名前を変える"):
        new_name = st.text_input("新しい名前", value=pet["name"], max_chars=20)
        if st.button("名前を変更する"):
            pet["name"] = new_name.strip() or pet["name"]
            st.session_state.pet = pet; save_pet(pet)
            st.success(f"名前を「{pet['name']}」に変更しました！")

# ══════════════════════════════════════════════════════════════════════════════
#  フッター
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, "/workspace")
try:
    from pipeline_widget import render_pipeline_widget
    render_pipeline_widget()
except Exception:
    pass
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass

if not pet.get("is_dead"):
    import time as _t
    last_refresh = st.session_state.get("_pet_last_refresh", 0)
    if _t.time() - last_refresh > 30:
        st.session_state["_pet_last_refresh"] = _t.time()
        _t.sleep(30)
        st.rerun()
