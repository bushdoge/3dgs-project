# ペット育成ゲーム「ガウスくん」
# たまごっちスタイル。実時間でステータスが変化し、研究の進捗でアイテムがもらえる。

import json
import random
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="ガウスくん", page_icon="🐾", layout="wide")

SAVE_FILE          = Path("/workspace/tmp/pet_save.json")
PIPELINE_STATE_FILE = Path("/workspace/tmp/pipeline_state.json")

# ══════════════════════════════════════════════════════════════════════════════
#  定数
# ══════════════════════════════════════════════════════════════════════════════

STAGE_NAMES = ["たまご", "ちびガウス", "こガウス", "ガウス博士", "✨スーパーガウス✨"]

# アイテム定義 {stat変化, 世話スコア}
ITEMS = {
    "normal_food": {"name": "普通のごはん", "icon": "🍙",
                    "hunger": 25,  "happiness":  0, "cleanliness":  0, "health":  0, "care":  4},
    "fancy_food":  {"name": "豪華なごはん", "icon": "🍱",
                    "hunger": 50,  "happiness": 15, "cleanliness":  0, "health":  5, "care": 12},
    "toy":         {"name": "おもちゃ",     "icon": "🧸",
                    "hunger": -5,  "happiness": 20, "cleanliness": -5, "health":  0, "care":  5},
    "medicine":    {"name": "くすり",       "icon": "💊",
                    "hunger":  0,  "happiness":  0, "cleanliness":  0, "health": 25, "care":  5,
                    "cure_sick": True},
    "shampoo":     {"name": "シャンプー",   "icon": "🛁",
                    "hunger":  0,  "happiness":  5, "cleanliness": 35, "health":  0, "care":  4},
}

MINIGAME_SAVE = Path("/workspace/tmp/minigame_save.json")

# 研究ステップ完了報酬
STEP_REWARDS = {
    "extracting": {"toy":        1},
    "colmap":     {"shampoo":    1, "medicine": 1},
    "training":   {"fancy_food": 3},
}

# 世話スコアマイルストーン報酬 (threshold, items, message)
MILESTONES = [
    ( 30, {"normal_food": 3},                         "世話スコア30達成！ごはん×3をプレゼント🎁"),
    ( 60, {"toy": 1, "shampoo": 1},                   "60達成！おもちゃ＆シャンプーをプレゼント🎁"),
    (100, {"medicine": 1, "fancy_food": 1},            "100達成！くすり＆豪華ごはんをプレゼント🎁"),
    (160, {"normal_food": 5, "shampoo": 2},            "160達成！ごはん×5＆シャンプー×2をプレゼント🎁"),
    (250, {"medicine": 2, "fancy_food": 1},            "250達成！くすり×2＆豪華ごはんをプレゼント🎁"),
    (400, {"fancy_food": 2, "medicine": 1, "toy": 1},  "400達成！豪華セットをプレゼント🎁"),
    (600, {"fancy_food": 3, "medicine": 2, "shampoo": 3}, "600達成！スペシャルパックをプレゼント🎁"),
]

# プレステージ報酬（1プレステージポイントあたり）
PRESTIGE_REWARD = {
    "fancy_food":  2,
    "medicine":    1,
    "shampoo":     2,
    "toy":         1,
}

DEFAULT_PET = {
    "name":              "ガウスくん",
    "stage":             0,
    "hunger":            100.0,
    "happiness":         80.0,
    "cleanliness":       100.0,
    "health":            100.0,
    "age_hours":         0.0,
    "last_update":       0.0,        # time.time() で初期化
    "is_sick":           False,
    "is_dead":           False,
    "care_score":        0.0,
    "sick_hours":        0.0,
    "items": {
        "normal_food": 5,
        "fancy_food":  0,
        "toy":         2,
        "medicine":    1,
        "shampoo":     3,
    },
    "last_login_bonus":       0.0,
    "step_reward_claimed":    {"extracting": 0.0, "colmap": 0.0, "training": 0.0},
    "milestones_claimed":     [],
    "prestige_reward_claimed": 0,
    "message":             "はじめまして！よろしくね！",
    "last_action_time":    0.0,
}

# ══════════════════════════════════════════════════════════════════════════════
#  セーブ / ロード
# ══════════════════════════════════════════════════════════════════════════════

def load_pet() -> dict:
    if SAVE_FILE.exists():
        try:
            data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            for k, v in DEFAULT_PET.items():
                data.setdefault(k, v)
            if isinstance(data.get("items"), dict):
                for k, v in DEFAULT_PET["items"].items():
                    data["items"].setdefault(k, v)
            return data
        except Exception:
            pass
    pet = DEFAULT_PET.copy()
    pet["items"] = DEFAULT_PET["items"].copy()
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

    elapsed = min((now - pet["last_update"]) / 3600.0, 168.0)  # 最大7日分
    if elapsed < 0.001:
        return pet

    # ── ステータス減少（難易度：2〜3日放置で危機的） ──
    pet["hunger"]      = max(0.0, pet["hunger"]      - 2.5 * elapsed)
    pet["happiness"]   = max(0.0, pet["happiness"]   - 1.2 * elapsed)
    pet["cleanliness"] = max(0.0, pet["cleanliness"] - 0.8 * elapsed)

    # ── 病気発症チェック ──
    if not pet["is_sick"]:
        risk = 0.0
        if pet["hunger"]      < 25: risk += 0.25
        if pet["cleanliness"] < 20: risk += 0.20
        if risk > 0:
            p = 1 - (1 - risk) ** elapsed
            if random.random() < p:
                pet["is_sick"] = True
                pet["message"] = "なんか体がだるい…くすりほしいな…"

    if pet["is_sick"]:
        pet["sick_hours"] = pet.get("sick_hours", 0.0) + elapsed

    # ── 体力減少（複合） ──
    decay = 0.10 * elapsed
    if pet["hunger"] < 30:
        decay += 1.20 * elapsed
    if pet["hunger"] < 5:
        decay += 2.50 * elapsed          # 空腹0近くで急速悪化
    if pet["is_sick"]:
        decay += 1.20 * elapsed
    if pet["happiness"] < 20:
        decay += 0.60 * elapsed

    pet["health"] = max(0.0, pet["health"] - decay)

    if pet["health"] <= 0.0:
        pet["is_dead"] = True
        pet["health"]  = 0.0
        pet["message"] = "ガウスくんは…天国へ旅立ちました…"
        pet["last_update"] = now
        return pet

    # ── 年齢加算 ──
    pet["age_hours"] = pet.get("age_hours", 0.0) + elapsed

    # ── 進化チェック ──
    _check_evolution(pet)

    pet["last_update"] = now
    return pet


def _check_evolution(pet: dict):
    """年齢と世話スコアに応じて段階を進化させる"""
    evolutions = [
        # (現段階, 最低年齢h, 最低スコア, 次段階, メッセージ)
        (0,  24,    0,   1, "たまごが孵化したよ！ちびガウス誕生！🐣"),
        (1,  72,   50,   2, "ちびガウスが成長した！こガウスになったよ！"),
        (2, 168,  160,   3, "立派に育った！ガウス博士になったよ！🧑‍🔬"),
        (3, 336,  450,   4, "なんと…！✨スーパーガウスに進化した！！✨"),
    ]
    for from_s, min_age, min_score, to_s, msg in evolutions:
        if (pet["stage"] == from_s
                and pet["age_hours"] >= min_age
                and pet["care_score"] >= min_score):
            pet["stage"] = to_s
            pet["message"] = msg
            pet["last_action_time"] = time.time()
            break

# ══════════════════════════════════════════════════════════════════════════════
#  気分・メッセージ
# ══════════════════════════════════════════════════════════════════════════════

def get_mood(pet: dict) -> str:
    if pet.get("is_dead"):           return "dead"
    if pet["stage"] == 0:            return "egg"
    if pet.get("is_sick") or pet["health"] < 20:
                                     return "sick"
    if 0 <= datetime.now().hour < 7: return "sleeping"
    if pet["hunger"]    < 25:        return "hungry"
    if pet["happiness"] < 25:        return "unhappy"
    if pet["hunger"] > 65 and pet["happiness"] > 65 and pet["cleanliness"] > 40:
                                     return "happy"
    return "normal"

def get_mood_message(pet: dict, mood: str) -> str:
    defaults = {
        "dead":     "……",
        "egg":      "あたたかく見守ってね…",
        "sick":     "うぅ、気持ち悪い…くすりが欲しいな",
        "sleeping": "すやすや…（おやすみ中）",
        "hungry":   "おなかすいた！ごはんちょうだい！！",
        "unhappy":  "かまってよ〜…つまんない",
        "happy":    "しあわせ〜！今日もいい日だね！",
        "normal":   "ふつうにすごしてるよ。",
    }
    # アクション直後5秒間はカスタムメッセージを優先表示
    if (time.time() - pet.get("last_action_time", 0)) < 5:
        return pet.get("message", defaults.get(mood, "…"))
    return defaults.get(mood, "…")

# ══════════════════════════════════════════════════════════════════════════════
#  画像生成（Pillow）
# ══════════════════════════════════════════════════════════════════════════════

_BG       = (10,  14, 26)
_CARD     = (15,  25, 40)
_OUTLINE  = (30,  60, 100)


def _circle(draw, cx, cy, r, fill, outline=None, width=2):
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=outline, width=width)

def _oval(draw, cx, cy, rx, ry, fill, outline=None, width=2):
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=fill, outline=outline, width=width)

def _body_color(stage: int, mood: str):
    if mood == "dead":    return (90, 90, 90),   (60, 60, 60)
    if mood == "sick":    return (130, 175, 100), (80, 130, 60)
    palette = [
        ((245, 240, 220), (200, 185, 150)),   # 0: 卵
        ((90,  185, 225), (50,  130, 180)),   # 1: 水色
        ((130,  80, 210), (85,  45, 160)),    # 2: 紫
        ((55,  165, 100), (30,  110, 65)),    # 3: 緑
        ((255, 210,  40), (200, 155,  0)),    # 4: 金
    ]
    return palette[min(stage, 4)]

def _draw_eyes(draw, lx, ly, rx, ry, mood, r=11):
    W, BK = (255, 255, 255), (20, 20, 20)
    if mood == "sleeping":
        for ex, ey in [(lx, ly), (rx, ry)]:
            draw.arc([ex-r, ey-r//3, ex+r, ey+r//3], 0, 180, fill=BK, width=3)
        return
    if mood == "dead":
        for ex, ey in [(lx, ly), (rx, ry)]:
            draw.line([ex-r, ey-r, ex+r, ey+r], fill=BK, width=3)
            draw.line([ex+r, ey-r, ex-r, ey+r], fill=BK, width=3)
        return
    if mood == "sick":
        for ex, ey in [(lx, ly), (rx, ry)]:
            _circle(draw, ex, ey, r, W, BK, 2)
            draw.arc([ex-r//2, ey-r//2, ex+r//2, ey+r//2], 0, 270, fill=BK, width=2)
        return
    if mood == "happy":
        for ex, ey in [(lx, ly), (rx, ry)]:
            draw.arc([ex-r, ey-r//2, ex+r, ey+r//2], 200, 340, fill=BK, width=3)
        return
    if mood in ("hungry", "unhappy"):
        for ex, ey in [(lx, ly), (rx, ry)]:
            _circle(draw, ex, ey, r, W, BK, 2)
            _circle(draw, ex+2, ey+3, r//2, BK)
        return
    # normal
    for ex, ey in [(lx, ly), (rx, ry)]:
        _circle(draw, ex, ey, r, W, BK, 2)
        _circle(draw, ex+3, ey, r//2, BK)
        _circle(draw, ex+r//2, ey-r//2+2, 2, W)

def _draw_mouth(draw, cx, cy, mood, s=1.0):
    r, BK = int(14*s), (20, 20, 20)
    if mood == "happy":
        draw.arc([cx-r, cy-r//2, cx+r, cy+r//2], 0, 180, fill=BK, width=3)
        for chx in [cx - int(28*s), cx + int(28*s)]:
            draw.ellipse([chx-9, cy-4, chx+9, cy+6], fill=(255, 140, 140, 120))
    elif mood in ("hungry", "unhappy", "sick", "dead"):
        draw.arc([cx-r, cy, cx+r, cy+r], 0, 180, fill=BK, width=3)
    elif mood == "sleeping":
        draw.ellipse([cx-7, cy-3, cx+7, cy+5], fill=BK)
    else:
        draw.line([cx - int(r*0.6), cy, cx + int(r*0.6), cy], fill=BK, width=3)

def _sparkles(draw, SIZE, color=(255, 215, 0)):
    rng = random.Random(42)
    for _ in range(10):
        sx, sy = rng.randint(20, SIZE-20), rng.randint(20, SIZE-20)
        r = rng.randint(3, 8)
        for dx, dy in [(r,0),(-r,0),(0,r),(0,-r)]:
            draw.line([sx, sy, sx+dx, sy+dy], fill=color, width=2)

# ─── ステージ別描画 ───────────────────────────────────────────────────────────

def _stage_egg(draw, cx, cy, mood):
    fill, out = _body_color(0, mood)
    _oval(draw, cx, cy+5, 62, 78, fill, out, 3)
    draw.ellipse([cx-30, cy-40, cx-8, cy-20], fill=tuple(min(v+15, 255) for v in fill))
    for sx, sy, sr in [(cx+22, cy-15, 9), (cx-28, cy+12, 7), (cx+12, cy+30, 6)]:
        _circle(draw, sx, sy, sr, tuple(max(v-20, 0) for v in fill))
    # 孵化直前ひび（孵化まで残りわずかのときだけ）
    age = st.session_state.get("_pet_age", 0)
    if age > 20:
        draw.line([cx+15, cy-30, cx+30, cy-10, cx+20, cy+5], fill=(160, 140, 110), width=2)

def _stage_baby(draw, cx, cy, mood):
    fill, out = _body_color(1, mood)
    _circle(draw, cx, cy+10, 68, fill, out, 3)
    for ax, ay in [(cx-78, cy+18), (cx+78, cy+18)]:
        _circle(draw, ax, ay, 22, fill, out, 2)
    _draw_eyes(draw, cx-24, cy-2, cx+24, cy-2, mood, 12)
    _draw_mouth(draw, cx, cy+26, mood, 0.85)
    if mood == "sick":
        for i, (tx, ty) in enumerate([(cx+25, cy-35), (cx+38, cy-50)]):
            draw.text((tx, ty), "~", fill=(120, 200, 80))

def _stage_child(draw, cx, cy, mood):
    fill, out = _body_color(2, mood)
    _oval(draw, cx, cy+5, 72, 78, fill, out, 3)
    for ax, ay in [(cx-84, cy+8), (cx+84, cy+8)]:
        _oval(draw, ax, ay, 24, 20, fill, out, 2)
    for fx, fy in [(cx-30, cy+80), (cx+30, cy+80)]:
        _oval(draw, fx, fy, 22, 14, fill, out, 2)
    # 3Dメガネ
    for gx in [cx-24, cx+24]:
        draw.rectangle([gx-15, cy-22, gx+15, cy-5], outline=(180, 220, 255), width=2)
    draw.line([cx-9, cy-13, cx+9, cy-13], fill=(180, 220, 255), width=2)
    _draw_eyes(draw, cx-24, cy-8, cx+24, cy-8, mood, 10)
    _draw_mouth(draw, cx, cy+24, mood, 0.9)

def _stage_adult(draw, cx, cy, mood):
    fill, out = _body_color(3, mood)
    # 白衣
    _oval(draw, cx, cy+8, 75, 80, (235, 242, 255), (190, 205, 225), 2)
    _oval(draw, cx, cy+5, 65, 72, fill, out, 3)
    for ax, ay in [(cx-82, cy+5), (cx+82, cy+5)]:
        _oval(draw, ax, ay, 24, 22, fill, out, 2)
    for fx, fy in [(cx-30, cy+80), (cx+30, cy+80)]:
        _oval(draw, fx, fy, 22, 15, (235, 242, 255), (190, 205, 225), 2)
    # 角帽
    draw.rectangle([cx-42, cy-98, cx+42, cy-62], fill=(20, 20, 35), outline=(10, 10, 20), width=2)
    draw.rectangle([cx-54, cy-70, cx+54, cy-62], fill=(20, 20, 35), outline=(10, 10, 20), width=2)
    draw.line([cx+42, cy-82, cx+58, cy-55], fill=(255, 215, 0), width=2)
    _circle(draw, cx+58, cy-52, 6, (255, 215, 0))
    # ネクタイ
    draw.polygon([(cx, cy+35), (cx-9, cy+56), (cx+9, cy+56)], fill=(210, 45, 45))
    _draw_eyes(draw, cx-24, cy-10, cx+24, cy-10, mood, 11)
    _draw_mouth(draw, cx, cy+20, mood, 1.0)

def _stage_special(draw, cx, cy, mood, SIZE):
    fill, out = _body_color(4, mood)
    # グロー
    for gr in range(90, 68, -5):
        a = int(255 * (90 - gr) / 22)
        glow_col = (255, 215, 0, a)
        try:
            tmp = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).ellipse([cx-gr, cy-gr+10, cx+gr, cy+gr+10], fill=glow_col)
            draw._image.alpha_composite(tmp)
        except Exception:
            pass
    _circle(draw, cx, cy+10, 68, fill, out, 3)
    # 王冠
    pts = [(cx-42, cy-77), (cx-42, cy-98), (cx-20, cy-88),
           (cx,    cy-105), (cx+20, cy-88), (cx+42, cy-98), (cx+42, cy-77)]
    draw.polygon(pts, fill=(255, 180, 0), outline=(200, 130, 0))
    for jx, jy, jc in [(cx-30, cy-84, (255,80,80)), (cx, cy-96, (80,200,255)), (cx+30, cy-84, (80,255,120))]:
        _circle(draw, jx, jy, 6, jc)
    for ax, ay in [(cx-82, cy+10), (cx+82, cy+10)]:
        _circle(draw, ax, ay, 24, fill, out, 2)
    _draw_eyes(draw, cx-24, cy+5, cx+24, cy+5, mood, 13)
    _draw_mouth(draw, cx, cy+30, mood, 1.1)
    _sparkles(draw, SIZE, (255, 215, 0))


def generate_pet_image(pet: dict) -> Image.Image:
    SIZE  = 280
    stage = pet["stage"]
    mood  = get_mood(pet)
    st.session_state["_pet_age"] = pet.get("age_hours", 0)

    img  = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # カード背景
    bg_img = Image.new("RGBA", (SIZE, SIZE), _BG)
    bg_draw = ImageDraw.Draw(bg_img)
    try:
        bg_draw.rounded_rectangle([4, 4, SIZE-4, SIZE-4], radius=22,
                                   fill=_CARD, outline=_OUTLINE, width=2)
    except AttributeError:
        bg_draw.rectangle([4, 4, SIZE-4, SIZE-4], fill=_CARD, outline=_OUTLINE, width=2)
    img = bg_img
    draw = ImageDraw.Draw(img)

    cx, cy = SIZE // 2, SIZE // 2 + 10

    if   stage == 0: _stage_egg(draw, cx, cy, mood)
    elif stage == 1: _stage_baby(draw, cx, cy, mood)
    elif stage == 2: _stage_child(draw, cx, cy, mood)
    elif stage == 3: _stage_adult(draw, cx, cy, mood)
    else:            _stage_special(draw, cx, cy, mood, SIZE)

    # 共通エフェクト
    if mood == "happy" and stage > 0:
        for hx, hy in [(cx-65, cy-75), (cx+58, cy-70)]:
            draw.text((hx, hy), "♥", fill=(255, 100, 150))
    if mood == "hungry":
        draw.text((cx+55, cy-40), "💧", fill=(100, 180, 255))
    if mood == "sleeping" and stage > 0:
        for i, (zx, zy) in enumerate([(cx+55, cy-55), (cx+68, cy-70), (cx+80, cy-85)]):
            draw.text((zx, zy), "z", fill=(160, 210, 255))

    return img


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

# ══════════════════════════════════════════════════════════════════════════════
#  報酬・ショップ関連
# ══════════════════════════════════════════════════════════════════════════════

def check_login_bonus(pet: dict) -> bool:
    """24時間経過していればログインボーナスが受取可能"""
    return (time.time() - pet.get("last_login_bonus", 0.0)) >= 86400

def claim_login_bonus(pet: dict) -> tuple[dict, dict]:
    """ログインボーナスを付与して (pet, items_given) を返す"""
    given = {"normal_food": 2}
    roll = random.random()
    if   roll < 0.10: given["medicine"] = 1
    elif roll < 0.30: given["toy"]      = 1
    elif roll < 0.60: given["shampoo"]  = 1
    for k, v in given.items():
        pet["items"][k] = pet["items"].get(k, 0) + v
    pet["last_login_bonus"] = time.time()
    bonus_str = "、".join(f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k, v in given.items())
    pet["message"]          = f"ログインボーナス受け取り！{bonus_str}"
    pet["last_action_time"] = time.time()
    return pet, given

def check_step_rewards(pet: dict) -> dict:
    """未受取の研究ステップ報酬を {step: items_dict} で返す"""
    if not PIPELINE_STATE_FILE.exists():
        return {}
    try:
        state = json.loads(PIPELINE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    step_times = state.get("step_times", {})
    claimed    = pet.get("step_reward_claimed", {})
    pending = {}
    for step, reward in STEP_REWARDS.items():
        done_time = step_times.get(step)
        if done_time and done_time > claimed.get(step, 0.0):
            pending[step] = reward
    return pending

def check_milestones(pet: dict) -> list:
    """未受取のマイルストーン一覧を返す"""
    score   = pet.get("care_score", 0.0)
    claimed = pet.get("milestones_claimed", [])
    return [(t, items, msg) for t, items, msg in MILESTONES
            if score >= t and t not in claimed]

def check_prestige_reward(pet: dict) -> int:
    """未受取のプレステージ数（ポイント差分）を返す"""
    try:
        if not MINIGAME_SAVE.exists():
            return 0
        data = json.loads(MINIGAME_SAVE.read_text(encoding="utf-8"))
        current_pp = int(data.get("prestige_points", 0))
        claimed_pp = int(pet.get("prestige_reward_claimed", 0))
        return max(0, current_pp - claimed_pp)
    except Exception:
        return 0

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a; color: #e0e6f0;
  }
  .block-container { padding: 1.5rem 2rem; }
  .stat-bar-wrap { margin: 4px 0; }
  .speech-bubble {
    background: #0d1b2e; border: 1px solid #1a3a5c; border-radius: 12px;
    padding: 10px 16px; font-size: 0.9rem; color: #a0d0f0; margin-top: 8px;
    position: relative;
  }
  .pet-info-card {
    background: #0d1b2e; border: 1px solid #1a3a5c; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 10px;
  }
  .action-title { font-size: 0.7rem; letter-spacing: 0.2em; color: #4a90b8;
                  text-transform: uppercase; margin: 12px 0 6px 0; }
  .item-badge {
    display: inline-block; background: #0d1b2e; border: 1px solid #1a3a5c;
    border-radius: 8px; padding: 3px 10px; margin: 3px; font-size: 0.8rem;
  }
  div[data-testid="stButton"] > button {
    background: #0d1b2e; border: 1px solid #1a3a5c; border-radius: 10px;
    color: #e0e6f0; font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem; letter-spacing: 0.05em; padding: 0.6rem 0.4rem; width: 100%;
    transition: border-color .2s, box-shadow .2s;
  }
  div[data-testid="stButton"] > button:hover {
    border-color: #00e5ff; box-shadow: 0 0 10px #00e5ff33; color: #00e5ff;
  }
  .reward-banner {
    background: linear-gradient(135deg, #1a2a10, #2a3a15);
    border: 1px solid #4a8a30; border-radius: 12px;
    padding: 12px 18px; margin: 10px 0; color: #a0e060;
  }
  .dead-banner {
    background: linear-gradient(135deg, #2a0a0a, #1a0505);
    border: 1px solid #8a2020; border-radius: 12px;
    padding: 16px 20px; margin: 10px 0; color: #e06060; text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ── データ読み込み・更新 ──────────────────────────────────────────────────────
if "pet" not in st.session_state:
    st.session_state.pet = load_pet()

pet = st.session_state.pet
pet = update_pet(pet)
st.session_state.pet = pet
save_pet(pet)

mood = get_mood(pet)

# ── ヘッダー ──────────────────────────────────────────────────────────────────
st.markdown("## 🐾 ガウスくんを育てよう")
st.markdown("---")

col_img, col_info = st.columns([1, 1.6], gap="large")

# ── 左：ペット画像 + 吹き出し ────────────────────────────────────────────────
with col_img:
    img = generate_pet_image(pet)
    st.image(pil_to_bytes(img), use_container_width=True)

    msg = get_mood_message(pet, mood)
    st.markdown(f'<div class="speech-bubble">💬 {msg}</div>', unsafe_allow_html=True)

    age_d = int(pet["age_hours"] // 24)
    age_h = int(pet["age_hours"] % 24)
    st.markdown(
        f'<div style="font-size:0.75rem;color:#4a90b8;margin-top:8px;text-align:center;">'
        f'{STAGE_NAMES[pet["stage"]]}　|　'
        f'年齢 {age_d}日{age_h}時間　|　'
        f'世話スコア {int(pet["care_score"])}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── 右：ステータス + アクション ───────────────────────────────────────────────
with col_info:
    # ── ステータスバー ──
    st.markdown('<div class="action-title">Status</div>', unsafe_allow_html=True)

    def stat_bar(label, icon, value, color="#00aaff", warn=30, danger=15):
        pct = max(0.0, min(100.0, value))
        bar_color = color if pct > warn else ("#ffaa00" if pct > danger else "#ff4444")
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        st.markdown(
            f'<div class="stat-bar-wrap">'
            f'{icon} <span style="font-size:0.78rem;color:#8ab0c8;">{label}</span>'
            f'<span style="font-size:0.75rem;color:{bar_color};margin-left:8px;">'
            f'{bar} {pct:.0f}%</span></div>',
            unsafe_allow_html=True,
        )

    stat_bar("体力",   "❤️",  pet["health"],      "#ff6688", warn=30, danger=20)
    stat_bar("お腹",   "🍎",  pet["hunger"],      "#00ccaa", warn=30, danger=15)
    stat_bar("機嫌",   "😊",  pet["happiness"],   "#ffcc00", warn=25, danger=10)
    stat_bar("清潔",   "🛁",  pet["cleanliness"], "#44aaff", warn=25, danger=10)

    if pet.get("is_sick"):
        st.markdown('<span style="color:#ffaa44;font-size:0.85rem;">⚠️ 病気中 — くすりをあげてください</span>',
                    unsafe_allow_html=True)

    # ── 死亡画面 ──
    if pet.get("is_dead"):
        st.markdown(
            '<div class="dead-banner">'
            '<div style="font-size:1.5rem;margin-bottom:8px;">💀</div>'
            'ガウスくんは天国へ旅立ちました…<br>'
            '<span style="font-size:0.8rem;color:#b08080;">もっとたくさん遊んであげてね</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("🥚 あたらしいたまごからやり直す", use_container_width=True):
            new_pet = DEFAULT_PET.copy()
            new_pet["items"] = DEFAULT_PET["items"].copy()
            new_pet["last_update"] = time.time()
            # アイテムは少し引き継ぐ（半分）
            for k in new_pet["items"]:
                new_pet["items"][k] = pet["items"].get(k, 0) // 2
            st.session_state.pet = new_pet
            save_pet(new_pet)
            st.rerun()

    else:
        # ── アクションボタン ──
        st.markdown('<div class="action-title">Actions</div>', unsafe_allow_html=True)

        items    = pet["items"]
        can_play = not pet.get("is_sick")

        ac1, ac2, ac3, ac4 = st.columns(4)

        def do_action(key, effect: dict, msg: str, cost_item: str | None = None):
            """アクション実行の共通処理"""
            if cost_item and items.get(cost_item, 0) <= 0:
                st.session_state._action_msg = f"{ITEMS[cost_item]['name']}が足りない！"
                return
            # ステータス適用
            for stat in ("hunger", "happiness", "cleanliness", "health"):
                if stat in effect:
                    pet[stat] = max(0.0, min(100.0, pet[stat] + effect[stat]))
            if effect.get("cure_sick"):
                pet["is_sick"]    = False
                pet["sick_hours"] = 0.0
            if cost_item:
                pet["items"][cost_item] -= 1
            pet["care_score"]   = pet.get("care_score", 0.0) + effect.get("care", 0)
            pet["message"]      = msg
            pet["last_action_time"] = time.time()
            _check_evolution(pet)
            st.session_state.pet = pet
            save_pet(pet)
            st.rerun()

        with ac1:
            has_food = items.get("normal_food", 0) + items.get("fancy_food", 0)
            if st.button(f"🍙 ごはん\n（{has_food}個）", disabled=(has_food == 0),
                         key="btn_food", use_container_width=True):
                st.session_state._feeding = True
                st.rerun()

        with ac2:
            has_toy = items.get("toy", 0)
            _play_label = f"🧸 あそぶ\n（おもちゃ{has_toy}個）" if has_toy > 0 else "🎮 あそぶ\n（手ぶら）"
            if st.button(_play_label, disabled=(not can_play),
                         key="btn_play", use_container_width=True):
                if has_toy > 0:
                    do_action("play",
                              {**ITEMS["toy"], "care": 5},
                              "おもちゃで遊んだ！たのし〜！！", cost_item="toy")
                else:
                    do_action("play",
                              {"happiness": 8, "hunger": -3, "cleanliness": -2, "care": 2},
                              "手ぶらで遊んだよ。まあ楽しかった…")

        with ac3:
            has_shampoo = items.get("shampoo", 0)
            if st.button(f"🛁 おそうじ\n（シャン{has_shampoo}）",
                         key="btn_clean", use_container_width=True):
                if has_shampoo > 0:
                    do_action("clean",
                              {**ITEMS["shampoo"], "care": 4},
                              "きれいになったよ！", cost_item="shampoo")
                else:
                    do_action("clean_bare",
                              {"cleanliness": 15, "care": 2},
                              "てでごしごし…きれいになったよ！")

        with ac4:
            has_med = items.get("medicine", 0)
            if st.button(f"💊 くすり\n（{has_med}個）",
                         disabled=(has_med == 0), key="btn_med",
                         use_container_width=True):
                do_action("medicine",
                          {**ITEMS["medicine"]},
                          "くすりのんだ！元気になってきたよ！", cost_item="medicine")

        # ── ごはん選択ポップアップ ──
        if st.session_state.get("_feeding"):
            st.markdown('<div class="action-title">どのごはんをあげますか？</div>',
                        unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            with f1:
                nf = items.get("normal_food", 0)
                if st.button(f"🍙 普通のごはん\n（{nf}個）",
                             disabled=(nf == 0), key="btn_nf",
                             use_container_width=True):
                    st.session_state._feeding = False
                    do_action("normal_food", {**ITEMS["normal_food"]},
                              "もぐもぐ…おいしい！", cost_item="normal_food")
            with f2:
                ff = items.get("fancy_food", 0)
                if st.button(f"🍱 豪華なごはん\n（{ff}個）",
                             disabled=(ff == 0), key="btn_ff",
                             use_container_width=True):
                    st.session_state._feeding = False
                    do_action("fancy_food", {**ITEMS["fancy_food"]},
                              "わあ！豪華なごはんだ！！最高！！", cost_item="fancy_food")
            with f3:
                if st.button("❌ キャンセル", key="btn_cancel_food",
                             use_container_width=True):
                    st.session_state._feeding = False
                    st.rerun()

# ── ログインボーナス ──────────────────────────────────────────────────────────
if not pet.get("is_dead") and check_login_bonus(pet):
    st.markdown(
        '<div class="reward-banner">'
        '🌅 <b>ログインボーナスが届いています！</b><br>'
        '<span style="font-size:0.85rem;">普通のごはん×2 ＋ ランダムアイテムを受け取れます。</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("🌅 ログインボーナスを受け取る", use_container_width=False):
        pet, _given = claim_login_bonus(pet)
        st.session_state.pet = pet
        save_pet(pet)
        st.rerun()

# ── 研究ステップ報酬 ──────────────────────────────────────────────────────────
if not pet.get("is_dead"):
    _STEP_JA = {"extracting": "フレーム抽出", "colmap": "COLMAP/HLoc", "training": "3DGS学習"}
    _pending_steps = check_step_rewards(pet)
    for _step, _reward in _pending_steps.items():
        _reward_str = "、".join(f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k, v in _reward.items())
        st.markdown(
            f'<div class="reward-banner">'
            f'🔬 <b>{_STEP_JA[_step]}完了報酬が届いています！</b><br>'
            f'<span style="font-size:0.85rem;">{_reward_str} を受け取れます。</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"🎁 受け取る（{_reward_str}）", key=f"claim_step_{_step}"):
            try:
                _state = json.loads(PIPELINE_STATE_FILE.read_text(encoding="utf-8"))
                _t = _state.get("step_times", {}).get(_step, time.time())
            except Exception:
                _t = time.time()
            pet.setdefault("step_reward_claimed", {})[_step] = _t
            for k, v in _reward.items():
                pet["items"][k] = pet["items"].get(k, 0) + v
            pet["message"]          = f"研究報酬受け取り！{_reward_str}"
            pet["last_action_time"] = time.time()
            st.session_state.pet = pet
            save_pet(pet)
            st.rerun()

# ── 世話スコアマイルストーン報酬 ──────────────────────────────────────────────
if not pet.get("is_dead"):
    for _threshold, _m_items, _m_msg in check_milestones(pet):
        _items_str = "、".join(f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k, v in _m_items.items())
        st.markdown(
            f'<div class="reward-banner">'
            f'⭐ <b>マイルストーン達成！（スコア {_threshold}）</b><br>'
            f'<span style="font-size:0.85rem;">{_items_str} を受け取れます。</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"⭐ 受け取る（スコア{_threshold}報酬）", key=f"claim_ms_{_threshold}"):
            pet.setdefault("milestones_claimed", []).append(_threshold)
            for k, v in _m_items.items():
                pet["items"][k] = pet["items"].get(k, 0) + v
            pet["message"]          = _m_msg
            pet["last_action_time"] = time.time()
            st.session_state.pet = pet
            save_pet(pet)
            st.rerun()

# ── アイテム一覧 ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="action-title">所持アイテム</div>', unsafe_allow_html=True)

badges = ""
for key, item in ITEMS.items():
    cnt = pet["items"].get(key, 0)
    color = "#1a3a5c" if cnt > 0 else "#111820"
    badges += (f'<span class="item-badge" style="border-color:{"#2a5a8c" if cnt > 0 else "#1a2a3c"};'
               f'color:{"#a0d0f0" if cnt > 0 else "#334455"};">'
               f'{item["icon"]} {item["name"]} × {cnt}</span>')
st.markdown(badges, unsafe_allow_html=True)

# ── プレステージ報酬 ──────────────────────────────────────────────────────────
if not pet.get("is_dead"):
    _new_prestiges = check_prestige_reward(pet)
    if _new_prestiges > 0:
        _total_items = {k: v * _new_prestiges for k, v in PRESTIGE_REWARD.items()}
        _items_str = "、".join(
            f'{ITEMS[k]["icon"]}{ITEMS[k]["name"]}×{v}' for k, v in _total_items.items()
        )
        st.markdown(
            f'<div class="reward-banner">'
            f'✦ <b>プレステージ報酬が届いています！（×{_new_prestiges}回分）</b><br>'
            f'<span style="font-size:0.85rem;">{_items_str} を受け取れます。</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"✦ 受け取る（{_items_str}）", key="claim_prestige"):
            try:
                _pp = int(json.loads(MINIGAME_SAVE.read_text(encoding="utf-8")).get("prestige_points", 0))
            except Exception:
                _pp = pet.get("prestige_reward_claimed", 0) + _new_prestiges
            pet["prestige_reward_claimed"] = _pp
            for k, v in _total_items.items():
                pet["items"][k] = pet["items"].get(k, 0) + v
            pet["message"]          = f"プレステージ報酬受け取り！研究、お疲れ様！"
            pet["last_action_time"] = time.time()
            st.session_state.pet = pet
            save_pet(pet)
            st.rerun()

# ── 進化ガイド ─────────────────────────────────────────────────────────────────
with st.expander("📖 進化の条件", expanded=False):
    st.markdown("""
| 段階 | 必要な年齢 | 必要な世話スコア |
|---|---|---|
| 🥚 たまご → 🔵 ちびガウス | 24時間 | なし |
| 🔵 ちびガウス → 💜 こガウス | 3日間 | 50以上 |
| 💜 こガウス → 🧑‍🔬 ガウス博士 | 7日間 | 160以上 |
| 🧑‍🔬 ガウス博士 → ⭐ スーパーガウス | 14日間 | 450以上 |

**世話スコアの貯め方：** ごはん+4、豪華なごはん+12、おもちゃで遊ぶ+5、手ぶら遊び+2、おそうじ+4、くすり+5

**⚠️ 注意：** ステータスの減少が速いので、**1〜2日以内**にごはんをあげてください。放置しすぎると体力が0になります。

**アイテム入手方法：**
- 🌅 **ログインボーナス** — 24時間ごとにごはん×2＋ランダムアイテム
- 🔬 **研究完了報酬** — 抽出→おもちゃ、COLMAP→シャンプー+くすり、学習→豪華ごはん×3
- ⭐ **マイルストーン** — 世話スコア30/60/100/160/250/400/600達成で報酬
- ✦ **プレステージ報酬** — ミニゲームでプレステージするたびに豪華ごはん×2・くすり×1・シャンプー×2・おもちゃ×1
""")

# ── 名前変更 ──────────────────────────────────────────────────────────────────
with st.expander("✏️ 名前を変える", expanded=False):
    new_name = st.text_input("新しい名前", value=pet["name"], max_chars=20)
    if st.button("名前を変更する"):
        pet["name"] = new_name.strip() or pet["name"]
        st.session_state.pet = pet
        save_pet(pet)
        st.success(f"名前を「{pet['name']}」に変更しました！")

# ── パイプライン進捗ウィジェット ───────────────────────────────────────────────
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

# ── 30秒ごと自動更新 ──────────────────────────────────────────────────────────
if not pet.get("is_dead"):
    import time as _t
    last_refresh = st.session_state.get("_pet_last_refresh", 0)
    if _t.time() - last_refresh > 30:
        st.session_state["_pet_last_refresh"] = _t.time()
        _t.sleep(30)
        st.rerun()
