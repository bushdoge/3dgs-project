# 実データ版のSfM細線欠乏分析（v3）。
# 「青空を背景にした細い暗線」を細線マスクの代用にし、その上のSfM観測点の割合を測る。
#  v1: 空判定に閉処理をかけ白い建物まで空に含めてしまい無効。
#  v2: 点の数え方に±2px膨張をかけたのに面積側は未膨張で不公平だった。
#  v3: 膨張マスクを面積・点の両方に使う。空検出が破綻したフレーム（空の面積<5%）は除外する。
#  2026-09-03: 実験dirを引数で受け取れるようにした（2本目の実データ用。既定は第1波のままなので
#              引数なしで実行すれば従来と同じ結果になる）。使い方: python3 research/wirebench/real/real_wire_sfm.py [実験dir]
import sys
import cv2, numpy as np, pycolmap
from pathlib import Path

EXP = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/20260823_wire_railside_01")
print(f"対象: {EXP}")
rec = pycolmap.Reconstruction(str(EXP / "sparse/0"))
img_dir = EXP / "images"

def masks(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sky = ((b - r > 18) & (v > 150)).astype(np.uint8)
    frac = cv2.blur(sky.astype(np.float32), (21, 21))
    bh = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    wire = ((bh > 25) & (frac > 0.85) & (sky == 0)).astype(np.uint8)
    wire = cv2.morphologyEx(wire, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.dilate(wire, np.ones((5, 5), np.uint8)), sky, frac

print(f"{'画像':<18}{'SfM観測':>8}{'細線画素%':>10}{'線上の点':>9}{'点の%':>8}{'欠乏倍率':>9}")
print("-" * 62)
tp = tw = 0; areas = []
for name in sorted(im.name for im in rec.images.values()):
    im = next(i for i in rec.images.values() if i.name == name)
    bgr = cv2.imread(str(img_dir / name))
    wire_d, sky, frac = masks(bgr)
    if sky.mean() < 0.05:            # 空が写っていない/検出破綻 → 除外
        continue
    H, W = wire_d.shape
    pts = np.array([p.xy for p in im.points2D if p.has_point3D()])
    xi = np.clip(pts[:, 0].astype(int), 0, W - 1); yi = np.clip(pts[:, 1].astype(int), 0, H - 1)
    on = int(wire_d[yi, xi].sum())
    a = wire_d.mean() * 100
    tp += len(pts); tw += on; areas.append(a)
    if len(areas) % 8 == 1:
        s = on / len(pts) * 100
        print(f"{name:<18}{len(pts):>8}{a:>10.3f}{on:>9}{s:>8.3f}{(a/s if s else np.inf):>9.1f}")
ma = float(np.mean(areas)); ts = tw / tp * 100
print("-" * 62)
print(f"{'有効'+str(len(areas))+'枚 合計':<18}{tp:>8}{ma:>10.3f}{tw:>9}{ts:>8.3f}{(ma/ts if ts else np.inf):>9.1f}")
print()
print("※ 細線マスクは±2px膨張済み（面積・点の両方に同じマスクを使用）")
print("※ 電柱の金具など線以外の構造も膨張マスクに一部含まれるため、欠乏倍率は控えめに出る")
