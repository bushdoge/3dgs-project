# 「帯 vs 半平面」（研究メモ §9as）の消失判定：物体マスク内のコントラスト保持率 = mean|pred−空| / mean|GT−空|。
# 空の色は物体の外側 4〜8px のリング（他物体を除く）の GT 中央値。1≈正しく見えている、0≈消失（空色で塗りつぶされた）。
# 境界誤差（eval_probe.py）は「物体が丸ごと消えた」場合と「見えているが境界が崩れた」場合を区別できないので、これで分ける。
# 使い方: python3 probe_presence.py <exp> <iter> <model_dir名> [...] [--split train|test] [--stride 3]
#   → <exp>/gt/probe_eval[_train]/presence.csv（モデル×物体の保持率の中央値）
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from testviews import test_view_names   # noqa: E402


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("iteration", type=int)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", choices=["test", "train"], default="train")
    ap.add_argument("--stride", type=int, default=3, help="視点を何枚おきに使うか")
    a = ap.parse_args()
    exp = Path(a.exp).resolve()
    objs = [o["id"] for o in json.load(open(exp / "gt" / "probe_objects.json"))]
    names = test_view_names(exp)
    if a.split == "train":
        tv = set(names)
        names = [n for n in test_view_names(exp, hold=1) if n not in tv]
    mdir = exp / "gt" / "mask_obj"
    rows = []
    for i in range(0, len(names), a.stride):
        v = names[i]
        fn = int(v.split("_")[-1])
        gt = cv2.imread(str(exp / "input" / f"{v}.png")).astype(np.float64) / 255
        soft = {o: cv2.imread(str(mdir / f"obj{o:02d}" / f"{fn:04d}.png"), cv2.IMREAD_GRAYSCALE) for o in objs}
        anyobj = np.zeros(gt.shape[:2], bool)
        for m in soft.values():
            anyobj |= m > 12
        preds = {m: cv2.imread(str(exp / m / a.split / f"ours_{a.iteration}" / "renders" / f"{i:05d}.png")).astype(np.float64) / 255
                 for m in a.models}
        for o in objs:
            core = soft[o] > 128
            if core.sum() < 30:
                continue
            own = (soft[o] > 12).astype(np.uint8)
            ring = (cv2.dilate(own, disk(8)) > 0) & (cv2.dilate(own, disk(4)) == 0) & ~anyobj
            if ring.sum() < 30:
                continue
            sky = np.median(gt[ring], 0)
            c_gt = np.abs(gt[core] - sky).mean()
            for m, p in preds.items():
                rows.append(dict(model=m, obj=o, view=v, ret=np.abs(p[core] - sky).mean() / c_gt))
    df = pd.DataFrame(rows)
    tab = df.pivot_table(index="obj", columns="model", values="ret", aggfunc="median")
    out = exp / "gt" / ("probe_eval" if a.split == "test" else "probe_eval_train") / "presence.csv"
    tab.to_csv(out)
    pd.set_option("display.width", 200)
    print(f"マスク内コントラスト保持率（{a.split}、{len(names[::a.stride])} 視点、中央値）")
    print(tab.round(2).to_string())
    low = tab[(tab < 0.5).any(axis=1)]
    print("\n保持率 <0.5 の物体（消失）:", {o: (r < 0.5).sum() for o, r in low.iterrows()} or "なし")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
