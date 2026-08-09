# シーンの見え方を定量するスクリプト（複数シーン検証用）
# GT深度EXRとGT細線マスクから、シーン間で比較したい「背景の性質」を測る:
#   sky_frac        画面に占める空（無限遠＝深度が有限でない画素）の割合
#   wire_bg_sky     細線マスクに隣接する背景画素のうち空である割合
#                   ＝「細線が空を背景にしている度合い」（§9mの裾＝空背景×近距離視点の統制指標）
#   thin_px_share   細線マスクの画素占有率
#
# 使い方: python3 scene_stats.py <実験dir> [<実験dir> ...]
#   例: python3 scene_stats.py experiments/20260726_wb_r15mm_f80_open{,_Lstreet}

import sys
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath


def read_depth(path):
    f = OpenEXR.InputFile(str(path))
    h = f.header()
    ch = list(h["channels"].keys())[0]
    dw = h["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    return np.frombuffer(f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT)),
                         np.float32).reshape(H, W)


def stats(exp):
    exp = Path(exp)
    masks = sorted((exp / "gt" / "mask_thin").glob("*.png"))
    if not masks:
        return None
    sky_f, bg_sky_f, thin_f = [], [], []
    k = np.ones((5, 5), np.uint8)
    for mp in masks:
        dp = exp / "gt" / "depth" / f"{mp.stem}.exr"
        if not dp.exists():
            continue
        depth = read_depth(dp)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
        sky = ~np.isfinite(depth) | (depth > 1e4)      # 無限遠＝空
        ring = (cv2.dilate(m.astype(np.uint8), k, iterations=1) > 0) & ~m
        sky_f.append(sky.mean())
        thin_f.append(m.mean())
        if ring.any():
            bg_sky_f.append(sky[ring].mean())
    return {"frames": len(sky_f),
            "sky_frac": float(np.mean(sky_f)),
            "wire_bg_sky": float(np.mean(bg_sky_f)) if bg_sky_f else float("nan"),
            "thin_px_share": float(np.mean(thin_f))}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__ or "使い方: python3 scene_stats.py <実験dir> [...]")
    print(f"{'exp':40s} {'frames':>6s} {'sky%':>7s} {'細線背後の空%':>13s} {'細線画素%':>9s}")
    for a in sys.argv[1:]:
        s = stats(a)
        if s is None:
            print(f"{Path(a).name:40s}  （マスクが無い）")
            continue
        print(f"{Path(a).name:40s} {s['frames']:6d} {s['sky_frac']*100:7.2f} "
              f"{s['wire_bg_sky']*100:13.2f} {s['thin_px_share']*100:9.3f}")
