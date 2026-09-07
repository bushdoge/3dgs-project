# A1: 「なぜ細線だけが他の高周波エッジより選択的に悪いのか」の機構分析。
#
# §9w-2 で、細線は他の高周波エッジより 4.5〜8.2dB 悪く、その差はどの処方でも縮まないと判明した。
# 原因を絞るため、thin と edge_other を**同じ土俵（画素単位）**で3つの観点から比較する：
#
#  (1) 裾集中度      … 領域内の二乗誤差の上位5%/10%画素が全誤差の何割を占めるか。
#                      §9j は細線の断面単位で「上位5%が33%」と測ったが、断面はエッジには定義できない。
#                      画素単位なら両領域で同じ定義が使える。
#  (2) 位置ずれ寄与  … レンダを±1px動かして画素ごとに誤差最小を取り、誤差が何割減るか。
#                      「形が合っているのに位置がずれているだけ」の割合。
#  (3) コントラスト統制比較（A1b・最重要）
#                    … 局所コントラスト（5x5のmax-min）で層別し、**同じコントラスト帯**の
#                      thin と edge_other を比べる。「空背景の電線はコントラストが最大だから
#                      誤差が大きいだけ」という代替説明を潰すための対照。
#
# 使い方: python3 analyze_selectivity.py <実験dir> <iteration> [model_dir]
#         環境変数 DIL でマスク膨張（既定0＝厳密マスク）
import sys, os, glob
from pathlib import Path
import numpy as np, cv2
_sys_path=__import__('sys').path; _sys_path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent)); from testviews import test_view_names

def read_depth_exr(path):
    """BlenderのZパス(EXR・単一チャンネル'V')を読む。空は無限大で入っている。"""
    import OpenEXR, Imath
    f = OpenEXR.InputFile(str(path))
    h = f.header(); dw = h["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    c = list(h["channels"].keys())[0]
    z = np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)),
                      dtype=np.float32).reshape(H, W)
    return z

EXP = Path(sys.argv[1]).resolve()
IT = int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "output"
DIL = int(os.environ.get("DIL", "0"))
EXCL = 6

md = EXP / MODEL / "test" / f"ours_{IT}"
gts = sorted(glob.glob(str(md / "gt" / "*.png")))
if not gts:
    raise SystemExit(f"レンダが見つからない: {md}")
rens = [str(md / "renders" / Path(g).name) for g in gts]   # 名前で対応付け（深度画像の混入対策）
masks = [str(EXP / "gt" / "mask_thin" / f"{n.split(chr(95))[1]}.png") for n in test_view_names(EXP)][:len(gts)]
k = np.ones((3, 3), np.uint8)

# A1d: edge_other を「物体シルエット（深度不連続＝片側境界）」と「面内テクスチャエッジ」に分割する。
# 両側性仮説（線は両側を背景に挟まれた有界な帯で、エッジは片側だけの境界）の検定。
DEPTH = [str(EXP / "gt" / "depth" / f"{n.split(chr(95))[1]}.exr") for n in test_view_names(EXP)][:len(gts)]
REGIONS = ("thin", "edge_sil", "edge_tex") if DEPTH else ("thin", "edge_other")
E0 = {r: [] for r in REGIONS}   # 画素ごとの二乗誤差
EM = {r: [] for r in REGIONS}   # ±1pxシフト後の最小二乗誤差
CT = {r: [] for r in REGIONS}   # 局所コントラスト

for gp, rp, mp in zip(gts, rens, masks):
    gt, ren = cv2.imread(gp).astype(np.float64), cv2.imread(rp).astype(np.float64)
    if gt.shape != ren.shape:
        ren = cv2.resize(ren, (gt.shape[1], gt.shape[0]))
    thin0 = (cv2.imread(mp, 0) > 127).astype(np.uint8)
    thin = (cv2.dilate(thin0, k, iterations=DIL) > 0) if DIL else thin0.astype(bool)
    thin_excl = cv2.dilate(thin0, k, iterations=EXCL) > 0
    g8 = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    edge = cv2.dilate((cv2.Canny(g8, 50, 150) > 0).astype(np.uint8), k, iterations=max(DIL, 1)) > 0
    edge_other = edge & ~thin_excl

    err0 = ((gt - ren) ** 2).mean(2)
    # ±1px の平行移動で画素ごとに誤差最小を取る（形は合うが位置がずれている分を測る）
    best = err0.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.roll(np.roll(ren, dy, axis=0), dx, axis=1)
            best = np.minimum(best, ((gt - sh) ** 2).mean(2))
    # 局所コントラスト＝5x5のmax-min（GT側で定義。全モデル共通の基準になる）
    gf = g8.astype(np.float32)
    ct = cv2.dilate(gf, np.ones((5, 5), np.uint8)) - cv2.erode(gf, np.ones((5, 5), np.uint8))

    if DEPTH:
        z = read_depth_exr(DEPTH[gts.index(gp)])
        zf = np.where(np.isfinite(z) & (z < 1e9), z, 100.0)   # 空は遠方の定数に置換
        disc = (cv2.dilate(zf, k) - cv2.erode(zf, k)) > 0.5   # 3x3の深度レンジ>0.5m＝不連続
        parts = (("thin", thin), ("edge_sil", edge_other & disc), ("edge_tex", edge_other & ~disc))
    else:
        parts = (("thin", thin), ("edge_other", edge_other))
    for name, m in parts:
        E0[name].append(err0[m]); EM[name].append(best[m]); CT[name].append(ct[m])

for d in (E0, EM, CT):
    for r in d:
        d[r] = np.concatenate(d[r])

print(f"実験 {EXP.name} / model {MODEL} / iter {IT} / DIL={DIL} / テスト{len(gts)}視点\n")
print(f"{'領域':12s}{'PSNR':>8s}{'上位5%':>9s}{'上位10%':>9s}{'位置ずれ寄与':>13s}{'画素数':>10s}")
for r in REGIONS:
    e = E0[r]; n = len(e)
    s = np.sort(e)[::-1]
    top5 = s[:max(1, n // 20)].sum() / e.sum()
    top10 = s[:max(1, n // 10)].sum() / e.sum()
    shift = 1 - EM[r].sum() / e.sum()
    psnr = 10 * np.log10(255.0 ** 2 / e.mean())
    print(f"{r:12s}{psnr:8.2f}{top5:9.1%}{top10:9.1%}{shift:13.1%}{n:10d}")

# ---- A1b: コントラストを揃えた比較 ----
print("\n=== コントラスト統制比較（局所コントラスト＝GTの5x5 max-min で層別）===")
allct = np.concatenate([CT[r] for r in REGIONS])
edges = np.percentile(allct, [0, 20, 40, 60, 80, 100])
hdr = f"{'コントラスト帯':>16s}" + "".join(f"{r:>12s}" for r in REGIONS)
if len(REGIONS) == 3:
    hdr += f"{'thin−sil':>10s}"
print(hdr)
for i in range(5):
    lo, hi = edges[i], edges[i + 1]
    vals = []
    for r in REGIONS:
        sel = (CT[r] >= lo) & (CT[r] <= hi if i == 4 else CT[r] < hi)
        e = E0[r][sel]
        vals.append(10 * np.log10(255.0 ** 2 / e.mean()) if e.size > 50 else np.nan)
    line = f"{lo:7.0f}〜{hi:5.0f}" + "".join(f"{v:12.2f}" for v in vals)
    if len(REGIONS) == 3:
        line += f"{vals[0]-vals[1]:10.2f}"
    print(line)
