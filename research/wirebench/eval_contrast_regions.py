# 細線劣化が「細線特有」かを検定する対照実験。
#
# 監査で出た指摘：「細線PSNRは背景より6.8〜11.3dB悪い」の背景は空を含むシーン全体なので、
# 『高周波で画素数の少ない領域なら何でも悪く出るのでは？』という反論に答えられない。
# そこで同一シーン・同一レンダの中を3領域に分けて同じ指標で測る：
#   thin      … GT細線マスク（電線＋フェンス棒）±2px膨張。eval_wires の md2 と同じ作り
#   edge_other… GT画像のCannyエッジ（箱テクスチャ・柵の輪郭など）から thin を除いた高周波領域
#   flat      … 上記いずれでもない領域（空・地面・面の内部）
# thin が edge_other より有意に悪ければ「細線特有」と言える。同程度なら
# 「高周波領域一般が悪いだけ」であり、主張を弱める必要がある。
#
# 使い方: python3 eval_contrast_regions.py <実験dir> <iteration> [model_dir]
import sys, glob
from pathlib import Path
import numpy as np, cv2
_sys_path=__import__('sys').path; _sys_path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent)); from testviews import test_view_names

EXP = Path(sys.argv[1]).resolve()
IT = int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "output"
DIL = int(__import__("os").environ.get("DIL", "2"))   # eval_wires の md2 と揃える。DIL=0 で厳密マスク
EXCL = 6         # edge_other から thin を除くときの余裕（ハロー混入を防ぐ）

md = EXP / MODEL / "test" / f"ours_{IT}"
gts = sorted(glob.glob(str(md / "gt" / "*.png")))
if not gts:
    raise SystemExit(f"レンダが見つからない: {md}")
# renders/ には深度画像などが混ざる実装がある（RAIN-GSは *_depth.png を同居させる）ので、
# ソート順で対応付けるとずれる。**GTと同名のファイルだけを拾う**。
rens = []
for g in gts:
    r = md / "renders" / Path(g).name
    if not r.exists():
        raise SystemExit(f"対応するレンダが無い: {r}")
    rens.append(str(r))
masks = sorted(glob.glob(str(EXP / "gt" / "mask_thin" / "*.png")))

# テスト視点はGT全体から8枚おき（3DGSのllffhold=8）。mask_thin は全フレーム分あるので対応付ける
test_masks = [str(EXP / "gt" / "mask_thin" / f"{n.split(chr(95))[1]}.png") for n in test_view_names(EXP)][:len(gts)]
if len(test_masks) != len(gts):
    raise SystemExit(f"マスクとレンダの枚数が不一致: {len(test_masks)} vs {len(gts)}")

k = np.ones((3, 3), np.uint8)

def psnr(a, b, m):
    d = ((a.astype(np.float64) - b.astype(np.float64)) ** 2)[m]
    return np.nan if d.size == 0 else 10 * np.log10(255.0 ** 2 / max(d.mean(), 1e-9))

acc = {r: [] for r in ("thin", "edge_other", "flat")}
px = {r: 0 for r in acc}
for gp, rp, mp in zip(gts, rens, test_masks):
    gt, ren = cv2.imread(gp), cv2.imread(rp)
    if gt.shape != ren.shape:
        ren = cv2.resize(ren, (gt.shape[1], gt.shape[0]))
    thin0 = (cv2.imread(mp, 0) > 127).astype(np.uint8)
    thin = cv2.dilate(thin0, k, iterations=DIL) > 0
    thin_excl = cv2.dilate(thin0, k, iterations=EXCL) > 0

    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    edge = cv2.dilate((cv2.Canny(gray, 50, 150) > 0).astype(np.uint8), k, iterations=DIL) > 0
    edge_other = edge & ~thin_excl
    flat = ~edge & ~thin_excl

    for name, m in (("thin", thin), ("edge_other", edge_other), ("flat", flat)):
        m3 = np.repeat(m[..., None], 3, 2)
        acc[name].append(psnr(gt, ren, m3))
        px[name] += int(m.sum())

n = len(gts)
print(f"実験 {EXP.name} / model {MODEL} / iter {IT} / テスト{n}視点")
print(f"{'領域':12s}{'PSNR平均':>10s}{'最悪視点':>10s}{'画素/視点':>12s}{'画面占有':>9s}")
tot = gt.shape[0] * gt.shape[1]
for name in ("thin", "edge_other", "flat"):
    a = np.array(acc[name])
    print(f"{name:12s}{np.nanmean(a):10.2f}{np.nanmin(a):10.2f}{px[name]/n:12.0f}{px[name]/n/tot*100:8.2f}%")
a_thin, a_edge = np.array(acc["thin"]), np.array(acc["edge_other"])
print(f"\n→ thin − edge_other = {np.nanmean(a_thin)-np.nanmean(a_edge):+.2f} dB "
      f"（負なら細線の方が悪い＝細線特有性の証拠）")
