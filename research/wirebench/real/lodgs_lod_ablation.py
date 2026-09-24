# LOD-GS の機構確認（研究メモ §9af 末尾 4 の仮説の検定）：学習済みモデルを「LOD あり／なし」でレンダし比べる。
# LOD-GS はガウシアンごとの係数 _coefs から、サンプリング間隔（深度/焦点距離）に応じた補正量を計算し、
# 不透明度（logit）・スケール（log）・色に足している（scene/gaussian_model.py の LOD_control）。
# render(..., LOD=False) にするとこの補正が丸ごと消えるので、「補正が電線の見え方にどれだけ効いているか」を
# 学習をやり直さずに測れる。合成（LOD-GSが電線で最良）と実データ（優位が消えた）で比べて反転の理由に迫る。
#
# 出力: <model_path>_noLOD/test/ours_<iter>/{renders,gt}（他のファイルは元モデルdirへのシンボリックリンク）
#       → eval_wires.py / research/wirebench/real/eval_real_lodgs.py でそのまま評価できる
# 検算: LOD ありで再レンダした画像が、保存済みの renders/ と一致するかを確認する（読み込みの正しさ）
# 使い方（LOD-GS の venv で、tools/LOD-GS から実行）:
#   cd /workspace/tools/LOD-GS && ./venv/bin/python /workspace/research/wirebench/real/lodgs_lod_ablation.py -m <model_path> --iteration 30000
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch, torchvision
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, GMMParams, get_combined_args
from utils.general_utils import safe_state
from PIL import Image

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser); gmmp = GMMParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
args = get_combined_args(parser)
safe_state(True)
try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SEP = True
except Exception:
    SEP = False

ds = model.extract(args)
src = os.path.abspath(ds.model_path.rstrip("/"))
dst = src + "_noLOD"
with torch.no_grad():
    g = GaussianModel(ds.sh_degree, args=gmmp.extract(args))
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    it = scene.loaded_iter
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    pipe = pipeline.extract(args)

    # 出力dir：test/ 以外は元へのシンボリックリンク（eval側が cameras.json 等を読んでも元と同じになる）
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name in ("test", "train") or name.startswith("chkpnt"):
            continue
        link = os.path.join(dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(src, name), link)
    rdir = os.path.join(dst, "test", f"ours_{it}", "renders"); gdir = os.path.join(dst, "test", f"ours_{it}", "gt")
    os.makedirs(rdir, exist_ok=True); os.makedirs(gdir, exist_ok=True)
    saved = os.path.join(src, "test", f"ours_{it}", "renders")

    # LOD 補正量を横取りして集計する（LOD_control の中で計算される LOD_residual の値）
    stats = []
    orig_res = g.LOD_residual
    def spy(x):
        r = orig_res(x)
        dop = r[:, 0]; dsc = r[:, 1:4].mean(1); dcol = r[:, 4:].abs().mean(1)
        stats.append(dict(
            dop_abs_med=float(dop.abs().median()), dop_mean=float(dop.mean()),
            scale_factor_med=float(torch.exp(dsc).median()),
            scale_shrink_frac=float((torch.exp(dsc) < 0.9).float().mean()),
            scale_grow_frac=float((torch.exp(dsc) > 1.1).float().mean()),
            dcol_abs_med=float(dcol.median())))
        return r
    g.LOD_residual = spy

    rows = []
    for idx, view in enumerate(scene.getTestCameras()):
        on = render(view, g, pipe, bg, use_trained_exp=ds.train_test_exp, separate_sh=SEP, LOD=True)["render"].clamp(0, 1)
        off = render(view, g, pipe, bg, use_trained_exp=ds.train_test_exp, separate_sh=SEP, LOD=False)["render"].clamp(0, 1)
        gt = view.original_image[0:3].cuda()
        torchvision.utils.save_image(off, os.path.join(rdir, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gdir, f"{idx:05d}.png"))
        # 検算：保存済みレンダ（8bit）との最大差
        ref = torch.from_numpy(np.asarray(Image.open(os.path.join(saved, f"{idx:05d}.png")).convert("RGB"))).permute(2, 0, 1).float().cuda() / 255
        on8 = torch.round(on * 255) / 255
        psnr = lambda a, b: float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12)))
        rows.append(dict(view=idx, recheck_maxdiff=float((on8 - ref).abs().max()), psnr_on=psnr(on, gt), psnr_off=psnr(off, gt),
                         on_vs_off=psnr(on, off), **stats[-1]))   # 直前の LOD 有レンダで記録した補正量
        r = rows[-1]
        print(f"view {idx:2d}  検算maxdiff={r['recheck_maxdiff']:.4f}  PSNR LOD有={r['psnr_on']:.2f} 無={r['psnr_off']:.2f}  "
              f"有vs無={r['on_vs_off']:.1f}dB  |Δ不透明度logit|中央値={r['dop_abs_med']:.3f}  "
              f"スケール倍率中央値={r['scale_factor_med']:.3f}（0.9未満 {r['scale_shrink_frac']*100:.0f}% / 1.1超 {r['scale_grow_frac']*100:.0f}%）",
              flush=True)
    json.dump(rows, open(os.path.join(dst, "lod_ablation_stats.json"), "w"), indent=1)
    m = lambda k: np.mean([r[k] for r in rows])
    print(f"平均: PSNR LOD有 {m('psnr_on'):.2f} / 無 {m('psnr_off'):.2f}（差 {m('psnr_on')-m('psnr_off'):+.2f}）  有vs無 {m('on_vs_off'):.1f}dB  "
          f"スケール倍率中央値 {m('scale_factor_med'):.3f}  → {dst}")
