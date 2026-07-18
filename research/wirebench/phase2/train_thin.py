# Phase 2「裾を潰すロス設計＋スーパーサンプル教師」の学習スクリプト（mip-splatting train.py ベース。
# 設計図: memo/design_20260717_phase2.md（第1波）・memo/design_20260718_p2wave2.md（第2波））
# 本家 train.py との差分は (1) network_gui の除去 (2) L1項の画素重み付け3方式 (3) 視点適応サンプリング
# (4) スーパーサンプル教師 のみ。densification・スケジュール・SSIM項は本家と同一。
#   --wire_weight W --wire_mask_dir gt/mask_thin  … P2-1 オラクル: 細線マスク画素のL1重みをW倍（上界診断）
#   --adaptive_gamma G                            … P2-2a mask-free: 視点ごとの誤差EMAマップの冪G乗を重みに
#   --topk_frac F --topk_weight W                 … P2-2b mask-free: 現在誤差の上位F%画素の重みをW倍
#   --view_adapt                                  … P2-3 mask-free: 視点選択確率∝視点ロスEMA（一様と半々で混合）
#   --supersample N --ss_filter {ss,base}         … P2-4a: N倍解像度でレンダ→N×N平均プールで1×に落として
#                                                   GT(1×)とロス。ss_filter=ss なら3DフィルタもN×カメラで計算
#   --centroid_weight L                           … P2-4c オラクル: 細線断面のソフト重心差|Δc|をロスに加算
#                                                   （位置ずれへの直接介入＝上界診断。設計図 roadmap §3.2）
# 実行は tools/mip-splatting へコピーして venv で:
#   cp research/wirebench/phase2/train_thin.py /workspace/tools/mip-splatting/
#   cd /workspace/tools/mip-splatting && venv/bin/python train_thin.py -s <exp> -m <exp>/output_mip_ss2_f2x \
#       --eval --iterations 7000 --supersample 2 --ss_filter ss

import os
import numpy as np
import cv2
import torch
import random
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def make_ss_camera(cam, n):
    """N倍解像度のレンダ用カメラプロキシ。render()/compute_3D_filter()が参照する属性のみ持つ"""
    from types import SimpleNamespace
    return SimpleNamespace(
        uid=cam.uid, image_name=cam.image_name, R=cam.R, T=cam.T,
        FoVx=cam.FoVx, FoVy=cam.FoVy,
        image_width=cam.image_width * n, image_height=cam.image_height * n,
        focal_x=cam.focal_x * n, focal_y=cam.focal_y * n,
        world_view_transform=cam.world_view_transform,
        full_proj_transform=cam.full_proj_transform,
        camera_center=cam.camera_center)


def load_wire_masks(source_path, mask_dir, cameras):
    """学習カメラごとの細線マスク(bool, H×W)をGPUに載せる。frame_0001 → <mask_dir>/0001.png"""
    masks = {}
    for cam in cameras:
        num = cam.image_name.split("_")[1]
        m = cv2.imread(os.path.join(source_path, mask_dir, num + ".png"), 0)
        if m is None:
            raise FileNotFoundError(f"細線マスクが見つからない: {mask_dir}/{num}.png ({cam.image_name})")
        if m.shape != (cam.image_height, cam.image_width):
            m = cv2.resize(m, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
        masks[cam.uid] = torch.from_numpy(m > 127).cuda()
    return masks


LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def soft_centroid(prof, eps=1e-6):
    """輝度プロファイル(n,L)のディップのソフト重心と質量。微分可能（tail_map系の重心のtorch版）"""
    bg = (prof[:, :3].mean(1) + prof[:, -3:].mean(1)) / 2
    dip = torch.relu(bg[:, None] - prof)
    mass = dip.sum(1)
    x = torch.arange(prof.shape[1], device=prof.device, dtype=prof.dtype)
    pos = (dip * x).sum(1) / (mass + eps)
    return pos, mass


def build_centroid_targets(cameras, masks, win=14):
    """視点ごとに細線断面のインデックスとGT重心を前計算。返り値: uid → dict(tensors on GPU)"""
    targets = {}
    x = torch.arange(2 * win + 1)
    for cam in cameras:
        mb = masks[cam.uid].cpu().numpy()
        h, w = mb.shape
        cols, cys = [], []
        for cx in range(0, w, 4):
            ys = np.where(mb[:, cx])[0]
            if len(ys) == 0:
                continue
            for run in np.split(ys, np.where(np.diff(ys) > 1)[0] + 1):
                if len(run) > win:
                    continue
                cy = int(run.mean())
                if cy - win < 0 or cy + win + 1 > h:
                    continue
                cols.append(cx)
                cys.append(cy)
        if not cols:
            continue
        cxs = torch.tensor(cols)
        idx_y = torch.tensor(cys)[:, None] - win + x[None, :]          # (n, 2w+1)
        gray_gt = (cam.original_image.cpu() * LUMA).sum(0)             # (H,W)
        prof_gt = gray_gt[idx_y, cxs[:, None]]
        c_gt, m_gt = soft_centroid(prof_gt)
        ok = m_gt > 0.03 * (2 * win + 1)      # GTディップが十分ある断面のみ（輝度0-1スケール）
        targets[cam.uid] = {"cx": cxs[ok].cuda(), "iy": idx_y[ok].cuda(),
                            "c_gt": c_gt[ok].cuda(), "m_gt": m_gt[ok].cuda()}
    return targets


def centroid_loss(image, tgt):
    """レンダ画像(3,H,W)に対する断面重心整合ロス（px単位のHuber平均）"""
    gray = (image * LUMA.to(image.device)).sum(0)
    prof = gray[tgt["iy"], tgt["cx"][:, None]]
    c_r, m_r = soft_centroid(prof)
    gate = (m_r.detach() > 0.3 * tgt["m_gt"])   # 線がまだ描けていない断面は位置を問わない
    if gate.sum() == 0:
        return image.sum() * 0.0
    return torch.nn.functional.huber_loss(c_r[gate], tgt["c_gt"][gate], delta=1.0)


def training(dataset, opt, pipe, args, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    trainCameras = scene.getTrainCameras().copy()
    n_ss = args.supersample
    ss_cams = {c.uid: make_ss_camera(c, n_ss) for c in trainCameras} if n_ss > 1 else None
    filterCameras = (list(ss_cams.values()) if (n_ss > 1 and args.ss_filter == "ss")
                     else trainCameras)
    if n_ss > 1:
        print(f"[thin] スーパーサンプル教師 ×{n_ss}（3Dフィルタ: {args.ss_filter}={'N×' if args.ss_filter=='ss' else '1×'}カメラ）")
    gaussians.compute_3D_filter(cameras=filterCameras)

    wire_masks = None
    if args.wire_weight != 1.0:
        wire_masks = load_wire_masks(dataset.source_path, args.wire_mask_dir, trainCameras)
        print(f"[thin] オラクルマスク重み w={args.wire_weight}（{len(wire_masks)}視点）")
    ctr_targets = None
    if args.centroid_weight > 0:
        cmasks = load_wire_masks(dataset.source_path, args.wire_mask_dir, trainCameras)
        ctr_targets = build_centroid_targets(trainCameras, cmasks)
        n_sec = sum(len(t["c_gt"]) for t in ctr_targets.values())
        print(f"[thin] 断面重心整合ロス λ={args.centroid_weight}（{len(ctr_targets)}視点・{n_sec}断面）")
    err_ema = {}      # P2-2a: uid → 誤差EMAマップ(H×W, fp16)
    view_ema = {}     # P2-3:  uid → 視点ロスEMA(スカラー)
    if args.adaptive_gamma > 0:
        print(f"[thin] mask-free画素適応重み γ={args.adaptive_gamma}")
    if args.topk_frac > 0:
        print(f"[thin] mask-free上位{args.topk_frac}%画素強調 ×{args.topk_weight}")
    if args.view_adapt:
        print("[thin] 視点適応サンプリング（p = 0.5·一様 + 0.5·∝ロスEMA）")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 視点選択：既定は本家と同じepochスタック。view_adapt時はロスEMA比例（一様と半々）
        if args.view_adapt and len(view_ema) == len(trainCameras):
            w = torch.tensor([view_ema[c.uid] for c in trainCameras], dtype=torch.float64)
            p = 0.5 / len(trainCameras) + 0.5 * (w / w.sum())
            viewpoint_cam = trainCameras[int(torch.multinomial(p.float(), 1))]
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_cam = ss_cams[viewpoint_cam.uid] if n_ss > 1 else viewpoint_cam
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(render_cam.image_height), int(render_cam.image_width), 2),
                                         dtype=torch.float32, device="cuda") - 0.5
        else:
            subpixel_offset = None
        render_pkg = render(render_cam, gaussians, pipe, background,
                            kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"],
            render_pkg["visibility_filter"], render_pkg["radii"])
        if n_ss > 1:
            image = torch.nn.functional.avg_pool2d(image[None], n_ss)[0]   # N×→1×（教師は1×のまま）

        gt_image = viewpoint_cam.original_image.cuda()

        # ---- L1項の画素重み付け（差分の本体） ----
        d = torch.abs(image - gt_image)                       # (3,H,W)
        uid = viewpoint_cam.uid
        if wire_masks is not None:
            w = torch.ones_like(d[0])
            w[wire_masks[uid]] = args.wire_weight
        elif args.adaptive_gamma > 0 and uid in err_ema:
            e = err_ema[uid].float()
            w = (e / e.mean().clamp_min(1e-8)).pow(args.adaptive_gamma).clamp(0.25, 8.0)
        elif args.topk_frac > 0:
            dm = d.mean(0).detach()
            k = max(1, int(dm.numel() * args.topk_frac / 100))
            thresh = dm.flatten().topk(k).values[-1]
            w = torch.where(dm >= thresh, torch.full_like(dm, args.topk_weight), torch.ones_like(dm))
        else:
            w = None
        if w is None:
            Ll1 = d.mean()
        else:
            Ll1 = (d * w[None]).sum() / (w.sum() * 3)         # 重み付き平均＝ロススケールを保つ
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if ctr_targets is not None and uid in ctr_targets:
            loss = loss + args.centroid_weight * centroid_loss(image, ctr_targets[uid])
        loss.backward()

        with torch.no_grad():
            if args.adaptive_gamma > 0:
                dm = d.mean(0).half()
                err_ema[uid] = dm if uid not in err_ema else (0.7 * err_ema[uid].float() + 0.3 * dm.float()).half()
            if args.view_adapt:
                li = float(loss)
                view_ema[uid] = li if uid not in view_ema else 0.6 * view_ema[uid] + 0.4 * li

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                report_test_psnr(iteration, scene, render, (pipe, background, dataset.kernel_size))
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians (n={})".format(iteration, gaussians.get_xyz.shape[0]))
                scene.save(iteration)

            # Densification（本家と同一）
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=filterCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    gaussians.compute_3D_filter(cameras=filterCameras)

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return None


def report_test_psnr(iteration, scene, renderFunc, renderArgs):
    from utils.image_utils import psnr
    torch.cuda.empty_cache()
    cams = scene.getTestCameras()
    if not cams:
        return
    psnr_test = sum(psnr(torch.clamp(renderFunc(vp, scene.gaussians, *renderArgs)["render"], 0, 1),
                         torch.clamp(vp.original_image.to("cuda"), 0, 1)).mean().double()
                    for vp in cams) / len(cams)
    print("\n[ITER {}] Evaluating test: PSNR {}".format(iteration, psnr_test))
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Phase2 thin-structure training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--wire_mask_dir", type=str, default="gt/mask_thin")
    parser.add_argument("--wire_weight", type=float, default=1.0)
    parser.add_argument("--adaptive_gamma", type=float, default=0.0)
    parser.add_argument("--topk_frac", type=float, default=0.0)
    parser.add_argument("--topk_weight", type=float, default=10.0)
    parser.add_argument("--view_adapt", action="store_true")
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--ss_filter", choices=["ss", "base"], default="ss")
    parser.add_argument("--centroid_weight", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.wire_weight != 1.0 and (args.adaptive_gamma > 0 or args.topk_frac > 0):
        raise SystemExit("重み方式は1つだけ指定する（wire_weight / adaptive_gamma / topk_frac）")

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), pp.extract(args), args,
             args.test_iterations, args.save_iterations)
    print("\nTraining complete.")
