# バッチジョブのサブプロセスコマンドを組み立てる共通モジュール
# batch_daemon.py と pages/00_batch.py の両方から import して使う
# （同じロジックを二重実装すると片方だけ修正されるバグの温床になるため一本化）

import os
import sys
from pathlib import Path


def build_extract_cmd(cfg: dict, exp: str) -> tuple[list, str]:
    """フレーム抽出（通常動画 / 360度動画）のコマンドとログパスを返す"""
    input_dir = str(Path(exp) / "input")
    os.makedirs(input_dir, exist_ok=True)
    log = str(Path(exp) / "extract_log.txt")
    if cfg.get("is_360"):
        cmd = [sys.executable, "/workspace/scripts/convert_360.py",
               "--input", cfg["video_path"], "--output", input_dir,
               "--fov", str(cfg.get("fov", 90)),
               "--width", str(cfg.get("out_w", 1024)),
               "--height", str(cfg.get("out_h", 1024)),
               "--fps", str(cfg.get("fps", 1.0)),
               "--angles", *[f"{y},{p}" for y, p in cfg.get("angles", [(0, 0), (90, 0), (180, 0), (270, 0)])]]
    else:
        cmd = [sys.executable, "/workspace/scripts/extract_frames.py",
               "--input", cfg["video_path"], "--output", input_dir,
               "--fps", str(cfg.get("fps", 2.0))]
    return cmd, log


def build_colmap_cmd(cfg: dict, exp: str) -> tuple[list, str]:
    """カメラ姿勢推定（COLMAP / HLoc）のコマンドとログパスを返す"""
    log = str(Path(exp) / "colmap_log.txt")
    if cfg.get("use_hloc"):
        cmd = [sys.executable, "/workspace/scripts/run_hloc.py",
               "--source_path", exp,
               "--feature_type", cfg.get("feature_type", "superpoint_aachen"),
               "--matcher_type", cfg.get("matcher_type", "superpoint+lightglue"),
               "--pair_method", cfg.get("pair_method", "exhaustive"),
               "--retrieval_model", cfg.get("retrieval_model", "netvlad"),
               "--num_matched", str(cfg.get("num_matched", 20))]
    else:
        cmd = [sys.executable, "/workspace/scripts/run_colmap.py",
               "--source_path", exp,
               "--camera_model", cfg.get("camera_model", "OPENCV")]
    return cmd, log


def build_train_cmd(cfg: dict, exp: str) -> tuple[list, str]:
    """3DGS学習のコマンドとログパスを返す"""
    model_path = cfg.get("model_path", str(Path(exp) / "output"))
    os.makedirs(model_path, exist_ok=True)
    log = str(Path(model_path) / "train_log.txt")
    cmd = [sys.executable, "/workspace/scripts/run_train.py",
           "--source", exp, "--model_path", model_path,
           "--iterations", str(cfg.get("iterations", 30000)),
           "--save_iterations", *[str(i) for i in cfg.get("save_iterations", [7000, 30000])],
           "--test_iterations", *[str(i) for i in cfg.get("test_iterations", [1000, 7000, 15000, 30000])]]
    if cfg.get("eval"):
        cmd.append("--eval")
    if cfg.get("resolution"):
        cmd += ["--resolution", str(cfg["resolution"])]
    return cmd, log


def build_render_cmd(cfg: dict, exp: str) -> tuple[list, str]:
    """レンダリングのコマンドとログパスを返す"""
    model_path = cfg.get("model_path", str(Path(exp) / "output"))
    log = str(Path(model_path) / "render_log.txt")
    cmd = [sys.executable, "/workspace/scripts/run_render.py",
           "-m", model_path, "-s", exp,
           "--iteration", str(cfg.get("iteration", -1))]
    if cfg.get("skip_train"):
        cmd.append("--skip_train")
    if cfg.get("skip_test"):
        cmd.append("--skip_test")
    if cfg.get("white_background"):
        cmd.append("--white_background")
    return cmd, log
