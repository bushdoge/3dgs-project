# バッチキューを監視してジョブを順次自動実行するデーモン
# Streamlit ブラウザを開いていなくてもキューを消化し続ける
# 使用方法: nohup python3 /workspace/scripts/batch_daemon.py &
#           または Streamlit のキューページから起動ボタンで開始

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/workspace")
from queue_helper import QUEUE_FILE, load_queue, save_queue

# ── ファイルパス定数 ──────────────────────────────────────────────────────────
BATCH_STATE_FILE  = "/workspace/tmp/batch_state.json"
DAEMON_PID_FILE   = "/workspace/tmp/batch_daemon.pid"
DAEMON_LOG_FILE   = "/workspace/tmp/batch_daemon.log"
POLL_INTERVAL     = 5   # 秒

# ── ロガー設定 ────────────────────────────────────────────────────────────────
Path(DAEMON_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DAEMON_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("batch_daemon")


# ── 状態ファイル操作 ──────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        p = Path(BATCH_STATE_FILE)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def save_state(state: dict):
    Path(BATCH_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(BATCH_STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )


def clear_state():
    Path(BATCH_STATE_FILE).unlink(missing_ok=True)


# ── プロセス管理 ──────────────────────────────────────────────────────────────

def is_pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        status = Path(f"/proc/{pid}/status")
        if not status.exists():
            return False
        for line in status.read_text().splitlines():
            if line.startswith("State:"):
                return "Z" not in line
        return False
    except Exception:
        return False


def proc_succeeded(log_path: str) -> bool:
    """ログ末尾でエラーがないか確認する"""
    if not log_path or not Path(log_path).exists():
        return True
    tail = Path(log_path).read_text(errors="replace").split("\n")[-10:]
    return not any(
        "ERROR:" in l or ("error" in l.lower() and "failed" in l.lower())
        for l in tail
    )


# ── コマンド構築 ──────────────────────────────────────────────────────────────

def _build_extract_cmd(cfg: dict, exp: str) -> tuple[list, str]:
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
               "--angles", *[f"{y},{p}" for y, p in cfg.get("angles", [(0,0),(90,0),(180,0),(270,0)])]]
    else:
        cmd = [sys.executable, "/workspace/scripts/extract_frames.py",
               "--input", cfg["video_path"], "--output", input_dir,
               "--fps", str(cfg.get("fps", 2.0))]
    return cmd, log


def _build_colmap_cmd(cfg: dict, exp: str) -> tuple[list, str]:
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


def _build_train_cmd(cfg: dict, exp: str) -> tuple[list, str]:
    model_path = cfg.get("model_path", str(Path(exp) / "output"))
    os.makedirs(model_path, exist_ok=True)
    log = str(Path(model_path) / "train_log.txt")
    cmd = [sys.executable, "/workspace/scripts/run_train.py",
           "--source", exp, "--model_path", model_path,
           "--iterations", str(cfg.get("iterations", 30000)),
           "--save_iterations", *[str(i) for i in cfg.get("save_iterations", [7000, 30000])],
           "--test_iterations", *[str(i) for i in cfg.get("test_iterations", [1000, 7000, 15000, 30000])]]
    if cfg.get("eval"):       cmd.append("--eval")
    if cfg.get("resolution"): cmd += ["--resolution", str(cfg["resolution"])]
    return cmd, log


def _build_render_cmd(cfg: dict, exp: str) -> tuple[list, str]:
    model_path = cfg.get("model_path", str(Path(exp) / "output"))
    log = str(Path(model_path) / "render_log.txt")
    cmd = [sys.executable, "/workspace/scripts/run_render.py",
           "-m", model_path, "-s", exp,
           "--iteration", str(cfg.get("iteration", -1))]
    if cfg.get("skip_train"):       cmd.append("--skip_train")
    if cfg.get("skip_test"):        cmd.append("--skip_test")
    if cfg.get("white_background"): cmd.append("--white_background")
    return cmd, log


# ── ジョブ起動 ────────────────────────────────────────────────────────────────

def launch(cmd: list, log_path: str, job: dict, queue: list, step: str) -> int:
    """サブプロセスを起動し、状態を保存する。起動した PID を返す。"""
    os.makedirs(Path(log_path).parent, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)

    pid = proc.pid
    job["status"]       = "running"
    job["current_step"] = step
    job["log_path"]     = log_path
    job["started_at"]   = time.time()
    save_queue(queue)

    save_state({"active": True, "pid": pid, "step": step, "log": log_path,
                "runner": "daemon"})
    log.info(f"[{job.get('exp_name','')}] {step} 開始 (pid={pid})")
    return pid


def start_job(job: dict, queue: list):
    """ジョブの最初のステップを起動する"""
    jtype = job.get("type", "pipeline")
    cfg   = job["config"]
    exp   = job["exp_dir"]
    os.makedirs(exp, exist_ok=True)

    # note.md / pipeline_config.json を書き込む
    note = cfg.get("note_md", "")
    note_path = Path(exp) / "note.md"
    if note or not note_path.exists():
        note_path.write_text(note, encoding="utf-8")

    pcfg_path = Path(exp) / "pipeline_config.json"
    existing_pcfg = {}
    if pcfg_path.exists():
        try:
            existing_pcfg = json.loads(pcfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 既存データを保持しつつ新ジョブのconfigをマージ（既存キーが消えないようにする）
    merged_pcfg = dict(existing_pcfg)
    merged_pcfg.update({k: v for k, v in cfg.items() if k != "note_md"})
    merged_pcfg["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pcfg_path.write_text(
        json.dumps(merged_pcfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if jtype == "pipeline":
        cmd, lp = _build_extract_cmd(cfg, exp)
        launch(cmd, lp, job, queue, "extracting")
    elif jtype == "extract":
        cmd, lp = _build_extract_cmd(cfg, exp)
        launch(cmd, lp, job, queue, "extract")
    elif jtype == "colmap":
        cmd, lp = _build_colmap_cmd(cfg, exp)
        launch(cmd, lp, job, queue, "colmap")
    elif jtype == "train":
        cmd, lp = _build_train_cmd(cfg, exp)
        launch(cmd, lp, job, queue, "training")
    elif jtype == "render":
        cmd, lp = _build_render_cmd(cfg, exp)
        launch(cmd, lp, job, queue, "render")
    else:
        log.warning(f"未対応のジョブ種別: {jtype}")
        job["status"] = "failed"
        save_queue(queue)


def advance(state: dict, queue: list):
    """現在のステップが完了していれば次へ進む"""
    pid      = state.get("pid")
    step     = state.get("step", "")
    log_path = state.get("log", "")

    # まだ実行中
    if is_pid_alive(pid):
        return

    # 完了 — 実行中ジョブを探す
    job = next((j for j in queue if j["status"] == "running"), None)
    if job is None:
        log.info("実行中ジョブなし → キュー停止")
        clear_state()
        return

    jtype = job.get("type", "pipeline")
    ok    = proc_succeeded(log_path)

    if not ok:
        log.error(f"[{job.get('exp_name','')}] {step} でエラー → スキップ")
        job["status"] = "failed"
        save_queue(queue)
        run_next(queue)
        return

    log.info(f"[{job.get('exp_name','')}] {step} 完了")

    if jtype == "pipeline":
        cfg = job["config"]
        exp = job["exp_dir"]
        if step == "extracting":
            cmd, lp = _build_colmap_cmd(cfg, exp)
            launch(cmd, lp, job, queue, "colmap")
        elif step == "colmap":
            cmd, lp = _build_train_cmd(cfg, exp)
            launch(cmd, lp, job, queue, "training")
        elif step == "training":
            job["status"] = "done"
            save_queue(queue)
            run_next(queue)
    else:
        job["status"] = "done"
        save_queue(queue)
        run_next(queue)


def run_next(queue: list):
    """次の pending ジョブを開始する。なければキューを停止する。"""
    # 最新のキューを読み直す（他プロセスが追加した可能性）
    queue = load_queue()
    for job in queue:
        if job["status"] == "pending":
            job["exp_dir"] = job.get("exp_dir") or str(
                Path("/workspace/experiments") / job["exp_name"]
            )
            os.makedirs(job["exp_dir"], exist_ok=True)
            start_job(job, queue)
            return
    log.info("全ジョブ完了 → デーモン待機中")
    clear_state()


# ── シグナルハンドラ ──────────────────────────────────────────────────────────

_shutdown = False

def _handle_sigterm(signum, frame):
    global _shutdown
    log.info("SIGTERM 受信 → シャットダウン")
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT,  _handle_sigterm)


# ── メインループ ──────────────────────────────────────────────────────────────

def main():
    # PID ファイルに自身を登録
    Path(DAEMON_PID_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(DAEMON_PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    log.info(f"バッチデーモン起動 (pid={os.getpid()})")

    try:
        while not _shutdown:
            queue = load_queue()
            state = load_state()

            pid = state.get("pid")
            active = state.get("active", False)

            if active and is_pid_alive(pid):
                # 実行中 — 待機
                pass
            elif active:
                # プロセスが終了した → advance
                advance(state, queue)
            else:
                # アイドル — pending があれば開始
                if any(j["status"] == "pending" for j in queue):
                    run_next(queue)

            time.sleep(POLL_INTERVAL)

    finally:
        Path(DAEMON_PID_FILE).unlink(missing_ok=True)
        log.info("バッチデーモン終了")


if __name__ == "__main__":
    main()
