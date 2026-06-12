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
from queue_helper import QUEUE_FILE, load_queue, update_job
from job_commands import (
    build_extract_cmd, build_colmap_cmd, build_train_cmd, build_render_cmd,
)

# ── ファイルパス定数 ──────────────────────────────────────────────────────────
BATCH_STATE_FILE  = "/workspace/tmp/batch_state.json"
DAEMON_PID_FILE   = "/workspace/tmp/batch_daemon.pid"
DAEMON_LOG_FILE   = "/workspace/tmp/batch_daemon.log"
POLL_INTERVAL     = 5   # 秒

# ── ロガー設定 ────────────────────────────────────────────────────────────────
# stdoutがログファイルにリダイレクトされて起動されることがあるため、
# 端末から直接実行されたときだけStreamHandlerを追加する（二重出力防止）
Path(DAEMON_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
_handlers = [logging.FileHandler(DAEMON_LOG_FILE, encoding="utf-8")]
if sys.stdout.isatty():
    _handlers.append(logging.StreamHandler(sys.stdout))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=_handlers,
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

# launch() が起動した Popen ハンドル。生死・成否を returncode で正確に判定するために保持する。
# （デーモン再起動などでハンドルがない場合のみ /proc とログのフォールバック判定を使う）
_active_proc: subprocess.Popen | None = None


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


def proc_running(pid) -> bool:
    """対象プロセスが実行中か。Popenハンドルがあればpoll()（ゾンビ回収も兼ねる）"""
    if _active_proc is not None and _active_proc.pid == pid:
        return _active_proc.poll() is None
    return is_pid_alive(pid)


def proc_succeeded(log_path: str) -> bool:
    """ログ末尾でエラーがないか確認する（Popenハンドルがない場合のフォールバック）"""
    if not log_path or not Path(log_path).exists():
        return True
    tail = Path(log_path).read_text(errors="replace").split("\n")[-30:]
    joined = "\n".join(tail)
    # Pythonの未捕捉例外は "ERROR:" を含まないためトレースバックも検出する
    if "Traceback (most recent call last):" in joined:
        return False
    return not any(
        "ERROR:" in l or ("error" in l.lower() and "failed" in l.lower())
        for l in tail
    )


def proc_result(pid, log_path: str) -> bool:
    """終了したプロセスの成否を返す。returncode が取れればそれを優先する"""
    global _active_proc
    if _active_proc is not None and _active_proc.pid == pid:
        rc = _active_proc.poll()
        _active_proc = None
        if rc is not None:
            if rc != 0:
                log.error(f"プロセス(pid={pid})が終了コード {rc} で終了")
            return rc == 0
    return proc_succeeded(log_path)


# ── ジョブ起動 ────────────────────────────────────────────────────────────────

def launch(cmd: list, log_path: str, job: dict, step: str) -> int:
    """サブプロセスを起動し、状態を保存する。起動した PID を返す。"""
    global _active_proc
    os.makedirs(Path(log_path).parent, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    _active_proc = proc

    pid = proc.pid
    update_job(job["id"], status="running", current_step=step,
               log_path=log_path, started_at=time.time())

    save_state({"active": True, "pid": pid, "step": step, "log": log_path,
                "runner": "daemon"})
    log.info(f"[{job.get('exp_name','')}] {step} 開始 (pid={pid})")
    return pid


def start_job(job: dict):
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
        cmd, lp = build_extract_cmd(cfg, exp)
        launch(cmd, lp, job, "extracting")
    elif jtype == "extract":
        cmd, lp = build_extract_cmd(cfg, exp)
        launch(cmd, lp, job, "extract")
    elif jtype == "colmap":
        cmd, lp = build_colmap_cmd(cfg, exp)
        launch(cmd, lp, job, "colmap")
    elif jtype == "train":
        cmd, lp = build_train_cmd(cfg, exp)
        launch(cmd, lp, job, "training")
    elif jtype == "render":
        cmd, lp = build_render_cmd(cfg, exp)
        launch(cmd, lp, job, "render")
    else:
        log.warning(f"未対応のジョブ種別: {jtype}")
        update_job(job["id"], status="failed")


def advance(state: dict, queue: list):
    """現在のステップが完了していれば次へ進む"""
    pid      = state.get("pid")
    step     = state.get("step", "")
    log_path = state.get("log", "")

    # まだ実行中
    if proc_running(pid):
        return

    # 完了 — 実行中ジョブを探す
    job = next((j for j in queue if j["status"] == "running"), None)
    if job is None:
        log.info("実行中ジョブなし → キュー停止")
        clear_state()
        return

    jtype = job.get("type", "pipeline")
    ok    = proc_result(pid, log_path)

    if not ok:
        log.error(f"[{job.get('exp_name','')}] {step} でエラー → スキップ")
        update_job(job["id"], status="failed")
        run_next()
        return

    log.info(f"[{job.get('exp_name','')}] {step} 完了")

    if jtype == "pipeline":
        cfg = job["config"]
        exp = job["exp_dir"]
        if step == "extracting":
            cmd, lp = build_colmap_cmd(cfg, exp)
            launch(cmd, lp, job, "colmap")
        elif step == "colmap":
            cmd, lp = build_train_cmd(cfg, exp)
            launch(cmd, lp, job, "training")
        elif step == "training":
            update_job(job["id"], status="done")
            run_next()
    else:
        update_job(job["id"], status="done")
        run_next()


def run_next():
    """次の pending ジョブを開始する。なければキューを停止する。"""
    # 最新のキューを読み直す（他プロセスが追加した可能性）
    queue = load_queue()
    for job in queue:
        if job["status"] == "pending":
            job["exp_dir"] = job.get("exp_dir") or str(
                Path("/workspace/experiments") / job["exp_name"]
            )
            os.makedirs(job["exp_dir"], exist_ok=True)
            update_job(job["id"], exp_dir=job["exp_dir"])
            start_job(job)
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
    # 多重起動ガード（既に別のデーモンが動いていれば終了）
    try:
        existing_pid = int(Path(DAEMON_PID_FILE).read_text().strip())
        if existing_pid != os.getpid() and is_pid_alive(existing_pid):
            log.warning(f"既にデーモンが稼働中 (pid={existing_pid}) → 終了します")
            return
    except (FileNotFoundError, ValueError):
        pass

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

            if active and proc_running(pid):
                # 実行中 — 待機
                pass
            elif active:
                # プロセスが終了した → advance
                advance(state, queue)
            else:
                # アイドル — pending があれば開始
                if any(j["status"] == "pending" for j in queue):
                    run_next()

            time.sleep(POLL_INTERVAL)

    finally:
        Path(DAEMON_PID_FILE).unlink(missing_ok=True)
        log.info("バッチデーモン終了")


if __name__ == "__main__":
    main()
