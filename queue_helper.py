# バッチキュー共通ユーティリティ
# 各ページから import して add_to_queue() でジョブを追加する
# Streamlitページとバッチデーモンが同じキューファイルを同時に触るため、
# 変更系の操作は queue_lock()（flock）で排他する

import json
import fcntl
import uuid
from contextlib import contextmanager
from pathlib import Path

QUEUE_FILE = "/workspace/tmp/batch_queue.json"
QUEUE_LOCK_FILE = "/workspace/tmp/batch_queue.lock"


@contextmanager
def queue_lock():
    """キューファイルの read-modify-write を排他するための flock"""
    Path(QUEUE_LOCK_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def edit_queue():
    """ロック下で最新キューを読み込み、ブロック終了時に保存する"""
    with queue_lock():
        q = load_queue()
        yield q
        save_queue(q)


def update_job(job_id: str, **fields):
    """ロック下で最新キューを読み直し、該当ジョブのフィールドだけ更新する。
    古いキューのスナップショットで save_queue すると他プロセスの追加分が
    消えるため、状態更新はこの関数を使うこと。"""
    with edit_queue() as q:
        for job in q:
            if job.get("id") == job_id:
                job.update(fields)
                break

JOB_ICONS = {
    "pipeline": "🚀",
    "extract":  "🎞️",
    "colmap":   "📷",
    "train":    "🧠",
    "render":   "🎬",
}

def load_queue() -> list:
    try:
        p = Path(QUEUE_FILE)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except Exception:
        return []

def save_queue(q: list):
    Path(QUEUE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(QUEUE_FILE).write_text(
        json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def add_to_queue(job_type: str, label: str, exp_name: str, exp_dir: str, config: dict) -> dict:
    job = {
        "id":           str(uuid.uuid4())[:8],
        "type":         job_type,
        "label":        label,
        "exp_name":     exp_name,
        "exp_dir":      str(exp_dir),
        "status":       "pending",
        "current_step": "",
        "log_path":     "",
        "config":       config,
    }
    with edit_queue() as q:
        q.append(job)
    return job

def queue_size() -> int:
    return len(load_queue())

def pending_size() -> int:
    return sum(1 for j in load_queue() if j["status"] == "pending")


ACTIVE_TASK_FILE = "/workspace/tmp/active_task.json"

def save_active_task_file(task: dict):
    Path(ACTIVE_TASK_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(ACTIVE_TASK_FILE).write_text(
        json.dumps(task, ensure_ascii=False), encoding="utf-8"
    )

def load_active_task_file() -> dict:
    try:
        p = Path(ACTIVE_TASK_FILE)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}

def clear_active_task_file():
    try:
        Path(ACTIVE_TASK_FILE).unlink(missing_ok=True)
    except Exception:
        pass


def next_exp_name(scene_name: str, base_dir: str = "/workspace/experiments") -> str:
    """
    YYYYMMDD_<scene>_01 形式で次に使える実験名を返す。
    同じ日・シーン名の実験が存在する場合は _02, _03 ... と連番を増やす。
    ディスク上のフォルダだけでなくキュー内の名前も重複チェックする。
    """
    from datetime import datetime as _dt
    date_prefix = _dt.now().strftime("%Y%m%d")
    base        = f"{date_prefix}_{scene_name}"

    # ディスク上の既存フォルダ
    existing = {p.name for p in Path(base_dir).iterdir() if p.is_dir()} \
               if Path(base_dir).exists() else set()

    # キュー内の名前も追加（まだ実行されていないものも含む）
    for job in load_queue():
        existing.add(job.get("exp_name", ""))

    n = 1
    while True:
        candidate = f"{base}_{n:02d}"
        if candidate not in existing:
            return candidate
        n += 1
