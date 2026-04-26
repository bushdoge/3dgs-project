# バッチキュー共通ユーティリティ
# 各ページから import して add_to_queue() でジョブを追加する

import json
import uuid
from pathlib import Path

QUEUE_FILE = "/workspace/tmp/batch_queue.json"

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
    q = load_queue()
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
    q.append(job)
    save_queue(q)
    return job

def queue_size() -> int:
    return len(load_queue())

def pending_size() -> int:
    return sum(1 for j in load_queue() if j["status"] == "pending")
