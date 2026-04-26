# バッチ実験ページ
# 複数の実験設定をキューに登録し、フレーム抽出→COLMAP→学習を順番に自動実行する

import json
import os
import signal
import sys
import subprocess
import time
import uuid
from pathlib import Path

import streamlit as st

QUEUE_FILE = "/workspace/tmp/batch_queue.json"
LOG_DIR    = Path("/workspace/tmp/batch_logs")

# ─── キュー永続化 ─────────────────────────────────────────────────────────────

def _load_queue() -> list:
    try:
        p = Path(QUEUE_FILE)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def _save_queue(q: list):
    Path(QUEUE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(QUEUE_FILE).write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")

# ─── セッション初期化 ──────────────────────────────────────────────────────────

if "bq_active" not in st.session_state:
    st.session_state.bq_active  = False
if "bq_proc"   not in st.session_state:
    st.session_state.bq_proc    = None
if "bq_pid"    not in st.session_state:
    st.session_state.bq_pid     = None
if "bq_step"   not in st.session_state:
    st.session_state.bq_step    = "idle"
if "bq_log"    not in st.session_state:
    st.session_state.bq_log     = None

# 実行中でない時は常にファイルから再読み込みして他ページの変更を反映する
if not st.session_state.bq_active:
    st.session_state.bq_queue = _load_queue()
elif "bq_queue" not in st.session_state:
    st.session_state.bq_queue = _load_queue()

# ─── ステップ進行ロジック ──────────────────────────────────────────────────────

def _current_job():
    """現在実行中（または次に実行する）ジョブを返す"""
    for job in st.session_state.bq_queue:
        if job["status"] in ("pending", "running"):
            return job
    return None

def _proc_done() -> bool:
    """現在のサブプロセスが終了しているか"""
    proc = st.session_state.bq_proc
    pid  = st.session_state.bq_pid
    if proc is not None:
        return proc.poll() is not None
    if pid:
        p = Path(f"/proc/{pid}/status")
        if not p.exists():
            return True
        for line in p.read_text().splitlines():
            if line.startswith("State:"):
                return "Z" in line
    return True

def _proc_ok() -> bool:
    """終了したプロセスが正常終了か（ログ末尾でERROR判定）"""
    proc = st.session_state.bq_proc
    if proc is not None:
        return proc.returncode == 0
    log = st.session_state.bq_log
    if log and Path(log).exists():
        tail = Path(log).read_text(errors="replace").split("\n")[-10:]
        return not any("ERROR:" in l for l in tail)
    return True

def _start_step(job: dict, step: str):
    """指定ステップのサブプロセスを起動する"""
    exp_dir   = job["exp_dir"]
    cfg       = job["config"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if step == "extracting":
        log_path = str(Path(exp_dir) / "extract_log.txt")
        input_dir = str(Path(exp_dir) / "input")
        os.makedirs(input_dir, exist_ok=True)
        if cfg.get("is_360"):
            cmd = [
                sys.executable, "/workspace/scripts/convert_360.py",
                "--input", cfg["video_path"],
                "--output", input_dir,
                "--fov",    str(cfg.get("fov", 90)),
                "--width",  str(cfg.get("out_w", 1024)),
                "--height", str(cfg.get("out_h", 1024)),
                "--fps",    str(cfg.get("fps", 1.0)),
                "--angles", *[f"{y},{p}" for y, p in cfg.get("angles", [(0,0),(90,0),(180,0),(270,0)])],
            ]
        else:
            cmd = [
                sys.executable, "/workspace/scripts/extract_frames.py",
                "--input",  cfg["video_path"],
                "--output", input_dir,
                "--fps",    str(cfg.get("fps", 2.0)),
            ]

    elif step == "colmap":
        log_path = str(Path(exp_dir) / "colmap_log.txt")
        if cfg.get("use_hloc"):
            cmd = [
                sys.executable, "/workspace/scripts/run_hloc.py",
                "--source_path",     exp_dir,
                "--feature_type",    cfg.get("feature_type", "superpoint_aachen"),
                "--matcher_type",    cfg.get("matcher_type", "superpoint+lightglue"),
                "--pair_method",     cfg.get("pair_method", "exhaustive"),
                "--retrieval_model", cfg.get("retrieval_model", "netvlad"),
                "--num_matched",     str(cfg.get("num_matched", 20)),
            ]
        else:
            cmd = [
                sys.executable, "/workspace/scripts/run_colmap.py",
                "--source_path",  exp_dir,
                "--camera_model", cfg.get("camera_model", "OPENCV"),
            ]

    elif step == "training":
        model_path = str(Path(exp_dir) / "output")
        log_path   = str(Path(model_path) / "train_log.txt")
        os.makedirs(model_path, exist_ok=True)
        cmd = [
            sys.executable, "/workspace/scripts/run_train.py",
            "--source",      exp_dir,
            "--model_path",  model_path,
            "--iterations",  str(cfg.get("iterations", 30000)),
            "--save_iterations", *[str(i) for i in cfg.get("save_iterations", [7000, 30000])],
            "--test_iterations", *[str(i) for i in cfg.get("test_iterations", [1000, 7000, 15000, 30000])],
        ]
        if cfg.get("eval"):
            cmd.append("--eval")
    else:
        return

    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    st.session_state.bq_proc = proc
    st.session_state.bq_pid  = proc.pid
    st.session_state.bq_step = step
    st.session_state.bq_log  = log_path
    job["status"]       = "running"
    job["current_step"] = step
    job["log_path"]     = log_path
    _save_queue(st.session_state.bq_queue)

def _advance():
    """現在のステップが完了したら次のステップ or 次のジョブへ進む"""
    if not _proc_done():
        return

    job  = _current_job()
    if job is None:
        st.session_state.bq_active = False
        st.session_state.bq_step   = "idle"
        return

    step = st.session_state.bq_step
    ok   = _proc_ok()
    st.session_state.bq_proc = None

    if not ok:
        job["status"] = "failed"
        job["current_step"] = step
        _save_queue(st.session_state.bq_queue)
        # 失敗しても次のジョブへ
        _next_job()
        return

    # 正常完了 → 次ステップへ
    if step == "extracting":
        _start_step(job, "colmap")
    elif step == "colmap":
        _start_step(job, "training")
    elif step == "training":
        job["status"] = "done"
        _save_queue(st.session_state.bq_queue)
        _next_job()

def _next_job():
    """次のpendingジョブを開始する"""
    for job in st.session_state.bq_queue:
        if job["status"] == "pending":
            exp_dir = str(Path("/workspace/experiments") / job["exp_name"])
            Path(exp_dir).mkdir(parents=True, exist_ok=True)
            job["exp_dir"] = exp_dir
            _start_step(job, "extracting")
            return
    # 全ジョブ完了
    st.session_state.bq_active = False
    st.session_state.bq_step   = "idle"
    _save_queue(st.session_state.bq_queue)

def _stop_batch():
    proc = st.session_state.bq_proc
    pid  = st.session_state.bq_pid
    kill_pid = proc.pid if (proc and proc.poll() is None) else pid
    if kill_pid:
        try: os.kill(kill_pid, signal.SIGTERM)
        except Exception: pass
    st.session_state.bq_active = False
    st.session_state.bq_proc   = None
    st.session_state.bq_step   = "idle"
    # 実行中のジョブをpendingに戻す
    for job in st.session_state.bq_queue:
        if job["status"] == "running":
            job["status"] = "pending"
            job["current_step"] = ""
    _save_queue(st.session_state.bq_queue)

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🗂️ バッチ実験")
st.caption("複数の実験設定をキューに登録し、順番に自動実行します")

st.divider()

# 実行中なら進捗を確認してから描画
if st.session_state.bq_active:
    _advance()

# ── 実行状況 ──────────────────────────────────────────────────────────────────
if st.session_state.bq_active:
    job  = _current_job()
    step = st.session_state.bq_step
    step_ja = {"extracting": "フレーム抽出", "colmap": "COLMAP", "training": "3DGS学習"}.get(step, step)

    done_n  = sum(1 for j in st.session_state.bq_queue if j["status"] == "done")
    total_n = len(st.session_state.bq_queue)

    st.markdown(
        f'<span style="color:#00cc66">● 実行中</span>　'
        f'<span style="color:#4a90b8">{done_n}/{total_n} 完了　</span>'
        f'<b>{job["exp_name"] if job else ""}</b>　{step_ja}',
        unsafe_allow_html=True,
    )
    st.progress(done_n / max(total_n, 1), text=f"{done_n} / {total_n} ジョブ完了")

    log = st.session_state.bq_log
    if log and Path(log).exists():
        lines = [l for l in Path(log).read_text(errors="replace").replace("\r","\n").splitlines() if l.strip()]
        with st.expander("📋 現在のログ（末尾）", expanded=True):
            st.code("\n".join(lines[-12:]) or "（ログ待機中）", language=None)

    if st.button("⏹ バッチを中断", type="secondary"):
        _stop_batch()
        st.rerun()

    time.sleep(3)
    st.rerun()
    st.stop()

# ── キュー一覧 ────────────────────────────────────────────────────────────────
st.subheader("📋 実行キュー")

queue = st.session_state.bq_queue
if not queue:
    st.info("キューは空です。下のフォームから実験を追加してください。")
else:
    STATUS_ICON = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}
    for i, job in enumerate(queue):
        icon = STATUS_ICON.get(job["status"], "⏳")
        cfg  = job["config"]
        method = "HLoc" if cfg.get("use_hloc") else "COLMAP"
        summary = (f"{cfg.get('fps')} fps / {method} / "
                   f"{cfg.get('iterations',30000):,} iter")
        c1, c2, c3 = st.columns([4, 3, 1])
        c1.markdown(f"{icon} **{job['exp_name']}**")
        c2.caption(summary)
        if job["status"] == "pending":
            if c3.button("✕", key=f"del_{job['id']}"):
                st.session_state.bq_queue = [j for j in queue if j["id"] != job["id"]]
                _save_queue(st.session_state.bq_queue)
                st.rerun()
        elif job["status"] in ("done", "failed"):
            if c3.button("🗑", key=f"rm_{job['id']}", help="履歴から削除"):
                st.session_state.bq_queue = [j for j in queue if j["id"] != job["id"]]
                _save_queue(st.session_state.bq_queue)
                st.rerun()

    pending = [j for j in queue if j["status"] == "pending"]
    if pending:
        if st.button(f"▶ バッチ実行開始（{len(pending)} 件）", type="primary", use_container_width=True):
            st.session_state.bq_active = True
            _next_job()
            st.rerun()

    if st.button("🗑 完了・失敗をまとめて削除", use_container_width=False):
        st.session_state.bq_queue = [j for j in queue if j["status"] == "pending"]
        _save_queue(st.session_state.bq_queue)
        st.rerun()

st.divider()
st.info("➕ 実験の追加は「🚀 Pipeline Runner」ページの「📋 バッチキューに追加」ボタンから行えます。")

# ─── 固定フッター ─────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
