# バッチ実験キュー管理・実行ページ
# 各ページから追加されたジョブを順番に自動実行する。
# 対応ジョブ: pipeline / extract / colmap / train / render

import os
import signal
import sys
import subprocess
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, "/workspace")
from queue_helper import (
    QUEUE_FILE, JOB_ICONS,
    load_queue, save_queue, pending_size,
)

# ─── セッション初期化 ──────────────────────────────────────────────────────────

if "bq_active" not in st.session_state: st.session_state.bq_active = False
if "bq_proc"   not in st.session_state: st.session_state.bq_proc   = None
if "bq_pid"    not in st.session_state: st.session_state.bq_pid    = None
if "bq_step"   not in st.session_state: st.session_state.bq_step   = ""
if "bq_log"    not in st.session_state: st.session_state.bq_log    = None

# 実行中でない場合は常にファイルから読み込む（他ページの追加を反映）
if not st.session_state.bq_active:
    st.session_state.bq_queue = load_queue()
elif "bq_queue" not in st.session_state:
    st.session_state.bq_queue = load_queue()

# ─── プロセス管理 ─────────────────────────────────────────────────────────────

def _proc_done() -> bool:
    proc = st.session_state.bq_proc
    pid  = st.session_state.bq_pid
    if proc is not None:
        return proc.poll() is not None
    if pid:
        p = Path(f"/proc/{pid}/status")
        if not p.exists(): return True
        for line in p.read_text().splitlines():
            if line.startswith("State:"):
                return "Z" in line
    return True

def _proc_ok() -> bool:
    proc = st.session_state.bq_proc
    if proc is not None:
        return proc.returncode == 0
    log = st.session_state.bq_log
    if log and Path(log).exists():
        tail = Path(log).read_text(errors="replace").split("\n")[-10:]
        return not any("ERROR:" in l for l in tail)
    return True

def _current_job():
    for job in st.session_state.bq_queue:
        if job["status"] == "running":
            return job
    return None

# ─── ジョブ起動 ───────────────────────────────────────────────────────────────

def _launch(job: dict, step: str, cmd: list, log_path: str):
    os.makedirs(Path(log_path).parent, exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    st.session_state.bq_proc  = proc
    st.session_state.bq_pid   = proc.pid
    st.session_state.bq_step  = step
    st.session_state.bq_log   = log_path
    job["status"]       = "running"
    job["current_step"] = step
    job["log_path"]     = log_path
    save_queue(st.session_state.bq_queue)

def _start_job(job: dict):
    jtype = job.get("type", "pipeline")
    cfg   = job["config"]
    exp   = job["exp_dir"]
    os.makedirs(exp, exist_ok=True)

    if jtype == "pipeline":
        _do_extract(job)

    elif jtype == "extract":
        log = str(Path(exp) / "extract_log.txt")
        input_dir = str(Path(exp) / "input")
        os.makedirs(input_dir, exist_ok=True)
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
        _launch(job, "extract", cmd, log)

    elif jtype == "colmap":
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
        _launch(job, "colmap", cmd, log)

    elif jtype == "train":
        model_path = cfg.get("model_path", str(Path(exp) / "output"))
        os.makedirs(model_path, exist_ok=True)
        log = str(Path(model_path) / "train_log.txt")
        cmd = [sys.executable, "/workspace/scripts/run_train.py",
               "--source", exp, "--model_path", model_path,
               "--iterations", str(cfg.get("iterations", 30000)),
               "--save_iterations", *[str(i) for i in cfg.get("save_iterations", [7000, 30000])],
               "--test_iterations", *[str(i) for i in cfg.get("test_iterations", [1000, 7000, 15000, 30000])]]
        if cfg.get("eval"): cmd.append("--eval")
        if cfg.get("resolution"): cmd += ["--resolution", str(cfg["resolution"])]
        _launch(job, "train", cmd, log)

    elif jtype == "render":
        model_path = cfg.get("model_path", str(Path(exp) / "output"))
        log = str(Path(model_path) / "render_log.txt")
        cmd = [sys.executable, "/workspace/scripts/run_render.py",
               "-m", model_path, "-s", exp,
               "--iteration", str(cfg.get("iteration", -1))]
        if cfg.get("skip_train"): cmd.append("--skip_train")
        if cfg.get("skip_test"):  cmd.append("--skip_test")
        if cfg.get("white_background"): cmd.append("--white_background")
        _launch(job, "render", cmd, log)

def _do_extract(job: dict):
    cfg = job["config"]
    exp = job["exp_dir"]
    log = str(Path(exp) / "extract_log.txt")
    input_dir = str(Path(exp) / "input")
    os.makedirs(input_dir, exist_ok=True)
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
    _launch(job, "extracting", cmd, log)

# ─── ステップ進行 ──────────────────────────────────────────────────────────────

def _advance():
    if not _proc_done():
        return

    job  = _current_job()
    if job is None:
        st.session_state.bq_active = False
        st.session_state.bq_step   = ""
        return

    step  = st.session_state.bq_step
    jtype = job.get("type", "pipeline")
    ok    = _proc_ok()
    st.session_state.bq_proc = None

    if not ok:
        job["status"] = "failed"
        save_queue(st.session_state.bq_queue)
        _run_next()
        return

    if jtype == "pipeline":
        if step == "extracting":
            _start_job({**job, "type": "colmap"})
            job["current_step"] = "colmap"
            # colmapとして起動し直す
            _start_colmap_for_pipeline(job)
            return
        elif step == "colmap":
            _start_train_for_pipeline(job)
            return
        elif step == "training":
            job["status"] = "done"
            save_queue(st.session_state.bq_queue)
            _run_next()
            return
    else:
        job["status"] = "done"
        save_queue(st.session_state.bq_queue)
        _run_next()

def _start_colmap_for_pipeline(job: dict):
    cfg = job["config"]
    exp = job["exp_dir"]
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
    _launch(job, "colmap", cmd, log)

def _start_train_for_pipeline(job: dict):
    cfg = job["config"]
    exp = job["exp_dir"]
    model_path = str(Path(exp) / "output")
    os.makedirs(model_path, exist_ok=True)
    log = str(Path(model_path) / "train_log.txt")
    cmd = [sys.executable, "/workspace/scripts/run_train.py",
           "--source", exp, "--model_path", model_path,
           "--iterations", str(cfg.get("iterations", 30000)),
           "--save_iterations", *[str(i) for i in cfg.get("save_iterations", [7000, 30000])],
           "--test_iterations", *[str(i) for i in cfg.get("test_iterations", [1000, 7000, 15000, 30000])]]
    if cfg.get("eval"): cmd.append("--eval")
    if cfg.get("resolution"): cmd += ["--resolution", str(cfg["resolution"])]
    _launch(job, "training", cmd, log)

def _run_next():
    # ファイルから最新キューを読み、次のpendingを開始
    q = load_queue()
    st.session_state.bq_queue = q
    for job in q:
        if job["status"] == "pending":
            job["exp_dir"] = job.get("exp_dir") or str(
                Path("/workspace/experiments") / job["exp_name"]
            )
            os.makedirs(job["exp_dir"], exist_ok=True)
            _start_job(job)
            return
    st.session_state.bq_active = False
    st.session_state.bq_step   = ""
    save_queue(q)

def _stop_batch():
    proc = st.session_state.bq_proc
    pid  = st.session_state.bq_pid
    kill = proc.pid if (proc and proc.poll() is None) else pid
    if kill:
        try: os.kill(kill, signal.SIGTERM)
        except Exception: pass
    st.session_state.bq_active = False
    st.session_state.bq_proc   = None
    st.session_state.bq_step   = ""
    for job in st.session_state.bq_queue:
        if job["status"] == "running":
            job["status"] = "pending"
            job["current_step"] = ""
    save_queue(st.session_state.bq_queue)

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🗂️ バッチキュー")
st.caption("各ページから追加されたジョブを順番に自動実行します")

# 実行中なら進行チェック
if st.session_state.bq_active:
    _advance()

st.divider()

# ── 実行中ステータス ──────────────────────────────────────────────────────────
if st.session_state.bq_active:
    job     = _current_job()
    step_ja = {
        "extracting": "フレーム抽出", "colmap": "COLMAP/HLoc",
        "training": "3DGS学習", "train": "3DGS学習",
        "render": "レンダリング", "extract": "フレーム抽出",
    }.get(st.session_state.bq_step, st.session_state.bq_step)

    done_n  = sum(1 for j in st.session_state.bq_queue if j["status"] == "done")
    total_n = len(st.session_state.bq_queue)

    st.markdown(
        f'<span style="color:#00cc66;font-weight:bold;">● 実行中</span>'
        f'　{done_n}/{total_n} 完了'
        f'　<b>{job["exp_name"] if job else "—"}</b>　{step_ja}',
        unsafe_allow_html=True,
    )
    st.progress(done_n / max(total_n, 1))

    log = st.session_state.bq_log
    if log and Path(log).exists():
        lines = [l for l in Path(log).read_text(errors="replace")
                 .replace("\r", "\n").splitlines() if l.strip()]
        with st.expander("📋 実行中ログ", expanded=True):
            st.code("\n".join(lines[-12:]) or "（待機中）", language=None)

    if st.button("⏹ バッチを中断", type="secondary"):
        _stop_batch()
        st.rerun()

    time.sleep(3)
    st.rerun()
    st.stop()

# ── キュー一覧 ────────────────────────────────────────────────────────────────
st.subheader("📋 キュー")

queue   = st.session_state.bq_queue
pending = [j for j in queue if j["status"] == "pending"]
done_f  = [j for j in queue if j["status"] in ("done", "failed")]

STATUS_ICON = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}

if not queue:
    st.info("キューは空です。各ページの「📋 キューに追加」ボタンから追加してください。")
else:
    # pending ジョブ（順番入れ替え可）
    if pending:
        st.caption(f"実行待ち: {len(pending)} 件")
        for i, job in enumerate(pending):
            icon  = JOB_ICONS.get(job.get("type", "pipeline"), "▪")
            c1, c2, c3, c4 = st.columns([5, 1, 1, 1])
            c1.markdown(f"{icon} **{job['exp_name']}**　`{job.get('label','')}`")
            # ↑ 上へ
            if c2.button("↑", key=f"up_{job['id']}",
                         disabled=(i == 0), use_container_width=True):
                idx = next(k for k, j in enumerate(queue) if j["id"] == job["id"])
                prev_idx = next(k for k in range(idx-1, -1, -1)
                                if queue[k]["status"] == "pending")
                queue[idx], queue[prev_idx] = queue[prev_idx], queue[idx]
                save_queue(queue)
                st.rerun()
            # ↓ 下へ
            if c3.button("↓", key=f"dn_{job['id']}",
                         disabled=(i == len(pending)-1), use_container_width=True):
                idx = next(k for k, j in enumerate(queue) if j["id"] == job["id"])
                next_idx = next(k for k in range(idx+1, len(queue))
                                if queue[k]["status"] == "pending")
                queue[idx], queue[next_idx] = queue[next_idx], queue[idx]
                save_queue(queue)
                st.rerun()
            # 削除
            if c4.button("✕", key=f"del_{job['id']}", use_container_width=True):
                st.session_state.bq_queue = [j for j in queue if j["id"] != job["id"]]
                save_queue(st.session_state.bq_queue)
                st.rerun()

    # 完了・失敗ジョブ
    if done_f:
        with st.expander(f"完了・失敗 ({len(done_f)} 件)", expanded=False):
            for job in done_f:
                icon = STATUS_ICON.get(job["status"], "▪")
                type_icon = JOB_ICONS.get(job.get("type", "pipeline"), "▪")
                dc1, dc2 = st.columns([5, 1])
                dc1.caption(f"{icon} {type_icon} {job['exp_name']}　({job['status']})")
                if dc2.button("🗑", key=f"rm_{job['id']}", use_container_width=True):
                    st.session_state.bq_queue = [j for j in queue if j["id"] != job["id"]]
                    save_queue(st.session_state.bq_queue)
                    st.rerun()

    st.divider()
    bc1, bc2 = st.columns(2)
    with bc1:
        if pending and st.button(
            f"▶ バッチ実行開始（{len(pending)} 件）",
            type="primary", use_container_width=True,
        ):
            st.session_state.bq_active = True
            _run_next()
            st.rerun()
    with bc2:
        if done_f and st.button("🗑 完了・失敗を削除", use_container_width=True):
            st.session_state.bq_queue = [j for j in queue if j["status"] == "pending"]
            save_queue(st.session_state.bq_queue)
            st.rerun()

st.divider()
st.info("➕ 各ページの「📋 キューに追加」ボタンから追加できます。実行中でも追加可能です。")

# ─── 固定フッター ─────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
