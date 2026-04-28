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

import json as _json

sys.path.insert(0, "/workspace")
from queue_helper import (
    QUEUE_FILE, JOB_ICONS,
    load_queue, save_queue, pending_size,
    load_active_task_file, clear_active_task_file,
)

BATCH_STATE_FILE  = "/workspace/tmp/batch_state.json"
DAEMON_PID_FILE   = "/workspace/tmp/batch_daemon.pid"
DAEMON_LOG_FILE   = "/workspace/tmp/batch_daemon.log"

# ─── バッチ実行状態の永続化 ────────────────────────────────────────────────────

def _save_batch_state():
    Path(BATCH_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(BATCH_STATE_FILE).write_text(_json.dumps({
        "active": st.session_state.bq_active,
        "pid":    st.session_state.bq_pid,
        "step":   st.session_state.bq_step,
        "log":    st.session_state.bq_log,
    }, ensure_ascii=False), encoding="utf-8")

def _load_batch_state() -> dict:
    try:
        p = Path(BATCH_STATE_FILE)
        if p.exists():
            return _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _clear_batch_state():
    try:
        Path(BATCH_STATE_FILE).unlink(missing_ok=True)
    except Exception:
        pass

# ─── セッション初期化（ページ再訪時にファイルから復元） ───────────────────────

if "bq_active" not in st.session_state:
    _bs = _load_batch_state()
    if _bs.get("active"):
        st.session_state.bq_active = True
        st.session_state.bq_pid    = _bs.get("pid")
        st.session_state.bq_step   = _bs.get("step", "")
        st.session_state.bq_log    = _bs.get("log")
        st.session_state.bq_proc   = None
    else:
        st.session_state.bq_active = False
        st.session_state.bq_pid    = None
        st.session_state.bq_step   = ""
        st.session_state.bq_log    = None
        st.session_state.bq_proc   = None

# キューは常にファイルから読み込む（他ページの追加・バッチ実行中の追加も反映）
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
    _save_batch_state()

def _start_job(job: dict):
    jtype = job.get("type", "pipeline")
    cfg   = job["config"]
    exp   = job["exp_dir"]
    os.makedirs(exp, exist_ok=True)

    note_content = cfg.get("note_md", "")
    note_path = Path(exp) / "note.md"
    if note_content or not note_path.exists():
        note_path.write_text(note_content, encoding="utf-8")

    import json as _json_bq
    from datetime import datetime as _dt_bq
    pipeline_cfg = {k: v for k, v in cfg.items() if k != "note_md"}
    pipeline_cfg.setdefault("saved_at", _dt_bq.now().strftime("%Y-%m-%d %H:%M:%S"))
    (Path(exp) / "pipeline_config.json").write_text(
        _json_bq.dumps(pipeline_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
    _clear_batch_state()

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
    _clear_batch_state()

# ─── コマンドライン生成 ───────────────────────────────────────────────────────

def _job_cmdline(job: dict) -> str:
    """ジョブ設定からコマンドライン文字列を生成する"""
    jtype = job.get("type", "pipeline")
    cfg   = job.get("config", {})
    exp   = job.get("exp_dir", "?")
    parts = []

    if jtype in ("pipeline", "extract"):
        if cfg.get("is_360"):
            angles = " ".join(f"{y},{p}" for y, p in cfg.get("angles", []))
            lines = [
                "python3 /workspace/scripts/convert_360.py",
                f"  --input {cfg.get('video_path','')}",
                f"  --output {exp}/input",
                f"  --fps {cfg.get('fps',1.0)}",
                f"  --fov {cfg.get('fov',90)} --width {cfg.get('out_w',1024)} --height {cfg.get('out_h',1024)}",
                f"  --angles {angles}",
            ]
        else:
            lines = [
                "python3 /workspace/scripts/extract_frames.py",
                f"  --input {cfg.get('video_path','')}",
                f"  --output {exp}/input",
                f"  --fps {cfg.get('fps',2.0)}",
            ]
        parts.append("# フレーム抽出\n" + " \\\n".join(lines))

    if jtype in ("pipeline", "colmap"):
        if cfg.get("use_hloc"):
            lines = [
                "python3 /workspace/scripts/run_hloc.py",
                f"  --source_path {exp}",
                f"  --feature_type {cfg.get('feature_type','superpoint_aachen')}",
                f"  --matcher_type {cfg.get('matcher_type','superpoint+lightglue')}",
                f"  --pair_method {cfg.get('pair_method','exhaustive')}",
            ]
            if cfg.get("pair_method") == "retrieval":
                lines += [
                    f"  --retrieval_model {cfg.get('retrieval_model','netvlad')}",
                    f"  --num_matched {cfg.get('num_matched',20)}",
                ]
        else:
            lines = [
                "python3 /workspace/scripts/run_colmap.py",
                f"  --source_path {exp}",
                f"  --camera_model {cfg.get('camera_model','OPENCV')}",
            ]
        parts.append("# 姿勢推定\n" + " \\\n".join(lines))

    if jtype in ("pipeline", "train"):
        mp = cfg.get("model_path", f"{exp}/output")
        save_it = " ".join(str(i) for i in cfg.get("save_iterations", [7000, 30000]))
        test_it = " ".join(str(i) for i in cfg.get("test_iterations", [1000, 7000, 30000]))
        lines = [
            "python3 /workspace/scripts/run_train.py",
            f"  --source {exp}",
            f"  --model_path {mp}",
            f"  --iterations {cfg.get('iterations',30000)}",
            f"  --save_iterations {save_it}",
            f"  --test_iterations {test_it}",
        ]
        if cfg.get("eval"):       lines.append("  --eval")
        if cfg.get("resolution"): lines.append(f"  --resolution {cfg['resolution']}")
        parts.append("# 3DGS学習\n" + " \\\n".join(lines))

    if jtype == "render":
        mp = cfg.get("model_path", f"{exp}/output")
        lines = [
            "python3 /workspace/scripts/run_render.py",
            f"  -m {mp}",
            f"  -s {exp}",
            f"  --iteration {cfg.get('iteration',-1)}",
        ]
        if cfg.get("skip_train"):       lines.append("  --skip_train")
        if cfg.get("skip_test"):        lines.append("  --skip_test")
        if cfg.get("white_background"): lines.append("  --white_background")
        parts.append("# レンダリング\n" + " \\\n".join(lines))

    return "\n\n".join(parts) if parts else "（コマンド生成不可）"


def _daemon_pid() -> int | None:
    """デーモンの PID を返す。起動していなければ None"""
    try:
        p = Path(DAEMON_PID_FILE)
        if p.exists():
            return int(p.read_text().strip())
    except Exception:
        pass
    return None

def _daemon_alive() -> bool:
    return _is_pid_alive_local(_daemon_pid())

def _start_daemon():
    import subprocess as _sp
    _sp.Popen(
        [sys.executable, "/workspace/scripts/batch_daemon.py"],
        stdout=open(DAEMON_LOG_FILE, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,   # ブラウザを閉じても生き続ける
    )

def _stop_daemon():
    pid = _daemon_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    Path(DAEMON_PID_FILE).unlink(missing_ok=True)

def _is_pid_alive_local(pid) -> bool:
    if not pid: return False
    try:
        p = Path(f"/proc/{pid}/status")
        if not p.exists(): return False
        for line in p.read_text().splitlines():
            if line.startswith("State:"):
                return "Z" not in line
        return False
    except Exception:
        return False

def _log_tail(log_path, n=15):
    try:
        p = Path(log_path or "")
        if not p.exists(): return ""
        lines = [l for l in p.read_text(errors="replace").replace("\r","\n").splitlines() if l.strip()]
        return "\n".join(lines[-n:])
    except Exception:
        return ""

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🗂️ バッチキュー")
st.caption("各ページから追加されたジョブを順番に自動実行します")

# ── デーモン制御パネル ────────────────────────────────────────────────────────
_d_alive = _daemon_alive()
_d_col1, _d_col2, _d_col3 = st.columns([3, 1, 1])
with _d_col1:
    if _d_alive:
        st.success(f"🟢 バックグラウンドデーモン 稼働中　(pid={_daemon_pid()})　— ブラウザを閉じてもキューは実行され続けます")
    else:
        st.warning("🔴 デーモン停止中　— ブラウザを開いている間のみキューが進みます")
with _d_col2:
    if not _d_alive:
        if st.button("▶ デーモン起動", use_container_width=True, type="primary"):
            _start_daemon()
            time.sleep(1)
            st.rerun()
with _d_col3:
    if _d_alive:
        if st.button("⏹ デーモン停止", use_container_width=True):
            _stop_daemon()
            st.rerun()

# デーモンログをエキスパンダーで表示
if Path(DAEMON_LOG_FILE).exists():
    with st.expander("📋 デーモンログ", expanded=False):
        _dlog = _log_tail(DAEMON_LOG_FILE, n=20)
        st.code(_dlog, language=None)

st.divider()

# デーモンが動いていればページ側の進行チェックをスキップ
if st.session_state.bq_active and not _daemon_alive():
    _advance()

_need_rerun = False  # ページ末尾でst.rerun()するかどうかのフラグ

# pipeline_widget からプログレス解析関数を取得
try:
    from pipeline_widget import _parse_progress, _parse_colmap_substeps, _render_substep_bars
    _has_widget = True
except Exception:
    _has_widget = False

queue   = st.session_state.bq_queue
running = [j for j in queue if j["status"] == "running"]
pending = [j for j in queue if j["status"] == "pending"]
done_f  = [j for j in queue if j["status"] in ("done", "failed")]

STEP_JA = {
    "extracting": "フレーム抽出", "extract": "フレーム抽出",
    "colmap": "COLMAP/HLoc", "training": "3DGS学習",
    "train": "3DGS学習", "render": "レンダリング",
}
STATUS_ICON = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}

# ════════════════════════════════════════════════════════════════════════════
#  1. 実行中ジョブ（バッチ）
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.bq_active:
    job     = _current_job()
    step    = st.session_state.bq_step
    step_ja = STEP_JA.get(step, step)
    elapsed = time.time() - (job.get("start_time") if job and job.get("start_time") else time.time())
    scene   = job["exp_name"] if job else "—"
    jtype   = job.get("type", "pipeline") if job else "pipeline"
    icon    = JOB_ICONS.get(jtype, "▪")

    # ヘッダー
    ha, hb = st.columns([5, 1])
    ha.markdown(
        f'<span style="color:#00cc66;font-weight:bold;">● 実行中</span>'
        f'　{icon} **{scene}**　— {step_ja}',
        unsafe_allow_html=True,
    )
    if hb.button("⏹ 中断", key="bq_stop", type="secondary"):
        _stop_batch()
        st.rerun()

    # プログレスバー
    log_path = st.session_state.bq_log or ""
    if _has_widget and log_path:
        _state = {"step": step, "log_path": log_path,
                  "use_hloc": job.get("config", {}).get("use_hloc", False) if job else False,
                  "iterations": job.get("config", {}).get("iterations", 30000) if job else 30000}
        try:
            if step == "colmap":
                _subs = _parse_colmap_substeps(_state)
                if _subs: _render_substep_bars(_subs)
            else:
                _pct, _lbl = _parse_progress(_state)
                if _pct is not None: st.progress(_pct, text=_lbl)
        except Exception:
            pass

    # ログ
    _tl = _log_tail(log_path)
    if _tl:
        st.code(_tl, language=None)
    else:
        st.caption("ログ待機中...")

    st.divider()
    _need_rerun = True  # ページ末尾でrerunする

# ════════════════════════════════════════════════════════════════════════════
#  2. パイプライン・単品ジョブ（他ページから起動されたもの）
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.bq_active:
    _at = load_active_task_file()
    _at_alive = bool(_at) and _is_pid_alive_local(_at.get("pid"))
    if _at and not _at_alive:
        clear_active_task_file(); _at = {}

    try:
        from pipeline_widget import _load_state as _pll
        _pl = _pll()
        _pl_alive = _pl.get("active") and _pl.get("step") not in ("done","failed","setup",None)
    except Exception:
        _pl = {}; _pl_alive = False

    if _pl_alive or _at_alive:
        with st.expander("🔄 他ページの実行中ジョブ", expanded=True):
            if _pl_alive:
                _step = _pl.get("step","")
                _scene = Path(_pl.get("experiment_dir","")).name
                st.caption(f"🚀 Pipeline Runner — {_scene} — {STEP_JA.get(_step,_step)}")
                _exp = _pl.get("experiment_dir","")
                _lp = {"extracting": f"{_exp}/extract_log.txt",
                       "colmap": f"{_exp}/colmap_log.txt",
                       "training": f"{_exp}/output/train_log.txt"}.get(_step, _pl.get("log_path",""))
                if _has_widget:
                    try:
                        _s = {**_pl, "log_path": _lp, "step": _step}
                        if _step == "colmap":
                            _subs = _parse_colmap_substeps(_s)
                            if _subs: _render_substep_bars(_subs)
                        else:
                            _p, _l = _parse_progress(_s)
                            if _p is not None: st.progress(_p, text=_l)
                    except Exception: pass
                _tl = _log_tail(_lp)
                if _tl: st.code(_tl, language=None)
            if _at_alive:
                _icon = JOB_ICONS.get(_at.get("step",""), "▪")
                st.caption(f"{_icon} {_at.get('label','')} — {_at.get('scene','')}")
                if _has_widget:
                    try:
                        _p, _l = _parse_progress(_at)
                        if _p is not None: st.progress(_p, text=_l)
                    except Exception: pass
                _tl = _log_tail(_at.get("log_path",""))
                if _tl: st.code(_tl, language=None)
        st.divider()

# ════════════════════════════════════════════════════════════════════════════
#  3. 待機中キュー
# ════════════════════════════════════════════════════════════════════════════
st.subheader("📋 待機中")

if not pending and not st.session_state.bq_active:
    st.info("キューは空です。各ページの「📋 キューに追加」ボタンから追加してください。")
else:
    # バッチ実行開始ボタン
    if not st.session_state.bq_active and pending:
        if st.button(f"▶ バッチ実行開始（{len(pending)} 件）",
                     type="primary", use_container_width=True):
            st.session_state.bq_active = True
            _run_next()
            st.rerun()

    # 待機中ジョブ一覧
    for i, job in enumerate(pending):
        icon  = JOB_ICONS.get(job.get("type", "pipeline"), "▪")
        label = job.get("label", "")
        c1, c2, c3, c4 = st.columns([5, 1, 1, 1])
        c1.markdown(f"{icon} **{job['exp_name']}**　<small style='color:#4a90b8'>{label}</small>",
                    unsafe_allow_html=True)
        if c2.button("↑", key=f"up_{job['id']}", disabled=(i == 0), use_container_width=True):
            idx      = next(k for k, j in enumerate(queue) if j["id"] == job["id"])
            prev_idx = next(k for k in range(idx-1, -1, -1) if queue[k]["status"] == "pending")
            queue[idx], queue[prev_idx] = queue[prev_idx], queue[idx]
            save_queue(queue); st.rerun()
        if c3.button("↓", key=f"dn_{job['id']}", disabled=(i == len(pending)-1), use_container_width=True):
            idx      = next(k for k, j in enumerate(queue) if j["id"] == job["id"])
            next_idx = next(k for k in range(idx+1, len(queue)) if queue[k]["status"] == "pending")
            queue[idx], queue[next_idx] = queue[next_idx], queue[idx]
            save_queue(queue); st.rerun()
        if c4.button("✕", key=f"del_{job['id']}", use_container_width=True):
            st.session_state.bq_queue = [j for j in queue if j["id"] != job["id"]]
            save_queue(st.session_state.bq_queue); st.rerun()

        # 設定コマンドライン（デフォルト非表示）
        with st.expander("設定を表示", expanded=False):
            st.code(_job_cmdline(job), language="bash")

# ════════════════════════════════════════════════════════════════════════════
#  4. 完了・失敗一覧
# ════════════════════════════════════════════════════════════════════════════
if done_f:
    st.divider()
    st.subheader("📁 完了・失敗")
    for job in done_f:
        s_icon  = STATUS_ICON.get(job["status"], "▪")
        j_icon  = JOB_ICONS.get(job.get("type","pipeline"), "▪")
        dc1, dc2 = st.columns([5, 1])
        dc1.markdown(f"{s_icon} {j_icon} **{job['exp_name']}**　<small style='color:#4a90b8'>{job['status']}</small>",
                     unsafe_allow_html=True)
        if dc2.button("🗑", key=f"rm_{job['id']}", use_container_width=True):
            st.session_state.bq_queue = [j for j in queue if j["id"] != job["id"]]
            save_queue(st.session_state.bq_queue); st.rerun()

    if st.button("🗑 完了・失敗をまとめて削除", use_container_width=False):
        st.session_state.bq_queue = [j for j in queue if j["status"] == "pending"]
        save_queue(st.session_state.bq_queue); st.rerun()

# ページ末尾で自動更新（途中でst.rerun()しないことで全コンテンツを描画してから更新）
if _need_rerun:
    time.sleep(3)
    st.rerun()

# ─── 固定フッター ─────────────────────────────────────────────────────────────