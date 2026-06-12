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
    load_queue, pending_size,
    edit_queue, update_job,
    load_active_task_file, clear_active_task_file,
)
from job_commands import (
    build_extract_cmd, build_colmap_cmd, build_train_cmd, build_render_cmd,
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
    update_job(job["id"], status="running", current_step=step, log_path=log_path)
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
    _pcfg_path = Path(exp) / "pipeline_config.json"
    _existing_pcfg = {}
    if _pcfg_path.exists():
        try:
            _existing_pcfg = _json_bq.loads(_pcfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 既存データを保持しつつ新ジョブのconfigをマージ（既存キーが消えないようにする）
    _merged_pcfg = dict(_existing_pcfg)
    _merged_pcfg.update({k: v for k, v in cfg.items() if k != "note_md"})
    _merged_pcfg["saved_at"] = _dt_bq.now().strftime("%Y-%m-%d %H:%M:%S")
    _pcfg_path.write_text(
        _json_bq.dumps(_merged_pcfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if jtype == "pipeline":
        cmd, log = build_extract_cmd(cfg, exp)
        _launch(job, "extracting", cmd, log)
    elif jtype == "extract":
        cmd, log = build_extract_cmd(cfg, exp)
        _launch(job, "extract", cmd, log)
    elif jtype == "colmap":
        cmd, log = build_colmap_cmd(cfg, exp)
        _launch(job, "colmap", cmd, log)
    elif jtype == "train":
        cmd, log = build_train_cmd(cfg, exp)
        _launch(job, "train", cmd, log)
    elif jtype == "render":
        cmd, log = build_render_cmd(cfg, exp)
        _launch(job, "render", cmd, log)

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
        update_job(job["id"], status="failed")
        _run_next()
        return

    if jtype == "pipeline":
        if step == "extracting":
            cmd, log = build_colmap_cmd(job["config"], job["exp_dir"])
            _launch(job, "colmap", cmd, log)
            return
        elif step == "colmap":
            cmd, log = build_train_cmd(job["config"], job["exp_dir"])
            _launch(job, "training", cmd, log)
            return
        elif step == "training":
            job["status"] = "done"
            update_job(job["id"], status="done")
            _run_next()
            return
    else:
        job["status"] = "done"
        update_job(job["id"], status="done")
        _run_next()

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
            update_job(job["id"], exp_dir=job["exp_dir"])
            _start_job(job)
            return
    st.session_state.bq_active = False
    st.session_state.bq_step   = ""
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
    with edit_queue() as q:
        for job in q:
            if job["status"] == "running":
                job["status"] = "pending"
                job["current_step"] = ""
    st.session_state.bq_queue = load_queue()
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
            with edit_queue() as q:
                idx      = next((k for k, j in enumerate(q) if j["id"] == job["id"]), None)
                prev_idx = next((k for k in range((idx or 0)-1, -1, -1)
                                 if q[k]["status"] == "pending"), None)
                if idx is not None and prev_idx is not None:
                    q[idx], q[prev_idx] = q[prev_idx], q[idx]
            st.rerun()
        if c3.button("↓", key=f"dn_{job['id']}", disabled=(i == len(pending)-1), use_container_width=True):
            with edit_queue() as q:
                idx      = next((k for k, j in enumerate(q) if j["id"] == job["id"]), None)
                next_idx = next((k for k in range((idx or 0)+1, len(q))
                                 if q[k]["status"] == "pending"), None)
                if idx is not None and next_idx is not None:
                    q[idx], q[next_idx] = q[next_idx], q[idx]
            st.rerun()
        if c4.button("✕", key=f"del_{job['id']}", use_container_width=True):
            with edit_queue() as q:
                q[:] = [j for j in q if j["id"] != job["id"]]
            st.session_state.bq_queue = load_queue()
            st.rerun()

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
            with edit_queue() as q:
                q[:] = [j for j in q if j["id"] != job["id"]]
            st.session_state.bq_queue = load_queue()
            st.rerun()

    if st.button("🗑 完了・失敗をまとめて削除", use_container_width=False):
        # 実行中（running）ジョブは消さずに残す（消すとデーモンがキューを停止してしまう）
        with edit_queue() as q:
            q[:] = [j for j in q if j["status"] in ("pending", "running")]
        st.session_state.bq_queue = load_queue()
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  使い方ガイド
# ════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📖 使い方：360度動画から3DGSを作る流れ（撮影者マスクあり）", expanded=False):
    st.markdown("""
## 全体の流れ

```
[Step 1] フレーム抽出（🎞️ フレーム抽出ページ）        … 数分・自動
[Step 2] SAM2マスク生成（🎭 SAM2マスクページ）         … クリック数十秒＋実行数分 ★唯一の手作業
[Step 3] 姿勢推定（📷 姿勢推定ページ → キュー）        … 数十分〜・自動
[Step 4] 3DGS学習（🧠 3DGS学習ページ → キュー）        … 30〜60分・自動（マスク自動適用）
[Step 5] 結果確認（🖼️ 結果確認ページ）
```

> **なぜ🚀パイプライン一括実行を使わないの？**
> パイプラインジョブは抽出→姿勢推定→学習までノンストップで進むため、途中にマスク作成
> （人間のクリック）を挟む隙がありません。学習開始時点で `masks/` が間に合っていないと
> マスクなし（または未完成のマスク）で学習が始まってしまいます。
> 撮影者を消したい場合は下記の分割手順が確実です。

---

## Step 1: フレーム抽出のみ実行

**🎞️ フレーム抽出ページ**で：

1. 動画を選択（`data/360movies/` に置いたもの。例: `Ylab_room_v2_mid.mp4`）
2. 「360度動画」をON → FPS・FOV・出力解像度・切り出し方向を設定
   - 例: FPS=1.0、FOV=90°、水平4方向（0/90/180/270°）→ 2分の動画なら約480枚
   - 45°刻み8方向にすると枚数が2倍になり姿勢推定時間も伸びるので、まず4方向がおすすめ
3. 実行 → `experiments/YYYYMMDD_<シーン名>_NN/input/` に
   `frame_000001_y000_p+0.jpg` のような連番画像が生成される
   - ファイル名は「時刻 + 方向」。`frame_000123_y090_p+0` = 123秒目・右90°・水平

---

## Step 2: SAM2マスク生成（手作業はここだけ）

**🎭 SAM2マスクページ**で：

1. Step 1 の実験フォルダを選択 → 方向ごとのタブが並ぶ（`y000_p+0`, `y090_p+0`, ...）
2. 各タブに**その方向の1枚目のフレーム**が表示されるので、
   **撮影者が写っているタブだけ**開いて体の上を🔴で1〜3点クリック
   - 撮影者はカメラと一緒に動くので、写る方向・位置は毎フレームほぼ固定です
   - 360度の重複により2〜3方向に写るのが普通 → 写っている方向は全部クリック
   - 写っていない方向は何もしなくてOK（自動でスキップされます）
   - マスクが体からはみ出すときは🔵背景点で抑制
3. ▶実行 → SAM2が1枚目で教えた人を全フレームへ自動追跡（GPUで1枚 0.1〜0.3秒）
4. **プレビューを必ず確認**：各方向の先頭・中間・末尾に赤いマスクが正しく
   人に乗っているか。ずれていたら点を修正して再実行（マスクは上書きされます）

→ `masks/` フォルダに白黒マスクPNGが生成される（白=撮影者=学習から除外）

---

## Step 3: 姿勢推定をキューに追加

**📷 姿勢推定ページ**で実験フォルダを選び、HLocを設定してキューに追加：

- 推奨: `superpoint_aachen` + `superpoint+lightglue`
- 画像が300枚を超える場合は **Retrieval（netvlad, top-20）** を推奨
  （Exhaustiveは480枚で約11.5万ペアになり非常に時間がかかります）

キューに入れたらこのページのデーモンが自動実行します。**ブラウザを閉じてもOK**。

→ `sparse/0/`（姿勢+点群）が生成される。SIMPLE_RADIAL等の場合は学習時に
  自動でundistortionされ `dense/` が作られます

任意: 姿勢推定が終わったら🎭ページの「SORのみ実行」で点群のノイズ除去もできます。

---

## Step 4: 3DGS学習をキューに追加

**🧠 3DGS学習ページ**で実験フォルダを選び、キューに追加（30,000 iter が標準）。

学習開始時に `masks/` が**自動検出**されます。ログでここを確認：

```
[masks] masks/ フォルダを検出。マスク合成を実行します...
[masks] undistortion検出 → マスクもカメラモデルに合わせて再マップします
[masks] 480/480 枚にマスクを適用 → .../images_masked
```

マスク領域は損失計算から除外され、他視点の観測だけでその空間が再構成される
→ 結果として撮影者が消えます。

---

## Step 5: 結果確認・やり直し

**🖼️ 結果確認ページ**でレンダリングし、撮影者がいた場所が消えているか確認。

- **足やスティックが残った** → Step 2 に戻って点を追加しマスク再生成 →
  **学習だけ再実行**でOK（`images_masked/` は学習起動のたびに作り直されるため、
  抽出・姿勢推定のやり直しは不要）
- **マスク以外の品質が悪い** → 姿勢推定の設定（特徴点・ペア方式）や
  FPS・方向数を見直して Step 1 / Step 3 から

---

## 補足

| 項目 | 内容 |
|---|---|
| 所要時間の目安 | 抽出 数分 / マスク 数分 / 姿勢推定 数十分〜 / 学習30k 30〜60分（A6000） |
| 手作業 | Step 2 のクリックとプレビュー確認のみ。残りは全部キュー+デーモン任せ |
| マスク不要なシーン | Step 2 を丸ごとスキップすれば通常のパイプライン一括実行でOK |
| デーモン | このページ上部で起動・停止。稼働中はブラウザを閉じてもキューが進みます |
| 同じ撮り方の2本目以降 | 撮影者はほぼ同じ画素位置に写るため、同じクリック座標が使い回せます |
""")

# ページ末尾で自動更新（途中でst.rerun()しないことで全コンテンツを描画してから更新）
if _need_rerun:
    time.sleep(3)
    st.rerun()

# ─── 固定フッター ─────────────────────────────────────────────────────────────