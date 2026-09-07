# 実行中ジョブの自動検出と進捗解析（GUI非経由のCLIジョブも拾う）
#
# 既存の pipeline_widget.py は「GUI/バッチデーモンが書いた state ファイル」を読むため、
# nohup で直接叩いた学習・COLMAP・Blender などは一切見えない。
# このモジュールは /proc を直接走査して「いま動いている作業」を種類ごとに同定し、
# 各プロセスの標準出力(/proc/<pid>/fd/1)からログを逆引きして進捗を解析する。
# Streamlit に依存しないので、python3 job_monitor.py で単体確認できる。

import os
import re
import time
from pathlib import Path

WORKSPACE = Path("/workspace")
TMP       = WORKSPACE / "tmp"
EXP_ROOT  = WORKSPACE / "experiments"

_CLK_TCK = os.sysconf("SC_CLK_TCK")


# ══════════════════════════════════════════════════════════════════════════════
#  ジョブ種別の定義
#   (cmdline に対する正規表現, 種別キー, 表示名)  … 上から順に最初にマッチしたもの
# ══════════════════════════════════════════════════════════════════════════════

JOB_KINDS = [
    (r"mip-splatting/train\.py",                    "train",    "Mip-Splatting 学習"),
    (r"(train_custom|gaussian-splatting/train)\.py", "train",   "3DGS 学習"),
    (r"mip-splatting/render\.py",                   "render",   "Mip レンダリング"),
    (r"(run_render|gaussian-splatting/render)\.py", "render",   "レンダリング"),
    (r"run_hloc\.py",                               "colmap",   "姿勢推定（HLoc）"),
    (r"(run_colmap\.py|/colmap\b|\bcolmap )",       "colmap",   "姿勢推定（COLMAP）"),
    (r"generate_masks\.py",                         "masks",    "SAM2 マスク生成"),
    (r"(extract_frames|convert_360)\.py",           "extract",  "フレーム抽出"),
    (r"blender.*make_scene\.py",                    "blender",  "シーン生成（Blender）"),
    (r"blender",                                    "blender",  "Blender"),
    (r"wirebench/\w+\.py",                          "analysis", "分析スクリプト"),
    (r"(eval_|analyze_|plot_|collect_)\w*\.py",     "analysis", "分析スクリプト"),
    (r"batch_daemon\.py",                           "daemon",   "バッチデーモン"),
    (r"^(?:\S*/)?(?:ba)?sh\s+\S*\.sh\b",           "runner",   "実行スクリプト"),
]

# 表示色は利用側（pages/92_jobs.py）が持つ。ここは検出と解析だけを担う。

# 自分自身・シェル・計測用コマンドなど、作業として数えないもの
_IGNORE = re.compile(
    r"(claude/shell-snapshots|/bin/claude|\bclaude\b|streamlit run|"
    r"\bps -eo|\bugrep\b|\bgrep\b|\btail\b|\bsleep\b|job_monitor\.py|"
    r"Xvfb|fluxbox|x11vnc|websockify|tmux)"
)


# ══════════════════════════════════════════════════════════════════════════════
#  /proc ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

def _read(path, binary=False):
    try:
        return Path(path).read_bytes() if binary else Path(path).read_text(errors="replace")
    except Exception:
        return b"" if binary else ""


def _boot_time() -> float:
    """システム起動時刻（epoch秒）"""
    try:
        for line in _read("/proc/stat").splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except Exception:
        pass
    return time.time()


def _proc_start_epoch(pid: int, boot: float):
    """プロセス開始時刻（epoch秒）。/proc/<pid>/stat の22番目フィールドから求める"""
    try:
        stat = _read(f"/proc/{pid}/stat")
        # comm に空白や ')' が入りうるので最後の ')' 以降を使う
        rest = stat[stat.rfind(")") + 2:].split()
        return boot + float(rest[19]) / _CLK_TCK      # field 22 = index 19 of rest
    except Exception:
        return None


def _cmdline(pid: int) -> str:
    raw = _read(f"/proc/{pid}/cmdline", binary=True)
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def _stdout_log(pid: int):
    """標準出力のリダイレクト先を実ファイルとして取得する（nohup ... > x.log の逆引き）"""
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except Exception:
        return None
    if target.startswith("/") and not target.startswith(("/dev/", "/proc/")):
        p = Path(target)
        if p.is_file():
            return str(p)
    return None


def _fallback_log(cmdline: str, started: float):
    """fd/1 がパイプ等で辿れない場合、tmp/*.log の冒頭に model_path 等が
    書かれているものを探して対応付ける"""
    key = _model_path(cmdline) or _source_path(cmdline)
    if not key or not TMP.is_dir():
        return None
    for lg in sorted(TMP.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
        try:
            if lg.stat().st_mtime < (started or 0) - 60:
                continue
            with open(lg, "rb") as f:
                if key.encode() in f.read(8192):
                    return str(lg)
        except Exception:
            continue
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  コマンドラインからの情報抽出
# ══════════════════════════════════════════════════════════════════════════════

def _arg_after(cmdline: str, *flags):
    toks = cmdline.split()
    for i, t in enumerate(toks):
        if t in flags and i + 1 < len(toks):
            return toks[i + 1]
    return None


def _model_path(cmdline: str):
    return _arg_after(cmdline, "--model_path", "-m")


def _source_path(cmdline: str):
    return _arg_after(cmdline, "--source", "-s", "--source_path", "--out")


def _experiment(cmdline: str):
    """コマンドラインから実験名（experiments/ 直下のディレクトリ名）を拾う"""
    m = re.search(r"/workspace/experiments/([^/\s]+)", cmdline)
    return m.group(1) if m else None


def _output_tag(cmdline: str):
    """output_30k のような出力ディレクトリ名（実験内で run を区別する札）"""
    mp = _model_path(cmdline)
    if mp:
        name = Path(mp).name
        if name and name != _experiment(cmdline):
            return name
    return None


def classify(cmdline: str, cwd: str = ""):
    for pat, kind, label in JOB_KINDS:
        if re.search(pat, cmdline):
            if kind == "runner" and not _is_workspace_script(cmdline, cwd):
                return None, None        # noVNC 等、プロジェクト外のシェルスクリプト
            return kind, label
    return None, None


def _is_workspace_script(cmdline: str, cwd: str) -> bool:
    """実行しているシェルスクリプトが /workspace 配下かどうか"""
    m = re.search(r"(\S*\.sh)\b", cmdline)
    if not m:
        return False
    path = m.group(1)
    if path.startswith("/"):
        return path.startswith("/workspace/")
    return str(cwd).startswith("/workspace")


# ══════════════════════════════════════════════════════════════════════════════
#  ログ解析
# ══════════════════════════════════════════════════════════════════════════════

def tail_text(path, nbytes=400_000) -> str:
    """ログ末尾を読む。tqdm は \\r で上書きするので改行に正規化する"""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            raw = f.read()
    except Exception:
        return ""
    return raw.decode("utf-8", "replace").replace("\r", "\n")


def last_line(text: str) -> str:
    for ln in reversed(text.splitlines()):
        s = ln.strip()
        if s and not s.startswith("Training progress"):
            return s[:300]
    for ln in reversed(text.splitlines()):
        if ln.strip():
            return ln.strip()[:300]
    return ""


_TQDM = re.compile(
    r"(\d+)/(\d+)\s*\[([\d:?]+)<([\d:?]+|\?)"        # cur/total [経過<残り
    r"(?:,\s*([\d.]+)\s*(it/s|s/it))?"               # 速度
)
_LOSS = re.compile(r"(?<!Depth )Loss=([\d.]+)")
_ITER_PSNR = re.compile(r"\[ITER (\d+)\] Evaluating (test|train): .*?PSNR ([\d.]+)")
_COLMAP_STEP = re.compile(r"\[COLMAP (\d+)/(\d+)\]\s*(.*)")
_COLMAP_FILE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_MASK_PROG = re.compile(r"PROGRESS (\S+) (\d+)/(\d+)")
_EXTRACT_PROG = re.compile(r"PROGRESS (\d+)/(\d+)")
_EXTRACT_360 = re.compile(r"\[(\d+)/(\d+)\] 変換中")
_BLENDER_FRA = re.compile(r"Fra:(\d+)\b")


def parse_progress(kind: str, text: str) -> dict:
    """種別ごとに進捗を解析する。
    Returns {pct, headline, detail, eta, marks}  （取れないものは None/空）"""
    out = {"pct": None, "headline": "", "detail": "", "eta": None, "marks": []}
    if not text:
        return out

    if kind in ("train", "render"):
        tq = _TQDM.findall(text)
        if tq:
            cur, total, elapsed, eta, speed, unit = tq[-1]
            cur, total = int(cur), int(total)
            if total > 0:
                out["pct"] = min(cur / total, 1.0)
            unit_ja = "iter" if kind == "train" else "枚"
            out["headline"] = f"{cur:,} / {total:,} {unit_ja}"
            out["eta"] = eta if eta and eta != "?" else None
            bits = []
            if speed:
                bits.append(f"{speed} {unit}")
            lo = _LOSS.findall(text)
            if lo:
                bits.append(f"Loss {lo[-1]}")
            out["detail"] = " · ".join(bits)
        if kind == "train":
            # テスト評価の履歴＝研究上いちばん見たい数字なので拾っておく
            seen = {}
            for it, split, psnr in _ITER_PSNR.findall(text):
                if split == "test":
                    seen[int(it)] = float(psnr)
            out["marks"] = [(k, seen[k]) for k in sorted(seen)][-6:]

    elif kind == "colmap":
        steps = _COLMAP_STEP.findall(text)
        cur_step, total_step, step_name = (0, 4, "")
        if steps:
            cur_step, total_step, step_name = int(steps[-1][0]), int(steps[-1][1]), steps[-1][2]
        files = _COLMAP_FILE.findall(text)
        inner = None
        if files:
            fc, ft = int(files[-1][0]), int(files[-1][1])
            inner = fc / ft if ft else None
            out["detail"] = f"画像 {fc}/{ft}"
        else:
            tq = _TQDM.findall(text)
            if tq:
                fc, ft = int(tq[-1][0]), int(tq[-1][1])
                inner = fc / ft if ft else None
                out["detail"] = f"{fc:,}/{ft:,}"
                if tq[-1][3] not in ("?", ""):
                    out["eta"] = tq[-1][3]
        if cur_step:
            done = cur_step - 1 + (inner or 0)
            out["pct"] = min(done / total_step, 1.0)
            out["headline"] = f"[{cur_step}/{total_step}] {step_name.strip() or '処理中'}"

    elif kind == "masks":
        m = _MASK_PROG.findall(text)
        if m:
            tag, cur, tot = m[-1][0], int(m[-1][1]), int(m[-1][2])
            out["pct"] = min(cur / tot, 1.0) if tot else None
            out["headline"] = f"{cur} / {tot} 枚"
            out["detail"] = f"対象 {tag}"

    elif kind == "extract":
        m = _EXTRACT_360.findall(text) or _EXTRACT_PROG.findall(text)
        if m:
            cur, tot = int(m[-1][0]), int(m[-1][1])
            out["pct"] = min(cur / tot, 1.0) if tot else None
            out["headline"] = f"{cur} / {tot} 枚"

    elif kind == "blender":
        fr = _BLENDER_FRA.findall(text)
        saved = text.count("Saved:")
        if fr:
            out["headline"] = f"フレーム {fr[-1]}"
            out["detail"] = f"書き出し {saved} 枚" if saved else ""

    return out


_MARKS_CACHE = {}     # log_path -> (size, marks)


def _cached_marks(log: str):
    """学習ログの test PSNR 履歴。ログサイズが変わるまで再解析しない"""
    try:
        size = Path(log).stat().st_size
    except Exception:
        return []
    hit = _MARKS_CACHE.get(log)
    if hit and hit[0] == size:
        return hit[1]
    marks = parse_progress("train", tail_text(log, 4_000_000))["marks"]
    _MARKS_CACHE[log] = (size, marks)
    if len(_MARKS_CACHE) > 64:
        _MARKS_CACHE.clear()
    return marks


# ══════════════════════════════════════════════════════════════════════════════
#  実行中ジョブの走査
# ══════════════════════════════════════════════════════════════════════════════

def scan_jobs(include_daemon: bool = False) -> list:
    """いま動いている作業を列挙する（親子関係つき）"""
    boot = _boot_time()
    now = time.time()
    jobs = []

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        cmd = _cmdline(pid)
        if not cmd or _IGNORE.search(cmd):
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except Exception:
            cwd = ""
        kind, label = classify(cmd, cwd)
        if kind is None:
            continue
        if kind == "daemon" and not include_daemon:
            continue

        started = _proc_start_epoch(pid, boot)
        log = _stdout_log(pid) or _fallback_log(cmd, started)
        text = tail_text(log) if log else ""
        prog = parse_progress(kind, text)
        if kind == "train" and log:
            # tqdm がログを膨らませるので、評価行だけは広い窓で取り直す。
            # 数秒ごとの自動更新で毎回4MB読むのは無駄なので、サイズ変化まではキャッシュする
            prog["marks"] = _cached_marks(log)

        ppid = 0
        try:
            for line in _read(f"/proc/{pid}/status").splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
        except Exception:
            pass

        jobs.append({
            "pid": pid, "ppid": ppid, "kind": kind, "label": label,
            "cmdline": cmd,
            "experiment": _experiment(cmd),
            "output_tag": _output_tag(cmd),
            "script": Path(cmd.split()[-1]).name if kind == "runner" else None,
            "started": started,
            "elapsed": (now - started) if started else None,
            "log": log,
            "log_age": (now - Path(log).stat().st_mtime) if log and Path(log).exists() else None,
            "last_line": last_line(text),
            **prog,
        })

    # 親（runner）を先に、その下に子を並べる
    jobs.sort(key=lambda j: (j["started"] or 0))
    return jobs


def _fmt_script_name(cmd: str) -> str:
    m = re.search(r"([\w.\-]+\.sh)", cmd)
    return m.group(1) if m else "スクリプト"


# ══════════════════════════════════════════════════════════════════════════════
#  実行スクリプトの工程表（tmp/*_status.txt）
#   run_*.sh が「[名前] start 日時 / [名前] exit=0 日時」を追記する規約を読む
# ══════════════════════════════════════════════════════════════════════════════

_STATUS_LINE = re.compile(r"^\[([^\]]+)\]\s+(start|exit=(-?\d+))\s+(.*)$")


def read_status_files(max_age_h: float = 48) -> list:
    """tmp/*_status.txt を工程表として解析する"""
    if not TMP.is_dir():
        return []
    now = time.time()
    result = []
    for f in sorted(TMP.glob("*status*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if (now - f.stat().st_mtime) > max_age_h * 3600:
                continue
            lines = f.read_text(errors="replace").splitlines()
        except Exception:
            continue

        steps, order = {}, []
        done_all = False
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("ALL DONE"):
                done_all = True
                continue
            m = _STATUS_LINE.match(ln)
            if not m:
                continue
            name, kind, code, when = m.group(1), m.group(2), m.group(3), m.group(4)
            if name not in steps:
                steps[name] = {"name": name, "status": "waiting", "start": None,
                               "end": None, "code": None}
                order.append(name)
            if kind == "start":
                steps[name]["status"] = "running"
                steps[name]["start"] = when
            else:
                steps[name]["code"] = int(code)
                steps[name]["status"] = "done" if int(code) == 0 else "failed"
                steps[name]["end"] = when
        if not order:
            continue
        result.append({
            "file": str(f), "name": f.stem, "mtime": f.stat().st_mtime,
            "all_done": done_all,
            "steps": [steps[n] for n in order],
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  直近に終わったジョブ（生きているプロセスが無いログ）
# ══════════════════════════════════════════════════════════════════════════════

# ログ名からの種別推定。mip-splatting の train.py のように、ログ先頭に
# 実行コマンドが残らないものはファイル名で判断する（上から順に最初の一致）
_LOG_NAME_HINTS = [
    (r"render",                     "render",   "レンダリング"),
    (r"colmap",                     "colmap",   "姿勢推定（COLMAP）"),
    (r"hloc",                       "colmap",   "姿勢推定（HLoc）"),
    (r"mask|sam2",                  "masks",    "SAM2 マスク生成"),
    (r"extract|frame|360",          "extract",  "フレーム抽出"),
    (r"scene|blender|plateau|lod",  "blender",  "シーン生成"),
    (r"train|3dgs|wave|sweep|inject", "train",  "学習"),
    (r"eval|analy|plot|stat|budget", "analysis", "分析スクリプト"),
]


def classify_log(head: str, name: str):
    """ログ本文の先頭＋ファイル名から種別を推定する"""
    kind, label = classify(head)
    if kind is None:
        for pat, k, lab in _LOG_NAME_HINTS:
            if re.search(pat, name, re.IGNORECASE):
                kind, label = k, lab
                break
    if kind == "train" and re.search(r"mip", name, re.IGNORECASE):
        label = "Mip-Splatting 学習"
    return kind, label


def find_status_for(script_name: str):
    """実行スクリプト run_xxx.sh に対応する工程表 tmp/xxx_status.txt を探す"""
    if not script_name:
        return None
    stem = re.sub(r"^run_|\.sh$", "", script_name)
    for st in read_status_files():
        key = re.sub(r"_?status$", "", st["name"])
        if key and (key in stem or stem in key):
            return st
    return None


# 常駐サービスのログ＝「作業」ではないので終了一覧に出さない
_NOT_A_JOB = {"batch_daemon", "streamlit", "daemon"}

_FAIL_PAT = re.compile(
    r"(Traceback \(most recent call last\)|CUDA out of memory|"
    r"RuntimeError|Killed|No such file or directory|Error: |ERROR:)"
)
_OK_PAT = re.compile(r"(Training complete|完了:|ALL DONE|Rendering progress: *100%|全て完了)")


def recent_finished(hours: float = 48, limit: int = 12, live_logs=()) -> list:
    """tmp/*.log のうち、対応プロセスが居ないものを「終わった作業」として並べる"""
    if not TMP.is_dir():
        return []
    now = time.time()
    live = set(live_logs)
    rows = []
    for lg in TMP.glob("*.log"):
        try:
            st = lg.stat()
        except Exception:
            continue
        if str(lg) in live or (now - st.st_mtime) > hours * 3600:
            continue
        if lg.stem in _NOT_A_JOB:
            continue
        text = tail_text(lg, 60_000)
        try:
            with open(lg, "rb") as f:
                head = f.read(4000).decode("utf-8", "replace")
        except Exception:
            head = ""
        kind, label = classify_log(head, lg.name)
        tail_only = "\n".join([l for l in text.splitlines() if l.strip()][-25:])
        if _OK_PAT.search(tail_only):
            status = "done"
        elif _FAIL_PAT.search(tail_only):
            status = "failed"
        else:
            status = "unknown"     # 中断（GPU消失など）はここに落ちる
        m = re.search(r"/workspace/experiments/([^/\s]+)", text)
        rows.append({
            "log": str(lg), "name": lg.stem, "mtime": st.st_mtime,
            "age": now - st.st_mtime, "status": status,
            "label": label or "ジョブ", "kind": kind or "analysis",
            "experiment": m.group(1) if m else None,
            "last_line": last_line(text),
            "size": st.st_size,
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows[:limit]


# ══════════════════════════════════════════════════════════════════════════════
#  GPU（学習ジョブの健康状態を見るための最小限。詳細はシステムモニターページ）
# ══════════════════════════════════════════════════════════════════════════════

def gpu_brief() -> dict:
    import subprocess
    try:
        q = "name,memory.used,memory.total,utilization.gpu,temperature.gpu"
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=6)
        if r.returncode != 0:
            return {"ok": False, "error": (r.stderr or r.stdout).strip()[:200]}
        name, used, total, util, temp = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
        return {"ok": True, "name": name, "used": int(used), "total": int(total),
                "util": int(util), "temp": int(temp)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ══════════════════════════════════════════════════════════════════════════════
#  表示用の整形
# ══════════════════════════════════════════════════════════════════════════════

def fmt_duration(sec) -> str:
    """秒 → 「1時間35分」形式"""
    if sec is None:
        return "—"
    sec = int(sec)
    if sec < 60:
        return f"{sec}秒"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}分"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}時間{m}分" if m else f"{h}時間"
    d, h = divmod(h, 24)
    return f"{d}日{h}時間"


def fmt_eta(eta: str) -> str:
    """tqdm の 1:35:39 / 12:34 → 日本語"""
    if not eta or eta == "?":
        return "—"
    parts = eta.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return eta
    if len(nums) == 3:
        return fmt_duration(nums[0] * 3600 + nums[1] * 60 + nums[2])
    if len(nums) == 2:
        return fmt_duration(nums[0] * 60 + nums[1])
    return eta


if __name__ == "__main__":
    print("── いま動いているもの " + "─" * 40)
    js = scan_jobs()
    if not js:
        print("  （なし）")
    for j in js:
        pct = f"{j['pct']*100:5.1f}%" if j["pct"] is not None else "   —  "
        print(f"  [{j['kind']:8s}] pid={j['pid']:<6d} {pct}  {j['label']}"
              f"  {j['experiment'] or ''}/{j['output_tag'] or ''}")
        print(f"      {j['headline']}  残り{fmt_eta(j['eta'])}  経過{fmt_duration(j['elapsed'])}"
              f"  {j['detail']}")
        print(f"      log={j['log']}  最終行: {j['last_line'][:90]}")
        if j["marks"]:
            print("      test PSNR: " + "  ".join(f"{i}→{v:.2f}" for i, v in j["marks"]))
    print()
    print("── 工程表 " + "─" * 46)
    for s in read_status_files():
        print(f"  {s['name']}  (all_done={s['all_done']})")
        for st_ in s["steps"]:
            print(f"      {st_['status']:8s} {st_['name']}  code={st_['code']}")
    print()
    print("── 直近に終わったもの " + "─" * 38)
    for r in recent_finished(hours=48, limit=8, live_logs=[j["log"] for j in js if j["log"]]):
        print(f"  {r['status']:8s} {fmt_duration(r['age'])}前  {r['name']}  ({r['label']})")
    print()
    print("── GPU " + "─" * 50)
    print(" ", gpu_brief())
