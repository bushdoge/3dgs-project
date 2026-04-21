# パイプライン進捗ウィジェット
# monitor / minigame などのページから呼び出せる共通コンポーネント

import json
import re
import time
from pathlib import Path

import streamlit as st

_STATE_FILE = "/workspace/tmp/pipeline_state.json"

_STEP_JA = {
    "extracting": "① フレーム抽出",
    "colmap":     "② COLMAP / HLoc",
    "training":   "③ 3DGS 学習",
    "done":       "完了",
    "failed":     "失敗",
}
_COLMAP_SUB = {1: "特徴点抽出", 2: "マッチング", 3: "3D再構成", 4: "undistortion"}


def _load_state() -> dict:
    """session_state → ファイル の順で最新のパイプライン状態を取得する"""
    pl = st.session_state.get("pipeline", {})
    if pl.get("active"):
        return pl
    try:
        p = Path(_STATE_FILE)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _parse_progress(pl: dict) -> tuple:
    """(pct: float|None, label: str) を返す"""
    step     = pl.get("step", "")
    log_path = pl.get("log_path", "")

    if not log_path or not Path(log_path).exists():
        return None, ""

    content = Path(log_path).read_text(errors="replace")
    pct, label = None, ""

    if step == "extracting":
        if pl.get("is_360"):
            m = re.findall(r'\[(\d+)/(\d+)\]', content)
            if m:
                cur, tot = int(m[-1][0]), int(m[-1][1])
                pct = min(cur / tot, 1.0)
                label = f"フレーム変換 {cur}/{tot} 枚 ({pct*100:.0f}%)"
        else:
            tm = re.search(r'PROGRESS_TOTAL (\d+)', content)
            pm = re.findall(r'PROGRESS (\d+)/(\d+)', content)
            if tm and pm:
                tot = int(tm.group(1))
                cur = int(pm[-1][0])
                if tot > 0:
                    pct = min(cur / tot, 1.0)
                    label = f"フレーム抽出 {cur}/{tot} 枚 ({pct*100:.0f}%)"

    elif step == "colmap":
        if pl.get("use_hloc"):
            m = re.findall(r'\[(\d+)/4\]', content)
        else:
            m = re.findall(r'\[COLMAP (\d+)/4\]', content)
        if m:
            cur = int(m[-1])
            pct = min(cur / 4, 1.0)
            label = f"ステップ {cur}/4: {_COLMAP_SUB.get(cur, '')} ({pct*100:.0f}%)"

    elif step == "training":
        total = pl.get("iterations", 30000)
        tm = re.findall(rf'(\d+)/{total}', content)
        if not tm:
            tm = re.findall(r'\[ITER\s+(\d+)\]', content)
        if tm:
            cur = int(tm[-1])
            pct = min(cur / total, 1.0)
            label = f"学習 {cur:,}/{total:,} iter ({pct*100:.0f}%)"

    return pct, label


def render_pipeline_status(compact: bool = False):
    """
    パイプライン進捗ウィジェットを描画する。

    compact=True のときはシンプルな1行表示＋進捗バーのみ。
    compact=False のときはステップバッジ・経過時間・ログも含む。
    """
    pl = _load_state()

    active = pl.get("active") and pl.get("step") not in ("done", "failed", "setup", None)
    step   = pl.get("step", "")

    # ── 非アクティブ ─────────────────────────────────────────────────────────
    if not active:
        done = (step == "done")
        color  = "#00cc66" if done else "#2a6080"
        icon   = "✅" if done else "💤"
        msg    = "パイプライン完了" if done else "実行中のパイプラインはありません"
        if pl.get("experiment_dir") and done:
            scene = Path(pl["experiment_dir"]).name
            msg   = f"完了: {scene}"
        st.markdown(
            f'<span style="color:{color};font-size:0.8rem;">{icon} {msg}</span>',
            unsafe_allow_html=True,
        )
        return

    # ── アクティブ ───────────────────────────────────────────────────────────
    exp_dir   = pl.get("experiment_dir", "")
    scene     = Path(exp_dir).name if exp_dir else "不明"
    step_ja   = _STEP_JA.get(step, step)
    elapsed   = time.time() - pl.get("start_time", time.time())
    pct, prog_label = _parse_progress(pl)

    if compact:
        # 1行ヘッダー + プログレスバー
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0d1b2e,#0a1520);'
            f'border:1px solid #00aaff33;border-radius:8px;padding:0.5rem 0.8rem;">'
            f'<span style="color:#00aaff;font-size:0.75rem;">🚀 {scene}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#4a90b8;font-size:0.72rem;">{step_ja} 実行中</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#2a6080;font-size:0.7rem;">{elapsed/60:.1f} 分経過</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if pct is not None:
            st.caption(prog_label)
            st.progress(pct)
        else:
            st.caption("進捗を取得中...")
        return

    # フルビュー
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0d1b2e,#091420);'
        f'border:1px solid #00aaff44;border-radius:10px;padding:0.7rem 1rem;'
        f'margin-bottom:0.5rem;">'
        f'<b style="color:#00aaff;">🚀 {scene}</b>'
        f'&nbsp;&nbsp;<span style="color:#e0e6f0;font-size:0.85rem;">{step_ja} 実行中</span>'
        f'&nbsp;&nbsp;<span style="color:#4a90b8;font-size:0.75rem;">{elapsed/60:.1f} 分経過</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ステップバッジ
    STEPS = [("extracting", "① フレーム抽出"), ("colmap", "② COLMAP"), ("training", "③ 学習")]
    step_status = pl.get("step_status", {})
    cols = st.columns(3)
    for col, (sk, slabel) in zip(cols, STEPS):
        st_s = step_status.get(sk, "waiting")
        if step == sk and st_s != "done":
            st_s = "running"
        color = {"done": "#00cc66", "running": "#00aaff", "error": "#ff4444"}.get(st_s, "#334455")
        icon  = {"done": "✅", "running": "🔄", "error": "❌"}.get(st_s, "⏳")
        col.markdown(
            f'<span style="color:{color};font-size:0.78rem;">{icon} {slabel}</span>',
            unsafe_allow_html=True,
        )

    # プログレスバー
    if pct is not None:
        st.caption(prog_label)
        st.progress(pct)
    else:
        st.caption("進捗を取得中...")

    # ログ（折りたたみ）
    log_path = pl.get("log_path", "")
    if log_path and Path(log_path).exists():
        lines = [l for l in Path(log_path).read_text(errors="replace").split("\n") if l.strip()]
        if lines:
            with st.expander("最新ログ（直近5行）", expanded=False):
                st.code("\n".join(lines[-5:]), language=None)
