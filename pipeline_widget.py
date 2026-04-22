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
_COLMAP_SUB = {
    4: {1: "特徴点抽出",    2: "マッチング",          3: "3D再構成",    4: "undistortion"},
    5: {1: "局所特徴点抽出", 2: "グローバル特徴量抽出", 3: "ペアリスト生成", 4: "マッチング", 5: "SfM再構成"},
}


def _load_state() -> dict:
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


# ══════════════════════════════════════════════════════════════════════════════
#  COLMAPサブステップ解析
# ══════════════════════════════════════════════════════════════════════════════

def _parse_colmap_substeps(pl: dict) -> list:
    """
    ログからHLoc/COLMAPの各サブステップ進捗を解析する。
    Returns list of dicts:
      {num, total, name, status: "done"|"running"|"waiting", cur, items, pct}
    """
    log_path = pl.get("log_path", "")
    if not log_path or not Path(log_path).exists():
        return []

    content = Path(log_path).read_text(errors="replace")
    use_hloc = pl.get("use_hloc", True)

    # ステップ数と名称を決定
    if use_hloc:
        m5 = re.findall(r'\[(\d+)/5\]', content)
        m4 = re.findall(r'\[(\d+)/4\]', content)
        total_steps = 5 if m5 else (4 if m4 else None)
        markers     = m5 if m5 else m4
        step_pat    = re.compile(r'\[(\d+)/' + str(total_steps or 5) + r'\]') if total_steps else None
    else:
        markers     = re.findall(r'\[COLMAP (\d+)/4\]', content)
        total_steps = 4
        step_pat    = re.compile(r'\[COLMAP (\d+)/4\]')

    if not total_steps:
        return []

    current_step = int(markers[-1]) if markers else 0
    step_names   = _COLMAP_SUB.get(total_steps, {})

    # ステップ開始位置をコンテンツ内で特定
    positions = {}  # step_num -> char index
    if step_pat:
        for m in step_pat.finditer(content):
            positions[int(m.group(1))] = m.start()

    result = []
    for i in range(1, total_steps + 1):
        name   = step_names.get(i, f"ステップ{i}")
        status = "done" if i < current_step else ("running" if i == current_step else "waiting")

        cur = items = pct = None

        if status in ("done", "running") and i in positions:
            start = positions[i]
            end   = positions.get(i + 1, len(content))
            section = content[start:end]

            # tqdmの進捗: "| N/TOTAL [elapsed<remaining"
            tqdm_m = re.findall(r'\|\s*(\d+)/(\d+)\s+\[[\d?]', section)
            if tqdm_m:
                cur   = int(tqdm_m[-1][0])
                items = int(tqdm_m[-1][1])
                pct   = min(cur / items, 1.0) if items > 0 else None
            elif status == "done":
                pct = 1.0

        result.append({"num": i, "total": total_steps, "name": name,
                        "status": status, "cur": cur, "items": items, "pct": pct})
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  全体進捗パース（非COLMAP用 / compact表示用）
# ══════════════════════════════════════════════════════════════════════════════

def _parse_progress(pl: dict) -> tuple:
    """(pct: float|None, label: str) を返す"""
    step     = pl.get("step", "")
    log_path = pl.get("log_path", "")

    if not log_path or not Path(log_path).exists():
        return None, ""

    content = Path(log_path).read_text(errors="replace")

    if step == "extracting":
        if pl.get("is_360"):
            m = re.findall(r'\[(\d+)/(\d+)\]', content)
            if m:
                cur, tot = int(m[-1][0]), int(m[-1][1])
                pct = min(cur / tot, 1.0)
                return pct, f"フレーム変換 {cur}/{tot} 枚 ({pct*100:.0f}%)"
        else:
            tm = re.search(r'PROGRESS_TOTAL (\d+)', content)
            pm = re.findall(r'PROGRESS (\d+)/(\d+)', content)
            if tm and pm:
                tot = int(tm.group(1))
                cur = int(pm[-1][0])
                if tot > 0:
                    pct = min(cur / tot, 1.0)
                    return pct, f"フレーム抽出 {cur}/{tot} 枚 ({pct*100:.0f}%)"

    elif step == "colmap":
        # サブステップから全体進捗を合成
        substeps = _parse_colmap_substeps(pl)
        if substeps:
            done_steps = sum(1 for s in substeps if s["status"] == "done")
            running    = next((s for s in substeps if s["status"] == "running"), None)
            total      = substeps[0]["total"]
            if running and running["pct"] is not None:
                overall = (done_steps + running["pct"]) / total
                label   = (f'[{running["num"]}/{total}] {running["name"]} '
                           f'{running["cur"]:,}/{running["items"]:,} ({running["pct"]*100:.0f}%)'
                           if running["cur"] is not None
                           else f'[{running["num"]}/{total}] {running["name"]}')
            else:
                overall = done_steps / total
                label   = f'ステップ {done_steps}/{total} 完了'
            return min(overall, 1.0), label

    elif step == "training":
        total = pl.get("iterations", 30000)
        tm = re.findall(rf'(\d+)/{total}', content)
        if not tm:
            tm = re.findall(r'\[ITER\s+(\d+)\]', content)
        if tm:
            cur = int(tm[-1])
            pct = min(cur / total, 1.0)
            return pct, f"学習 {cur:,}/{total:,} iter ({pct*100:.0f}%)"

    return None, ""


# ══════════════════════════════════════════════════════════════════════════════
#  描画関数
# ══════════════════════════════════════════════════════════════════════════════

def _render_substep_bars(substeps: list):
    """COLMAPサブステップを個別プログレスバーで描画する"""
    COLOR  = {"done": "#00cc66", "running": "#00aaff", "waiting": "#334455"}
    ICON   = {"done": "✅", "running": "🔄", "waiting": "⏳"}

    for s in substeps:
        c = COLOR[s["status"]]
        ico = ICON[s["status"]]
        tag = f'[{s["num"]}/{s["total"]}]'

        if s["status"] == "waiting":
            st.markdown(
                f'<span style="color:{c};font-size:0.78rem;">'
                f'{ico} {tag} {s["name"]}</span>',
                unsafe_allow_html=True,
            )
            continue

        # 進捗テキスト
        pct_val = s["pct"] if s["pct"] is not None else (1.0 if s["status"] == "done" else 0.0)
        if s["cur"] is not None and s["items"] is not None:
            detail = f'{s["cur"]:,} / {s["items"]:,} ({pct_val*100:.0f}%)'
        elif s["status"] == "done":
            detail = "完了"
        else:
            detail = "処理中..."

        st.markdown(
            f'<span style="color:{c};font-size:0.78rem;">'
            f'{ico} {tag} {s["name"]} &nbsp;'
            f'<span style="color:#4a90b8;">{detail}</span></span>',
            unsafe_allow_html=True,
        )
        st.progress(pct_val)


def render_pipeline_status(compact: bool = False):
    """
    パイプライン進捗ウィジェットを描画する。

    compact=True: 1行ヘッダー + 全体プログレスバー
    compact=False: ステップバッジ + サブステップ詳細（COLMAP時）+ ログ
    """
    pl   = _load_state()
    active = pl.get("active") and pl.get("step") not in ("done", "failed", "setup", None)
    step   = pl.get("step", "")

    # ── 非アクティブ ─────────────────────────────────────────────────────────
    if not active:
        done  = (step == "done")
        color = "#00cc66" if done else "#2a6080"
        icon  = "✅" if done else "💤"
        msg   = "パイプライン完了" if done else "実行中のパイプラインはありません"
        if pl.get("experiment_dir") and done:
            msg = f"完了: {Path(pl['experiment_dir']).name}"
        st.markdown(
            f'<span style="color:{color};font-size:0.8rem;">{icon} {msg}</span>',
            unsafe_allow_html=True,
        )
        return

    # ── アクティブ ───────────────────────────────────────────────────────────
    scene   = Path(pl.get("experiment_dir", "")).name or "不明"
    step_ja = _STEP_JA.get(step, step)
    elapsed = time.time() - pl.get("start_time", time.time())
    pct, prog_label = _parse_progress(pl)

    if compact:
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

    # ── フルビュー ────────────────────────────────────────────────────────────
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
    STEPS       = [("extracting", "① フレーム抽出"), ("colmap", "② COLMAP"), ("training", "③ 学習")]
    step_status = pl.get("step_status", {})
    cols        = st.columns(3)
    for col, (sk, slabel) in zip(cols, STEPS):
        st_s  = step_status.get(sk, "waiting")
        if step == sk and st_s != "done":
            st_s = "running"
        color = {"done": "#00cc66", "running": "#00aaff", "error": "#ff4444"}.get(st_s, "#334455")
        icon  = {"done": "✅", "running": "🔄", "error": "❌"}.get(st_s, "⏳")
        col.markdown(
            f'<span style="color:{color};font-size:0.78rem;">{icon} {slabel}</span>',
            unsafe_allow_html=True,
        )

    # ── COLMAPステップのみサブステップ詳細表示 ────────────────────────────────
    if step == "colmap":
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        substeps = _parse_colmap_substeps(pl)
        if substeps:
            _render_substep_bars(substeps)
        else:
            if pct is not None:
                st.caption(prog_label)
                st.progress(pct)
            else:
                st.caption("ログを解析中...")
    else:
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


# ══════════════════════════════════════════════════════════════════════════════
#  固定フッター（全ページ共通）
# ══════════════════════════════════════════════════════════════════════════════

def render_sticky_footer():
    """
    ページ下部に固定表示するプログレスバー。
    パイプライン実行中のみ表示。ページのリロード・操作時に更新される。
    各ページの末尾で呼び出す。
    """
    pl = _load_state()
    if not pl.get("active") or pl.get("step") in ("done", "failed", "setup", None):
        return

    pct, label = _parse_progress(pl)
    step_ja    = _STEP_JA.get(pl.get("step", ""), "")
    scene      = Path(pl.get("experiment_dir", "")).name or ""
    pct_val    = (pct or 0.0) * 100

    # COLMAPの場合は最も進んでいるサブステップ名を表示
    detail_label = label
    if pl.get("step") == "colmap":
        substeps = _parse_colmap_substeps(pl)
        running  = next((s for s in substeps if s["status"] == "running"), None)
        if running:
            sub_detail = (f'{running["cur"]:,}/{running["items"]:,}'
                          if running["cur"] is not None else "処理中")
            detail_label = (f'[{running["num"]}/{running["total"]}] '
                            f'{running["name"]} {sub_detail}')

    st.markdown(f"""
<style>
  #pipeline-sticky-footer {{
    position: fixed; bottom: 0; left: 0; right: 0;
    background: rgba(8, 12, 22, 0.96);
    border-top: 1px solid #1a3a5c;
    padding: 5px 20px;
    z-index: 99999;
    display: flex; align-items: center; gap: 14px;
    backdrop-filter: blur(6px);
    font-family: 'Share Tech Mono', monospace;
  }}
  #pipeline-sticky-footer .psf-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: #00e5ff; box-shadow: 0 0 6px #00e5ff;
    animation: psf-pulse 1.5s infinite; flex-shrink: 0;
  }}
  @keyframes psf-pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
  #pipeline-sticky-footer .psf-scene {{
    font-size: 0.7rem; color: #00aaff; white-space: nowrap; flex-shrink: 0;
  }}
  #pipeline-sticky-footer .psf-step {{
    font-size: 0.68rem; color: #4a90b8; white-space: nowrap; flex-shrink: 0;
  }}
  #pipeline-sticky-footer .psf-bar-wrap {{
    flex: 1; height: 3px; background: #0a1520; border-radius: 2px;
    overflow: hidden; min-width: 60px;
  }}
  #pipeline-sticky-footer .psf-bar {{
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #0055bb, #00e5ff);
    box-shadow: 0 0 5px #00e5ff55;
    width: {pct_val:.1f}%;
  }}
  #pipeline-sticky-footer .psf-pct {{
    font-size: 0.7rem; color: #00e5ff; white-space: nowrap; flex-shrink: 0;
  }}
  #pipeline-sticky-footer .psf-detail {{
    font-size: 0.65rem; color: #2a6080; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 300px;
  }}
  /* フッターと重ならないようにコンテンツ下余白を確保 */
  .block-container {{ padding-bottom: 2.5rem !important; }}
</style>
<div id="pipeline-sticky-footer">
  <div class="psf-dot"></div>
  <span class="psf-scene">🚀 {scene}</span>
  <span class="psf-step">{step_ja}</span>
  <div class="psf-bar-wrap"><div class="psf-bar"></div></div>
  <span class="psf-pct">{pct_val:.0f}%</span>
  <span class="psf-detail">{detail_label}</span>
</div>
""", unsafe_allow_html=True)


# 後方互換性のため旧名を残す
def render_pipeline_widget():
    render_pipeline_status(compact=True)
