# 作業モニター：いま何が動いていて、どこまで進んでいるかを一覧するページ
#
# 既存の「システムモニター」がハードウェア（GPU/CPU/メモリ）を見るのに対し、
# こちらは「作業（ジョブ）」を見る。GUI経由のジョブだけでなく、nohup で直接起動した
# 学習・COLMAP・Blender も /proc から自動検出する（検出ロジックは job_monitor.py）。
#
# 意匠：streamlit_app.py の共通スタイルに合わせる。
#   - カード      = surfaceContainer・角丸20px・ホバーで面が明るくなる（stMetric と同じ処方）
#   - 進捗バー    = 高さ14px・primary→tertiary のグラデ（stProgress と同じ処方）
#   - 見出し      = st.subheader（共通CSSが左のコーラルバーを付ける）
#   - 作業種別の色 = config.toml に列挙されたアクセント候補から割り当てる

import sys
sys.path.insert(0, "/workspace")

import html
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

import job_monitor as jm


# ══════════════════════════════════════════════════════════════════════════════
#  配色（.streamlit/config.toml のアクセント候補から）
#    Material 3 Expressive のカラーロールに合わせる（定義は .streamlit/config.toml）
#    primary #FFB59B / tertiary #EFC77A / secondary #E7BDAC
#    拡張アクセント: スカイ #A8C7FA / ラベンダー #CFBCFF / ミント #8ED9A8 / 無彩 #9E948A
# ══════════════════════════════════════════════════════════════════════════════

KIND_ACCENT = {
    "train":    "#FFB59B",   # primary（主役）
    "render":   "#CFBCFF",   # ラベンダー
    "colmap":   "#A8C7FA",   # スカイ
    "masks":    "#EFC77A",   # tertiary
    "extract":  "#8ED9A8",   # ミント
    "blender":  "#8ED9A8",
    "analysis": "#A8C7FA",
    "runner":   "#9E948A",   # M3 outline（束ねているだけなので目立たせない）
    "daemon":   "#9E948A",
}
ST_ACCENT = {"done": "#8ED9A8", "running": "#FFB59B", "failed": "#FFB4AB",
             "waiting": "#6F675E", "unknown": "#EFC77A"}
ST_JA     = {"done": "完了", "running": "実行中", "failed": "失敗",
             "waiting": "待機", "unknown": "中断?"}

# 種別・状態ごとの色は「クラスで持つ」（インラインのCSS変数は環境により落ちるため）
_KIND_CSS = "\n".join(
    f'  .k-{k} {{ --a:{c}; --a-soft:{c}22; }}' for k, c in KIND_ACCENT.items())
_ST_CSS = "\n".join(
    f'  .s-{k} {{ --a:{c}; --a-soft:{c}22; }}' for k, c in ST_ACCENT.items())

st.markdown(f"""
<style>
{_KIND_CSS}
{_ST_CSS}

  /* ── カード：stMetric / stExpander と同じ処方に、左のアクセントバーを足す ── */
  .jm-card {{
    position: relative;
    background: var(--m3-surface-c, #1E232B);
    border: none;
    border-radius: var(--m3-r-l, 20px);
    padding: .95rem 1.2rem 1rem 1.5rem;
    margin-bottom: .6rem;
    transition: background .25s cubic-bezier(.2,0,0,1),
                transform .35s cubic-bezier(.34,1.56,.64,1),
                box-shadow .25s cubic-bezier(.2,0,0,1);
  }}
  .jm-card:hover {{ background: var(--m3-surface-high, #282D35); transform: translateY(-2px);
                    box-shadow: 0 2px 8px 2px rgba(0,0,0,.26); }}
  /* 見出しの左バー（h2::before）と同じ意匠 */
  .jm-card::before {{
    content: ""; position: absolute; left: .55rem; top: 1rem; bottom: 1rem;
    width: 5px; border-radius: 999px; background: var(--a);
  }}

  .jm-head {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }}
  .jm-badge {{
    font-size:.7rem; font-weight:700; letter-spacing:.03em; white-space:nowrap;
    padding:2px 10px; border-radius:999px;
    color:var(--a); background:var(--a-soft); border:1px solid var(--a-soft);
  }}
  .jm-name {{ font-size:1.02rem; font-weight:650; letter-spacing:-.01em;
              color:var(--m3-on-surface,#EAE6DC); }}
  .jm-tag  {{ font-size:.78rem; color:var(--m3-tertiary,#EFC77A); }}
  .jm-pid  {{ margin-left:auto; font-size:.72rem; color:var(--m3-on-surface-var,#CFC7BC);
              white-space:nowrap; }}

  /* ── 進捗バー：stProgress と同じ処方 ── */
  .jm-barrow {{ display:flex; align-items:center; gap:.7rem; margin:.65rem 0 .1rem 0; }}
  .jm-bar {{ flex:1; height:14px; border-radius:999px;
             background:var(--m3-surface-high,#282D35); overflow:hidden; }}
  .jm-bar i {{ display:block; height:100%; border-radius:999px;
               background:linear-gradient(90deg,var(--m3-primary,#FFB59B),var(--m3-tertiary,#EFC77A));
               transition:width .5s cubic-bezier(.22,1.2,.36,1); }}
  .jm-barpct {{ font-size:.82rem; font-weight:700; color:var(--m3-on-surface,#EAE6DC);
                min-width:3.2em; text-align:right; }}

  /* ── 数値の並び ── */
  .jm-figs {{ display:flex; gap:1.6rem; flex-wrap:wrap; margin-top:.6rem; }}
  .jm-fig .k {{ display:block; font-size:.68rem; color:var(--m3-on-surface-var,#CFC7BC);
                letter-spacing:.06em; text-transform:uppercase; }}
  .jm-fig .v {{ font-size:.92rem; font-weight:650; color:var(--m3-on-surface,#EAE6DC); }}

  .jm-note {{ font-size:.75rem; color:#A8C7FA; margin-top:.55rem; }}
  .jm-warn {{ font-size:.75rem; color:var(--m3-error,#FFB4AB); font-weight:600; margin-top:.55rem; }}
  .jm-marks {{ font-size:.78rem; color:var(--m3-on-surface-var,#CFC7BC); margin-top:.55rem;
               border-top:1px solid var(--m3-outline-var,#4B443C); padding-top:.5rem; }}
  .jm-marks i {{ font-style:normal; color:var(--m3-outline,#998F84); }}
  .jm-marks b {{ color:var(--m3-tertiary,#EFC77A); }}

  /* ── ログ最終行：stCode と同じ質感 ── */
  .jm-log {{
    font-family: ui-monospace,"SFMono-Regular",Menlo,monospace;
    font-size:.75rem; color:var(--m3-on-surface-var,#CFC7BC);
    background:var(--m3-surface-lowest,#0D1117); border-radius:var(--m3-r-s,12px);
    padding:.4rem .6rem; margin-top:.6rem;
    white-space:pre-wrap; word-break:break-all;
  }}

  /* ── 工程表 ── */
  .jm-steps {{ margin-top:.6rem; border-top:1px solid rgba(255,255,255,.07);
               padding-top:.5rem; }}
  .jm-step  {{ display:flex; align-items:center; gap:.6rem;
               font-size:.82rem; color:#c8c2b6; padding:.16rem 0; }}
  .jm-dot   {{ width:9px; height:9px; border-radius:50%; flex:none;
               background:var(--a); box-shadow:0 0 0 3px var(--a-soft); }}
  .jm-step .n {{ flex:1; }}
  .jm-step .s {{ font-size:.74rem; font-weight:700; color:var(--a); }}

  /* ── 完了一覧 ── */
  .jm-row {{ display:flex; align-items:center; gap:.7rem; font-size:.82rem;
             color:#8f97a4; padding:.42rem .7rem; border-radius:10px;
             transition: background .12s ease; }}
  .jm-row:hover {{ background:rgba(255,255,255,.035); }}
  .jm-row .st {{ font-size:.72rem; font-weight:700; color:var(--a);
                 background:var(--a-soft); border-radius:999px;
                 padding:1px 9px; min-width:4.2em; text-align:center; flex:none; }}
  .jm-row .nm {{ color:#eae6dc; }}
  .jm-row .ag {{ margin-left:auto; color:#77808e; white-space:nowrap; }}

  /* ── 空表示 ── */
  .jm-empty {{ border:1px dashed rgba(255,255,255,.14); border-radius:14px;
               padding:1.5rem; text-align:center; color:#77808e; font-size:.9rem; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  部品
# ══════════════════════════════════════════════════════════════════════════════

def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def bar(pct) -> str:
    if pct is None:
        return ""
    w = max(0.0, min(pct, 1.0)) * 100
    return (f'<div class="jm-barrow"><div class="jm-bar">'
            f'<i style="width:{w:.1f}%"></i></div>'
            f'<span class="jm-barpct">{w:.0f}%</span></div>')


def figs(pairs) -> str:
    items = "".join(f'<div class="jm-fig"><span class="k">{esc(k)}</span>'
                    f'<span class="v">{esc(v)}</span></div>'
                    for k, v in pairs if v not in (None, "", "—"))
    return f'<div class="jm-figs">{items}</div>' if items else ""


def steps_html(steps) -> str:
    rows = []
    for s in steps:
        extra = f'（exit={s["code"]}）' if s.get("code") not in (None, 0) else ""
        rows.append(
            f'<div class="jm-step s-{s["status"]}"><span class="jm-dot"></span>'
            f'<span class="n">{esc(s["name"])}</span>'
            f'<span class="s">{ST_JA.get(s["status"], s["status"])}{esc(extra)}</span></div>')
    return f'<div class="jm-steps">{"".join(rows)}</div>' if rows else ""


def _eta_seconds(eta):
    try:
        n = [int(x) for x in eta.split(":")]
    except Exception:
        return None
    if len(n) == 3:
        return n[0] * 3600 + n[1] * 60 + n[2]
    if len(n) == 2:
        return n[0] * 60 + n[1]
    return None


def ticket(job, status_pack=None) -> str:
    if job["kind"] == "runner":
        title, tag = jm._fmt_script_name(job["cmdline"]), ""
    else:
        title = job["experiment"] or "（実験ディレクトリ不明）"
        tag = f'<span class="jm-tag">{esc(job["output_tag"])}</span>' if job["output_tag"] else ""

    body = (f'<div class="jm-head"><span class="jm-badge">{esc(job["label"])}</span>'
            f'<span class="jm-name">{esc(title)}</span>{tag}'
            f'<span class="jm-pid">PID {job["pid"]}</span></div>')

    body += bar(job["pct"])

    pairs = []
    if job["headline"]:
        pairs.append(("進捗", job["headline"]))
    pairs.append(("経過", jm.fmt_duration(job["elapsed"])))
    if job["eta"]:
        pairs.append(("残り", jm.fmt_eta(job["eta"])))
        eta_s = _eta_seconds(job["eta"])
        if eta_s is not None:
            pairs.append(("終了予定",
                          datetime.fromtimestamp(time.time() + eta_s).strftime("%m/%d %H:%M")))
    if job["detail"]:
        pairs.append(("状況", job["detail"]))
    body += figs(pairs)

    if status_pack:
        done_n = sum(1 for s in status_pack["steps"] if s["status"] == "done")
        body += (f'<div class="jm-note">工程 {done_n} / {len(status_pack["steps"])} 完了'
                 f'（{esc(Path(status_pack["file"]).name)}）</div>')
        body += steps_html(status_pack["steps"])

    if job.get("marks"):
        cells = "　".join(f'<i>{it:,}</i> <b>{v:.2f}</b>' for it, v in job["marks"])
        body += f'<div class="jm-marks">test PSNR　{cells}</div>'

    if job["log_age"] is not None and job["log_age"] > 180:
        body += (f'<div class="jm-warn">ログが {jm.fmt_duration(job["log_age"])} '
                 f'更新されていません（停止している可能性）</div>')

    if job["last_line"]:
        body += f'<div class="jm-log">{esc(job["last_line"])}</div>'
    elif job["log"]:
        body += f'<div class="jm-log">{esc(Path(job["log"]).name)}（出力なし）</div>'

    return f'<div class="jm-card k-{job["kind"]}">{body}</div>'


# ══════════════════════════════════════════════════════════════════════════════
#  ヘッダー・操作
# ══════════════════════════════════════════════════════════════════════════════

st.title("📋 作業モニター")
st.caption("実行中のプロセスとそのログから作業内容を推定して表示します"
           "（GUI経由でないCLIジョブも検出）。ハードウェアの推移は「システムモニター」ページ。")

c1, c2, c3 = st.columns([2, 1, 3], vertical_alignment="bottom")
with c1:
    interval = st.select_slider("自動更新", options=["切", "2秒", "5秒", "15秒", "60秒"],
                                value="5秒", help="この一覧だけを更新します")
with c2:
    st.button("いま更新", use_container_width=True)

run_every = {"切": None, "2秒": 2, "5秒": 5, "15秒": 15, "60秒": 60}[interval]


# ══════════════════════════════════════════════════════════════════════════════
#  本体（fragment：この部分だけ自動再実行される）
# ══════════════════════════════════════════════════════════════════════════════

@st.fragment(run_every=run_every)
def board():
    jobs = jm.scan_jobs()
    live_logs = [j["log"] for j in jobs if j["log"]]

    # ── GPU（学習中の健康状態。詳細はシステムモニター）──
    g = jm.gpu_brief()
    if g["ok"]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("GPU使用率", f'{g["util"]} %')
        m2.metric("VRAM", f'{g["used"]/1024:.1f} GB', f'/ {g["total"]/1024:.0f} GB',
                  delta_color="off")
        m3.metric("温度", f'{g["temp"]} ℃')
        m4.metric("最終確認", datetime.now().strftime("%H:%M:%S"), g["name"].replace("NVIDIA ", ""),
                  delta_color="off")
    else:
        st.error(f'GPUが見えません: {g["error"]}', icon="⚠️")

    # ── 実行中 ──
    st.subheader(f"実行中の作業　{len(jobs)} 件")
    if not jobs:
        st.markdown('<div class="jm-empty">動いている作業はありません</div>',
                    unsafe_allow_html=True)
    else:
        pids = {j["pid"] for j in jobs}
        roots = [j for j in jobs if j["ppid"] not in pids]
        ordered = []
        for r in roots:
            ordered.append(r)
            ordered += [j for j in jobs if j["ppid"] == r["pid"]]
        ordered += [j for j in jobs if j not in ordered]

        for j in ordered:
            pack = jm.find_status_for(j.get("script")) if j["kind"] == "runner" else None
            st.markdown(ticket(j, pack), unsafe_allow_html=True)

    # ── 工程表（スクリプトは終わったが記録が残っているもの）──
    shown = set()
    for j in jobs:
        if j["kind"] == "runner":
            sp = jm.find_status_for(j.get("script"))
            if sp:
                shown.add(sp["file"])
    packs = [s for s in jm.read_status_files(max_age_h=72) if s["file"] not in shown]
    if packs:
        st.subheader("実行スクリプトの工程表")
        for s in packs[:4]:
            done_n = sum(1 for x in s["steps"] if x["status"] == "done")
            cls = "s-done" if s["all_done"] else "s-unknown"
            head = (f'<div class="jm-head">'
                    f'<span class="jm-badge">{"完了" if s["all_done"] else "記録"}</span>'
                    f'<span class="jm-name">{esc(s["name"])}</span>'
                    f'<span class="jm-pid">{done_n} / {len(s["steps"])} 工程</span></div>')
            st.markdown(f'<div class="jm-card {cls}">{head}{steps_html(s["steps"])}</div>',
                        unsafe_allow_html=True)

    # ── 直近に終わったもの ──
    fin = jm.recent_finished(hours=72, limit=12, live_logs=live_logs)
    st.subheader("直近に終わった作業")
    if not fin:
        st.markdown('<div class="jm-empty">記録がありません</div>', unsafe_allow_html=True)
    else:
        rows = []
        for r in fin:
            where = f'<span>{esc(r["experiment"])}</span>' if r["experiment"] else ""
            rows.append(
                f'<div class="jm-row s-{r["status"]}">'
                f'<span class="st">{ST_JA.get(r["status"], "?")}</span>'
                f'<span class="nm">{esc(r["name"])}</span>'
                f'<span>{esc(r["label"])}</span>{where}'
                f'<span class="ag">{jm.fmt_duration(r["age"])}前</span></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)

        with st.expander("終わった作業の最終行を見る"):
            for r in fin:
                st.markdown(f'**{esc(r["name"])}** — {ST_JA.get(r["status"], "?")}')
                st.code(r["last_line"] or "(出力なし)", language=None)


board()
