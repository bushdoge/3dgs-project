# GPU / CPU / メモリのリアルタイムモニタリングページ

import sys
sys.path.insert(0, "/workspace")

import streamlit as st
import subprocess
import time
import re
import os
from pathlib import Path
from collections import deque
from datetime import datetime

import altair as alt
import pandas as pd

from pipeline_widget import render_pipeline_status


# ── カラーパレット ────────────────────────────────────────────────────────────
C_GPU  = "#00e5ff"
C_CPU  = "#00e878"
C_MEM  = "#b040ff"
C_SWP  = "#ff8c00"
C_TEMP = "#ffaa00"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a;
    color: #e0e6f0;
  }}
  .block-container {{ padding: 1.5rem 2rem; }}

  .header-title {{
    font-size: 2rem; font-weight: 700; letter-spacing: 0.15em;
    color: {C_GPU}; text-shadow: 0 0 12px {C_GPU}88; margin-bottom: 0;
  }}
  .header-sub {{
    font-size: 0.75rem; color: #4a90b8; letter-spacing: 0.2em; margin-top: 0.1rem;
  }}

  /* ── メトリクスカード ── */
  .mc {{
    background: linear-gradient(135deg, #0d1b2e 0%, #0f2340 100%);
    border: 1px solid #1a3a5c; border-radius: 12px;
    padding: 1.1rem 1.3rem; position: relative; overflow: hidden;
    margin-bottom: 0.4rem;
  }}
  .mc::before {{ content:""; position:absolute; top:0; left:0; right:0; height:2px; }}
  .mc-gpu::before {{ background: linear-gradient(90deg, {C_GPU}, #7b2ff7); }}
  .mc-cpu::before {{ background: linear-gradient(90deg, {C_CPU}, #00ff99); }}
  .mc-mem::before {{ background: linear-gradient(90deg, #8800ee, {C_MEM}); }}
  .mc-swp::before {{ background: linear-gradient(90deg, #cc4400, {C_SWP}); }}

  .mc-label {{ font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
               color:#4a90b8; margin-bottom:0.25rem; }}
  .mc-val    {{ font-size:1.9rem; font-weight:700; line-height:1.1; }}
  .mc-val-sm {{ font-size:1.3rem; font-weight:700; line-height:1.1; }}
  .mc-unit   {{ font-size:0.8rem; color:#4a90b8; margin-left:0.15rem; }}
  .mc-sub    {{ font-size:0.65rem; color:#2a6080; margin-top:0.15rem; }}

  /* ── プログレスバー ── */
  .bar-wrap {{ background:#0a1520; border-radius:4px; height:7px;
               width:100%; overflow:hidden; margin-top:0.45rem; }}
  .bar-gpu  {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#0066cc,{C_GPU});
               box-shadow:0 0 8px {C_GPU}66; transition:width 0.4s ease; }}
  .bar-vram {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#4a00cc,#7b2ff7);
               box-shadow:0 0 8px #7b2ff766; transition:width 0.4s ease; }}
  .bar-cpu  {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#009944,{C_CPU});
               box-shadow:0 0 8px {C_CPU}66; transition:width 0.4s ease; }}
  .bar-mem  {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#6600bb,{C_MEM});
               box-shadow:0 0 8px {C_MEM}66; transition:width 0.4s ease; }}
  .bar-swp  {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#cc4400,{C_SWP});
               box-shadow:0 0 8px {C_SWP}66; transition:width 0.4s ease; }}
  .bar-ok   {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#006633,#00cc66);
               box-shadow:0 0 8px #00cc6666; transition:width 0.4s ease; }}
  .bar-warn {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#996600,{C_TEMP});
               box-shadow:0 0 8px {C_TEMP}66; transition:width 0.4s ease; }}
  .bar-hot  {{ height:100%; border-radius:4px;
               background:linear-gradient(90deg,#990000,#ff3333);
               box-shadow:0 0 8px #ff333366; transition:width 0.4s ease; }}

  /* ── プロセステーブル ── */
  .process-table {{ width:100%; border-collapse:collapse; font-size:0.75rem; }}
  .process-table th {{
    color:#4a90b8; text-align:left; padding:0.4rem 0.6rem;
    border-bottom:1px solid #1a3a5c; letter-spacing:0.1em;
    font-size:0.65rem; text-transform:uppercase;
  }}
  .process-table td {{
    padding:0.4rem 0.6rem; border-bottom:1px solid #0d1b2e; color:#b0c8e0;
  }}
  .process-table tr:hover td {{ background:#0f2340; }}

  /* ── タイムスタンプ ── */
  .timestamp {{ font-size:0.65rem; color:#2a6080; letter-spacing:0.1em; }}
  .dot-live {{
    display:inline-block; width:6px; height:6px; border-radius:50%;
    background:{C_GPU}; box-shadow:0 0 6px {C_GPU};
    margin-right:6px; animation:pulse 1.5s infinite;
  }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}

  div[data-testid="stMetric"] {{ display:none; }}
</style>
""", unsafe_allow_html=True)


# ── パース関数 ────────────────────────────────────────────────────────────────

def _get_nvml_error_detail() -> str:
    """NVMLが失敗した理由を /proc や ctypes で補足調査して返す"""
    lines = []
    # カーネルドライバ確認
    try:
        ver = Path("/proc/driver/nvidia/version").read_text().splitlines()[0]
        lines.append(f"カーネルドライバ: {ver.split('Module')[1].strip().split()[0]}")
    except Exception:
        lines.append("カーネルドライバ: 情報取得不可")
    # デバイスファイル確認
    devs = [str(p) for p in Path("/dev").glob("nvidia*")]
    lines.append(f"デバイスファイル: {', '.join(sorted(devs)) if devs else 'なし'}")
    # NVMLエラーコード確認
    try:
        import ctypes
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        code = lib.nvmlInit_v2()
        meaning = {0: "SUCCESS", 3: "権限不足", 6: "ドライバ未ロード", 999: "UNKNOWN（再起動が必要）"}
        lines.append(f"NVMLエラーコード: {code} — {meaning.get(code, '不明')}")
    except Exception as e:
        lines.append(f"NVML直接呼び出し失敗: {e}")
    return " / ".join(lines)


def parse_nvidia_smi():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
             "utilization.memory,memory.used,memory.total,power.draw,"
             "power.limit,fan.speed,clocks.current.graphics,clocks.current.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 12:
                continue
            def sf(v):
                try: return float(re.sub(r"[^\d.]", "", v))
                except: return 0.0
            gpus.append({
                "index": int(parts[0]), "name": parts[1],
                "temp": sf(parts[2]),  "gpu_util": sf(parts[3]),
                "mem_util": sf(parts[4]), "mem_used": sf(parts[5]),
                "mem_total": sf(parts[6]), "power_draw": sf(parts[7]),
                "power_limit": sf(parts[8]), "fan": sf(parts[9]),
                "clk_gpu": sf(parts[10]), "clk_mem": sf(parts[11]),
            })
        return gpus, None
    except Exception as e:
        return [], str(e)


def parse_processes():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        procs = []
        for line in result.stdout.strip().splitlines():
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4: continue
            procs.append({"pid": parts[1], "mem_mb": parts[2],
                          "name": parts[3].split("/")[-1][:40]})
        return procs
    except:
        return []


def parse_cpu():
    try:
        with open("/proc/stat") as f:
            lines = f.readlines()
        def stat(line):
            v = list(map(int, line.split()[1:]))
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            return sum(v), idle
        total, idle = stat(lines[0])
        cores = [stat(l) for l in lines[1:] if l.startswith("cpu")]
        prev = st.session_state.get("_cpu_prev")
        st.session_state["_cpu_prev"] = {"total": total, "idle": idle, "cores": cores}
        if prev is None:
            return {"util": 0.0, "cores": [0.0]*len(cores),
                    "load": os.getloadavg(), "n": len(cores)}
        dt = total - prev["total"]; di = idle - prev["idle"]
        util = (1 - di/dt)*100 if dt else 0
        cu = []
        for i, (ct, ci) in enumerate(cores):
            pt, pi = prev["cores"][i] if i < len(prev["cores"]) else (ct, ci)
            d = ct-pt; e = ci-pi
            cu.append((1-e/d)*100 if d else 0)
        return {"util": max(0.0, min(100.0, util)),
                "cores": [max(0.0, min(100.0, u)) for u in cu],
                "load": os.getloadavg(), "n": len(cores)}
    except:
        return {"util": 0.0, "cores": [], "load": (0, 0, 0), "n": 0}


def parse_memory():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.strip().split()[0])
        mt = info.get("MemTotal", 0); ma = info.get("MemAvailable", 0)
        mu = mt - ma
        st_ = info.get("SwapTotal", 0); sf_ = info.get("SwapFree", 0)
        su = st_ - sf_
        g = lambda kb: kb/1024/1024
        return {"mt": g(mt), "mu": g(mu), "mp": mu/mt*100 if mt else 0,
                "st": g(st_), "su": g(su), "sp": su/st_*100 if st_ else 0}
    except:
        return {"mt":0,"mu":0,"mp":0,"st":0,"su":0,"sp":0}


# ── UI ヘルパー ───────────────────────────────────────────────────────────────
def bar(pct, cls):
    w = min(max(pct, 0), 100)
    return f'<div class="bar-wrap"><div class="{cls}" style="width:{w}%"></div></div>'

def temp_bar(t):
    return bar(t, "bar-ok" if t < 50 else ("bar-warn" if t < 75 else "bar-hot"))

def card(label, value, unit, sub="", bar_html="", extra_style="", card_cls="mc-gpu"):
    return f"""
    <div class="mc {card_cls}">
      <div class="mc-label">{label}</div>
      <div class="mc-val{extra_style}">{value}<span class="mc-unit">{unit}</span></div>
      {'<div class="mc-sub">'+sub+'</div>' if sub else ''}
      {bar_html}
    </div>"""


# ── グラフ（Altair、固定30ポイント・0秒左端固定） ────────────────────────────
GRAPH_N = 30

def make_chart(values, color, y_max=None, y_min=0, refresh_rate=2):
    data = list(values)
    if len(data) < GRAPH_N:
        data = data + [float("nan")] * (GRAPH_N - len(data))
    else:
        data = data[-GRAPH_N:]
    seconds = [i * refresh_rate for i in range(GRAPH_N)]
    df = pd.DataFrame({"t": seconds, "v": data})
    x_max = (GRAPH_N - 1) * refresh_rate
    y_scale = alt.Scale(domain=[y_min, y_max]) if y_max is not None else alt.Scale()
    return (
        alt.Chart(df)
        .mark_area(
            line={"color": color, "strokeWidth": 1.5},
            color=alt.Gradient(
                gradient="linear", x1=0, x2=0, y1=1, y2=0,
                stops=[alt.GradientStop(color=color+"00", offset=0),
                       alt.GradientStop(color=color+"44", offset=1)],
            ),
        )
        .encode(
            x=alt.X("t:Q", scale=alt.Scale(domain=[0, x_max]),
                    axis=alt.Axis(title="sec", labelFontSize=9,
                                  tickCount=5, grid=False, domain=False)),
            y=alt.Y("v:Q", scale=y_scale,
                    axis=alt.Axis(labelFontSize=9, tickCount=3,
                                  grid=True, gridColor="#1a3a5c",
                                  domain=False, ticks=False)),
        )
        .properties(height=110)
    )


# ── 初期化 ────────────────────────────────────────────────────────────────────
if "gpu_hist" not in st.session_state:
    st.session_state.gpu_hist = {}
if "sys_hist" not in st.session_state:
    st.session_state.sys_hist = {k: deque(maxlen=GRAPH_N) for k in ("cpu", "mem", "swp")}

# ── ヘッダー ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="header-title">⚡ SYSTEM MONITOR</div>'
    '<div class="header-sub">REAL-TIME GPU / CPU / MEMORY DASHBOARD</div>',
    unsafe_allow_html=True,
)

# ── コントロール ──────────────────────────────────────────────────────────────
cc1, cc2, cc3, _ = st.columns([1, 1, 1, 5])
with cc1:
    refresh_rate = st.selectbox("Refresh", [1, 2, 5, 10], index=2,
                                label_visibility="collapsed")
with cc2:
    auto = st.toggle("Auto", value=True)
with cc3:
    show_graph = st.toggle("Graph", value=False)

placeholder = st.empty()

# ═════════════════════════════════════════════════════════════════════════════
while True:
    gpus, gpu_err  = parse_nvidia_smi()
    procs = parse_processes()
    cpu   = parse_cpu()
    mem   = parse_memory()
    now   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # 履歴更新
    for gpu in gpus:
        idx = gpu["index"]
        if idx not in st.session_state.gpu_hist:
            st.session_state.gpu_hist[idx] = {
                k: deque(maxlen=GRAPH_N) for k in ("util", "vram", "temp", "power")
            }
        h = st.session_state.gpu_hist[idx]
        h["util"].append(gpu["gpu_util"])
        h["vram"].append(gpu["mem_used"])
        h["temp"].append(gpu["temp"])
        h["power"].append(gpu["power_draw"])

    sh = st.session_state.sys_hist
    sh["cpu"].append(cpu["util"])
    sh["mem"].append(mem["mu"])
    sh["swp"].append(mem["su"])

    with placeholder.container():

        # ══════════════════════════════════════
        #  パイプライン進捗
        # ══════════════════════════════════════
        st.markdown(
            '<div style="font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;'
            'color:#4a90b8;border-bottom:1px solid #1a3a5c;padding-bottom:0.3rem;'
            'margin-bottom:0.6rem;">Pipeline Status</div>',
            unsafe_allow_html=True,
        )
        render_pipeline_status(compact=True)

        st.divider()

        # ══════════════════════════════════════
        #  GPU セクション（シアン） — トグル
        # ══════════════════════════════════════
        if not gpus:
            detail = _get_nvml_error_detail()
            st.warning(
                "⚠️ **nvidia-smi からGPUデータを取得できません**\n\n"
                f"{detail}\n\n"
                "**対処法：** ホスト側で `docker restart <コンテナ名>` を実行してください。"
                "コンテナ再起動後に GPU が認識されます。"
            )
        else:
            for gpu in gpus:
                mp  = gpu["mem_used"] / gpu["mem_total"] * 100 if gpu["mem_total"] else 0
                pwp = gpu["power_draw"] / gpu["power_limit"] * 100 if gpu["power_limit"] else 0

                with st.expander(f"⚡  GPU {gpu['index']}  —  {gpu['name']}", expanded=True):
                    g1, g2, g3, g4 = st.columns(4)
                    with g1:
                        st.markdown(card("GPU Utilization", f"{gpu['gpu_util']:.0f}", "%",
                            bar_html=bar(gpu["gpu_util"], "bar-gpu")), unsafe_allow_html=True)
                    with g2:
                        st.markdown(card("VRAM Usage", f"{gpu['mem_used']:.0f}", "MiB",
                            sub=f"/ {gpu['mem_total']:.0f} MiB &nbsp;({mp:.1f}%)",
                            bar_html=bar(mp, "bar-vram")), unsafe_allow_html=True)
                    with g3:
                        tc = C_GPU if gpu["temp"] < 50 else (C_TEMP if gpu["temp"] < 75 else "#ff3333")
                        st.markdown(card("Temperature",
                            f"<span style='color:{tc}'>{gpu['temp']:.0f}</span>", "°C",
                            sub=f"Fan &nbsp;{gpu['fan']:.0f}%",
                            bar_html=temp_bar(gpu["temp"])), unsafe_allow_html=True)
                    with g4:
                        st.markdown(card("Power Draw", f"{gpu['power_draw']:.0f}", "W",
                            sub=f"/ {gpu['power_limit']:.0f} W &nbsp;({pwp:.1f}%)",
                            bar_html=bar(pwp, "bar-swp")), unsafe_allow_html=True)

                    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

                    g5, g6, g7, g8 = st.columns(4)
                    with g5:
                        st.markdown(card("Graphics Clock", f"{gpu['clk_gpu']:.0f}", "MHz",
                            extra_style="-sm"), unsafe_allow_html=True)
                    with g6:
                        st.markdown(card("Memory Clock", f"{gpu['clk_mem']:.0f}", "MHz",
                            extra_style="-sm"), unsafe_allow_html=True)
                    with g7:
                        st.markdown(card("Mem Bandwidth", f"{gpu['mem_util']:.0f}", "%",
                            extra_style="-sm"), unsafe_allow_html=True)
                    with g8:
                        st.markdown(card("Active Processes", str(len(procs)), "",
                            extra_style="-sm"), unsafe_allow_html=True)

                    if show_graph:
                        h = st.session_state.gpu_hist.get(gpu["index"], {})
                        if h and len(h["util"]) > 1:
                            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                            cg1, cg2, cg3, cg4 = st.columns(4)
                            with cg1:
                                st.caption("GPU Util (%)")
                                st.altair_chart(make_chart(h["util"], C_GPU, y_max=100,
                                    refresh_rate=refresh_rate), use_container_width=True)
                            with cg2:
                                st.caption(f"VRAM (MiB) / {gpu['mem_total']:.0f}")
                                st.altair_chart(make_chart(h["vram"], C_GPU,
                                    y_max=gpu["mem_total"], refresh_rate=refresh_rate),
                                    use_container_width=True)
                            with cg3:
                                st.caption("Temperature (°C)")
                                st.altair_chart(make_chart(h["temp"], C_TEMP, y_max=100,
                                    refresh_rate=refresh_rate), use_container_width=True)
                            with cg4:
                                st.caption(f"Power (W) / {gpu['power_limit']:.0f}")
                                st.altair_chart(make_chart(h["power"], C_SWP,
                                    y_max=gpu["power_limit"], refresh_rate=refresh_rate),
                                    use_container_width=True)

        # ══════════════════════════════════════
        #  CPU セクション（グリーン） — トグル
        # ══════════════════════════════════════
        with st.expander("🖥  CPU", expanded=True):
            load1, load5, load15 = cpu["load"]

            ca1, ca2, ca3, ca4 = st.columns(4)
            with ca1:
                st.markdown(card("CPU Utilization", f"{cpu['util']:.1f}", "%",
                    sub=f"Load avg &nbsp;{load1:.2f} / {load5:.2f} / {load15:.2f}",
                    bar_html=bar(cpu["util"], "bar-cpu"),
                    card_cls="mc-cpu"), unsafe_allow_html=True)
            with ca2:
                st.markdown(card("Logical Cores", str(cpu["n"]), " cores",
                    extra_style="-sm", card_cls="mc-cpu"), unsafe_allow_html=True)
            with ca3:
                if show_graph and len(sh["cpu"]) > 1:
                    st.caption("CPU Util (%)")
                    st.altair_chart(make_chart(sh["cpu"], C_CPU, y_max=100,
                        refresh_rate=refresh_rate), use_container_width=True)

            cores = cpu["cores"]
            if cores:
                half = len(cores) // 2
                for row_cores in [cores[:half], cores[half:]]:
                    bar_cols = st.columns(len(row_cores))
                    for col, u in zip(bar_cols, row_cores):
                        c_col = C_CPU if u < 50 else (C_TEMP if u < 80 else "#ff4444")
                        with col:
                            st.markdown(
                                f'<div style="height:18px;background:#0a1520;'
                                f'border-radius:3px;overflow:hidden;">'
                                f'<div style="height:100%;background:{c_col};width:{u:.0f}%;'
                                f'transition:width 0.4s;"></div></div>',
                                unsafe_allow_html=True)

        # ══════════════════════════════════════
        #  メモリ セクション（パープル） — トグル
        # ══════════════════════════════════════
        with st.expander("💾  Memory", expanded=True):
            ma1, ma2, ma3, ma4 = st.columns(4)
            with ma1:
                st.markdown(card("RAM Usage", f"{mem['mu']:.1f}", "GiB",
                    sub=f"/ {mem['mt']:.1f} GiB &nbsp;({mem['mp']:.1f}%)",
                    bar_html=bar(mem["mp"], "bar-mem"),
                    card_cls="mc-mem"), unsafe_allow_html=True)
            with ma2:
                if show_graph and len(sh["mem"]) > 1:
                    st.caption(f"RAM (GiB) / {mem['mt']:.1f}")
                    st.altair_chart(make_chart(sh["mem"], C_MEM, y_max=mem["mt"],
                        refresh_rate=refresh_rate), use_container_width=True)
            with ma3:
                st.markdown(card("Swap Usage", f"{mem['su']:.1f}", "GiB",
                    sub=f"/ {mem['st']:.1f} GiB &nbsp;({mem['sp']:.1f}%)",
                    bar_html=bar(mem["sp"], "bar-swp"),
                    card_cls="mc-swp"), unsafe_allow_html=True)
            with ma4:
                if show_graph and len(sh["swp"]) > 1:
                    st.caption(f"Swap (GiB) / {mem['st']:.1f}")
                    st.altair_chart(make_chart(sh["swp"], C_SWP,
                        y_max=max(mem["st"], 0.1), refresh_rate=refresh_rate),
                        use_container_width=True)

        # ══════════════════════════════════════
        #  プロセステーブル — トグル
        # ══════════════════════════════════════
        with st.expander("⚙  Compute Processes", expanded=True):
            if procs:
                rows = "".join(
                    f"<tr><td>{p['pid']}</td><td>{p['name']}</td>"
                    f"<td>{p['mem_mb']} MiB</td></tr>"
                    for p in procs
                )
                st.markdown(f"""
                <table class="process-table">
                  <thead><tr><th>PID</th><th>Process</th><th>VRAM</th></tr></thead>
                  <tbody>{rows}</tbody>
                </table>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    '<span style="color:#2a6080;font-size:0.75rem;">'
                    'No active compute processes</span>',
                    unsafe_allow_html=True)

        # ── タイムスタンプ ──
        st.markdown(
            f"<div style='margin-top:1.2rem'>"
            f"<span class='dot-live'></span>"
            f"<span class='timestamp'>LAST UPDATE: {now}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not auto:
        break

    time.sleep(refresh_rate)
