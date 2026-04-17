import streamlit as st
import subprocess
import time
import re
from datetime import datetime

st.set_page_config(
    page_title="GPU Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #0a0e1a;
    color: #e0e6f0;
  }

  .block-container { padding: 1.5rem 2rem; }

  .header-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #00e5ff;
    text-shadow: 0 0 12px #00e5ff88;
    margin-bottom: 0;
  }
  .header-sub {
    font-size: 0.75rem;
    color: #4a90b8;
    letter-spacing: 0.2em;
    margin-top: 0.1rem;
  }

  .metric-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #0f2340 100%);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00e5ff, #7b2ff7);
  }
  .metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #4a90b8;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00e5ff;
    text-shadow: 0 0 8px #00e5ff66;
    line-height: 1.1;
  }
  .metric-unit {
    font-size: 0.85rem;
    color: #4a90b8;
    margin-left: 0.2rem;
  }
  .metric-sub {
    font-size: 0.7rem;
    color: #2a6080;
    margin-top: 0.2rem;
  }

  .gpu-name-badge {
    display: inline-block;
    background: linear-gradient(90deg, #001a33, #00264d);
    border: 1px solid #00e5ff44;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    color: #00e5ff;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
  }

  .bar-wrap {
    background: #0a1520;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 0.5rem;
  }
  .bar-fill-blue {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #0066cc, #00e5ff);
    box-shadow: 0 0 8px #00e5ff88;
    transition: width 0.4s ease;
  }
  .bar-fill-purple {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4a00cc, #7b2ff7);
    box-shadow: 0 0 8px #7b2ff766;
    transition: width 0.4s ease;
  }
  .bar-fill-green {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #006633, #00cc66);
    box-shadow: 0 0 8px #00cc6666;
    transition: width 0.4s ease;
  }
  .bar-fill-orange {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #cc4400, #ff7700);
    box-shadow: 0 0 8px #ff770066;
    transition: width 0.4s ease;
  }

  .process-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
  }
  .process-table th {
    color: #4a90b8;
    text-align: left;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid #1a3a5c;
    letter-spacing: 0.1em;
    font-size: 0.65rem;
    text-transform: uppercase;
  }
  .process-table td {
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid #0d1b2e;
    color: #b0c8e0;
  }
  .process-table tr:hover td { background: #0f2340; }

  .timestamp {
    font-size: 0.65rem;
    color: #2a6080;
    letter-spacing: 0.1em;
  }
  .dot-live {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00e5ff;
    box-shadow: 0 0 6px #00e5ff;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .section-title {
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #4a90b8;
    text-transform: uppercase;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 0.3rem;
    margin-bottom: 0.8rem;
  }

  div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)


def parse_nvidia_smi():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,"
                "utilization.memory,memory.used,memory.total,power.draw,"
                "power.limit,fan.speed,clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 12:
                continue

            def safe_float(v, default=0.0):
                try:
                    return float(re.sub(r"[^\d.]", "", v))
                except ValueError:
                    return default

            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "temp": safe_float(parts[2]),
                "gpu_util": safe_float(parts[3]),
                "mem_util": safe_float(parts[4]),
                "mem_used": safe_float(parts[5]),
                "mem_total": safe_float(parts[6]),
                "power_draw": safe_float(parts[7]),
                "power_limit": safe_float(parts[8]),
                "fan": safe_float(parts[9]),
                "clk_gpu": safe_float(parts[10]),
                "clk_mem": safe_float(parts[11]),
            })
        return gpus
    except Exception as e:
        return []


def parse_processes():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        procs = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            procs.append({
                "gpu": parts[0][:8] + "…",
                "pid": parts[1],
                "mem_mb": parts[2],
                "name": parts[3].split("/")[-1][:40],
            })
        return procs
    except Exception:
        return []


def bar_html(pct, color_class):
    w = min(max(pct, 0), 100)
    return f"""
    <div class="bar-wrap">
      <div class="{color_class}" style="width:{w}%"></div>
    </div>
    """


def temp_color(t):
    if t < 50:
        return "#00cc66"
    elif t < 75:
        return "#ffaa00"
    else:
        return "#ff3333"


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="header-title">⚡ GPU MONITOR</div>'
    '<div class="header-sub">REAL-TIME SYSTEM DASHBOARD</div>',
    unsafe_allow_html=True,
)

# ── Auto-refresh controls ────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 6])
with col_ctrl1:
    refresh_rate = st.selectbox("Refresh", [1, 2, 5, 10], index=1, label_visibility="collapsed")
with col_ctrl2:
    auto = st.toggle("Auto", value=True)

placeholder = st.empty()

while True:
    gpus = parse_nvidia_smi()
    procs = parse_processes()
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    with placeholder.container():

        if not gpus:
            st.error("nvidia-smi からデータを取得できませんでした。")
        else:
            for gpu in gpus:
                mem_pct = (gpu["mem_used"] / gpu["mem_total"] * 100) if gpu["mem_total"] else 0
                pwr_pct = (gpu["power_draw"] / gpu["power_limit"] * 100) if gpu["power_limit"] else 0

                st.markdown(
                    f'<div class="gpu-name-badge">GPU {gpu["index"]} — {gpu["name"]}</div>',
                    unsafe_allow_html=True,
                )

                # ── Row 1: 4 main metrics ──
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">GPU Utilization</div>
                      <div class="metric-value">{gpu['gpu_util']:.0f}<span class="metric-unit">%</span></div>
                      {bar_html(gpu['gpu_util'], 'bar-fill-blue')}
                    </div>""", unsafe_allow_html=True)

                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">VRAM Usage</div>
                      <div class="metric-value">{gpu['mem_used']:.0f}<span class="metric-unit">MiB</span></div>
                      <div class="metric-sub">/ {gpu['mem_total']:.0f} MiB &nbsp;({mem_pct:.1f}%)</div>
                      {bar_html(mem_pct, 'bar-fill-purple')}
                    </div>""", unsafe_allow_html=True)

                with c3:
                    tc = temp_color(gpu['temp'])
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">Temperature</div>
                      <div class="metric-value" style="color:{tc};text-shadow:0 0 8px {tc}88;">{gpu['temp']:.0f}<span class="metric-unit">°C</span></div>
                      <div class="metric-sub">Fan &nbsp;{gpu['fan']:.0f}%</div>
                      {bar_html(gpu['temp'] / 100 * 100, 'bar-fill-green' if gpu['temp'] < 75 else 'bar-fill-orange')}
                    </div>""", unsafe_allow_html=True)

                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">Power Draw</div>
                      <div class="metric-value">{gpu['power_draw']:.0f}<span class="metric-unit">W</span></div>
                      <div class="metric-sub">/ {gpu['power_limit']:.0f} W &nbsp;({pwr_pct:.1f}%)</div>
                      {bar_html(pwr_pct, 'bar-fill-orange')}
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

                # ── Row 2: clocks ──
                c5, c6, c7, c8 = st.columns(4)

                with c5:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">Graphics Clock</div>
                      <div class="metric-value" style="font-size:1.4rem">{gpu['clk_gpu']:.0f}<span class="metric-unit">MHz</span></div>
                    </div>""", unsafe_allow_html=True)

                with c6:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">Memory Clock</div>
                      <div class="metric-value" style="font-size:1.4rem">{gpu['clk_mem']:.0f}<span class="metric-unit">MHz</span></div>
                    </div>""", unsafe_allow_html=True)

                with c7:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">Mem Bandwidth Util</div>
                      <div class="metric-value" style="font-size:1.4rem">{gpu['mem_util']:.0f}<span class="metric-unit">%</span></div>
                    </div>""", unsafe_allow_html=True)

                with c8:
                    proc_count = len(procs)
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">Active Processes</div>
                      <div class="metric-value" style="font-size:1.4rem">{proc_count}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Process table ──
        st.markdown('<div class="section-title">Compute Processes</div>', unsafe_allow_html=True)

        if procs:
            rows = "".join(
                f"<tr><td>{p['pid']}</td><td>{p['name']}</td><td>{p['mem_mb']} MiB</td></tr>"
                for p in procs
            )
            st.markdown(f"""
            <table class="process-table">
              <thead><tr><th>PID</th><th>Process</th><th>VRAM</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#2a6080;font-size:0.75rem;">No active compute processes</span>', unsafe_allow_html=True)

        # ── Timestamp ──
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
