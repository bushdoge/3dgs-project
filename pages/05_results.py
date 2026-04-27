# 3DGS学習結果（レンダリング実行・画像・ログ・ファイル構造）を確認するページ

import ast
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


# ── セッション状態の初期化 ────────────────────────────────────────────────────
if "render_proc" not in st.session_state:
    st.session_state.render_proc     = None
    st.session_state.render_log_path = None
    st.session_state.render_exp      = None


def is_rendering() -> bool:
    p = st.session_state.render_proc
    return p is not None and p.poll() is None

st.title("🖼️ 結果確認")
st.caption("学習結果のレンダリング画像・ログ・ファイル構造を確認します")

st.divider()

# ── 実験フォルダ選択 ──────────────────────────────────────────────────────────
experiments_dir = Path("/workspace/experiments")
experiment_list = []
if experiments_dir.exists():
    experiment_list = sorted(
        [str(p) for p in experiments_dir.iterdir() if p.is_dir()],
        reverse=True,
    )

if not experiment_list:
    st.warning("experiments/ フォルダに実験結果が見つかりません。")
    st.stop()

selected_exp = st.selectbox(
    "実験フォルダを選択",
    experiment_list,
    format_func=lambda x: Path(x).name,
)
exp_path = Path(selected_exp)

st.divider()

# ── フォルダ内の状態サマリー ──────────────────────────────────────────────────
st.subheader("📁 パイプラインの進捗")

def count_images(folder):
    p = Path(folder)
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".png", ".jpeg"})

col1, col2, col3, col4 = st.columns(4)

with col1:
    n_input = count_images(exp_path / "input")
    st.metric("入力フレーム", f"{n_input} 枚" if n_input else "未実行")

with col2:
    has_colmap = (exp_path / "sparse" / "0").exists()
    st.metric("COLMAP", "✅ 完了" if has_colmap else "❌ 未実行")

with col3:
    has_output = (exp_path / "output").exists()
    # 最新のpoint_cloudを探す
    ply_files = list((exp_path / "output").rglob("*.ply")) if has_output else []
    st.metric("3DGS学習", f"✅ {len(ply_files)}ply" if ply_files else ("⏳ フォルダあり" if has_output else "❌ 未実行"))

with col4:
    n_renders = count_images(exp_path / "renders")
    # gaussian-splatting の render.py 出力先も探す
    if n_renders == 0 and has_output:
        n_renders = sum(
            count_images(p)
            for p in (exp_path / "output").rglob("renders")
            if p.is_dir()
        )
    st.metric("レンダリング画像", f"{n_renders} 枚" if n_renders else "なし")

# ── COLMAP 再構成品質 ────────────────────────────────────────────────────────
if has_colmap:
    st.divider()
    st.subheader("📐 COLMAP 再構成品質")
    st.caption("sparse/0/ の cameras / images / points3D ファイルから再構成の品質を表示します。"
               "登録率 ≥ 80%・再投影誤差 < 1.0 px が目安です。")

    sparse_dir = exp_path / "sparse" / "0"

    def _parse_cameras_txt(p):
        cameras = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    cameras.append({"model": parts[1], "width": int(parts[2]), "height": int(parts[3])})
        return cameras

    def _parse_images_txt(p):
        count = 0
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            count += 1
            i += 2
        return count

    def _parse_points3d_txt(p):
        errors = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    try:
                        errors.append(float(parts[7]))
                    except ValueError:
                        pass
        return len(errors), (sum(errors) / len(errors) if errors else 0.0)

    cam_txt  = sparse_dir / "cameras.txt"
    img_txt  = sparse_dir / "images.txt"
    pts_txt  = sparse_dir / "points3D.txt"
    cam_bin  = sparse_dir / "cameras.bin"
    img_bin  = sparse_dir / "images.bin"
    pts_bin  = sparse_dir / "points3D.bin"

    use_text = cam_txt.exists() and img_txt.exists() and pts_txt.exists()
    use_bin  = cam_bin.exists() and img_bin.exists() and pts_bin.exists()

    if use_text or use_bin:
        try:
            if use_text:
                cameras = _parse_cameras_txt(cam_txt)
                n_reg   = _parse_images_txt(img_txt)
                n_pts, mean_err = _parse_points3d_txt(pts_txt)
            else:
                import sys as _sys
                _sys.path.insert(0, "/opt/gaussian-splatting/scene")
                from colmap_loader import (read_intrinsics_binary,
                                           read_extrinsics_binary,
                                           read_points3D_binary)
                cams_d  = read_intrinsics_binary(str(cam_bin))
                imgs_d  = read_extrinsics_binary(str(img_bin))
                pts_d   = read_points3D_binary(str(pts_bin))
                cameras = [{"model": c.model, "width": c.width, "height": c.height}
                           for c in cams_d.values()]
                n_reg   = len(imgs_d)
                _errs   = [p.error for p in pts_d.values()]
                n_pts   = len(_errs)
                mean_err = sum(_errs) / n_pts if n_pts else 0.0

            n_total  = count_images(exp_path / "input")
            reg_rate = (n_reg / n_total * 100) if n_total > 0 else 0.0

            cq1, cq2, cq3, cq4 = st.columns(4)
            cq1.metric("登録カメラ数",   f"{n_reg} 枚")
            cq2.metric("登録率",
                       f"{reg_rate:.1f}%",
                       delta=f"全 {n_total} 枚中" if n_total else None,
                       delta_color="off")
            cq3.metric("3D点数",         f"{n_pts:,} 点")
            cq4.metric("平均再投影誤差", f"{mean_err:.3f} px")

            if mean_err == 0.0 and n_pts == 0:
                st.info("points3D ファイルが空です（再構成に失敗している可能性があります）。")
            elif mean_err < 1.0 and reg_rate >= 80:
                st.success(f"✅ 再構成品質: 良好（誤差 {mean_err:.3f} px · 登録率 {reg_rate:.1f}%）")
            elif mean_err < 2.0 and reg_rate >= 50:
                st.warning(f"⚠️ 再構成品質: 普通（誤差 {mean_err:.3f} px · 登録率 {reg_rate:.1f}%）")
            else:
                st.error(f"❌ 再構成品質: 要確認（誤差 {mean_err:.3f} px · 登録率 {reg_rate:.1f}%）")

            if cameras:
                cam = cameras[0]
                fmt = "テキスト" if use_text else "バイナリ"
                st.caption(f"カメラモデル: {cam['model']} · 解像度: {cam['width']} × {cam['height']} px · 形式: {fmt}")

        except Exception as _e:
            st.warning(f"COLMAP ファイルの読み込みに失敗しました: {_e}")
    else:
        st.info("sparse/0/ に cameras / images / points3D ファイルが見つかりません。")

# ── point_cloud 情報 ─────────────────────────────────────────────────────────
if has_output:
    st.divider()
    st.subheader("💾 保存済みpoint_cloud")

    pc_dir = exp_path / "output" / "point_cloud"
    if pc_dir.exists():
        iters = sorted(pc_dir.iterdir(), key=lambda p: p.name)
        cols = st.columns(len(iters)) if iters else []
        for col, it_dir in zip(cols, iters):
            ply = it_dir / "point_cloud.ply"
            size_mb = ply.stat().st_size / 1e6 if ply.exists() else 0
            col.metric(it_dir.name, f"{size_mb:.1f} MB" if ply.exists() else "なし")
    else:
        st.info("point_cloud/ フォルダがまだありません。")

# ── ローカルテスト用ZIPダウンロード ───────────────────────────────────────────
st.divider()
st.subheader("📦 ローカルテスト用 ZIP ダウンロード")
st.caption("point_cloud.ply・cameras.json・cfg_args をまとめてダウンロードできます。")

_dl_pc_dir = exp_path / "output" / "point_cloud"
_dl_iters  = [d.name for d in sorted(_dl_pc_dir.iterdir()) if (_dl_pc_dir / d.name / "point_cloud.ply").exists()] if _dl_pc_dir.exists() else []

if not _dl_iters:
    st.info("ダウンロード可能な point_cloud.ply が見つかりません。先に学習を完了してください。")
else:
    _dc1, _dc2, _dc3 = st.columns(3)
    with _dc1:
        _dl_sel_iter = st.selectbox("イテレーション", _dl_iters,
                                    index=len(_dl_iters) - 1, key="res_dl_iter")
    with _dc2:
        _dl_renders = st.checkbox("レンダリング画像を含める", value=False, key="res_dl_renders")
    with _dc3:
        _dl_ply_size = (_dl_pc_dir / _dl_sel_iter / "point_cloud.ply").stat().st_size / 1e6
        st.metric("PLYサイズ", f"{_dl_ply_size:.1f} MB")

    if st.button("📦 ZIP を作成", key="res_dl_btn"):
        import zipfile as _zf
        _zip_path = Path("/workspace/tmp") / f"{exp_path.name}_{_dl_sel_iter}.zip"
        _zip_path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("ZIP 作成中..."):
            with _zf.ZipFile(_zip_path, "w", _zf.ZIP_DEFLATED, compresslevel=1) as _z:
                _ply = _dl_pc_dir / _dl_sel_iter / "point_cloud.ply"
                if _ply.exists():
                    _z.write(_ply, f"output/point_cloud/{_dl_sel_iter}/point_cloud.ply")
                for _fn in ["cameras.json", "cfg_args"]:
                    _fp = exp_path / "output" / _fn
                    if _fp.exists():
                        _z.write(_fp, f"output/{_fn}")
                if _dl_renders:
                    for _split in ["test", "train"]:
                        _sd = exp_path / "output" / _split
                        if _sd.exists():
                            for _id in sorted(_sd.iterdir()):
                                _rd = _id / "renders"
                                for _img in sorted(_rd.glob("*.png"))[:50] if _rd.exists() else []:
                                    _z.write(_img, f"output/{_split}/{_id.name}/renders/{_img.name}")
        st.session_state["res_zip_path"] = str(_zip_path)
        st.session_state["res_zip_name"] = _zip_path.name
        st.rerun()

    if st.session_state.get("res_zip_path") and Path(st.session_state["res_zip_path"]).exists():
        _zsize = Path(st.session_state["res_zip_path"]).stat().st_size / 1e6
        st.download_button(
            f"⬇ ダウンロード（{_zsize:.1f} MB）",
            data=Path(st.session_state["res_zip_path"]).read_bytes(),
            file_name=st.session_state["res_zip_name"],
            mime="application/zip",
            key="res_dl_download",
        )

# ── レンダリング実行 ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🎬 レンダリング実行")

pc_dir = exp_path / "output" / "point_cloud"
iter_dirs = sorted(pc_dir.iterdir(), key=lambda p: p.name) if pc_dir.exists() else []
iter_names = [d.name for d in iter_dirs if (d / "point_cloud.ply").exists()]

if not iter_names:
    st.info("学習済みモデル（point_cloud.ply）が見つかりません。先に3DGS学習を完了してください。")
else:
    # 別実験のレンダリングが走っていたらリセット
    if st.session_state.render_exp and st.session_state.render_exp != str(exp_path):
        st.session_state.render_proc = None
        st.session_state.render_log_path = None
        st.session_state.render_exp = None

    rendering = is_rendering()
    proc = st.session_state.render_proc

    if rendering or (proc is not None and proc.poll() is not None):
        # ── 実行中 / 完了後ビュー ──────────────────────────────────────────
        done     = proc.poll() is not None
        success  = done and proc.returncode == 0
        failed   = done and proc.returncode != 0

        if rendering:
            st.markdown('<span style="color:#00aaff">🔄 レンダリング実行中...</span>',
                        unsafe_allow_html=True)
        elif success:
            st.success("✅ レンダリング完了！")
        else:
            st.error(f"❌ エラーで終了しました（終了コード: {proc.returncode}）")

        # ログ表示
        log_path = Path(st.session_state.render_log_path or "")
        if log_path.exists():
            log_text = log_path.read_text(errors="replace")

            # tqdm 進捗パース: "Rendering progress: N/M"
            prog_m = re.findall(r'Rendering progress.*?(\d+)/(\d+)', log_text)
            if prog_m:
                cur, tot = int(prog_m[-1][0]), int(prog_m[-1][1])
                st.progress(min(cur / tot, 1.0),
                            text=f"フレーム {cur} / {tot} ({cur/tot*100:.0f}%)")

            with st.expander("レンダリングログ", expanded=rendering):
                lines = [l for l in log_text.split("\n") if l.strip()]
                st.code("\n".join(lines[-30:]), language=None)

        col_r1, col_r2 = st.columns([1, 5])
        with col_r1:
            if rendering:
                if st.button("⏹ 中断", type="secondary"):
                    try:
                        os.kill(proc.pid, signal.SIGTERM)
                    except Exception:
                        pass
                    st.session_state.render_proc = None
                    st.session_state.pop("active_task", None)
                    st.rerun()
            else:
                st.session_state.pop("active_task", None)
                if st.button("← 設定に戻る"):
                    st.session_state.render_proc = None
                    st.rerun()

        if rendering:
            time.sleep(2)
            st.rerun()

    else:
        # ── 設定ビュー ────────────────────────────────────────────────────
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sel_iter = st.selectbox(
                "イテレーション",
                ["最新（自動）"] + iter_names,
                help="レンダリングに使うチェックポイントを選択",
            )
        with col_s2:
            skip_train = st.checkbox("学習視点をスキップ",  value=False,
                                     help="train用カメラのレンダリングを省略")
            skip_test  = st.checkbox("テスト視点をスキップ", value=False,
                                     help="test用カメラのレンダリングを省略")
        with col_s3:
            white_bg = st.checkbox("背景を白にする", value=False)

        # コマンドプレビュー
        iter_arg = "-1" if sel_iter == "最新（自動）" else sel_iter.replace("iteration_", "")
        model_path = str(exp_path / "output")
        cmd_preview = (
            f"python scripts/run_render.py \\\n"
            f"  -m {model_path} \\\n"
            f"  -s {exp_path} \\\n"
            f"  --iteration {iter_arg}"
        )
        if skip_train: cmd_preview += " \\\n  --skip_train"
        if skip_test:  cmd_preview += " \\\n  --skip_test"
        if white_bg:   cmd_preview += " \\\n  --white_background"
        with st.expander("実行コマンド（プレビュー）", expanded=False):
            st.code(cmd_preview, language="bash")

        # キューに追加
        import sys as _sys; _sys.path.insert(0, "/workspace")
        from queue_helper import add_to_queue as _add_q, pending_size as _psize
        if st.button(
            f"📋 バッチキューに追加（待ち: {_psize()} 件）",
            disabled=(skip_train and skip_test),
            use_container_width=True,
        ):
            _add_q(
                job_type="render",
                label=f"レンダリング {sel_iter}",
                exp_name=exp_path.name,
                exp_dir=str(exp_path),
                config={
                    "model_path":       model_path,
                    "iteration":        int(iter_arg) if iter_arg != "-1" else -1,
                    "skip_train":       skip_train,
                    "skip_test":        skip_test,
                    "white_background": white_bg,
                },
            )
            st.success("バッチキューに追加しました。")

        st.warning("⚠️ レンダリングはGPUを使用します。学習中は実行しないでください。")
        if st.button("🎬 レンダリング開始", type="primary",
                     disabled=(skip_train and skip_test)):
            log_path = str(exp_path / "output" / "render_log.txt")
            cmd = [
                sys.executable, "/workspace/scripts/run_render.py",
                "-m", model_path,
                "-s", str(exp_path),
                "--iteration", iter_arg,
            ]
            if skip_train: cmd.append("--skip_train")
            if skip_test:  cmd.append("--skip_test")
            if white_bg:   cmd.append("--white_background")

            log_file = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            st.session_state.render_proc     = proc
            st.session_state.render_log_path = log_path
            st.session_state.render_exp      = str(exp_path)
            st.session_state.active_task = {
                "step":       "rendering",
                "label":      "レンダリング",
                "scene":      exp_path.name,
                "log_path":   log_path,
                "pid":        proc.pid,
                "start_time": time.time(),
            }
            st.rerun()

# ── 学習ログとPSNR ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 学習ログ・メトリクス")

log_files = (
    list((exp_path / "output").rglob("*.txt")) +
    list((exp_path / "logs").rglob("*.txt")) +
    list((exp_path / "output").rglob("*.log"))
) if (exp_path / "output").exists() or (exp_path / "logs").exists() else []

# train_log.txt を優先
train_logs = [f for f in log_files if "train_log" in f.name]
log_files = train_logs + [f for f in log_files if f not in train_logs]

if log_files:
    selected_log = st.selectbox("ログファイルを選択", log_files,
                                format_func=lambda x: str(x.relative_to(exp_path)))
    log_content = selected_log.read_text(errors="replace")

    # PSNR を抽出して表示
    psnr_records = []
    for m in re.finditer(
        r"\[ITER (\d+)\] Evaluating (\w+): L1 [^\s]+ PSNR [^(\n]*\(([\d.]+)\)",
        log_content,
    ):
        psnr_records.append({
            "iteration": int(m.group(1)),
            "split": m.group(2),
            "PSNR": float(m.group(3)),
        })

    if psnr_records:
        st.subheader("📈 PSNR推移")
        df = pd.DataFrame(psnr_records)
        pivot = df.pivot(index="iteration", columns="split", values="PSNR")
        st.line_chart(pivot)
        st.dataframe(df, use_container_width=True)

    with st.expander("ログ全文"):
        st.text_area("", log_content, height=300, label_visibility="collapsed")
else:
    st.info("ログファイルが見つかりません。")

# ── レンダリング画像 ──────────────────────────────────────────────────────────
st.divider()
st.subheader("🖼️ レンダリング画像")

# renders/ または output/**/renders/ を探す
render_dirs = [exp_path / "renders"]
if has_output:
    render_dirs += list((exp_path / "output").rglob("renders"))

images = []
for rd in render_dirs:
    if rd.is_dir():
        images += sorted(rd.rglob("*.png")) + sorted(rd.rglob("*.jpg"))

# gt/ が存在する split/iteration を収集（比較タブ用）
_compare_splits = []
if has_output:
    for _split in ["test", "train"]:
        _sdir = exp_path / "output" / _split
        if _sdir.exists():
            for _idir in sorted(_sdir.iterdir()):
                if (_idir / "renders").exists() and (_idir / "gt").exists():
                    _compare_splits.append((_split, _idir.name))

if not images and not _compare_splits:
    st.info("レンダリング画像が見つかりません。render.py を実行すると生成されます。")
else:
    _tab_labels = ["▶ 動画再生", "🖼️ 画像一覧"]
    if _compare_splits:
        _tab_labels.append("🔍 元画像 vs レンダリング")
    _tabs = st.tabs(_tab_labels)

    # ── 動画再生タブ ──────────────────────────────────────────────────────────
    with _tabs[0]:
        if not images:
            st.info("レンダリング画像が見つかりません。")
        else:
            video_path = Path("/workspace/tmp") / f"render_{exp_path.name}.mp4"
            col_v1, col_v2, col_v3 = st.columns(3)
            fps_v      = col_v1.number_input("再生FPS", min_value=1, max_value=60, value=10)
            max_frames = col_v2.number_input("最大フレーム数", min_value=10, max_value=1000,
                                             value=min(300, len(images)))
            if col_v3.button("🎬 動画を生成", use_container_width=True):
                import tempfile, shutil
                tmp_dir  = Path(tempfile.mkdtemp())
                use_imgs = images[:max_frames]
                for i, img in enumerate(use_imgs):
                    shutil.copy(img, tmp_dir / f"frame_{i:06d}{img.suffix}")
                result_v = subprocess.run([
                    "ffmpeg", "-y", "-framerate", str(fps_v),
                    "-i", str(tmp_dir / f"frame_%06d{use_imgs[0].suffix}"),
                    # vflip: render.py の上下反転を補正
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path),
                ], capture_output=True)
                shutil.rmtree(tmp_dir)
                if result_v.returncode == 0:
                    st.success(f"{len(use_imgs)} フレームから動画を生成しました。")
                else:
                    st.error("動画生成に失敗しました。")

            if video_path.exists():
                # 動画は幅60%に抑える
                _, vid_col, _ = st.columns([1, 3, 1])
                with vid_col:
                    st.video(str(video_path))
                st.caption(f"保存先: `{video_path}`")
            else:
                st.info("「動画を生成」ボタンを押すとここに再生ビューアが表示されます。")

    # ── 画像一覧タブ ──────────────────────────────────────────────────────────
    with _tabs[1]:
        if not images:
            st.info("レンダリング画像が見つかりません。")
        else:
            cols_per_row = st.slider("1行あたりの表示枚数", 3, 8, 6, key="grid_cols")
            for i in range(0, min(len(images), 48), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(images):
                        col.image(str(images[i + j]),
                                  caption=images[i + j].name,
                                  use_container_width=True)

    # ── 比較タブ（gt/ がある場合のみ） ───────────────────────────────────────
    if _compare_splits:
        with _tabs[2]:
            _split_labels = {
                "test":  "🧪 test（未学習視点）",
                "train": "🎓 train（学習視点）",
            }
            _cmp_options = [f"{s}/{i}" for s, i in _compare_splits]
            _cmp_labels  = [
                f"{_split_labels.get(s, s)} / {i.replace('ours_', 'iter ')}"
                for s, i in _compare_splits
            ]
            _cmp_ctrl1, _cmp_ctrl2, _cmp_ctrl3 = st.columns(3)
            with _cmp_ctrl1:
                _cmp_sel = st.selectbox(
                    "split / iteration",
                    _cmp_options,
                    format_func=lambda x: _cmp_labels[_cmp_options.index(x)],
                )
            _sel_split, _sel_iter = _cmp_sel.split("/", 1)
            _renders_dir = exp_path / "output" / _sel_split / _sel_iter / "renders"
            _gt_dir      = exp_path / "output" / _sel_split / _sel_iter / "gt"
            _render_imgs = sorted(
                list(_renders_dir.glob("*.png")) + list(_renders_dir.glob("*.jpg"))
            )

            if not _render_imgs:
                st.info("レンダリング画像が見つかりません。")
            else:
                with _cmp_ctrl2:
                    _pairs_per_row = st.slider("1行あたりのペア数", 1, 3, 2, key="cmp_pairs")
                with _cmp_ctrl3:
                    _n_pages = max(1, (len(_render_imgs) + _pairs_per_row - 1) // _pairs_per_row)
                    _page    = st.number_input("ページ", min_value=1, max_value=_n_pages,
                                               value=1, key="cmp_page")

                _start = (_page - 1) * _pairs_per_row
                _end   = min(_start + _pairs_per_row, len(_render_imgs))
                st.caption(
                    f"全 {len(_render_imgs)} 枚中 {_start + 1}〜{_end} 枚　"
                    f"| 左: 元画像（gt）　右: レンダリング"
                )

                # 1行に _pairs_per_row ペア（= _pairs_per_row×2 列）
                _row_cols = st.columns(_pairs_per_row * 2)
                for _idx, _img_path in enumerate(_render_imgs[_start:_end]):
                    _gt_path = _gt_dir / _img_path.name
                    _col_gt  = _row_cols[_idx * 2]
                    _col_rd  = _row_cols[_idx * 2 + 1]
                    if _gt_path.exists():
                        _col_gt.image(str(_gt_path),
                                      caption=f"元画像 {_img_path.name}",
                                      use_container_width=True)
                    else:
                        _col_gt.warning("gt なし")
                    _col_rd.image(str(_img_path),
                                  caption=f"レンダリング {_img_path.name}",
                                  use_container_width=True)

# ── メモ（note.md） ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📝 実験メモ（note.md）")

note_path = exp_path / "note.md"
current_note = note_path.read_text(encoding="utf-8") if note_path.exists() else ""

new_note = st.text_area("メモを自由に記入できます（気づき・失敗原因・パラメータの感想など）",
                         value=current_note, height=150, label_visibility="visible")

if st.button("💾 メモを保存"):
    note_path.write_text(new_note, encoding="utf-8")
    st.success("メモを保存しました。")

# ── config.yaml / cfg_args ────────────────────────────────────────────────────
st.divider()
st.subheader("⚙️ 実験設定")

def _parse_namespace(text: str) -> dict:
    text = text.strip()
    if text.startswith("Namespace(") and text.endswith(")"):
        text = text[len("Namespace("):-1]
    result = {}
    pattern = re.compile(
        r"(\w+)="
        r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|"
        r"True|False|None|"
        r"-?\d+\.\d+e[+-]?\d+|-?\d+\.\d+|-?\d+|"
        r"\[[^\]]*\])"
    )
    for m in pattern.finditer(text):
        key, val_str = m.group(1), m.group(2)
        try:
            result[key] = ast.literal_eval(val_str)
        except Exception:
            result[key] = val_str
    return result

_CFG_LABELS = {
    "source_path":      ("入力パス",                "パス"),
    "model_path":       ("出力パス",                "パス"),
    "images":           ("画像フォルダ名",           "パス"),
    "depths":           ("深度フォルダ名",           "パス"),
    "sh_degree":        ("SH次数",                  "学習設定"),
    "resolution":       ("解像度縮小倍率",           "学習設定"),
    "white_background": ("白背景",                  "学習設定"),
    "eval":             ("--eval（train/test分割）",        "学習設定"),
    "train_test_exp":   ("train_test_exp（独立実験モード）", "学習設定"),
    "data_device":      ("データデバイス",           "その他"),
}

config_path = exp_path / "config.yaml"
cfg_args_list = list((exp_path / "output").rglob("cfg_args")) if has_output else []

if config_path.exists():
    st.markdown("**`config.yaml`**")
    try:
        cfg_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(cfg_data, dict):
            st.dataframe(
                pd.DataFrame([{"設定項目": k, "値": str(v)} for k, v in cfg_data.items()]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.code(config_path.read_text(encoding="utf-8"), language="yaml")
    except Exception:
        st.code(config_path.read_text(encoding="utf-8", errors="replace"), language="yaml")
    with st.expander("生YAML", expanded=False):
        st.code(config_path.read_text(encoding="utf-8", errors="replace"), language="yaml")

if cfg_args_list:
    st.markdown("**`cfg_args`**　学習引数（gaussian-splatting）")
    for cfg_file in cfg_args_list:
        raw = cfg_file.read_text(errors="replace").strip()
        parsed = _parse_namespace(raw)
        if parsed:
            categories: dict = {}
            for key, val in parsed.items():
                label, cat = _CFG_LABELS.get(key, (key, "その他"))
                categories.setdefault(cat, []).append({"設定項目": label, "キー": key, "値": str(val)})
            for cat_name in ["パス", "学習設定", "その他"]:
                if cat_name not in categories:
                    continue
                st.caption(f"**{cat_name}**")
                st.dataframe(
                    pd.DataFrame(categories[cat_name]),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "設定項目": st.column_config.TextColumn(width="medium"),
                        "キー":     st.column_config.TextColumn(width="small"),
                        "値":       st.column_config.TextColumn(width="large"),
                    },
                )
        with st.expander("生テキスト", expanded=False):
            st.code(raw, language="text")

if not config_path.exists() and not cfg_args_list:
    st.info("config ファイルが見つかりません。")

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### PSNR（Peak Signal-to-Noise Ratio）

レンダリング画像と正解画像の類似度を示す指標です。**高いほど高品質**。

| PSNR | 品質の目安 |
|---|---|
| < 20 dB | 低品質（学習不足・問題あり） |
| 20〜25 dB | 普通 |
| 25〜30 dB | 良好（通常の3DGS） |
| > 30 dB | 高品質 |

---

### train split vs test split

- **train split**：学習に使った画像から評価したPSNR（過学習気味になりがち）
- **test split**：学習に使っていない画像から評価したPSNR（汎化性能の指標）
- `--eval` オプションを有効にして学習した場合のみ、testデータが存在します。
- `--eval` なしで学習した場合はtrain PSNRのみ表示されます。

---

### L1 Loss

学習中の平均絶対誤差です。**低いほど良好**。学習の進み具合の確認に使います。

---

### レンダリング
- 「レンダリング実行」ボタンで学習済みモデルからテスト視点の画像を生成できます。
- `renders/` フォルダに保存されます。
- 生成にはGPUを使用します（数分程度）。
- PINHOLE以外のカメラモデルで学習した実験（undistortion済み）は、`dense/` フォルダを自動検出してレンダリングに使用します。手動でのパス変更は不要です。

---

### 実験設定（cfg_args）の見方
- **`--eval`（train/test分割）**：`True` なら `--eval` が有効で、test分割でPSNRを評価しています。
- **`train_test_exp`（独立実験モード）**：通常は `False` で問題ありません。別フラグです。
- **解像度縮小倍率**：`4` なら 1/4 サイズで学習しています。

---

### 点群プレビュー
- `point_cloud/` 以下の `.ply` ファイルを読み込んで簡易可視化します。
- ブラウザ上での3D確認が可能です（重いシーンでは表示に時間がかかります）。
""")