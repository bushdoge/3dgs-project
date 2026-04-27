# 実験フォルダの一覧管理ページ
# 各実験のステータス・ディスク使用量・メモを一覧表示し、削除やメモ編集・ログ閲覧・設定確認をGUIで行う

import re
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st


def _filter_tqdm(text: str) -> str:
    """tqdmのプログレスバー行（大量のログ）を先頭1行・末尾1行に圧縮して返す"""
    lines = text.replace("\r", "\n").splitlines()
    result = []
    buf = []
    for line in lines:
        if re.search(r"\d+%\|", line):
            buf.append(line)
        else:
            if buf:
                result.append(buf[0])
                if len(buf) > 2:
                    omitted = len(buf) - 2
                    result.append("")
                    result.append(f"... ({omitted} 行省略) ...")
                    result.append("")
                if len(buf) > 1:
                    result.append(buf[-1])
                buf = []
            result.append(line)
    if buf:
        result.append(buf[0])
        if len(buf) > 2:
            omitted = len(buf) - 2
            result.append("")
            result.append(f"... ({omitted} 行省略) ...")
            result.append("")
        if len(buf) > 1:
            result.append(buf[-1])
    return "\n".join(result)


st.title("🗂️ 実験管理")
st.caption("実験フォルダの一覧・ディスク使用量・メモ管理・削除")

st.divider()

EXPERIMENTS_DIR = Path("/workspace/experiments")


@st.cache_data(ttl=120)
def get_dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


@st.cache_data(ttl=60)
def get_exp_status(exp: Path) -> dict:
    has_frames = (exp / "input").exists() and bool(
        list((exp / "input").glob("*.jpg")) + list((exp / "input").glob("*.png"))
    )
    has_colmap = (exp / "sparse" / "0").exists()
    has_output = bool(list((exp / "output").rglob("*.ply"))) if (exp / "output").exists() else False
    note_path = exp / "note.md"
    note = note_path.read_text(encoding="utf-8").strip() if note_path.exists() else ""
    note_preview = note.splitlines()[0][:50] if note else ""
    size_mb = get_dir_size_mb(exp)
    return {
        "フレーム抽出": "✅" if has_frames else "❌",
        "COLMAP": "✅" if has_colmap else "❌",
        "3DGS学習": "✅" if has_output else "❌",
        "ディスク使用量": f"{size_mb:.1f} MB",
        "メモ": note_preview,
        "_note_full": note,
        "_size_mb": size_mb,
    }


# ── 実験一覧の読み込み ────────────────────────────────────────────────────────
if not EXPERIMENTS_DIR.exists() or not any(EXPERIMENTS_DIR.iterdir()):
    st.info("experiments/ フォルダに実験がまだありません。")
    st.stop()

exps = sorted([p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()], reverse=True)

rows = []
_status_cache = {}
with st.spinner("実験データを読み込み中..."):
    for exp in exps:
        s = get_exp_status(exp)
        _status_cache[exp.name] = s
        rows.append({
            "実験名": exp.name,
            "フレーム抽出": s["フレーム抽出"],
            "COLMAP": s["COLMAP"],
            "3DGS学習": s["3DGS学習"],
            "ディスク使用量": s["ディスク使用量"],
            "メモ": s["メモ"],
        })

df = pd.DataFrame(rows)

# ── ディスク使用量サマリー ────────────────────────────────────────────────────
total_mb = sum(s["_size_mb"] for s in _status_cache.values())  # get_exp_status の結果を再利用
disk = shutil.disk_usage("/workspace")
free_gb = disk.free / 1e9
used_gb = disk.used / 1e9
total_gb = disk.total / 1e9

c1, c2, c3, c4 = st.columns(4)
c1.metric("実験数", f"{len(exps)} 件")
c2.metric("実験合計サイズ", f"{total_mb/1024:.2f} GB" if total_mb > 1024 else f"{total_mb:.0f} MB")
c3.metric("ディスク空き容量", f"{free_gb:.1f} GB")
c4.metric("ディスク使用率", f"{100*disk.used/disk.total:.1f} %")

st.divider()

# ── 一覧テーブル ──────────────────────────────────────────────────────────────
st.subheader("📋 実験一覧")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ── 詳細操作パネル ────────────────────────────────────────────────────────────
st.subheader("🔧 実験の詳細操作")
st.caption("📝 メモ編集 / 📋 ログ閲覧（フレーム抽出・COLMAP・学習・レンダリング）"
           " / ⚙️ 設定確認（config.yaml・学習引数） / 🗑️ フォルダ削除")

exp_names = [e.name for e in exps]
selected_name = st.selectbox("操作する実験を選択", exp_names)

if not selected_name:
    st.stop()

selected_exp = EXPERIMENTS_DIR / selected_name

tab_note, tab_logs, tab_config, tab_dl, tab_rename, tab_delete = st.tabs(
    ["📝 メモ編集", "📋 ログ閲覧", "⚙️ 設定確認", "📦 ダウンロード", "✏️ 名前変更", "🗑️ フォルダ削除"]
)

# ── メモ編集タブ ─────────────────────────────────────────────────────────────
with tab_note:
    note_path = selected_exp / "note.md"
    current = note_path.read_text(encoding="utf-8") if note_path.exists() else ""
    new_note = st.text_area("メモ", value=current, height=200, label_visibility="visible")
    if st.button("💾 保存", key="save_note"):
        note_path.write_text(new_note, encoding="utf-8")
        st.success("メモを保存しました。")
        st.rerun()

# ── ログ閲覧タブ ─────────────────────────────────────────────────────────────
with tab_logs:
    LOG_DEFS = [
        ("📥 フレーム抽出", selected_exp / "extract_log.txt"),
        ("📐 COLMAP",       selected_exp / "colmap_log.txt"),
        ("🧠 3DGS学習",     selected_exp / "output" / "train_log.txt"),
        ("🎬 レンダリング", selected_exp / "output" / "render_log.txt"),
    ]
    available = [(label, path) for label, path in LOG_DEFS if path.exists()]

    if not available:
        st.info("ログファイルが見つかりません。パイプラインを実行するとここに表示されます。")
    else:
        log_tabs = st.tabs([label for label, _ in available])
        for ltab, (label, log_path) in zip(log_tabs, available):
            with ltab:
                log_text = log_path.read_text(errors="replace")
                st.caption(f"`{log_path.relative_to(selected_exp)}`　｜　{len(log_text.splitlines())} 行")

                # 3DGS学習ログの場合は PSNR チャートも表示
                if "train_log" in log_path.name and log_text:
                    psnr_records = []
                    for m in re.finditer(
                        r"\[ITER (\d+)\] Evaluating (\w+): L1 [^\s]+ PSNR [^(\n]*\(([\d.]+)\)",
                        log_text,
                    ):
                        psnr_records.append({
                            "iteration": int(m.group(1)),
                            "split":     m.group(2),
                            "PSNR":      float(m.group(3)),
                        })
                    if psnr_records:
                        import pandas as _pd
                        st.markdown("**📈 PSNR 推移**")
                        _df = _pd.DataFrame(psnr_records)
                        _pivot = _df.pivot(index="iteration", columns="split", values="PSNR")
                        st.line_chart(_pivot)

                # COLMAP ログの場合は再構成サマリー行を強調
                if "colmap_log" in log_path.name and log_text:
                    summary_lines = [
                        l for l in log_text.splitlines()
                        if any(kw in l for kw in ("Registered", "registered", "points", "error", "残差"))
                    ]
                    if summary_lines:
                        with st.expander("🔍 再構成サマリー行"):
                            st.code("\n".join(summary_lines[-20:]), language=None)

                with st.expander("ログ全文", expanded=True):
                    filtered = _filter_tqdm(log_text)
                    lines = [l for l in filtered.splitlines() if l.strip()]
                    st.code("\n".join(lines[-100:]) if len(lines) > 100 else "\n".join(lines),
                            language=None)
                    if len(lines) > 100:
                        st.caption(f"最新 100 行を表示（全 {len(lines)} 行）")

# ── 設定確認タブ ─────────────────────────────────────────────────────────────
with tab_config:
    import ast
    import json as _json_cfg
    import yaml

    # ── パイプライン設定（pipeline_config.json） ──────────────────────────────
    _pipeline_cfg_path = selected_exp / "pipeline_config.json"
    if _pipeline_cfg_path.exists():
        st.markdown("#### `pipeline_config.json`　パイプライン設定")
        try:
            _pc = _json_cfg.loads(_pipeline_cfg_path.read_text(encoding="utf-8"))

            _PIPELINE_LABELS = {
                "saved_at":        ("保存日時",          "メタ情報"),
                "video_path":      ("入力動画",          "フレーム抽出"),
                "fps":             ("抽出FPS",           "フレーム抽出"),
                "is_360":          ("360度変換",         "フレーム抽出"),
                "fov":             ("水平視野角（FOV）",  "フレーム抽出"),
                "out_w":           ("出力幅",            "フレーム抽出"),
                "out_h":           ("出力高さ",          "フレーム抽出"),
                "angles":          ("変換方向",          "フレーム抽出"),
                "use_hloc":        ("HLoc使用",          "姿勢推定"),
                "feature_type":    ("特徴点抽出器",       "姿勢推定"),
                "matcher_type":    ("マッチャー",         "姿勢推定"),
                "pair_method":     ("ペアリスト方式",     "姿勢推定"),
                "retrieval_model": ("Retrievalモデル",   "姿勢推定"),
                "num_matched":     ("top-K",            "姿勢推定"),
                "camera_model":    ("カメラモデル（COLMAP）", "姿勢推定"),
                "use_gpu":         ("GPU使用",           "姿勢推定"),
                "iterations":      ("学習ステップ数",     "3DGS学習"),
                "save_iterations": ("保存タイミング",     "3DGS学習"),
                "test_iterations": ("評価タイミング",     "3DGS学習"),
                "eval":            ("train/test分割",    "3DGS学習"),
                "resolution":      ("解像度縮小倍率",     "3DGS学習"),
            }
            _CAT_ORDER = ["メタ情報", "フレーム抽出", "姿勢推定", "3DGS学習"]

            # カテゴリ別に行を振り分け
            _pc_cats: dict[str, list] = {}
            for key, val in _pc.items():
                if key.startswith("_"):  # _recovered_from 等の内部メタキーはスキップ
                    continue
                label, cat = _PIPELINE_LABELS.get(key, (key, "その他"))
                # 360度変換なしのときは無関係な360設定を非表示
                if not _pc.get("is_360") and key in ("fov", "out_w", "out_h", "angles"):
                    continue
                # HLoc無しのときはHLoc専用設定を非表示
                if not _pc.get("use_hloc") and key in ("feature_type", "matcher_type",
                                                         "pair_method", "retrieval_model", "num_matched"):
                    continue
                # COLMAP使用時はカメラモデル・GPU を表示
                if _pc.get("use_hloc") and key in ("camera_model", "use_gpu"):
                    continue
                def _fmt_iter(v):
                    return f"{v // 1000}k" if isinstance(v, int) and v % 1000 == 0 else str(v)
                if key in ("save_iterations", "test_iterations") and isinstance(val, list):
                    val_str = ", ".join(_fmt_iter(v) for v in val)
                elif isinstance(val, list):
                    val_str = ", ".join(str(v) for v in val)
                else:
                    val_str = str(val)
                _pc_cats.setdefault(cat, []).append({"設定項目": label, "値": val_str})

            for _cat in _CAT_ORDER + ["その他"]:
                if _cat not in _pc_cats:
                    continue
                st.caption(f"**{_cat}**")
                st.dataframe(
                    pd.DataFrame(_pc_cats[_cat]),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "設定項目": st.column_config.TextColumn(width="medium"),
                        "値":       st.column_config.TextColumn(width="large"),
                    },
                )

            # 復元元情報があればキャプション表示
            _recovered_from = _pc.get("_recovered_from")
            if _recovered_from:
                st.caption(f"⚠️ このファイルはログから復元されました（取得元: {', '.join(_recovered_from)}）")

        except Exception as _e:
            st.warning(f"pipeline_config.json の解析に失敗しました: {_e}")

        with st.expander("生JSON", expanded=False):
            st.code(_pipeline_cfg_path.read_text(encoding="utf-8", errors="replace"), language="json")

    else:
        st.info("pipeline_config.json が見つかりません。次回のパイプライン実行から自動で保存されます。")
        if st.button("🔍 ログから設定を復元", key="recover_pipeline_cfg"):
            import sys as _sys_r
            _sys_r.path.insert(0, "/workspace/scripts")
            from recover_pipeline_config import recover as _recover
            _cfg_recovered, _sources = _recover(selected_exp)
            if len(_cfg_recovered) > 1:  # saved_at 以外に何かあれば保存
                _pipeline_cfg_path.write_text(
                    __import__("json").dumps(_cfg_recovered, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                st.success(f"復元しました（取得元: {', '.join(_sources) if _sources else 'なし'}）")
                st.rerun()
            else:
                st.warning("ログから読み取れる設定情報がありませんでした。")

    st.divider()

    config_path   = selected_exp / "config.yaml"
    cfg_args_list = list((selected_exp / "output").rglob("cfg_args")) \
                    if (selected_exp / "output").exists() else []

    # cfg_args の Namespace(...) 文字列をパースして dict を返す
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

    # cfg_args のキーを日本語ラベル・カテゴリに変換するマッピング
    _CFG_LABELS = {
        "source_path":      ("入力パス",           "パス"),
        "model_path":       ("出力パス",           "パス"),
        "images":           ("画像フォルダ名",      "パス"),
        "depths":           ("深度フォルダ名",      "パス"),
        "sh_degree":        ("SH次数",             "学習設定"),
        "resolution":       ("解像度縮小倍率",      "学習設定"),
        "white_background": ("白背景",             "学習設定"),
        "eval":             ("--eval（train/test分割）",        "学習設定"),
        "train_test_exp":   ("train_test_exp（独立実験モード）", "学習設定"),
        "data_device":      ("データデバイス",      "その他"),
    }

    # ── config.yaml ──
    if config_path.exists():
        st.markdown("#### `config.yaml`　実験設定")
        try:
            cfg_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(cfg_data, dict):
                cfg_rows = [{"設定項目": k, "値": str(v)} for k, v in cfg_data.items()]
                st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True, hide_index=True)
            else:
                st.code(config_path.read_text(encoding="utf-8"), language="yaml")
        except Exception:
            st.code(config_path.read_text(encoding="utf-8", errors="replace"), language="yaml")
        with st.expander("生YAML", expanded=False):
            st.code(config_path.read_text(encoding="utf-8", errors="replace"), language="yaml")
    elif not cfg_args_list:
        st.info("config.yaml が見つかりません。")

    # ── cfg_args ──
    if cfg_args_list:
        st.markdown("#### `cfg_args`　学習引数（gaussian-splatting）")
        for cfg_file in cfg_args_list:
            raw = cfg_file.read_text(errors="replace").strip()
            parsed = _parse_namespace(raw)
            if parsed:
                # カテゴリ別にグループ化
                categories = {}
                for key, val in parsed.items():
                    label, cat = _CFG_LABELS.get(key, (key, "その他"))
                    categories.setdefault(cat, []).append({"設定項目": label, "キー": key, "値": str(val)})

                for cat_name in ["パス", "学習設定", "その他"]:
                    if cat_name not in categories:
                        continue
                    st.caption(f"**{cat_name}**")
                    st.dataframe(
                        pd.DataFrame(categories[cat_name]),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "設定項目": st.column_config.TextColumn(width="medium"),
                            "キー":     st.column_config.TextColumn(width="small"),
                            "値":       st.column_config.TextColumn(width="large"),
                        },
                    )
            with st.expander("生テキスト", expanded=False):
                st.code(raw, language="text")

    if not config_path.exists() and not cfg_args_list:
        st.info("設定ファイルが見つかりません。パイプラインを実行すると config.yaml が生成されます。")

# ── ダウンロードタブ ──────────────────────────────────────────────────────────
with tab_dl:
    import zipfile as _zf

    st.caption("point_cloud.ply・cameras.json・cfg_args をまとめてダウンロードできます。")

    _em_pc_dir = selected_exp / "output" / "point_cloud"
    _em_iters  = (
        [d.name for d in sorted(_em_pc_dir.iterdir())
         if (_em_pc_dir / d.name / "point_cloud.ply").exists()]
        if _em_pc_dir.exists() else []
    )

    if not _em_iters:
        st.info("ダウンロード可能な point_cloud.ply が見つかりません。先に学習を完了してください。")
    else:
        _ec1, _ec2, _ec3 = st.columns(3)
        with _ec1:
            _em_sel_iter = st.selectbox("イテレーション", _em_iters,
                                        index=len(_em_iters) - 1, key="em_dl_iter")
        with _ec2:
            _em_renders = st.checkbox("レンダリング画像を含める", value=False, key="em_dl_renders")
        with _ec3:
            _em_ply_size = (_em_pc_dir / _em_sel_iter / "point_cloud.ply").stat().st_size / 1e6
            st.metric("PLYサイズ", f"{_em_ply_size:.1f} MB")

        if st.button("📦 ZIP を作成", key="em_dl_btn"):
            _em_zip_path = Path("/workspace/tmp") / f"{selected_name}_{_em_sel_iter}.zip"
            _em_zip_path.parent.mkdir(parents=True, exist_ok=True)
            with st.spinner("ZIP 作成中..."):
                with _zf.ZipFile(_em_zip_path, "w", _zf.ZIP_DEFLATED, compresslevel=1) as _z:
                    _em_ply = _em_pc_dir / _em_sel_iter / "point_cloud.ply"
                    if _em_ply.exists():
                        _z.write(_em_ply, f"output/point_cloud/{_em_sel_iter}/point_cloud.ply")
                    for _fn in ["cameras.json", "cfg_args"]:
                        _fp = selected_exp / "output" / _fn
                        if _fp.exists():
                            _z.write(_fp, f"output/{_fn}")
                    if _em_renders:
                        for _split in ["test", "train"]:
                            _sd = selected_exp / "output" / _split
                            if _sd.exists():
                                for _id in sorted(_sd.iterdir()):
                                    _rd = _id / "renders"
                                    for _img in (sorted(_rd.glob("*.png"))[:50] if _rd.exists() else []):
                                        _z.write(_img, f"output/{_split}/{_id.name}/renders/{_img.name}")
            st.session_state["em_zip_path"] = str(_em_zip_path)
            st.session_state["em_zip_name"] = _em_zip_path.name
            st.rerun()

        if st.session_state.get("em_zip_path") and Path(st.session_state["em_zip_path"]).exists():
            _em_zsize = Path(st.session_state["em_zip_path"]).stat().st_size / 1e6
            st.download_button(
                f"⬇ ダウンロード（{_em_zsize:.1f} MB）",
                data=Path(st.session_state["em_zip_path"]).read_bytes(),
                file_name=st.session_state["em_zip_name"],
                mime="application/zip",
                key="em_dl_download",
            )

# ── 名前変更タブ ──────────────────────────────────────────────────────────────
with tab_rename:
    st.caption("実験フォルダ名を変更します。メモ・ログ・学習結果はそのまま引き継がれます。")
    new_name = st.text_input(
        "新しいフォルダ名",
        value=selected_name,
        key="rename_input",
        help="日時プレフィックス（例: 20260421_）は任意で変更できます。",
    )
    new_name = new_name.strip()

    # バリデーション
    rename_error = None
    if not new_name:
        rename_error = "フォルダ名を入力してください。"
    elif new_name == selected_name:
        rename_error = "現在と同じ名前です。"
    elif "/" in new_name or "\\" in new_name or ".." in new_name:
        rename_error = "使用できない文字が含まれています（/ \\ ..）"
    elif (EXPERIMENTS_DIR / new_name).exists():
        rename_error = f"「{new_name}」はすでに存在します。"

    if rename_error:
        st.warning(rename_error)
    else:
        st.info(f"`{selected_name}` → `{new_name}`")

    if st.button("✏️ 名前を変更する", type="primary",
                 disabled=(rename_error is not None)):
        import subprocess as _sp
        result = _sp.run(
            ["mv", str(selected_exp), str(EXPERIMENTS_DIR / new_name)],
            capture_output=True,
        )
        if result.returncode == 0:
            st.success(f"✅ `{selected_name}` → `{new_name}` に変更しました。")
            st.rerun()
        else:
            st.error(f"変更に失敗しました: {result.stderr.decode()}")

# ── 削除タブ ─────────────────────────────────────────────────────────────────
with tab_delete:
    size_mb = get_dir_size_mb(selected_exp)
    st.warning(f"⚠️ `{selected_name}` を削除します（{size_mb:.1f} MB）。この操作は元に戻せません。")
    confirm_text = st.text_input("確認のため実験フォルダ名を入力してください",
                                  placeholder=selected_name)
    if st.button("🗑️ 削除する", type="primary",
                 disabled=(confirm_text != selected_name)):
        import subprocess as _sp
        result = _sp.run(["rm", "-rf", str(selected_exp)], capture_output=True)
        if result.returncode == 0:
            st.success(f"`{selected_name}` を削除しました。")
            st.rerun()
        else:
            st.error(f"削除に失敗しました: {result.stderr.decode()}")

# ── 使い方（詳細） ────────────────────────────────────────────────────────────
with st.expander("📖 使い方（詳細）", expanded=False):
    st.markdown("""
### 実験一覧の見方

| 列 | 説明 |
|---|---|
| フォルダ名 | `YYYYMMDD_HHMMSS_<シーン名>` 形式 |
| フレーム | 抽出済み画像の有無 |
| COLMAP | カメラ姿勢推定の完了有無（sparse/0/ の存在） |
| 3DGS | 学習済みモデルの有無（output/*.ply の存在） |
| サイズ | フォルダの総ディスク使用量 |
| メモ | note.md の先頭行（編集可能） |

---

### ログタブ
- **学習ログ**：Loss・PSNRの推移グラフと生ログを表示します
- **COLMAPログ**：再構成サマリー（登録画像数・点群数・再投影誤差）を強調表示します
- tqdmプログレスバー行は自動的に圧縮されます（先頭1行＋末尾1行のみ表示）

---

### 削除
- 実験フォルダ名を手動入力することで誤削除を防止しています
- 削除した実験は**元に戻せません**。必要なデータは事前にバックアップしてください
- `data/` 配下の元動画は削除されません（実験フォルダのみ削除）
""")

# ── 固定フッター ──────────────────────────────────────────────────────────────
try:
    from pipeline_widget import render_sticky_footer
    render_sticky_footer()
except Exception:
    pass
