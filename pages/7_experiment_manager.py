# 実験フォルダの一覧管理ページ
# 各実験のステータス・ディスク使用量・メモを一覧表示し、削除やメモ編集をGUIで行う

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="実験管理", page_icon="🗂️", layout="wide")

st.title("🗂️ 実験管理")
st.caption("実験フォルダの一覧・ディスク使用量・メモ管理・削除")

st.divider()

EXPERIMENTS_DIR = Path("/workspace/experiments")


def get_dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


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
for exp in exps:
    s = get_exp_status(exp)
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
total_mb = sum(get_dir_size_mb(e) for e in exps)
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

exp_names = [e.name for e in exps]
selected_name = st.selectbox("操作する実験を選択", exp_names)
selected_exp = EXPERIMENTS_DIR / selected_name

tab_note, tab_delete = st.tabs(["📝 メモ編集", "🗑️ フォルダ削除"])

# ── メモ編集タブ ─────────────────────────────────────────────────────────────
with tab_note:
    note_path = selected_exp / "note.md"
    current = note_path.read_text(encoding="utf-8") if note_path.exists() else ""
    new_note = st.text_area("メモ", value=current, height=200, label_visibility="visible")
    if st.button("💾 保存", key="save_note"):
        note_path.write_text(new_note, encoding="utf-8")
        st.success("メモを保存しました。")
        st.rerun()

# ── 削除タブ ─────────────────────────────────────────────────────────────────
with tab_delete:
    size_mb = get_dir_size_mb(selected_exp)
    st.warning(f"⚠️ `{selected_name}` を削除します（{size_mb:.1f} MB）。この操作は元に戻せません。")
    confirm_text = st.text_input("確認のため実験フォルダ名を入力してください",
                                  placeholder=selected_name)
    if st.button("🗑️ 削除する", type="primary",
                 disabled=(confirm_text != selected_name)):
        shutil.rmtree(selected_exp)
        st.success(f"`{selected_name}` を削除しました。")
        st.rerun()
