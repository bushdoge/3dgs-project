# 実験ディレクトリ内のログ・cfg_argsから pipeline_config.json を復元するスクリプト
# pipeline_config.json が存在しない実験（パイプライン未使用 / 実装前のもの）向け

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path


# ── 構造化ヘッダー（_write_settings_header 形式）のパース ────────────────────

def _parse_structured_header(text: str) -> dict:
    """================================================================ で囲まれた設定ヘッダーをパースする"""
    result = {}

    m = re.search(r'入力動画/ソース\s*:\s*(.+)', text)
    if m:
        result["video_path"] = m.group(1).strip()

    m = re.search(r'FPS\s*:\s*([\d.]+)', text)
    if m:
        result["fps"] = float(m.group(1))

    m = re.search(r'360度変換\s*:\s*(あり|なし)', text)
    if m:
        result["is_360"] = m.group(1) == "あり"
        if result["is_360"]:
            fm = re.search(r'FOV=(\d+)', text)
            if fm:
                result["fov"] = int(fm.group(1))
            sm = re.search(r'(\d+)x(\d+)', text)
            if sm:
                result["out_w"] = int(sm.group(1))
                result["out_h"] = int(sm.group(2))
            am = re.search(r'向き\s*:\s*(.+)', text)
            if am:
                angles = []
                for a in re.findall(r'y=(-?\d+)/p=(-?\d+)', am.group(1)):
                    angles.append([int(a[0]), int(a[1])])
                if angles:
                    result["angles"] = angles

    m = re.search(r'カメラ推定\s*\((\w+)\)', text)
    if m:
        result["use_hloc"] = (m.group(1) == "HLoc")

    if result.get("use_hloc"):
        m = re.search(r'特徴量\s*:\s*(\S+)', text)
        if m:
            result["feature_type"] = m.group(1)
        m = re.search(r'マッチング\s*:\s*(\S+)', text)
        if m:
            result["matcher_type"] = m.group(1)
        m = re.search(r'ペアリスト\s*:\s*(\S+)', text)
        if m:
            result["pair_method"] = m.group(1)
        m = re.search(r'retrieval=(\S+),\s*(\d+)\s*pairs', text)
        if m:
            result["retrieval_model"] = m.group(1)
            result["num_matched"] = int(m.group(2))
    else:
        m = re.search(r'カメラモデル\s*:\s*(\S+)', text)
        if m:
            result["camera_model"] = m.group(1)
        m = re.search(r'GPU\s*:\s*(あり|なし)', text)
        if m:
            result["use_gpu"] = (m.group(1) == "あり")

    m = re.search(r'イテレーション\s*:\s*(\d+)', text)
    if m:
        result["iterations"] = int(m.group(1))

    m = re.search(r'保存\s*:\s*([\d,\s]+)', text)
    if m:
        vals = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
        if vals:
            result["save_iterations"] = vals

    m = re.search(r'テスト\s*:\s*([\d,\s]+)', text)
    if m:
        vals = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
        if vals:
            result["test_iterations"] = vals

    m = re.search(r'解像度\s*:\s*(.+)', text)
    if m:
        rt = m.group(1).strip()
        if '自動' in rt:
            result["resolution"] = None
        else:
            rm2 = re.search(r'(\d+)x', rt)
            if rm2:
                result["resolution"] = int(rm2.group(1))

    return result


# ── HLoc 生ログのパース ───────────────────────────────────────────────────────

def _parse_hloc_raw(text: str) -> dict:
    """run_hloc.py の生ログ出力から HLoc 設定を読み取る"""
    result = {"use_hloc": True}

    m = re.search(r'局所特徴点抽出:\s*(\S+)', text)
    if m:
        result["feature_type"] = m.group(1)

    m = re.search(r'ペアリスト生成方式:\s*(\S+)', text)
    if m:
        result["pair_method"] = m.group(1)

    m = re.search(r'グローバル特徴量抽出:\s*(\S+)', text)
    if m:
        result["retrieval_model"] = m.group(1)

    m = re.search(r'ペアリスト生成（retrieval top-(\d+)）', text)
    if m:
        result["num_matched"] = int(m.group(1))

    m = re.search(r'特徴点マッチング:\s*(\S+)', text)
    if m:
        result["matcher_type"] = m.group(1)

    return result


# ── extract_log からの FPS 取得 ───────────────────────────────────────────────

def _parse_extract_log(text: str) -> dict:
    """extract_log.txt / convert_360 ログから FPS・360 設定を読み取る"""
    result = {}

    m = re.search(r'動画からフレームを抽出中（([\d.]+)\s*fps）', text)
    if m:
        result["fps"] = float(m.group(1))
        result["is_360"] = False

    # 360変換ログ（convert_360.py）
    m = re.search(r'([\d.]+)\s*fps.*360\|360.*([\d.]+)\s*fps', text, re.IGNORECASE)
    if not m:
        # 「変換中」という文言があれば360変換とみなす
        if "変換中" in text and "fps" not in result:
            # FPS行がなければ360変換の可能性
            pass

    return result


# ── train_log からの学習設定取得 ─────────────────────────────────────────────

def _parse_train_log(text: str) -> dict:
    """train_log.txt の「実行コマンド:」行から学習パラメータを読み取る"""
    result = {}

    m = re.search(r'実行コマンド:\s*(.+)', text)
    if not m:
        return result
    cmd = m.group(1)

    m = re.search(r'--iterations\s+(\d+)', cmd)
    if m:
        result["iterations"] = int(m.group(1))

    m = re.search(r'--save_iterations\s+([\d\s]+?)(?:--|$)', cmd)
    if m:
        vals = [int(x) for x in m.group(1).split() if x.isdigit()]
        if vals:
            result["save_iterations"] = vals

    m = re.search(r'--test_iterations\s+([\d\s]+?)(?:--|$)', cmd)
    if m:
        vals = [int(x) for x in m.group(1).split() if x.isdigit()]
        if vals:
            result["test_iterations"] = vals

    result["eval"] = "--eval" in cmd

    m = re.search(r'--resolution\s+(\d+)', cmd)
    if m:
        result["resolution"] = int(m.group(1))
    else:
        result["resolution"] = None

    return result


# ── cfg_args からの補完 ───────────────────────────────────────────────────────

def _parse_cfg_args(path: Path) -> dict:
    """cfg_args の Namespace(...) から学習設定を読み取る"""
    result = {}
    try:
        text = path.read_text(errors="replace").strip()
        if text.startswith("Namespace(") and text.endswith(")"):
            text = text[len("Namespace("):-1]
        pattern = re.compile(
            r"(\w+)="
            r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|"
            r"True|False|None|-?\d+\.\d+e[+-]?\d+|-?\d+\.\d+|-?\d+|\[[^\]]*\])"
        )
        for m in pattern.finditer(text):
            key, val_str = m.group(1), m.group(2)
            try:
                result[key] = ast.literal_eval(val_str)
            except Exception:
                result[key] = val_str
    except Exception:
        pass
    return result


# ── メイン復元ロジック ────────────────────────────────────────────────────────

def recover(exp_dir: Path) -> tuple[dict, list[str]]:
    """
    実験ディレクトリのログから pipeline_config の内容を復元する。
    Returns: (config_dict, sources_list)
    """
    config: dict = {}
    sources: list[str] = []

    # ── 1. 構造化ヘッダーを探す ─────────────────────────────────────────────
    # ヘッダーは ={64} で3行囲まれた形式。キーワードが含まれるか確認してから
    # テキスト全体を渡してパースすることで、正規表現の境界ミスを防ぐ。
    for log_name in ("extract_log.txt", "colmap_log.txt"):
        log_path = exp_dir / log_name
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="replace")
        if "入力動画/ソース" in text or "実験ディレクトリ" in text:
            parsed = _parse_structured_header(text)
            if parsed:
                config.update(parsed)
                sources.append(f"{log_name}（設定ヘッダー）")
                break

    # ── 2. HLoc 生ログから姿勢推定設定を補完 ────────────────────────────────
    colmap_log = exp_dir / "colmap_log.txt"
    if colmap_log.exists():
        text = colmap_log.read_text(errors="replace")
        if "局所特徴点抽出" in text and "feature_type" not in config:
            hloc = _parse_hloc_raw(text)
            config.update({k: v for k, v in hloc.items() if k not in config})
            sources.append("colmap_log.txt（HLoc出力）")

    # ── 3. extract_log から FPS 補完 ────────────────────────────────────────
    extract_log = exp_dir / "extract_log.txt"
    if extract_log.exists() and "fps" not in config:
        text = extract_log.read_text(errors="replace")
        parsed = _parse_extract_log(text)
        config.update({k: v for k, v in parsed.items() if k not in config})
        if parsed:
            sources.append("extract_log.txt")

    # ── 4. train_log から学習設定を補完 ─────────────────────────────────────
    train_logs = list((exp_dir / "output").rglob("train_log.txt")) \
        if (exp_dir / "output").exists() else []
    if train_logs and "iterations" not in config:
        text = train_logs[0].read_text(errors="replace")
        parsed = _parse_train_log(text)
        config.update({k: v for k, v in parsed.items() if k not in config})
        if parsed:
            sources.append("train_log.txt（実行コマンド行）")

    # ── 5. input/ ファイル名パターンから is_360 を補完 ──────────────────────
    if "is_360" not in config:
        input_dir = exp_dir / "input"
        if input_dir.exists():
            samples = list(input_dir.glob("*.jpg"))[:5] + list(input_dir.glob("*.png"))[:5]
            config["is_360"] = any(
                re.search(r'_y\d+_p[+\-]?\d+', f.stem) for f in samples
            )

    # ── 6. cfg_args から学習設定を補完 ──────────────────────────────────────
    cfg_args_files = list((exp_dir / "output").rglob("cfg_args")) \
        if (exp_dir / "output").exists() else []
    if cfg_args_files:
        ca = _parse_cfg_args(cfg_args_files[0])
        for key in ("resolution", "eval"):
            if key not in config and key in ca:
                config[key] = ca[key]
        if ca:
            sources.append("cfg_args")

    config["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["_recovered_from"] = sources

    return config, sources


# ── CLI エントリポイント ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ログから pipeline_config.json を復元する"
    )
    parser.add_argument("exp_dir", help="実験ディレクトリのパス")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存の pipeline_config.json を上書きする")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} が見つかりません", file=sys.stderr)
        return 1

    out_path = exp_dir / "pipeline_config.json"
    if out_path.exists() and not args.overwrite:
        print("pipeline_config.json はすでに存在します。上書きするには --overwrite を指定してください。")
        return 0

    config, sources = recover(exp_dir)
    out_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存しました: {out_path}")
    print(f"取得元: {', '.join(sources) if sources else '（なし）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
