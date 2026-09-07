# テスト視点名を「学習側と同じ定義」で導出する共通ヘルパー。
#
# 2026-09-06 のコード監査で、多くのツールが `input/` や `mask_thin/` のファイル一覧を
# 8個おきに取ってテスト視点にしていたことが判明した。しかし 3DGS（llffhold=8）が並べるのは
# **COLMAPに登録された画像**であり、登録に失敗した画像があると並びがずれ、
# **別フレームのマスクで評価してしまう**（street f40 で5視点中3視点、crossing f40 で2視点）。
#
# 対応付けは必ずこの関数を通すこと。
from pathlib import Path


def test_view_names(exp, hold=8):
    """<exp>/sparse/0/images.bin の登録名を名前順に並べ、hold個おきを返す。
    images.bin が無い場合のみ input/ にフォールバックする（合成前の段階など）。"""
    exp = Path(exp)
    img_bin = exp / "sparse" / "0" / "images.bin"
    if img_bin.exists():
        try:
            import sys
            sys.path.insert(0, "/opt/gaussian-splatting")
            from scene.colmap_loader import read_extrinsics_binary
            names = sorted(Path(v.name).stem for v in read_extrinsics_binary(str(img_bin)).values())
            return names[::hold]
        except Exception as e:
            print(f"[警告] images.bin を読めないので input/ にフォールバック: {e}")
    names = sorted(p.stem for p in (exp / "input").glob("*.png"))
    if not names:
        names = sorted(p.stem for p in (exp / "input").glob("*.jpg"))
    return names[::hold]
