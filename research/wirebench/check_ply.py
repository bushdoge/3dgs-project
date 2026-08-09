# 3DGSが保存した point_cloud.ply が最後まで書けているかを検証するスクリプト。
# 学習中に電源断・強制終了が起きると書きかけのplyが残る。実行スクリプトの再開ガードが
# 「ファイルが存在する＝完了済み」で判定していると、そのrunをスキップして壊れたモデルのまま
# render → eval が走り、エラーも出さずに誤った数値が結果CSVに入る（2026-07-26/27に鯖落ちで実害の危険）。
# ヘッダの element vertex とプロパティ数から期待バイト数を計算し、実ファイルサイズと突き合わせる。
#
# 使い方: python3 check_ply.py <point_cloud.ply>
#   終了コード 0=完全 / 1=不完全・壊れている（メッセージは標準エラーへ）

import sys
from pathlib import Path

TYPE_SIZE = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
             "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
             "int": 4, "uint": 4, "int32": 4, "uint32": 4, "float": 4, "float32": 4,
             "double": 8, "float64": 8}


def check(path):
    path = Path(path)
    if not path.is_file():
        return f"存在しない: {path}"

    n_vertex, vertex_bytes, header_bytes = None, 0, 0
    in_vertex_element = False
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return "ヘッダが end_header に到達しない（書きかけ）"
            header_bytes += len(line)
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "element":
                # 3DGSのplyは vertex 要素のみだが、他要素があってもvertex分だけ数える
                in_vertex_element = tok[1] == "vertex"
                if in_vertex_element:
                    n_vertex = int(tok[2])
            elif tok[0] == "property" and in_vertex_element:
                if tok[1] == "list":
                    return "list プロパティは未対応（3DGSのplyではないかもしれない）"
                if tok[1] not in TYPE_SIZE:
                    return f"未知の型: {tok[1]}"
                vertex_bytes += TYPE_SIZE[tok[1]]
            elif tok[0] == "end_header":
                break

    if n_vertex is None:
        return "element vertex がヘッダに無い"
    if n_vertex == 0:
        return "頂点数が0"

    expected = header_bytes + n_vertex * vertex_bytes
    actual = path.stat().st_size
    if actual != expected:
        return (f"サイズ不一致: 実際 {actual} バイト / 期待 {expected} バイト"
                f"（頂点 {n_vertex} × {vertex_bytes} バイト + ヘッダ {header_bytes}）"
                f" → {'書きかけ' if actual < expected else '余分なデータ'}")
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__ or "使い方: python3 check_ply.py <point_cloud.ply>")
    err = check(sys.argv[1])
    if err:
        print(f"[check_ply] 不完全: {sys.argv[1]}\n  {err}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
