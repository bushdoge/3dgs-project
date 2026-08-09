# PLATEAU（国交省の実測3D都市モデル）のCityGMLをBlenderへ読み込むインポータ。
# 第1段階＋LOD3対応：建物(bldg)をテクスチャ付きで読み込み、近景のテスト画像をレンダする。
# 電線の生成やmake_scene.pyへの統合は次段階（本ファイルはそれらを行わない）。
# 詳細仕様は memo/design_20260729_plateau.md 参照（LOD3対応は「追記（2026-07-29 夕方）」節）。
#
# 座標系: CityGML(EPSG:6697, JGD2011)の<gml:posList>は「緯度 経度 標高」の順。
#   シーン中心(lat0, lon0)＝既定で --center-lat/--center-lon（LOD3建物が最密な地点）を使い、
#   簡易ENU近似でメートルに変換する（x=east, y=north, z=up。Blenderのワールド座標=Z-upに一致）。
#   標高原点h0だけはgml:boundedByのEnvelopeから取る（自己相殺するオフセットなので厳密さ不要）。
# 名前空間: ファイルによってプレフィクスが違うことがあるため、タグ名・属性名は
#   ElementTreeが解決した完全修飾名から「最後の}以降」だけを見るローカル名一致で拾う
#   （prefix非依存。正規表現より確実——実際の名前空間URIに関係なくプレフィクス表記のゆれを吸収する）。
# テクスチャ: <app:ParameterizedTexture>の<app:target uri="#poly_xxx">がPolygonのgml:idを指し、
#   その中の<app:textureCoordinates>がUV（posListと同じ頂点順・末尾重複あり）を持つ。
#   建物ごとにboundedBy(WallSurface/RoofSurface/GroundSurface等)のlodNMultiSurfaceを使う
#   （lodNSolidは全体形状のみでテクスチャが付かないため、boundedByがあればそちらを優先）。
#
# LOD3対応（2026-07-29追記）: 沼津メッシュ52385618は全3126棟中298棟がLOD3を持ち、
#   LOD3面には4096x4096アトラス(NUMZXXXXX.jpg)が貼られる。LOD2は256x128級(numzXXXXX_l2.jpg)で
#   壁面実効解像度が約5倍違う。--lod で建物ごとの優先順位を切り替える：
#     auto/3 = LOD3優先。無ければLOD2、それも無ければlodNSolid(テクスチャなし)にフォールバック
#     2      = LOD2固定（LOD3があっても使わない。撮り比べの旧挙動再現用）
#   auto と 3 は同じ挙動（3を明示できるのは--lod 2との対称性のため）。
#   どちらを使ったかは建物ごとに集計して標準出力に出す。
#
# 使い方:
#   blender -b -P plateau_import.py -- --bldg <mesh>_bldg_6697_op.gml [--tex-dir <jpg類のdir>]
#                                       [--out /workspace/tmp] [--radius 60] [--lod auto]
#                                       [--center-lat 35.097021] [--center-lon 138.857105]
#                                       [--cam-x 0] [--cam-y 0] [--cam-height 1.5] [--n-views 3]
#                                       [--far-min 10] [--far-max 15]
# 出力:
#   <out>/plateau_lod{2,3}_01.png ... 03.png  カメラ高さ1.5m・水平向きの近景プレビュー
#   <out>/plateau_lod{2,3}_far.png            壁から--far-min〜--far-max[m]離れた1枚
#   標準出力: 建物数・面数・テクスチャ枚数・LOD使用内訳・レンダ時間・pass_index検算結果

import argparse
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import bpy
import bmesh
import mathutils
import numpy as np


# ── 引数 ─────────────────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--bldg", required=True, help="<mesh>_bldg_6697_op.gml のパス")
parser.add_argument("--tex-dir", default=None,
                    help="appearanceのjpgがあるディレクトリ（省略時は--bldgと同じディレクトリ）")
parser.add_argument("--out", default="/workspace/tmp", help="preview画像の出力先ディレクトリ")
parser.add_argument("--radius", type=float, default=60.0, help="シーン中心からの取り込み半径[m]")
parser.add_argument("--lod", choices=["auto", "2", "3"], default="auto",
                    help="auto/3=LOD3優先+LOD2フォールバック（既定）。2=LOD2固定（撮り比べの旧挙動再現用）")
parser.add_argument("--center-lat", type=float, default=35.097021,
                    help="シーン中心 緯度（既定=沼津メッシュ52385618でLOD3建物が最密な地点）")
parser.add_argument("--center-lon", type=float, default=138.857105,
                    help="シーン中心 経度")
parser.add_argument("--res", type=int, nargs=2, default=[1024, 768])
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--cam-height", type=float, default=1.5, help="カメラ高さ[m]")
parser.add_argument("--cam-x", type=float, default=0.0, help="カメラ位置x[m]（シーン中心基準）")
parser.add_argument("--cam-y", type=float, default=0.0, help="カメラ位置y[m]（シーン中心基準）")
parser.add_argument("--n-views", type=int, default=3, help="水平方向に等間隔で撮る枚数")
parser.add_argument("--far-min", type=float, default=10.0,
                    help="追加の1視点(壁からの距離下限[m])。電線は5-8m上空に張るため壁との距離は10m以上が本番条件")
parser.add_argument("--far-max", type=float, default=15.0, help="追加視点(壁からの距離上限[m])")
args = parser.parse_args(argv)
LOD_TAG = "lod2" if args.lod == "2" else "lod3"

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
BLDG_PATH = Path(args.bldg)
TEX_DIR = Path(args.tex_dir) if args.tex_dir else BLDG_PATH.parent


# ── XMLユーティリティ（名前空間プレフィクス非依存） ──────────────────────────────
def local(tag):
    return tag.rsplit("}", 1)[-1]


def find_ln(elem, name):
    for c in elem:
        if local(c.tag) == name:
            return c
    return None


def gml_id(elem):
    for k, v in elem.attrib.items():
        if local(k) == "id":
            return v
    return None


def extract_ring(poly):
    """Polygonの外周(exterior)posListを(lat,lon,h)のリストで返す。内周(interior=穴)は無視。"""
    ext = find_ln(poly, "exterior")
    if ext is None:
        return None
    ring = find_ln(ext, "LinearRing")
    if ring is None:
        return None
    pl = find_ln(ring, "posList")
    if pl is None or not pl.text:
        return None
    vals = [float(v) for v in pl.text.split()]
    if len(vals) % 3 != 0:
        return None
    pts = list(zip(vals[0::3], vals[1::3], vals[2::3]))
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]           # CityGMLは始点=終点で閉じるので末尾を落とす
    return pts if len(pts) >= 3 else None


LOD_MS_RE = re.compile(r"^lod(\d)MultiSurface$")


def lod_priority_order(lod_mode):
    """--lod の値からこの建物で試す優先順位[int]を返す。
    2固定=LOD3を最初から候補に入れない（撮り比べの旧挙動を厳密に再現するため）。
    auto/3=LOD3優先、無ければLOD2、それも無ければLOD1、という段階的フォールバック。"""
    return [2, 1] if lod_mode == "2" else [3, 2, 1]


def collect_building_polys(bldg_elem, lod_mode="auto"):
    """建物1棟分のPolygon群を(polygon_id, [(lat,lon,h),...])のリストと、
    どのLOD由来を採用したか("lod3"/"lod2"/"lod1"/"solid3"/"solid2"/"solid1"/None)を返す。
    boundedBy(壁面/屋根面/地面等、テクスチャが付く単位)のlodNMultiSurfaceを優先し、
    無ければlodNSolid(全体形状のみ・通常テクスチャなし)にフォールバックする。
    lod_modeで指定した優先順位のうち、この建物に実際に存在する最初の候補を採用する。"""
    surfaces_by_lod = {}
    for c in bldg_elem:
        if local(c.tag) != "boundedBy":
            continue
        surf = next(iter(c), None)      # WallSurface/RoofSurface/GroundSurface等
        if surf is None:
            continue
        for c2 in surf:
            m = LOD_MS_RE.match(local(c2.tag))
            if not m:
                continue
            lod = int(m.group(1))
            for poly in c2.iter():
                if local(poly.tag) != "Polygon":
                    continue
                pts = extract_ring(poly)
                if pts:
                    surfaces_by_lod.setdefault(lod, []).append((gml_id(poly), pts))

    order = lod_priority_order(lod_mode)
    for lod in order:
        if lod in surfaces_by_lod:
            return surfaces_by_lod[lod], f"lod{lod}"

    solid_names = {3: "lod3Solid", 2: "lod2Solid", 1: "lod1Solid"}
    for lod in order:
        solid = find_ln(bldg_elem, solid_names[lod])
        if solid is None:
            continue
        out = []
        for poly in solid.iter():
            if local(poly.tag) != "Polygon":
                continue
            pts = extract_ring(poly)
            if pts:
                out.append((gml_id(poly), pts))
        if out:
            return out, f"solid{lod}"
    return [], None


def enu(lat, lon, h, lat0, lon0, h0):
    """緯度経度標高 → シーン中心基準の簡易ENU[m]（100m規模の範囲なので球面近似で十分）。"""
    x = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110540.0
    z = h - h0
    return (x, y, z)


# ── GML読み込み（1パス。buildingsが先・appearanceが後に出現する前提の単一走査） ──────
print(f"[plateau_import] parsing {BLDG_PATH} ({BLDG_PATH.stat().st_size/1e6:.1f}MB) ...")
print(f"[plateau_import] --lod={args.lod} --center-lat={args.center_lat:.6f} --center-lon={args.center_lon:.6f}")
t_parse0 = time.time()

# シーン中心は既定でLOD3建物が最密な地点（--center-lat/--center-lonで上書き可能）。
# 標高原点h0だけはEnvelopeから取る（h - h0 は一律オフセットで、後段のlocal_min_z補正で
# 相殺されるため、lat0/lon0のように引数で明示指定する必要はない）。
lat0, lon0 = args.center_lat, args.center_lon
h0 = None
buildings = []            # [{"id":..., "polys":[(pid, [(x,y,z),...]), ...], "bbox":(...), "lod_used":...}]
kept_pids = set()
tex_map = {}               # pid -> (image_filename, [(u,v),...])
n_seen_buildings = 0
n_skipped_no_geom = 0

for event, elem in ET.iterparse(str(BLDG_PATH), events=("end",)):
    tag = local(elem.tag)

    if tag == "Envelope" and h0 is None:
        lower_el, upper_el = find_ln(elem, "lowerCorner"), find_ln(elem, "upperCorner")
        if lower_el is not None and upper_el is not None:
            lower = [float(v) for v in lower_el.text.split()]
            upper = [float(v) for v in upper_el.text.split()]
            h0 = lower[2]
            env_lat0 = (lower[0] + upper[0]) / 2
            env_lon0 = (lower[1] + upper[1]) / 2
            print(f"[plateau_import] scene center(指定)lat0={lat0:.6f} lon0={lon0:.6f} h0={h0:.2f} "
                  f"(参考: タイルEnvelope中心lat={env_lat0:.6f} lon={env_lon0:.6f})")
        elem.clear()

    elif tag == "cityObjectMember":
        n_seen_buildings += 1
        bldg_elem = next((c for c in elem if local(c.tag) == "Building"), None)
        if bldg_elem is not None and h0 is not None:
            bid = gml_id(bldg_elem) or f"bldg_{n_seen_buildings}"
            raw_polys, lod_used = collect_building_polys(bldg_elem, args.lod)
            if raw_polys:
                polys_local = [(pid, [enu(lat, lon, h, lat0, lon0, h0) for lat, lon, h in pts])
                               for pid, pts in raw_polys]
                all_pts = [p for _, pts in polys_local for p in pts]
                cx = sum(p[0] for p in all_pts) / len(all_pts)
                cy = sum(p[1] for p in all_pts) / len(all_pts)
                if math.hypot(cx, cy) <= args.radius:
                    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
                    buildings.append({"id": bid, "polys": polys_local,
                                      "bbox": (min(xs), max(xs), min(ys), max(ys)),
                                      "lod_used": lod_used})
                    for pid, _ in polys_local:
                        if pid:
                            kept_pids.add(pid)
            else:
                n_skipped_no_geom += 1
        elem.clear()

    elif tag == "ParameterizedTexture":
        uri_el = find_ln(elem, "imageURI")
        image_name = os.path.basename(uri_el.text.strip()) if uri_el is not None and uri_el.text else None
        if image_name:
            for target in elem:
                if local(target.tag) != "target":
                    continue
                pid = target.attrib.get("uri", "").lstrip("#")
                if pid not in kept_pids:
                    continue
                tclist = find_ln(target, "TexCoordList")
                tc = find_ln(tclist, "textureCoordinates") if tclist is not None else None
                if tc is None or not tc.text:
                    continue
                vals = [float(v) for v in tc.text.split()]
                if len(vals) % 2 != 0:
                    continue
                uv = list(zip(vals[0::2], vals[1::2]))
                if len(uv) >= 2 and uv[0] == uv[-1]:
                    uv = uv[:-1]
                tex_map[pid] = (image_name, uv)
        elem.clear()

n_faces_total = sum(len(b["polys"]) for b in buildings)
n_faces_textured = sum(1 for b in buildings for pid, _ in b["polys"] if pid in tex_map)
print(f"[plateau_import] parse done in {time.time()-t_parse0:.1f}s: "
      f"cityObjectMember走査={n_seen_buildings} 半径内建物={len(buildings)} "
      f"(ジオメトリなしで除外={n_skipped_no_geom}) 面数={n_faces_total} "
      f"テクスチャ付き面={n_faces_textured} tex_mapエントリ={len(tex_map)}")

if not buildings:
    print("[plateau_import] 半径内に建物が見つからなかった。--radius を広げるか --bldg を確認してください。")
    sys.exit(1)

# LOD使用内訳（完了判定2: LOD3を使えた建物数 / LOD2にフォールバックした建物数）
lod_counts = Counter(b["lod_used"] for b in buildings)
n_lod3 = lod_counts.get("lod3", 0)
n_lod2 = lod_counts.get("lod2", 0)
n_lod1 = lod_counts.get("lod1", 0)
n_solid = lod_counts.get("solid3", 0) + lod_counts.get("solid2", 0) + lod_counts.get("solid1", 0)
print(f"[plateau_import] LOD使用内訳(半径内建物={len(buildings)}棟, --lod={args.lod}): "
      f"LOD3採用={n_lod3} LOD2フォールバック={n_lod2} LOD1={n_lod1} "
      f"Solidのみ(テクスチャなし)={n_solid}")
if args.lod == "2" and n_lod3 > 0:
    print("[plateau_import] 警告: --lod 2 指定なのにLOD3採用棟が0でない（ロジック不整合の可能性）")

# h0はタイル全体(gml:boundedByのEnvelope)の最低標高だが、我々が拾う半径内の建物群は
# タイル内のどこか一角に過ぎないため、地形起伏でここでの実際の地表高とはずれることがある
# （沼津は海沿いなのでタイル最低点=海抜0m付近、建物群が内陸側/高台なら数m浮いて見える）。
# 実際に建物が浮かんで見える不具合が出たため、地面基準は「半径内で採用した建物群自身の
# 最低点」を使う（=このローカルシーンでの実地表高の近似）。
local_min_z = min(p[2] for b in buildings for _, pts in b["polys"] for p in pts)
print(f"[plateau_import] ローカル地表高補正: タイル全体h0基準z=0 -> 採用建物群の最低点z={local_min_z:.2f}m "
      f"だけ地面・カメラをかさ上げする")


# ── テクスチャファイルの索引（jpgの実ファイル名ゆれ対応: 大小文字・空白差異を吸収） ─────
def normalize_name(name):
    return name.lower().replace(" ", "")


tex_index = {}
if TEX_DIR.is_dir():
    for p in TEX_DIR.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            tex_index[normalize_name(p.name)] = p
print(f"[plateau_import] tex-dir={TEX_DIR} 内のテクスチャ候補={len(tex_index)}枚")


def find_texture_path(image_name):
    return tex_index.get(normalize_name(image_name))


# ── Blenderシーン初期化 ───────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x, scene.render.resolution_y = args.res
scene.render.image_settings.file_format = "PNG"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
# 今回の目的は「近景でテクスチャがどう見えるか」の確認そのものなので、Filmicで色を
# 加工せずStandardのまま見る（make_scene.pyはSfM都合でFilmicを使うが、ここでは無関係）。
scene.view_settings.view_transform = "Standard"

prefs = bpy.context.preferences.addons["cycles"].preferences
gpu_found = False
for dev_type in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = dev_type
        prefs.get_devices()
        for d in prefs.devices:
            if d.type == dev_type:
                d.use = True
                gpu_found = True
        if gpu_found:
            break
    except Exception:
        continue
scene.cycles.device = "GPU" if gpu_found else "CPU"
print(f"[plateau_import] render device: {scene.cycles.device}"
      + (f" ({prefs.compute_device_type})" if gpu_found else ""))


# ── 地面（最小限の追加。カメラが立つ地表がないと絵にならないための実務上の判断。
#          設計書にはないが単一の無地グレー平面のみで、電線等の設計要素は一切足さない）───
ground_mat = bpy.data.materials.new("plateau_ground")
ground_mat.use_nodes = True
ground_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.15, 0.15, 0.15, 1)
ground_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
bpy.ops.mesh.primitive_plane_add(size=args.radius * 2.2, location=(0, 0, local_min_z))
ground_ob = bpy.context.object
ground_ob.name = "plateau_ground"
ground_ob.data.materials.append(ground_mat)
ground_ob.pass_index = 0


# ── マテリアル（画像1枚につき1個。全建物で共有してImageデータブロックの重複を避ける） ────
GLOBAL_MATERIALS = {}
n_missing_tex = 0


def get_material(image_name):
    global n_missing_tex
    key = image_name if image_name else "__notex__"
    if key in GLOBAL_MATERIALS:
        return GLOBAL_MATERIALS[key]
    mat = bpy.data.materials.new(name=f"tex_{key}"[:63])
    mat.use_nodes = True
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    if image_name:
        path = find_texture_path(image_name)
        if path is not None:
            # PLATEAUのapp:ParameterizedTextureは実写JPG(照明が焼き込み済みのアルベド)。
            # Principled BSDF + 自前のSun/Skyで再度照らすと二重露光で白飛びし、
            # 「近景でテクスチャがどう見えるか」の確認という目的とズレる。
            # Emissionで直結し、写真そのものの色をそのまま出す。
            nt.nodes.remove(nt.nodes["Principled BSDF"])
            img = bpy.data.images.load(str(path))
            teximg = nt.nodes.new("ShaderNodeTexImage")
            teximg.image = img
            emit = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(teximg.outputs["Color"], emit.inputs["Color"])
            nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        else:
            n_missing_tex += 1
            bsdf = nt.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 1.0, 1)   # マゼンタ=欠損マーカー
    else:
        bsdf = nt.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1)
    GLOBAL_MATERIALS[key] = mat
    return mat


# ── 建物ジオメトリ生成 ────────────────────────────────────────────────────────
n_faces_created = 0
n_faces_degenerate = 0

for b in buildings:
    mesh = bpy.data.meshes.new(f"{b['id']}_mesh")
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("uv")
    vert_cache = {}

    def get_vert(p):
        key = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
        v = vert_cache.get(key)
        if v is None:
            v = bm.verts.new(p)
            vert_cache[key] = v
        return v

    obj_materials = []
    obj_mat_index = {}
    for pid, pts in b["polys"]:
        verts = [get_vert(p) for p in pts]
        if len(set(v.index if v.index != 0 else id(v) for v in verts)) < 3:
            pass  # bmeshのindexは未確定なのでこのチェックは実質スキップ（下のtry/exceptで担保）
        try:
            face = bm.faces.new(verts)
        except ValueError:
            n_faces_degenerate += 1
            continue
        image_name, uv = tex_map.get(pid, (None, None))
        if image_name not in obj_mat_index:
            obj_mat_index[image_name] = len(obj_materials)
            obj_materials.append(image_name)
        face.material_index = obj_mat_index[image_name]
        if uv is not None and len(uv) == len(face.loops):
            for loop, (u, v) in zip(face.loops, uv):
                loop[uv_layer].uv = (u, v)
        n_faces_created += 1

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    ob = bpy.data.objects.new(b["id"], mesh)
    for image_name in obj_materials:
        ob.data.materials.append(get_material(image_name))
    ob.pass_index = 0                     # 設計書§3.5: 取り込んだ地物は全て背景扱い(0)
    bpy.context.collection.objects.link(ob)

n_unique_tex = len([k for k in GLOBAL_MATERIALS if k != "__notex__"])
print(f"[plateau_import] 建物オブジェクト={len(buildings)} 生成面={n_faces_created} "
      f"(退化面で除外={n_faces_degenerate}) ユニークテクスチャ={n_unique_tex} "
      f"テクスチャ欠損(マゼンタ代替)={n_missing_tex}")


# ── ライティング・空 ──────────────────────────────────────────────────────────
sun = bpy.data.lights.new("sun", type="SUN")
sun.energy = 3.0
sun_ob = bpy.data.objects.new("sun", sun)
sun_ob.rotation_euler = (math.radians(50), 0, math.radians(30))
bpy.context.collection.objects.link(sun_ob)

world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
sky = world.node_tree.nodes.new("ShaderNodeTexSky")
sky.sun_elevation = math.radians(50)
world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 0.8


# ── カメラ配置（高さ1.5m・水平向き。中心が建物内部に埋まらないよう
#    建物bboxとの重なりを見て周囲へ退避する簡易ヒューリスティック）────────────────
def inside_any_bbox(x, y):
    for b in buildings:
        xmin, xmax, ymin, ymax = b["bbox"]
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False


def dist_to_bbox(x, y, bbox):
    """点(x,y)からbbox矩形までの最短距離[m]（内部なら0）。"""
    xmin, xmax, ymin, ymax = bbox
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def min_dist_to_any_bbox(x, y):
    if not buildings:
        return float("inf")
    return min(dist_to_bbox(x, y, b["bbox"]) for b in buildings)


def find_far_camera(min_d, max_d):
    """壁からの最短距離がmin_d〜max_d[m]の範囲に収まる地点を、シーン中心から
    外向きに走査して探す（電線を5-8m上空に張る本番シーンでは壁との距離が
    10m以上になるため、その条件を近似的なカメラ位置で再現する）。
    見つかった点と、その最近傍建物を返す（無ければNone）。"""
    for r in np.arange(5.0, 80.0, 2.0):
        for k in range(24):
            th = 2 * math.pi * k / 24
            x, y = r * math.cos(th), r * math.sin(th)
            d = min_dist_to_any_bbox(x, y)
            if min_d <= d <= max_d:
                nearest_b = min(buildings, key=lambda b: dist_to_bbox(x, y, b["bbox"]))
                return x, y, d, nearest_b
    return None


cam_x, cam_y = args.cam_x, args.cam_y
if inside_any_bbox(cam_x, cam_y):
    found = False
    for r in np.arange(3.0, 30.0, 3.0):
        for k in range(16):
            th = 2 * math.pi * k / 16
            tx, ty = args.cam_x + r * math.cos(th), args.cam_y + r * math.sin(th)
            if not inside_any_bbox(tx, ty):
                cam_x, cam_y = tx, ty
                found = True
                break
        if found:
            break
    print(f"[plateau_import] 指定カメラ位置は建物内部だったため退避: "
          f"({args.cam_x:.1f},{args.cam_y:.1f}) -> ({cam_x:.1f},{cam_y:.1f})")

cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_width = 36.0
cam_data.lens = 30.0
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam
cam.pass_index = 0


def look_at_euler(fwd_xy):
    """水平方向fwd_xy(単位ベクトル,z=0)を向く、ロール無しのcam→world回転をeulerで返す
    （make_scene.pyの周回カメラと同じ一般的なlook-at構成。ファイル自体は流用していない）。"""
    fwd = np.array([fwd_xy[0], fwd_xy[1], 0.0])
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=1)
    return mathutils.Matrix(R.tolist()).to_euler()


cam.location = (cam_x, cam_y, local_min_z + args.cam_height)

# ── レンダリング ─────────────────────────────────────────────────────────────
scene.use_nodes = False
render_times = []
for i in range(args.n_views):
    th = 2 * math.pi * i / args.n_views
    cam.rotation_euler = look_at_euler((math.cos(th), math.sin(th)))
    # LOD2版とLOD3版で上書きし合わないようファイル名にLODを入れる（撮り比べのため）
    scene.render.filepath = str(OUT / f"plateau_{LOD_TAG}_{i+1:02d}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    render_times.append(dt)
    print(f"[plateau_import] rendered view {i+1}/{args.n_views} "
          f"({dt:.1f}s, device={scene.cycles.device}) -> {scene.render.filepath}")

# ── 遠景（壁から10〜15m）＝本番の実験条件に近い画を1枚 ──────────────────────
# 実験では電線を5〜8m上空に見るので壁との距離は10m以上になる。至近距離の画だけで
# 「使える/使えない」を判断すると誤るため、この1枚を必ず撮る。
far = find_far_camera(10.0, 15.0)
if far is None:
    print("[plateau_import] 壁から10〜15mの地点が見つからず遠景はスキップ")
else:
    fx, fy, fd, fb = far
    cam.location = (fx, fy, local_min_z + args.cam_height)
    bx = (fb["bbox"][0] + fb["bbox"][1]) / 2 - fx
    by = (fb["bbox"][2] + fb["bbox"][3]) / 2 - fy
    cam.rotation_euler = look_at_euler((bx, by))
    scene.render.filepath = str(OUT / f"plateau_{LOD_TAG}_far.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    render_times.append(time.time() - t0)
    print(f"[plateau_import] 遠景レンダ: 壁から{fd:.1f}m -> {scene.render.filepath}")

print(f"[plateau_import] レンダ合計時間={sum(render_times):.1f}s "
      f"(1枚あたり平均{sum(render_times)/len(render_times):.1f}s)")


# ── pass_index 検算（設計書§3.5・完了判定4: 取り込んだ全オブジェクトが0であること）──
all_objs = list(bpy.data.objects)
bad = [o.name for o in all_objs if o.pass_index != 0]
print(f"[plateau_import] pass_index検算: 対象オブジェクト数={len(all_objs)} "
      f"非0={len(bad)}" + (f" NG: {bad}" if bad else " (OK, 全て0)"))

print(f"[plateau_import] ==== 完了判定サマリ ====")
print(f"  建物数(半径{args.radius}m以内) = {len(buildings)}")
print(f"  面数(生成) = {n_faces_created}")
print(f"  ユニークテクスチャ枚数 = {n_unique_tex} (欠損マゼンタ代替 = {n_missing_tex})")
print(f"  レンダ時間合計 = {sum(render_times):.1f}s ({args.n_views}枚)")
print(f"  pass_index異常 = {len(bad)}")
