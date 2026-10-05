# 「帯 vs 半平面」検定（memo/design_20260922_band_vs_halfplane.md、研究メモ §9am・§9y-2）用の統制シーンを Blender で作る。
# 発光（Emission）マテリアルの円柱を半径6段×向き3種、半平面の基準として同じ発光の大きな水平板を空に浮かべ、
# 一様色の空を背景に周回カメラでレンダする。陰影が付かないので、どの物体・どの視点でも同じ色・同じコントラストで写り、
# 「境界の間隔（＝見かけ幅）」以外を揃えて比べられる。make_scene.py は書き換えない（別スクリプト）。
#
# 設計書から細部で決めたこと：
#   - 物体は半径3mの円周上に接線向きで並べ、高さ4.2m以上に置く（カメラは最高3.2m・建物は最高4m。
#     視線が上向きのまま物体を通るので、背後は必ず空か他の物体になる。他の物体との重なりは評価側で除外）
#   - 向き3種は水平からの傾き 0°/30°/60°。上下の画角に収めるため長さを 2.4/1.6/1.0m にする（評価は境界の単位長さあたり）
#   - カメラの注視点を z=1.4→2.4 に上げる（物体が画角の上端で切れないように。軌道 r=8.5・高さ 2.2±1.0 は base と同じ）
#   - 電柱・電線・フェンスは置かない（マスクの混ざりを防ぐ。§9ab）。SfM の特徴点は地面と建物3個（base と同じ）で取る
#   - GT は既定で 1024spp・デノイズ無し（§9ar-5 の教訓：デノイザは細い構造に視点独立な誤差を残す）
#
# 使い方:
#   blender -b -P make_probe_scene.py -- --out <出力dir> [--frames 80] [--samples 1024] [--test-one]
# 出力構成:
#   <out>/input/frame_####.png          学習用RGB
#   <out>/gt/mask_obj/obj##/####.png    物体ごとのマスク（pass_index = 物体ID。1〜18 円柱、19 板）
#   <out>/gt/depth/####.exr             深度
#   <out>/gt/poses.json                 GTカメラ姿勢（make_scene.py と同形式）
#   <out>/gt/probe_objects.json         物体の幾何（ID・種類・半径・傾き・軸の端点／板の角）
#   <out>/gt/scene_params.json          シーンの統制変数

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import mathutils
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--frames", type=int, default=80)
parser.add_argument("--samples", type=int, default=1024)
parser.add_argument("--denoise", action="store_true", help="デノイザを使う（既定は無し）")
parser.add_argument("--res", type=int, nargs=2, default=[1024, 768])
parser.add_argument("--test-one", action="store_true", help="1フレームだけレンダして終了（動作確認用）")
parser.add_argument("--radii-mm", type=float, nargs="+", default=[7.5, 15, 30, 60, 120, 240])
parser.add_argument("--tilts-deg", type=float, nargs="+", default=[0, 30, 60])
parser.add_argument("--obj-gray", type=float, default=0.02, help="発光物体の色（linear。暗い線＝電線と同じ向きのコントラスト）")
parser.add_argument("--sky", type=float, nargs=3, default=[0.45, 0.6, 0.85], help="一様な空の色（linear）")
args = parser.parse_args(argv)

OUT = Path(args.out)
for d in ("input", "gt/depth", "gt/mask_obj"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ── シーン初期化（make_scene.py と同じレンダ設定。デノイズの既定だけ逆） ─────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x, scene.render.resolution_y = args.res
scene.render.image_settings.file_format = "PNG"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = args.denoise
scene.view_settings.view_transform = "Filmic"

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
print(f"[probe] render device: {scene.cycles.device}")


def make_material(name, base_color, noise_scale, color2):
    """ノイズテクスチャ付きマテリアル（SfMの特徴点確保用。make_scene.py と同じ作り）"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.8
    coord = nt.nodes.new("ShaderNodeTexCoord")
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = noise_scale
    tex.inputs["Detail"].default_value = 10.0
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[1].position = 0.58
    ramp.color_ramp.elements[0].color = (*base_color, 1)
    ramp.color_ramp.elements[1].color = (*color2, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def make_emission(name, gray):
    """陰影の付かない発光マテリアル（どの向きから見ても同じ色）"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (gray, gray, gray, 1)
    em.inputs["Strength"].default_value = 1.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def add_box(name, size, loc, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0], loc[1], loc[2] + size[2] / 2))
    ob = bpy.context.object
    ob.name = name
    ob.scale = size
    ob.data.materials.append(mat)
    return ob


# ── 背景ジオメトリ（base と同じ地面・建物3個。SfM 用） ───────────────────────────
mat_ground = make_material("ground", (0.04, 0.035, 0.03), 7.0, (0.3, 0.27, 0.2))
mat_brickA = make_material("brickA", (0.25, 0.05, 0.03), 4.0, (0.65, 0.3, 0.15))
mat_brickB = make_material("brickB", (0.04, 0.08, 0.25), 3.5, (0.3, 0.45, 0.65))
mat_boxC = make_material("boxC", (0.2, 0.15, 0.03), 5.0, (0.65, 0.55, 0.2))
bpy.ops.mesh.primitive_plane_add(size=30)
bpy.context.object.data.materials.append(mat_ground)
add_box("bldgA", (3.0, 3.0, 4.0), (-3.5, 2.5, 0), mat_brickA)
add_box("bldgB", (2.2, 2.2, 3.0), (3.2, 3.0, 0), mat_brickB)
add_box("boxC", (1.5, 1.5, 1.5), (2.5, -2.8, 0), mat_boxC)

# ── 測定対象：発光円柱（半径×傾き）と発光板 ───────────────────────────────────
mat_obj = make_emission("probe", args.obj_gray)
LENGTH = {0: 2.4, 30: 1.6, 60: 1.0}         # 傾きごとの長さ[m]（上下の画角に収める）
ZC = {0: 4.45, 30: 4.85, 60: 4.85}          # 中心高さ（最下点が 4.1m 以上になる）
RING_R = 3.0
objects = []
combos = [(ti, ri) for ti in range(len(args.tilts_deg)) for ri in range(len(args.radii_mm))]
n_cyl = len(combos)
for k, (ti, ri) in enumerate(combos):
    tilt = float(args.tilts_deg[ti])
    R = args.radii_mm[ri] / 1000.0
    L = LENGTH.get(int(tilt), 1.6)
    zc = ZC.get(int(tilt), 4.85)
    # 半径の並びが円周上で偏らないよう、傾きのグループごとに開始位置をずらす
    az = 2 * math.pi * ((ri * len(args.tilts_deg) + ti) % n_cyl) / n_cyl
    c = np.array([RING_R * math.cos(az), RING_R * math.sin(az), zc])
    tangent = np.array([-math.sin(az), math.cos(az), 0.0])
    axis = math.cos(math.radians(tilt)) * tangent + np.array([0, 0, math.sin(math.radians(tilt))])
    p0, p1 = c - axis * L / 2, c + axis * L / 2
    assert min(p0[2], p1[2]) - R > 4.05, "物体の最下点が建物の高さに近すぎる"
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=R, depth=L, location=tuple(c))
    ob = bpy.context.object
    oid = k + 1
    ob.name = f"cyl{oid:02d}"
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = mathutils.Vector((0, 0, 1)).rotation_difference(mathutils.Vector(axis.tolist()))
    ob.data.materials.append(mat_obj)
    ob.pass_index = oid
    ob.visible_shadow = False   # 測定対象は影を落とさない（地面・建物の見え方を物体の有無で変えない）
    objects.append({"id": oid, "kind": "cylinder", "radius_m": R, "tilt_deg": tilt, "length_m": L,
                    "center": c.tolist(), "p0": p0.tolist(), "p1": p1.tolist()})

# 半平面の基準：水平な大きい板（下から見上げるので、4辺はどれも十分離れた1本ずつの境界）
PLATE = (2.0, 1.5, 0.02)
plate_c = (0.0, 0.0, 5.2)
oid = n_cyl + 1
bpy.ops.mesh.primitive_cube_add(size=1, location=plate_c)
ob = bpy.context.object
ob.name = "plate"
ob.scale = PLATE
ob.data.materials.append(mat_obj)
ob.pass_index = oid
ob.visible_shadow = False
hx, hy = PLATE[0] / 2, PLATE[1] / 2
corners = [[plate_c[0] + sx * hx, plate_c[1] + sy * hy, plate_c[2] - PLATE[2] / 2]
           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
objects.append({"id": oid, "kind": "plate", "size_m": list(PLATE), "center": list(plate_c), "corners": corners})

# ── ライティング・一様な空 ────────────────────────────────────────────────────
sun = bpy.data.lights.new("sun", type="SUN")
sun.energy = 1.6
sun_ob = bpy.data.objects.new("sun", sun)
sun_ob.rotation_euler = (math.radians(50), 0, math.radians(30))
bpy.context.collection.objects.link(sun_ob)
world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (*args.sky, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

# ── カメラ（base と同じ周回軌道。注視点だけ z=2.4） ────────────────────────────
cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_width = 36.0
cam_data.lens = 30.0
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam
target = np.array([0.0, 0.0, 2.4])
N = args.frames
poses = {}
for i in range(N):
    th = 2 * math.pi * i / N
    loc = np.array([8.5 * math.cos(th), 8.5 * math.sin(th), 2.2 + 1.0 * math.sin(2 * th)])
    fwd = target - loc
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    Rm = np.stack([right, up2, -fwd], axis=1)
    scene.frame_set(i + 1)
    cam.location = loc
    cam.rotation_euler = mathutils.Matrix(Rm.tolist()).to_euler()
    cam.keyframe_insert("location")
    cam.keyframe_insert("rotation_euler")
    M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = loc
    poses[f"frame_{i+1:04d}"] = M.tolist()
f_px = cam_data.lens / cam_data.sensor_width * args.res[0]
json.dump({"convention": "cam2world, Blender/OpenGL (camera looks -Z, up +Y)",
           "width": args.res[0], "height": args.res[1],
           "fx": f_px, "fy": f_px, "cx": args.res[0] / 2, "cy": args.res[1] / 2, "frames": poses},
          open(OUT / "gt" / "poses.json", "w"), indent=1)
json.dump(objects, open(OUT / "gt" / "probe_objects.json", "w"), indent=1)

# 見かけ幅の見込み（円柱ごと・全フレームで中心が画角内のとき 2R·f/深度）
W, H = args.res
width_est = {}
for o in objects:
    if o["kind"] != "cylinder":
        continue
    ws = []
    for M in poses.values():
        M = np.array(M)
        pc = M[:3, :3].T @ (np.array(o["center"]) - M[:3, 3])
        dep = -pc[2]
        if dep > 0.1:
            u, v = f_px * pc[0] / dep + W / 2, H / 2 - f_px * pc[1] / dep
            if 0 <= u < W and 0 <= v < H:
                ws.append(2 * o["radius_m"] * f_px / dep)
    width_est[o["id"]] = {"n_frames_in_view": len(ws),
                          "p50_px": round(float(np.median(ws)), 2) if ws else None}
json.dump({"frames": N, "samples": args.samples, "denoise": bool(args.denoise), "res": args.res,
           "radii_mm": args.radii_mm, "tilts_deg": args.tilts_deg, "obj_gray": args.obj_gray, "sky": args.sky,
           "target": target.tolist(), "pixel_filter": scene.cycles.pixel_filter_type,
           "filter_width": float(scene.cycles.filter_width), "width_est": width_est},
          open(OUT / "gt" / "scene_params.json", "w"), indent=1)
for o in objects:
    if o["kind"] == "cylinder":
        e = width_est[o["id"]]
        print(f"[probe] obj{o['id']:02d} r={o['radius_m']*1000:6.1f}mm tilt={o['tilt_deg']:4.0f}° "
              f"見かけ幅p50={e['p50_px']}px 画角内={e['n_frames_in_view']}/{N}")

# ── 出力パス（RGB・物体ごとのマスク・深度） ─────────────────────────────────────
vl = scene.view_layers[0]
vl.use_pass_object_index = True
vl.use_pass_z = True
scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()
rl = nt.nodes.new("CompositorNodeRLayers")
out_mask = nt.nodes.new("CompositorNodeOutputFile")
out_mask.base_path = str(OUT / "gt" / "mask_obj")
out_mask.format.file_format = "PNG"
out_mask.format.color_mode = "BW"
out_mask.file_slots.clear()
for o in objects:
    idm = nt.nodes.new("CompositorNodeIDMask")
    idm.index = o["id"]
    idm.use_antialiasing = True
    nt.links.new(rl.outputs["IndexOB"], idm.inputs["ID value"])
    out_mask.file_slots.new(f"obj{o['id']:02d}/")
    nt.links.new(idm.outputs["Alpha"], out_mask.inputs[len(out_mask.inputs) - 1])
out_depth = nt.nodes.new("CompositorNodeOutputFile")
out_depth.base_path = str(OUT / "gt" / "depth")
out_depth.file_slots[0].path = ""
out_depth.format.file_format = "OPEN_EXR"
out_depth.format.color_depth = "32"
nt.links.new(rl.outputs["Depth"], out_depth.inputs[0])
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], comp.inputs["Image"])

scene.render.filepath = str(OUT / "input" / "frame_")
scene.frame_start = 1
scene.frame_end = 1 if args.test_one else N
t0 = time.time()
bpy.ops.render.render(animation=True)
print(f"[probe] rendered {scene.frame_end} frame(s) in {time.time()-t0:.1f}s")
