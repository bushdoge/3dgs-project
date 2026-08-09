# 細線ベンチマーク用の合成シーンをBlenderで構築しレンダリングするスクリプト
# 電線（カテナリー）・フェンス（細い縦棒）・建物・地面を含むシーンを作り、
# 周回カメラで学習用RGB画像＋GT（カメラ姿勢・細線マスク・細線3D点群・深度）を出力する。
#
# シーン構造は --layout で切り替える（複数シーン検証用。既定 base は従来と完全に同一）:
#   base     電柱2本・平行な電線3本（従来シーン。既存の全実験と比較可能）
#   crossing 電柱4本（高さ2種）・2方向の電線6本が画像上で交差する（線方向の多様性・相互オクルージョン）
#   street   ジオメトリは base と完全同一のまま、r=11〜14mに高さ違いの建物リングを追加して
#            「電線の背後」を高周波の壁面で埋める（空背景という交絡の統制。--enclosedとは別物で
#             空自体は上方に残る）
# 柵の周期性は --fence で独立に切り替える（periodic=等間隔40本 / random=ランダム間隔）。
#
# 使い方:
#   blender -b -P make_scene.py -- --out <出力dir> [--frames 80] [--test-one] [--samples 64]
#                                   [--layout base|crossing|street] [--fence periodic|random]
# 出力構成:
#   <out>/input/frame_####.png      学習用RGB（COLMAP/3DGSパイプラインへの入力）
#   <out>/gt/mask_thin/####.png     細線マスク（白=電線・フェンス棒）
#   <out>/gt/depth/####.exr         深度（float EXR）
#   <out>/gt/poses.json             GTカメラ姿勢（cam2world 4x4, Blender OpenGL convention）+内部パラメータ
#   <out>/gt/wire_points.npz        細線の3D点群（world座標, ラベル: wire/fence）

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np


# ── 引数 ─────────────────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--frames", type=int, default=80)
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--res", type=int, nargs=2, default=[1024, 768])
parser.add_argument("--test-one", action="store_true", help="1フレームだけレンダして終了（動作確認用）")
parser.add_argument("--enclosed", action="store_true",
                    help="シーンをテクスチャ付き円筒壁+天井で囲い、無限遠の空を排除する"
                         "（空フローター交絡の統制版。他の要素はすべて同一）")
parser.add_argument("--wire-radius", type=float, default=0.015,
                    help="電線の半径[m]（スイープ用の統制変数。デフォルト0.015≒距離9mで3px幅。"
                         "フェンス棒は統制のため固定）")
parser.add_argument("--layout", default="base", choices=["base", "crossing", "street"],
                    help="シーン構造（複数シーン検証用。既定 base は従来シーンと完全に同一）")
parser.add_argument("--fence", default="periodic", choices=["periodic", "random"],
                    help="柵の縦棒の並び（periodic=等間隔・従来 / random=ランダム間隔。"
                         "SfM点欠乏が周期構造由来かを単独で検定するための直交フラグ）")
parser.add_argument("--fence-seed", type=int, default=0, help="--fence random の乱数種（再現用）")
args = parser.parse_args(argv)

OUT = Path(args.out)
(OUT / "input").mkdir(parents=True, exist_ok=True)
(OUT / "gt" / "mask_thin").mkdir(parents=True, exist_ok=True)
(OUT / "gt" / "depth").mkdir(parents=True, exist_ok=True)

# ── シーン初期化 ──────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x, scene.render.resolution_y = args.res
scene.render.image_settings.file_format = "PNG"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
# デフォルトのAgXは彩度・コントラストを潰しSfMに不利、Standardは白飛びするためFilmicを使う
scene.view_settings.view_transform = "Filmic"

# GPU があれば使う（無ければCPUにフォールバック）
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
print(f"[wirebench] render device: {scene.cycles.device}"
      + (f" ({prefs.compute_device_type})" if gpu_found else ""))


def make_material(name, base_color, roughness=0.8, metallic=0.0, noise_scale=None,
                  color2=None, brick=False):
    """ノイズ/ブリックのプロシージャルテクスチャ付きマテリアル（SfMの特徴点確保用）"""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    nt = mat.node_tree
    # テクスチャ座標はワールド座標のObjectでなくGeneratedだとスケールに引きずられるので
    # 常にワールド座標(Object出力)を使い、SfM向けに高周波・高コントラストにする
    coord = nt.nodes.new("ShaderNodeTexCoord")
    if brick:
        tex = nt.nodes.new("ShaderNodeTexBrick")
        tex.inputs["Scale"].default_value = 1.6
        tex.inputs["Color1"].default_value = (*base_color, 1)
        tex.inputs["Color2"].default_value = (*color2, 1)
        tex.inputs["Mortar"].default_value = (0.1, 0.1, 0.1, 1)
        tex.inputs["Mortar Size"].default_value = 0.015
        nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
        # ノイズを重ねて汚しを付ける（特徴点を増やす）
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 30.0
        noise.inputs["Detail"].default_value = 8.0
        nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 0.5
        nt.links.new(tex.outputs["Color"], mix.inputs["Color1"])
        nt.links.new(noise.outputs["Color"], mix.inputs["Color2"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    elif noise_scale is not None:
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
    else:
        bsdf.inputs["Base Color"].default_value = (*base_color, 1)
    return mat


def add_box(name, size, loc, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0], loc[1], loc[2] + size[2] / 2))
    ob = bpy.context.object
    ob.name = name
    ob.scale = (size[0], size[1], size[2])
    ob.data.materials.append(mat)
    return ob


def add_polyline_curve(name, pts, radius, mat, pass_index=0):
    """点列から太さradiusのカーブ（チューブ）を作る"""
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    sp = curve.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, (x, y, z) in zip(sp.points, pts):
        p.co = (x, y, z, 1)
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    ob = bpy.data.objects.new(name, curve)
    ob.data.materials.append(mat)
    ob.pass_index = pass_index
    bpy.context.collection.objects.link(ob)
    return ob


# ── マテリアル ────────────────────────────────────────────────────────────────
mat_ground = make_material("ground", (0.04, 0.035, 0.03), noise_scale=7.0, color2=(0.3, 0.27, 0.2))
mat_brickA = make_material("brickA", (0.25, 0.05, 0.03), noise_scale=4.0, color2=(0.65, 0.3, 0.15))
mat_brickB = make_material("brickB", (0.04, 0.08, 0.25), noise_scale=3.5, color2=(0.3, 0.45, 0.65))
mat_boxC   = make_material("boxC", (0.2, 0.15, 0.03), noise_scale=5.0, color2=(0.65, 0.55, 0.2))
mat_pole   = make_material("pole", (0.1, 0.08, 0.06), noise_scale=12.0, color2=(0.3, 0.25, 0.2))
mat_wire   = make_material("wire", (0.02, 0.02, 0.02), roughness=0.5, metallic=0.3)
mat_fence  = make_material("fence", (0.05, 0.05, 0.06), roughness=0.4, metallic=0.6)

# ── ジオメトリ ────────────────────────────────────────────────────────────────
# 地面
bpy.ops.mesh.primitive_plane_add(size=30)
bpy.context.object.data.materials.append(mat_ground)

# 建物（全レイアウト共通。street はこれに建物リングを足す）
add_box("bldgA", (3.0, 3.0, 4.0), (-3.5, 2.5, 0), mat_brickA)
add_box("bldgB", (2.2, 2.2, 3.0), (3.2, 3.0, 0), mat_brickB)
add_box("boxC", (1.5, 1.5, 1.5), (2.5, -2.8, 0), mat_boxC)

# 建物リング（--layout street: 電線の背後を高周波の壁面で埋め、空背景という交絡を外す）
# カメラ軌道 r=8.5 の外側(r=11〜14)に置くので、視線が建物に遮られて電線が隠れることはない。
if args.layout == "street":
    rng_ring = np.random.default_rng(7)
    N_RING = 20
    for i in range(N_RING):
        th = 2 * math.pi * i / N_RING + float(rng_ring.uniform(-0.05, 0.05))
        rr = float(rng_ring.uniform(11.0, 14.0))
        w, d = float(rng_ring.uniform(2.5, 4.5)), float(rng_ring.uniform(2.5, 4.5))
        h = float(rng_ring.uniform(6.0, 10.0))
        ob = add_box(f"ring{i}", (w, d, h),
                     (rr * math.cos(th), rr * math.sin(th), 0),
                     [mat_brickA, mat_brickB, mat_boxC][i % 3])
        ob.rotation_euler = (0, 0, th)   # 壁面を内向きに向ける

# 電柱・電線（レイアウト依存。poles = (x, y, 高さ)）
thin_points = []   # (x,y,z,label)  label 1=wire 2=fence-bar
WIRE_R = args.wire_radius
N_SAMP = 60


def add_catenary(name, p0, p1, sag, radius, mat):
    """2点間に放物線サグの電線を張り、GT中心線サンプルを thin_points に積む"""
    ts = np.linspace(0, 1, N_SAMP)
    pts = [(float(p0[0] + (p1[0] - p0[0]) * t),
            float(p0[1] + (p1[1] - p0[1]) * t),
            float(p0[2] + (p1[2] - p0[2]) * t - sag * 4 * t * (1 - t))) for t in ts]
    add_polyline_curve(name, pts, radius, mat, pass_index=1)
    thin_points.extend((x, y, z, 1) for x, y, z in pts)


if args.layout == "crossing":
    # 高さ2種の電柱4本。長スパン2組が画像上で交差する（3D上は約2m離れるので接触はしない）。
    # 建物3個（bldgA/B・boxC）の footprint と電線が干渉しない配置になっている。
    poles = [(-4.6, -3.2, 5.6), (4.6, 1.0, 5.6), (-1.0, 4.4, 3.6), (4.2, -2.6, 3.6)]
    spans = [((0, 1), 5.5), ((2, 3), 3.5)]   # (電柱indexの組, 最上段の高さ)
else:
    poles = [(-4.0, -1.8, 4.2), (4.0, -1.2, 4.2)]
    spans = None

for i, (px, py, ph) in enumerate(poles):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=ph, location=(px, py, ph / 2))
    bpy.context.object.name = f"pole{i}"
    bpy.context.object.data.materials.append(mat_pole)

if args.layout == "crossing":
    k = 0
    for (a, b), ztop in spans:
        for dz in (0.0, -0.3, -0.6):
            add_catenary(f"wire{k}", (poles[a][0], poles[a][1], ztop + dz),
                         (poles[b][0], poles[b][1], ztop + dz), 0.55, WIRE_R, mat_wire)
            k += 1
else:
    # 電線3本（カテナリー近似=放物線サグ。半径は--wire-radiusで統制）
    for k, (dz, dy) in enumerate([(0.0, 0.0), (-0.25, 0.12), (-0.5, -0.12)]):
        add_catenary(f"wire{k}",
                     (poles[0][0], poles[0][1] + dy, poles[0][2] + dz - 0.1),
                     (poles[1][0], poles[1][1] + dy, poles[1][2] + dz - 0.1),
                     0.55, WIRE_R, mat_wire)

# フェンス（上下レール＋細い縦棒）
FENCE_Y = -4.2
BAR_R = 0.012
fx0, fx1, fh = -3.0, 3.0, 1.2
for z in (0.15, fh):   # レール（やや太い・マスク対象外）
    add_polyline_curve(f"rail{z}", [(fx0, FENCE_Y, z), (fx1, FENCE_Y, z)], 0.03, mat_fence, pass_index=0)
n_bars = 40
if args.fence == "random":
    # 周期構造がSfMのマッチング曖昧性を生んでいる可能性の統制（最小間隔4cmで間引く）
    rng_f = np.random.default_rng(args.fence_seed)
    bar_xs = []
    for x in np.sort(rng_f.uniform(fx0, fx1, n_bars * 4)):
        if not bar_xs or x - bar_xs[-1] >= 0.04:
            bar_xs.append(float(x))
        if len(bar_xs) == n_bars + 1:
            break
else:
    bar_xs = [fx0 + (fx1 - fx0) * j / n_bars for j in range(n_bars + 1)]
for j, x in enumerate(bar_xs):
    add_polyline_curve(f"bar{j}", [(x, FENCE_Y, 0.15), (x, FENCE_Y, fh)], BAR_R, mat_fence, pass_index=1)
    for z in np.linspace(0.15, fh, 12):
        thin_points.append((x, FENCE_Y, float(z), 2))

np.savez(OUT / "gt" / "wire_points.npz",
         points=np.array([(x, y, z) for x, y, z, _ in thin_points], dtype=np.float32),
         labels=np.array([l for _, _, _, l in thin_points], dtype=np.int32))

# ── 囲い込み（--enclosed: 空フローター交絡の統制）────────────────────────────────
if args.enclosed:
    mat_wall = make_material("encl_wall", (0.25, 0.28, 0.33), noise_scale=2.5, color2=(0.6, 0.62, 0.68))
    mat_ceil = make_material("encl_ceil", (0.3, 0.28, 0.24), noise_scale=3.0, color2=(0.55, 0.52, 0.45))
    # 円筒壁（内向き法線でOK: Cyclesは両面シェーディング）
    bpy.ops.mesh.primitive_cylinder_add(radius=13.0, depth=10.0, location=(0, 0, 5.0),
                                        end_fill_type="NOTHING")   # 蓋なし（底面が地面とZファイトするため）
    wall = bpy.context.object
    wall.name = "encl_wall"
    wall.data.materials.append(mat_wall)
    # 天井（円盤）
    bpy.ops.mesh.primitive_circle_add(radius=13.2, fill_type="NGON", location=(0, 0, 9.9))
    ceil = bpy.context.object
    ceil.name = "encl_ceil"
    ceil.data.materials.append(mat_ceil)
    # 囲うと太陽光が遮られるので面光源を天井付近に追加
    for lx, ly in [(-5, -5), (5, 5), (5, -5), (-5, 5)]:
        area = bpy.data.lights.new("area", type="AREA")
        area.energy = 1500
        area.size = 4.0
        ob = bpy.data.objects.new("area", area)
        ob.location = (lx, ly, 9.0)
        bpy.context.collection.objects.link(ob)

# ── ライティング・空 ──────────────────────────────────────────────────────────
sun = bpy.data.lights.new("sun", type="SUN")
sun.energy = 1.6
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
bg.inputs["Strength"].default_value = 0.6

# ── カメラ（周回・2周期の高さ変化）─────────────────────────────────────────────
cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_width = 36.0
cam_data.lens = 30.0
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

target = np.array([0.0, 0.0, 1.4])
N = args.frames
poses = {}
for i in range(N):
    th = 2 * math.pi * i / N
    r = 8.5
    z = 2.2 + 1.0 * math.sin(2 * th)
    loc = np.array([r * math.cos(th), r * math.sin(th), z])
    fwd = target - loc
    fwd /= np.linalg.norm(fwd)
    # look-at回転（Blenderカメラは-Zが視線、+Yが上）
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=1)   # cam→world 回転
    scene.frame_set(i + 1)
    cam.location = loc
    cam.rotation_euler = __import__("mathutils").Matrix(R.tolist()).to_euler()
    cam.keyframe_insert("location")
    cam.keyframe_insert("rotation_euler")
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = loc
    poses[f"frame_{i+1:04d}"] = M.tolist()

f_px = cam_data.lens / cam_data.sensor_width * args.res[0]
json.dump({"convention": "cam2world, Blender/OpenGL (camera looks -Z, up +Y)",
           "width": args.res[0], "height": args.res[1],
           "fx": f_px, "fy": f_px, "cx": args.res[0] / 2, "cy": args.res[1] / 2,
           "frames": poses},
          open(OUT / "gt" / "poses.json", "w"), indent=1)


# ── 画面上の電線幅[px]の分布（レイアウトで被写体距離が変わるので統制変数として記録）─────
# 幅 = 2R*f/深度。全フレーム×全電線サンプルのうち画角内に入るものだけで集計する
# （オクルージョンは考慮しない解析値。実測は measure_wire_width.py）。
def wire_width_px_stats():
    wp = np.array([(x, y, z) for x, y, z, l in thin_points if l == 1], dtype=np.float64)
    W, H = args.res
    widths = []
    for M in poses.values():
        M = np.array(M)
        cam_pts = (M[:3, :3].T @ (wp - M[:3, 3]).T).T              # world → camera
        depth = -cam_pts[:, 2]                                     # 視線は -Z
        ok = depth > 0.1
        u = f_px * cam_pts[:, 0] / np.where(ok, depth, 1) + W / 2
        v = H / 2 - f_px * cam_pts[:, 1] / np.where(ok, depth, 1)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if ok.any():
            widths.append(2 * WIRE_R * f_px / depth[ok])
    if not widths:
        return {}
    w = np.concatenate(widths)
    return {"p05": round(float(np.percentile(w, 5)), 2), "p50": round(float(np.median(w)), 2),
            "p95": round(float(np.percentile(w, 95)), 2), "mean": round(float(w.mean()), 2),
            "min": round(float(w.min()), 2), "max": round(float(w.max()), 2),
            "n_samples": int(w.size)}


# シーンの統制変数を記録（スイープ結果の集計で使う）
width_px = wire_width_px_stats()
json.dump({"wire_radius": WIRE_R, "bar_radius": BAR_R, "frames": args.frames,
           "samples": args.samples, "enclosed": bool(args.enclosed),
           "res": args.res, "pole_positions": [[px, py] for px, py, _ in poles],
           "layout": args.layout, "fence": args.fence, "fence_seed": args.fence_seed,
           "traj": "circle", "n_wires": sum(1 for _, _, _, l in thin_points if l == 1) // N_SAMP,
           "n_bars": len(bar_xs), "wire_width_px": width_px},
          open(OUT / "gt" / "scene_params.json", "w"), indent=1)
print(f"[wirebench] layout={args.layout} fence={args.fence} "
      f"電線幅[px] p05/p50/p95 = {width_px.get('p05')}/{width_px.get('p50')}/{width_px.get('p95')}")

# ── 出力パス（RGB・細線マスク・深度）────────────────────────────────────────────
view_layer = scene.view_layers[0]
view_layer.use_pass_object_index = True
view_layer.use_pass_z = True
scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()
rl = nt.nodes.new("CompositorNodeRLayers")

idmask = nt.nodes.new("CompositorNodeIDMask")
idmask.index = 1
idmask.use_antialiasing = True
nt.links.new(rl.outputs["IndexOB"], idmask.inputs["ID value"])
out_mask = nt.nodes.new("CompositorNodeOutputFile")
out_mask.base_path = str(OUT / "gt" / "mask_thin")
out_mask.file_slots[0].path = ""
out_mask.format.file_format = "PNG"
out_mask.format.color_mode = "BW"
nt.links.new(idmask.outputs["Alpha"], out_mask.inputs[0])

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

# ── レンダリング ─────────────────────────────────────────────────────────────
import time
t0 = time.time()
bpy.ops.render.render(animation=True)
print(f"[wirebench] rendered {scene.frame_end} frame(s) in {time.time()-t0:.1f}s "
      f"({(time.time()-t0)/scene.frame_end:.1f}s/frame, device={scene.cycles.device})")
