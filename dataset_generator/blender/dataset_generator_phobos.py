"""Gerador de dataset COCO-pose usando o modelo NAO Phobos de alta qualidade.

Variante de dataset_generator_blender.py para o rig Phobos (`nao_blender.blend`),
onde cada link é um armature-objeto encadeado por parenting (ver
nao_poser_phobos.NaoPoserPhobos). Reaproveita câmera, projeção, visibilidade,
randomização e escrita COCO sem alterações.

Uso:
    blender --background nao_blender.blend --python dataset_generator_phobos.py \
        -- --num 100 --start 0

Saída (idêntica ao pipeline original):
    output/images/{train,val,test}/nao_NNNNN.png
    output/annotations/person_keypoints_{split}.json
"""
import sys
import math
import numpy as np
from pathlib import Path

import bpy
import mathutils

HERE      = Path(__file__).parent
PROJ_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ_ROOT / "src"))

from nao_poser_phobos         import (NaoPoserPhobos, hide_metadata_collections,
                                      COCO_KEYPOINTS_ORDER)
from nao_texture             import NaoTextureRandomizer
from blender_camera           import build_K, cam_to_world as get_c2w, BLENDER_AXIS_REMAP
from blender_scene_randomizer import SceneRandomizer
from nao_coco_pose.config        import load_dataset, load_randomization, make_rng
from nao_coco_pose.randomization import DomainRandomizer
from nao_coco_pose.coco_writer   import CocoDatasetBuilder
from nao_coco_pose.visibility    import bbox_from_keypoints, V_ABSENT, V_OCCLUDED, V_VISIBLE, in_frame
from nao_coco_pose.projection    import world_to_pixels

CFG_DIR = PROJ_ROOT / "config"

# ---------------------------------------------------------------------------
# Argumentos CLI (passados após '--' ao blender)
# ---------------------------------------------------------------------------
import argparse as _ap
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_parser = _ap.ArgumentParser()
_parser.add_argument("--start", type=int, default=0)
_parser.add_argument("--num",   type=int, default=None)
_parser.add_argument("--out",   type=str, default=None,
                     help="Sobrescreve output_dir do YAML (útil p/ testes)")
_args = _parser.parse_args(_argv)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
dcfg = load_dataset(CFG_DIR / "dataset.yaml")
rcfg = load_randomization(CFG_DIR / "randomization.yaml")

START_INDEX = _args.start
NUM_SAMPLES = _args.num if _args.num is not None else dcfg.num_samples

rng  = make_rng(dcfg.seed + START_INDEX)
rand = DomainRandomizer(rng, rcfg)

print(f"[GEN-PHOBOS] start={START_INDEX}  num={NUM_SAMPLES}  seed={dcfg.seed + START_INDEX}")

W, H        = 640, 480
SENSOR_W_MM = 36.0
LENS_MM     = (SENSOR_W_MM / 2.0) / math.tan(math.radians(60.0 / 2.0))
OUT_ROOT    = Path(_args.out) if _args.out else PROJ_ROOT / dcfg.output_dir

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def _split_indices(n: int, splits: dict):
    idx = np.arange(n)
    np.random.default_rng(42).shuffle(idx)
    result = {}; start = 0
    names = list(splits.keys())
    for i, name in enumerate(names):
        frac = splits[name]
        end  = start + (round(frac * n) if i < len(names) - 1 else n - start)
        result[name] = set(idx[start:end].tolist())
        start = end
    return result

split_map = _split_indices(NUM_SAMPLES, dcfg.splits)

ann_dir = OUT_ROOT / "annotations"
split_builders: dict[str, CocoDatasetBuilder] = {}
for s in dcfg.splits:
    b = CocoDatasetBuilder(f"NAO synthetic COCO-pose (Blender/Phobos) — {s}")
    b.load_existing(ann_dir / f"person_keypoints_{s}.json")
    split_builders[s] = b
    if b.images:
        print(f"[GEN-PHOBOS] {s}: carregou {len(b.images)} amostras existentes")

# ---------------------------------------------------------------------------
# Setup do rig Phobos
# ---------------------------------------------------------------------------
hide_metadata_collections()
poser = NaoPoserPhobos()
if not poser._joint_obj:
    sys.exit("[GEN-PHOBOS] nenhuma junta encontrada. Rode com nao_blender.blend.")

# Randomizador da cor de time do NAO (troca a textura por amostra).
tex_rand = NaoTextureRandomizer()
print(f"[GEN-PHOBOS] variantes de cor: {tex_rand.names}")

# Meshes do robô, capturados ANTES de criar chão/cena/oclusores. Usados para o
# bbox de enquadramento e aterramento (exclui Ground e oclusores da SceneRandomizer).
ROBOT_MESHES = [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render]
print(f"[GEN-PHOBOS] {len(ROBOT_MESHES)} meshes do robô")

def _apply_base_tilt(pitch_deg: float, roll_deg: float) -> None:
    """Inclina o tronco rotacionando base_link em torno de Y (pitch) e X (roll).

    Frame NAO == mundo Blender (X=frente, Y=esq, Z=cima): pitch inclina p/ frente
    ou trás, roll inclina lateralmente. Reescrito a cada amostra.
    """
    base = bpy.data.objects.get("base_link")
    if base is None:
        return
    base.rotation_mode = "QUATERNION"
    base.rotation_quaternion = (
        mathutils.Quaternion((0.0, 1.0, 0.0), math.radians(pitch_deg))
        @ mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(roll_deg))
    )

def _ground_and_target():
    """Assenta o robô (menor ponto em z=0) e devolve (alvo, raio_bbox).

    Alvo = centro XY do bounding box do robô e meia-altura em Z. raio_bbox = raio
    da esfera envolvente (metade da diagonal), usado para enquadrar a câmera.
    Deve ser chamado após aplicar a pose das juntas e a inclinação do tronco.
    """
    bpy.context.view_layer.update()
    base = bpy.data.objects.get("base_link")
    dg = bpy.context.evaluated_depsgraph_get()
    lo = mathutils.Vector(( 1e9,  1e9,  1e9))
    hi = mathutils.Vector((-1e9, -1e9, -1e9))
    for o in ROBOT_MESHES:
        ev = o.evaluated_get(dg)
        for corner in ev.bound_box:
            w = o.matrix_world @ mathutils.Vector(corner)
            for k in range(3):
                lo[k] = min(lo[k], w[k]); hi[k] = max(hi[k], w[k])
    base.location.z += -lo[2]
    bpy.context.view_layer.update()
    # Após reassentar: Z vai de 0 a (hi.z - lo.z); XY inalterado pelo shift.
    center = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (hi[2] - lo[2])])
    bbox_radius = 0.5 * float((hi - lo).length)
    return center, bbox_radius

def _fit_distance(bbox_radius: float) -> float:
    """Distância em que a esfera de raio bbox_radius encaixa no menor semi-FOV."""
    v_half = math.atan((0.5 * cam_data.sensor_width * H / W) / cam_data.lens)
    return bbox_radius / math.sin(v_half)

# Alvo inicial (recalculado por amostra no loop).
poser.apply_pose({})
robot_base, _ = _ground_and_target()
print(f"[GEN-PHOBOS] robot_base(inicial)={robot_base.round(3).tolist()}")

def _setup_scene():
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_obj  = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.location = (3, -2, 4); sun_obj.rotation_euler = (0.6, 0.2, -0.8)

    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 60.0
    fill_obj = bpy.data.objects.new("Fill", fill_data)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (-2, -1, 2)

    cam_data = bpy.data.cameras.new("GenCam")
    cam_data.lens = LENS_MM; cam_data.sensor_width = SENSOR_W_MM
    cam_obj = bpy.data.objects.new("GenCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world

    sc = bpy.context.scene
    sc.render.engine                     = "CYCLES"
    sc.cycles.samples                    = 64
    sc.cycles.use_denoising              = True
    sc.cycles.denoiser                   = "OPENIMAGEDENOISE"
    sc.render.resolution_x               = W
    sc.render.resolution_y               = H
    sc.render.image_settings.file_format = "PNG"
    return ground, sun_obj, cam_obj, cam_data, world

ground, sun_obj, cam_obj, cam_data, world = _setup_scene()
scene_rand = SceneRandomizer(rng, ground, world)

def _apply_lighting(p: dict):
    sun_obj.data.energy = p["intensity"] * 3.0
    c = p["color"]
    sun_obj.data.color  = (c[0], c[1], c[2])

def _place_camera(cam_pos, aim_pt):
    cam_obj.location = mathutils.Vector(cam_pos)
    cam_obj.rotation_euler = (
        mathutils.Vector(aim_pt) - mathutils.Vector(cam_pos)
    ).to_track_quat("-Z", "Y").to_euler()

def _apply_dof(cam_pos, target):
    focus_dist = float(np.linalg.norm(np.array(cam_pos) - np.asarray(target)))
    cam_data.dof.use_dof        = True
    cam_data.dof.focus_distance = focus_dist
    cam_data.dof.aperture_fstop = float(rng.uniform(2.8, 11.0))

# ---------------------------------------------------------------------------
# Visibilidade por ray cast
# ---------------------------------------------------------------------------
_RAY_TOL = 0.12

def _visibility_ray(cam_pos, kp_world, uv, depth, depsgraph) -> int:
    if depth <= 0 or not in_frame(uv, W, H):
        return V_ABSENT
    origin    = mathutils.Vector((float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
    target    = mathutils.Vector((float(kp_world[0]), float(kp_world[1]), float(kp_world[2])))
    direction = (target - origin).normalized()
    max_dist  = (target - origin).length - _RAY_TOL
    if max_dist <= 0:
        return V_VISIBLE
    hit, *_ = bpy.context.scene.ray_cast(depsgraph, origin, direction, distance=max_dist)
    return V_OCCLUDED if hit else V_VISIBLE

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
sc = bpy.context.scene

for i in range(NUM_SAMPLES):
    frame_idx = START_INDEX + i
    split   = next(s for s, idx_set in split_map.items() if i in idx_set)
    img_dir = OUT_ROOT / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    img_name = f"{dcfg.image_prefix}{frame_idx:05d}.{dcfg.image_format}"
    img_path = img_dir / img_name
    ann_name = f"{split}/{img_name}"

    rp = rand.sample_robot_pose()
    poser.apply_pose(rp.joints)
    _apply_base_tilt(*rp.tilt_deg)
    cam_target, bbox_r = _ground_and_target()
    color = tex_rand.randomize(rng)

    cam_pos, aim_pt = rand.sample_camera_pose(cam_target, _fit_distance(bbox_r))
    _place_camera(cam_pos, aim_pt)
    _apply_dof(cam_pos, cam_target)
    _apply_lighting(rand.sample_lighting())

    scene_rand.randomize(cam_pos)
    bg = rand.sample_background()
    if bg is not None:
        scene_rand.set_background(str(PROJ_ROOT / bg))

    bpy.context.view_layer.update()

    sc.render.filepath = str(img_path)
    bpy.ops.render.render(write_still=True)

    K      = build_K(cam_data, W, H)
    C2W    = get_c2w(cam_obj)
    kps_3d = poser.get_keypoints_world()
    pts    = [kps_3d.get(kp) or (0.0, 0.0, 0.0) for kp in COCO_KEYPOINTS_ORDER]
    uv_list, depths = world_to_pixels(pts, C2W, K, axis_remap=BLENDER_AXIS_REMAP)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    flags = [
        _visibility_ray(cam_pos, kps_3d.get(kp) or (0, 0, 0), uv, d, depsgraph)
        for kp, uv, d in zip(COCO_KEYPOINTS_ORDER, uv_list, depths)
    ]
    keypoints_xyv = [(float(uv[0]), float(uv[1]), f) for uv, f in zip(uv_list, flags)]
    bbox, area    = bbox_from_keypoints(uv_list, flags, W, H)

    if area >= 1.0:
        split_builders[split].add_sample(ann_name, W, H, keypoints_xyv, bbox, area)

    n_vis = sum(f == V_VISIBLE  for f in flags)
    n_occ = sum(f == V_OCCLUDED for f in flags)
    print(f"[GEN-PHOBOS] {i+1}/{NUM_SAMPLES}  #{frame_idx}  {split}  {(rp.category or '-'):7s} {color:11s} vis={n_vis}/17 occ={n_occ}/17  {img_name}")

# ---------------------------------------------------------------------------
# Salva JSONs
# ---------------------------------------------------------------------------
ann_dir.mkdir(parents=True, exist_ok=True)
for split, builder in split_builders.items():
    if not builder.images:
        continue
    out_json = ann_dir / f"person_keypoints_{split}.json"
    builder.save(out_json)
    print(f"[GEN-PHOBOS] {split}: {len(builder.images)} imagens → {out_json}")

scene_rand.cleanup()
print("\n[GEN-PHOBOS] Concluído.")
