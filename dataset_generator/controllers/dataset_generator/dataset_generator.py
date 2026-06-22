"""Controlador (Supervisor) do Webots — laço principal de geração do dataset.

Roda no robô-rig da câmera (DEF RIG em worlds/dataset.wbt), que tem poderes de
Supervisor: ele move a própria câmera, lê as juntas do NAO (DEF NAO) e aplica
as poses. Fluxo:

  configurar -> [loop N]: randomizar -> assentar -> capturar RGB
              -> ler juntas 3D -> projetar -> visibilidade/bbox -> anotar
              -> [fim] -> salvar JSON COCO

Pontos marcados como TODO dependem do ajuste fino da SUA cena.
"""
from __future__ import annotations

import sys
from pathlib import Path

# torna o pacote src/ importável quando o Webots roda este controlador
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # salvar imagens (instale no Python que o Webots usa)
from controller import Supervisor  # API do Webots

from nao_coco_pose import config as cfg
from nao_coco_pose.camera import CameraRig
from nao_coco_pose.coco_writer import CocoDatasetBuilder
from nao_coco_pose.keypoints import COCO_KEYPOINTS
from nao_coco_pose.nao_landmarks import KeypointResolver, NaoPoser, get_field  # , dump_solid_tree
from nao_coco_pose.projection import (
    look_at_axis_angle,
    rotation_to_axis_angle,
    world_to_pixels,
)
from nao_coco_pose.randomization import DomainRandomizer
from nao_coco_pose.visibility import bbox_from_keypoints, visibility_flag

# pose "de repouso" do NAO para reancorar a base a cada frame (em pé na arena)
NAO_HOME_TRANSLATION = [0.0, 0.0, 0.333]
NAO_HOME_ROTATION = [0.0, 0.0, 1.0, 0.0]
# mapa de eixos para reajustar a orientação da câmera

def main() -> None:
    conf = ROOT / "config"
    cam_cfg = cfg.load_camera(conf / "camera.yaml")
    rnd_cfg = cfg.load_randomization(conf / "randomization.yaml")
    ds_cfg = cfg.load_dataset(conf / "dataset.yaml")
    rng = cfg.make_rng(ds_cfg.seed)

    robot = Supervisor()                       # este controlador É o robô-rig
    timestep = int(robot.getBasicTimeStep())

    rig = CameraRig(robot, camera_name="camera", camera_def="CAMERA")
    rig.enable(timestep)
    rig_node = robot.getSelf()                 # para mover a câmera (o rig)

    nao_node = robot.getFromDef("NAO")
    if nao_node is None:
        raise RuntimeError("DEF NAO não encontrado na world")

    # Descomente UMA vez para inspecionar a árvore e conferir nomes de juntas:
    # dump_solid_tree(nao_node); return

    # Coleta os campos jointParameters.position das falanges para zerá-los a cada
    # resetPhysics(), evitando o warning "too low requested position < 0".
    _phalanx_pos_fields = []
    for _side in ("R", "L"):
        for _i in range(1, 9):
            _node = robot.getFromDef(f"{_side}Phalanx{_i}")
            if _node is None:
                continue
            _jp_f = _node.getField("jointParameters")
            if _jp_f is None:
                continue
            _jp = _jp_f.getSFNode()
            if _jp is None:
                continue
            _pos = _jp.getField("position")
            if _pos is not None:
                _phalanx_pos_fields.append(_pos)

    nao_translation = get_field(nao_node, "translation")
    nao_rotation = get_field(nao_node, "rotation")

    resolver = KeypointResolver(nao_node).resolve()
    poser = NaoPoser(nao_node, list(rnd_cfg.joint_limits))
    randomizer = DomainRandomizer(rng, rnd_cfg)

    builder = CocoDatasetBuilder()
    builder.info.update({
        "camera": {"width": rig.width, "height": rig.height},
        "robot": "NAO (MyNao.proto)",
        "seed": ds_cfg.seed,
        "pose_range_scale": rnd_cfg.pose_range_scale,
    })

    out_dir = ROOT / ds_cfg.output_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    K = rig.intrinsics()
    settle = int(rnd_cfg.camera.get("settle_steps", 3))

    n = 0
    while robot.step(timestep) != -1 and n < ds_cfg.num_samples:
        # 1) randomizar cena
        nao_translation.setSFVec3f(NAO_HOME_TRANSLATION)
        nao_rotation.setSFRotation(NAO_HOME_ROTATION)
        angles = randomizer.sample_pose()
        poser.apply(angles)

        # 2) re-ancora o NAO a cada passo para que a física não o tombe;
        #    avança a simulação para estabilizar a pose.
        #    resetPhysics() reseta o target interno dos motores para o valor padrão
        #    do PROTO; por isso re-aplicamos os ângulos e zeramos as falanges logo
        #    depois, antes do step(), para evitar warnings "too low/big requested pos".
        for _ in range(settle):
            nao_translation.setSFVec3f(NAO_HOME_TRANSLATION)
            nao_rotation.setSFRotation(NAO_HOME_ROTATION)
            nao_node.resetPhysics()
            poser.apply(angles)
            for _f in _phalanx_pos_fields:
                _f.setSFFloat(0.0)
            if robot.step(timestep) == -1:
                break

        # Centra a esfera na posição-home (não na física) para que o NAO em queda
        # não arraste a câmera para o chão. nao_pos_real só serve para diagnóstico.
        nao_pos_real = list(nao_node.getPosition())
        cam_pos, target = randomizer.sample_camera_pose(NAO_HOME_TRANSLATION)
        rig_node.getField("translation").setSFVec3f([float(v) for v in cam_pos])
        rig_node.getField("rotation").setSFRotation(look_at_axis_angle(cam_pos, target, up=(0, 0, 1)))
        # TODO: aplicar iluminação/fundo (randomizer.sample_lighting / .sample_background)
        if robot.step(timestep) == -1:
            break

        # 3) capturar as duas fontes do frame
        cam_pos_real = list(rig.camera_node.getPosition())

        rgb = rig.capture_rgb()                       # imagem (input)
        kps_world = resolver.get_keypoints_world()    # juntas 3D (ground truth)
        cam_to_world = rig.cam_to_world()

        if n == 0:
            nao_rot_aa = nao_node.getField("rotation").getSFRotation()  # [x,y,z,angle]
            cam_M = cam_to_world                                         # 4x4 cam->world
            cam_R = cam_M[:3, :3]
            cam_t = cam_M[:3,  3]
            _r = lambda v: [round(float(x), 4) for x in v]
            z_col = cam_R[:, 2]   # eixo +Z local da câmera (câmera olha em -Z)
            cam_opt = -z_col      # direção óptica real (para onde a câmera aponta)
            print("=" * 60)
            print("[diag] NAO  translação (world):  ", _r(nao_pos_real))
            print("[diag] NAO  rotação eixo-ângulo: ", _r(nao_rot_aa))
            print("[diag] CAM  posição (world):     ", _r(cam_pos_real))
            print("[diag] CAM  ponto de mira:       ", _r(list(target)))
            print("[diag] CAM  direção óptica (-Z): ", _r(cam_opt))
            print("[diag] CAM  matriz R (cam->world):")
            for row in cam_R:
                print("            ", _r(row))
            print("[diag] CAM  translação t (cam->world):", _r(cam_t))
            print("[diag] CAM  pose 4x4 completa (cam->world):")
            for row in cam_M:
                print("            ", _r(row))
            print("=" * 60)

        # 4) projetar 3D -> 2D na ordem COCO
        pts, present = [], []
        for name in COCO_KEYPOINTS:
            p = kps_world.get(name)
            present.append(p is not None)
            pts.append(p if p is not None else (0.0, 0.0, 0.0))
        uv, depth = world_to_pixels(pts, cam_to_world, K)

        if n == 0:
            print("[diag] keypoints frame 0 (nome: u, v, depth, flag):")
            for i, name in enumerate(COCO_KEYPOINTS):
                u, v = float(uv[i][0]), float(uv[i][1])
                d = float(depth[i])
                in_f = 0 <= u < rig.width and 0 <= v < rig.height
                print(f"  {name:20s}: u={u:7.1f} v={v:7.1f} depth={d:6.3f} {'IN' if in_f and d>0 else 'OUT'}")

        # 5) visibilidade + bbox
        flags, kp_xyv = [], []
        for i, _name in enumerate(COCO_KEYPOINTS):
            f = 0 if not present[i] else visibility_flag(
                uv[i], depth[i], rig.width, rig.height, depth_image=None
            )  # TODO: passar mapa de profundidade do RangeFinder p/ oclusão real
            flags.append(f)
            kp_xyv.append((uv[i][0], uv[i][1], f))
        bbox, area = bbox_from_keypoints(uv, flags, rig.width, rig.height)

        # 6) gravar imagem + anotação
        fname = f"{ds_cfg.image_prefix}{n:06d}.{ds_cfg.image_format}"
        cv2.imwrite(str(img_dir / fname), rgb[:, :, ::-1])  # RGB -> BGR p/ OpenCV
        builder.add_sample(fname, rig.width, rig.height, kp_xyv, bbox, area)
        n += 1

    builder.save(out_dir / "annotations" / "person_keypoints.json")
    print(f"[ok] {n} amostras salvas em {out_dir}")


if __name__ == "__main__":
    main()
