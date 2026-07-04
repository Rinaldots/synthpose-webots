"""Poser para o modelo NAO de alta qualidade exportado via Phobos.

Diferente de `nao_poser_blender.NaoPoserBlender` (um único armature com cadeia
de bones), este modelo (`nao_blender.blend`) tem UM armature-objeto por link,
encadeados por parenting de bone. Cada objeto carrega metadata de junta em
custom properties (`joint/name`, `joint/axis`, `joint/type`, `joint/limits/*`).

Convenção de eixos (Phobos/URDF nativo):
  X=frente, Y=esquerda, Z=cima — o mesmo "NAO frame" do CLAUDE.md, porém SEM
  a rotação -90° em Z do rig antigo. O mundo Blender coincide com o frame NAO.

Interface idêntica à de NaoPoserBlender para reaproveitar o restante do
pipeline (câmera, projeção, visibilidade, COCO):
    poser = NaoPoserPhobos()
    poser.apply_pose({"LShoulderPitch": 0.5, "HeadYaw": -0.3})
    kps = poser.get_keypoints_world()   # {coco_name: (x, y, z)} em mundo Blender

Como posa:
  Cada link-objeto tem, em repouso, rotação identidade e o eixo da junta
  (`joint/axis`) expresso no frame local (= mundo). Rotacionar a junta =
  aplicar Quaternion(axis, angle) ao objeto (pivô no próprio origin, que fica
  na junta). O parenting propaga a rotação para os filhos.
"""
from __future__ import annotations

# Offsets faciais no referencial da cabeça (frame NAO: X=frente, Y=esq, Z=cima),
# em metros, relativos à origem do objeto `Head` (junta HeadPitch).
# Reaproveitados de nao_poser_blender._HEAD_OFFSETS_NAO; calibrar pelo overlay.
# Olhos NÃO ficam aqui — são ancorados nos links dos sensores IR (COCO_TO_LINK).
_HEAD_OFFSETS_NAO = {
    "nose":       (0.058,  0.000,  0.042),
    "left_ear":   (0.010,  0.066,  0.062),
    "right_ear":  (0.010, -0.066,  0.062),
}

# Keypoint COCO -> nome do objeto-link cuja origem (matrix_world.translation)
# fornece a posição 3D do keypoint. Faciais restantes (nariz/orelhas) por offset
# a partir de Head. Olhos ancorados nos frames dos sensores IR (ficam nos olhos).
COCO_TO_LINK = {
    "left_eye":       "LInfraRed_frame",
    "right_eye":      "RInfraRed_frame",
    "left_shoulder":  "LShoulder",
    "right_shoulder": "RShoulder",
    "left_elbow":     "LElbow",
    "right_elbow":    "RElbow",
    "left_wrist":     "l_wrist",
    "right_wrist":    "r_wrist",
    "left_hip":       "LHip",
    "right_hip":      "RHip",
    "left_knee":      "LTibia",
    "right_knee":     "RTibia",
    "left_ankle":     "l_ankle",
    "right_ankle":    "r_ankle",
}

COCO_KEYPOINTS_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

_HEAD_OBJ = "Head"

# Coleções que não devem aparecer no render (metadata Phobos).
HIDDEN_COLLECTIONS = ("collision", "inertial")


class NaoPoserPhobos:
    """Posa o NAO Phobos (um armature-objeto por link) e lê keypoints COCO."""

    def __init__(self):
        import bpy
        from mathutils import Vector

        # Indexa objetos-junta por nome de junta (custom prop 'joint/name').
        self._joint_obj: dict[str, object] = {}
        self._rest_loc: dict[str, "Vector"] = {}
        self._rest_axis: dict[str, "Vector"] = {}
        for o in bpy.data.objects:
            if o.type != "ARMATURE":
                continue
            jname = o.get("joint/name")
            jtype = o.get("joint/type")
            if not jname or jtype != "revolute":
                continue
            axis = o.get("joint/axis")
            if axis is None:
                continue
            self._joint_obj[jname] = o
            self._rest_loc[jname] = o.location.copy()
            self._rest_axis[jname] = Vector((axis[0], axis[1], axis[2])).normalized()

        self._head = bpy.data.objects.get(_HEAD_OBJ)
        self._last_angles: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Posing
    # ------------------------------------------------------------------

    def reset_pose(self) -> None:
        import bpy
        from mathutils import Quaternion
        for jname, o in self._joint_obj.items():
            o.rotation_mode = "QUATERNION"
            o.rotation_quaternion = Quaternion()
            o.location = self._rest_loc[jname]
        bpy.context.view_layer.update()

    def apply_pose(self, joint_angles: dict[str, float]) -> None:
        """Aplica ângulos de junta (rad) rotacionando cada link-objeto.

        A rotação de cada objeto é independente das demais; o parenting de
        bone acumula a FK. Um único view_layer.update() no fim basta.
        """
        import bpy
        from mathutils import Quaternion

        self._last_angles = dict(joint_angles)
        self.reset_pose()

        for jname, angle in joint_angles.items():
            o = self._joint_obj.get(jname)
            if o is None or angle is None or abs(angle) < 1e-9:
                continue
            o.rotation_mode = "QUATERNION"
            o.rotation_quaternion = Quaternion(self._rest_axis[jname], float(angle))

        bpy.context.view_layer.update()

    # ------------------------------------------------------------------
    # Keypoint reading
    # ------------------------------------------------------------------

    def _head_offset_world(self, off_nao: tuple):
        """Rotaciona um offset (frame NAO) para o mundo pela orientação da cabeça."""
        from mathutils import Vector
        R_head = self._head.matrix_world.to_3x3()
        # off_nao já está em (X=frente, Y=esq, Z=cima) = frame do mundo em repouso.
        return R_head @ Vector((off_nao[0], off_nao[1], off_nao[2]))

    def get_keypoints_world(self) -> dict[str, tuple | None]:
        """{coco_name: (x, y, z)} em mundo Blender. None se link ausente."""
        import bpy
        from mathutils import Vector
        bpy.context.view_layer.update()

        out: dict[str, tuple | None] = {}
        head_pos = Vector(self._head.matrix_world.translation) if self._head else None

        for kp in COCO_KEYPOINTS_ORDER:
            try:
                if kp in _HEAD_OFFSETS_NAO:
                    out[kp] = tuple(head_pos + self._head_offset_world(_HEAD_OFFSETS_NAO[kp]))
                else:
                    obj = bpy.data.objects.get(COCO_TO_LINK[kp])
                    out[kp] = tuple(obj.matrix_world.translation) if obj else None
            except Exception as exc:  # noqa: BLE001
                print(f"[NaoPoserPhobos] erro keypoint '{kp}': {exc}")
                out[kp] = None
        return out


def hide_metadata_collections(scene=None) -> None:
    """Oculta coleções de colisão/inércia (metadata Phobos) do render."""
    import bpy
    for coll in bpy.data.collections:
        if coll.name in HIDDEN_COLLECTIONS:
            for o in coll.objects:
                o.hide_render = True
                o.hide_viewport = True
    # Objetos 'resource::*' (gizmos de junta) também não devem renderizar.
    for o in bpy.data.objects:
        if o.name.startswith("resource::"):
            o.hide_render = True
            o.hide_viewport = True
