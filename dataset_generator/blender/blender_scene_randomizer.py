"""Randomização procedural do ambiente Blender a cada frame.

Cenários (escolhidos aleatoriamente por randomize()):
  office       — mesa, cadeira, estante com livros, monitor
  living_room  — sofá, mesa de centro, rack + TV, tapete
  warehouse    — prateleiras metálicas, caixas, paletes
  plain        — primitivas coloridas aleatórias
"""
import math
import colorsys

import bpy

SCENE_TYPES = ("office", "living_room", "warehouse", "plain")


def _hsv(h, s, v, a=1.0):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, a)


class SceneRandomizer:

    def __init__(self, rng, ground_obj, world_obj):
        self.rng         = rng
        self._ground     = ground_obj
        self._world      = world_obj
        self._scene_objs: list = []
        self._floor_mat  = self._init_floor_mat()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def randomize(self, cam_pos: tuple) -> None:
        """Reconstrói a cena completa a cada frame."""
        self._clear_scene()
        self._randomize_sky()
        scene = str(self.rng.choice(SCENE_TYPES))
        if scene == "plain":
            self._build_plain(cam_pos)
        else:
            getattr(self, f"_build_{scene}")()
        self._add_occluder(cam_pos)

    def cleanup(self) -> None:
        self._clear_scene()

    # ------------------------------------------------------------------
    # Gestão de memória
    # ------------------------------------------------------------------

    def _clear_scene(self):
        mats:   set = set()
        meshes: set = set()
        for obj in self._scene_objs:
            if obj.type == "MESH":
                meshes.add(obj.data)
                for m in obj.data.materials:
                    if m: mats.add(m)
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in meshes:
            if mesh.users == 0: bpy.data.meshes.remove(mesh)
        for mat in mats:
            if mat.users  == 0: bpy.data.materials.remove(mat)
        self._scene_objs.clear()

    # ------------------------------------------------------------------
    # Primitivas
    # ------------------------------------------------------------------

    def _add_box(self, loc, scale, rot_z=0.0, color=None, roughness=0.7, metallic=0.0):
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        obj = bpy.context.active_object
        obj.scale = scale
        if rot_z: obj.rotation_euler[2] = rot_z
        self._mat(obj, color, roughness, metallic)
        self._scene_objs.append(obj)
        return obj

    def _add_cyl(self, loc, radius, depth, rot_z=0.0, color=None, roughness=0.5, metallic=0.0):
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=loc)
        obj = bpy.context.active_object
        if rot_z: obj.rotation_euler[2] = rot_z
        self._mat(obj, color, roughness, metallic)
        self._scene_objs.append(obj)
        return obj

    def _mat(self, obj, color, roughness, metallic=0.0):
        r   = self.rng
        mat = bpy.data.materials.new(f"_M{len(self._scene_objs)}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        if color is None:
            color = _hsv(float(r.uniform(0,1)), float(r.uniform(0.05,0.5)), float(r.uniform(0.2,0.8)))
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value  = float(roughness)
        bsdf.inputs["Metallic"].default_value   = float(metallic)
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    # ------------------------------------------------------------------
    # Chão — texturas procedurais
    # ------------------------------------------------------------------

    def _init_floor_mat(self):
        mat = bpy.data.materials.new("FloorRand")
        mat.use_nodes = True
        if self._ground.data.materials:
            self._ground.data.materials[0] = mat
        else:
            self._ground.data.materials.append(mat)
        return mat

    def _set_floor(self, floor_type: str | None = None) -> None:
        r = self.rng
        mat = self._floor_mat
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out  = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs[0], out.inputs["Surface"])

        coord   = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        links.new(coord.outputs["Generated"], mapping.inputs["Vector"])

        floor_type = floor_type or str(r.choice(["wood", "wood", "tile", "concrete", "carpet"]))

        if floor_type == "wood":
            mapping.inputs["Scale"].default_value = (
                float(r.uniform(3, 8)), float(r.uniform(0.5, 1.5)), 1.0)
            mapping.inputs["Rotation"].default_value = (0, 0, float(r.uniform(0, math.pi / 2)))
            wave = nodes.new("ShaderNodeTexWave")
            wave.wave_type = "BANDS"
            wave.bands_direction = "Y"
            wave.inputs["Scale"].default_value        = float(r.uniform(3, 8))
            wave.inputs["Distortion"].default_value   = float(r.uniform(1, 4))
            wave.inputs["Detail"].default_value       = float(r.uniform(2, 6))
            wave.inputs["Detail Scale"].default_value = float(r.uniform(1, 3))
            links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
            ramp = nodes.new("ShaderNodeValToRGB")
            dark = _hsv(float(r.uniform(0.06,0.12)), float(r.uniform(0.4,0.7)), float(r.uniform(0.15,0.35)))
            lite = _hsv(float(r.uniform(0.06,0.12)), float(r.uniform(0.3,0.6)), float(r.uniform(0.45,0.65)))
            ramp.color_ramp.elements[0].color = dark
            ramp.color_ramp.elements[1].color = lite
            links.new(wave.outputs["Color"], ramp.inputs["Fac"])
            links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = float(r.uniform(0.3, 0.6))

        elif floor_type == "tile":
            scale = float(r.uniform(5, 15))
            mapping.inputs["Scale"].default_value = (scale, scale, 1.0)
            checker = nodes.new("ShaderNodeTexChecker")
            c1 = _hsv(float(r.uniform(0,1)), float(r.uniform(0,0.15)), float(r.uniform(0.6,0.95)))
            c2 = _hsv(float(r.uniform(0,1)), float(r.uniform(0,0.15)), float(r.uniform(0.3,0.7)))
            checker.inputs["Color1"].default_value = c1
            checker.inputs["Color2"].default_value = c2
            checker.inputs["Scale"].default_value  = 1.0
            links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
            links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = float(r.uniform(0.05, 0.25))

        elif floor_type == "concrete":
            mapping.inputs["Scale"].default_value = (
                float(r.uniform(1, 3)), float(r.uniform(1, 3)), 1.0)
            noise = nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value     = float(r.uniform(5, 20))
            noise.inputs["Detail"].default_value    = float(r.uniform(4, 10))
            noise.inputs["Roughness"].default_value = float(r.uniform(0.5, 0.8))
            links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
            ramp = nodes.new("ShaderNodeValToRGB")
            g0 = float(r.uniform(0.25, 0.45))
            g1 = float(r.uniform(0.55, 0.75))
            ramp.color_ramp.elements[0].color = (g0, g0, g0, 1)
            ramp.color_ramp.elements[1].color = (g1, g1, g1, 1)
            links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
            links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = float(r.uniform(0.75, 0.95))

        else:  # carpet
            scale = float(r.uniform(8, 20))
            mapping.inputs["Scale"].default_value = (scale, scale, 1.0)
            noise = nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value  = float(r.uniform(20, 60))
            noise.inputs["Detail"].default_value = float(r.uniform(6, 12))
            links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
            ramp = nodes.new("ShaderNodeValToRGB")
            hue = float(r.uniform(0, 1))
            ramp.color_ramp.elements[0].color = _hsv(hue, float(r.uniform(0.3,0.7)), float(r.uniform(0.15,0.45)))
            ramp.color_ramp.elements[1].color = _hsv(hue, float(r.uniform(0.2,0.6)), float(r.uniform(0.35,0.65)))
            links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
            links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = 1.0

    # ------------------------------------------------------------------
    # Céu / HDRI
    # ------------------------------------------------------------------

    def set_background(self, img_path: str | None) -> None:
        """Define textura de ambiente a partir de um arquivo (HDRI, JPG, PNG).

        Se img_path for None, reverte para céu sólido aleatório.
        """
        nodes = self._world.node_tree.nodes
        links = self._world.node_tree.links
        nodes.clear()

        bg_node  = nodes.new("ShaderNodeBackground")
        out_node = nodes.new("ShaderNodeOutputWorld")

        if img_path is not None:
            from pathlib import Path as _Path
            p = _Path(img_path)
            if p.exists():
                tex_node  = nodes.new("ShaderNodeTexEnvironment")
                img = bpy.data.images.load(str(p), check_existing=True)
                tex_node.image = img

                # Rotação aleatória em Z para variar a orientação do HDRI
                coord_node   = nodes.new("ShaderNodeTexCoord")
                mapping_node = nodes.new("ShaderNodeMapping")
                mapping_node.inputs["Rotation"].default_value = (
                    0.0, 0.0, float(self.rng.uniform(0, 2 * math.pi)))
                links.new(coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
                links.new(mapping_node.outputs["Vector"],  tex_node.inputs["Vector"])
                links.new(tex_node.outputs["Color"],       bg_node.inputs["Color"])
                bg_node.inputs["Strength"].default_value = float(self.rng.uniform(0.6, 1.2))
                links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])
                return
            else:
                print(f"[SCENE] background não encontrado: {img_path}")

        # fallback: cor sólida
        rr = self.rng
        bg_node.inputs["Color"].default_value    = _hsv(float(rr.uniform(0,1)), float(rr.uniform(0,0.15)), float(rr.uniform(0.3,0.85)))
        bg_node.inputs["Strength"].default_value = float(rr.uniform(0.2, 0.6))
        links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    def _randomize_sky(self):
        self.set_background(None)

    # ------------------------------------------------------------------
    # Oclusores — objetos entre câmera e robô
    # ------------------------------------------------------------------

    def _add_occluder(self, cam_pos) -> None:
        """Coloca 0–2 objetos finos perto do caminho câmera→robô, para oclusão
        *parcial leve*.

        Enviesado para nenhum/poucos oclusores; postes finos deslocados
        lateralmente (perpendicular ao raio câmera→origem) para roçar a borda do
        robô em vez de cobrir o centro — evita oclusão total e reduz a parcial.
        """
        r = self.rng
        # 70% nenhum, 24% um, 6% dois.
        n = int(r.choice([0, 1, 2], p=[0.70, 0.24, 0.06]))
        if n == 0:
            return
        cx, cy = float(cam_pos[0]), float(cam_pos[1])
        # Direção lateral (perpendicular ao raio câmera→origem), normalizada.
        ln = math.hypot(cx, cy) or 1.0
        lx, ly = -cy / ln, cx / ln
        for _ in range(n):
            frac = float(r.uniform(0.35, 0.70))            # mais perto do robô
            side = float(r.uniform(0.16, 0.42)) * float(r.choice([-1.0, 1.0]))
            ox   = cx * frac + lx * side
            oy   = cy * frac + ly * side
            h    = float(r.uniform(0.35, 1.05))
            w    = float(r.uniform(0.05, 0.13))            # mais fino
            if bool(r.integers(0, 2)):
                self._add_box((ox, oy, h / 2), (w / 2, w / 2, h / 2),
                              rot_z=float(r.uniform(0, math.pi)))
            else:
                self._add_cyl((ox, oy, h / 2), w / 2, h)

    # ------------------------------------------------------------------
    # Paredes + teto
    # ------------------------------------------------------------------

    ROOM_W = 7.0; ROOM_D = 7.0; ROOM_H = 2.8

    def _build_room(self, wall_color=None, roughness=0.8):
        r = self.rng
        if wall_color is None:
            wall_color = _hsv(float(r.uniform(0,1)), float(r.uniform(0,0.2)), float(r.uniform(0.55,0.92)))
        t = 0.15; W, D, H = self.ROOM_W, self.ROOM_D, self.ROOM_H
        for loc, sc in [
            ((0,    D/2,  H/2), (W/2, t/2,  H/2)),
            ((0,   -D/2,  H/2), (W/2, t/2,  H/2)),
            ((-W/2, 0,    H/2), (t/2, D/2,  H/2)),
            (( W/2, 0,    H/2), (t/2, D/2,  H/2)),
            ((0,    0,    H  ), (W/2, D/2,  t/2)),
        ]:
            self._add_box(loc, sc, color=wall_color, roughness=roughness)

    # ------------------------------------------------------------------
    # Escritório
    # ------------------------------------------------------------------

    def _build_office(self):
        r = self.rng
        self._set_floor(str(r.choice(["wood", "tile"])))
        self._build_room()
        wood  = _hsv(float(r.uniform(0.06,0.12)), float(r.uniform(0.3,0.6)), float(r.uniform(0.3,0.6)))
        dark  = _hsv(0.6, 0.05, 0.12)
        chair = _hsv(0.0, 0.0, float(r.uniform(0.2,0.55)))
        sx = float(r.choice([-1,1])) * float(r.uniform(1.2,2.0))
        sy = float(r.uniform(1.5,2.8))
        self._add_box((sx,sy,0.74),(0.60,0.35,0.02),color=wood,roughness=0.4)
        for ox,oy in [(0.55,0.3),(-0.55,0.3),(0.55,-0.3),(-0.55,-0.3)]:
            self._add_box((sx+ox*0.95,sy+oy*0.85,0.36),(0.02,0.02,0.36),color=dark,roughness=0.3)
        self._add_box((sx,sy+0.20,0.92),(0.27,0.02,0.20),color=dark,roughness=0.15)
        self._add_box((sx,sy+0.22,0.76),(0.04,0.02,0.12),color=dark,roughness=0.3)
        cx=sx+float(r.choice([-1,1]))*0.55; cy=sy-0.65
        self._add_box((cx,cy,0.47),(0.22,0.22,0.04),color=chair,roughness=0.9)
        self._add_box((cx,cy+0.20,0.66),(0.22,0.04,0.22),color=chair,roughness=0.9)
        self._add_cyl((cx,cy,0.24),0.025,0.48,color=dark,roughness=0.2,metallic=0.8)
        bx=-sx*0.7+float(r.uniform(-0.3,0.3))
        self._add_box((bx,3.0,1.1),(0.20,0.12,1.10),color=wood,roughness=0.5)
        for sh_z in [0.3,0.75,1.20,1.65,2.05]:
            self._add_box((bx,3.06,sh_z),(0.20,0.13,0.01),color=wood,roughness=0.5)
        for j in range(int(r.integers(4,10))):
            bw=float(r.uniform(0.02,0.04)); bh=float(r.uniform(0.12,0.22))
            self._add_box((bx-0.16+j*0.038,2.96,0.32+bh/2),(bw,0.10,bh/2),
                          color=_hsv(float(r.uniform(0,1)),0.75,0.65),roughness=0.85)
        self._add_cyl((sx+float(r.uniform(-0.4,0.4)),sy-1.0,0.18),0.09,0.36,
                      color=_hsv(0.0,0.0,0.3),roughness=0.6)

    # ------------------------------------------------------------------
    # Sala de estar
    # ------------------------------------------------------------------

    def _build_living_room(self):
        r = self.rng
        self._set_floor(str(r.choice(["wood", "carpet", "tile"])))
        self._build_room(wall_color=_hsv(float(r.uniform(0,1)),float(r.uniform(0,0.12)),float(r.uniform(0.72,0.95))))
        sofa_col=_hsv(float(r.uniform(0,1)),float(r.uniform(0.2,0.6)),float(r.uniform(0.2,0.55)))
        wood=_hsv(float(r.uniform(0.06,0.12)),float(r.uniform(0.3,0.6)),float(r.uniform(0.3,0.6)))
        sy=float(r.uniform(2.0,3.0))
        self._add_box((0,sy,0.22),(0.90,0.45,0.22),color=sofa_col,roughness=0.95)
        self._add_box((0,sy+0.40,0.52),(0.90,0.10,0.35),color=sofa_col,roughness=0.95)
        for ox in [-1.0,1.0]:
            self._add_box((ox*0.92,sy,0.37),(0.08,0.45,0.37),color=sofa_col,roughness=0.95)
        for ox in [-0.35,0.0,0.35]:
            self._add_box((ox,sy+0.25,0.50),(0.14,0.06,0.14),
                          color=_hsv(float(r.uniform(0,1)),0.5,0.7),roughness=0.9)
        self._add_box((0,sy-1.1,0.21),(0.42,0.27,0.02),color=wood,roughness=0.4)
        for ox,oy in [(0.38,0.23),(-0.38,0.23),(0.38,-0.23),(-0.38,-0.23)]:
            self._add_box((ox*0.95,sy-1.1+oy*0.85,0.10),(0.02,0.02,0.10),color=wood,roughness=0.5)
        self._add_box((0,3.1,0.21),(0.80,0.12,0.21),color=wood,roughness=0.5)
        self._add_box((0,3.13,0.57),(0.58,0.03,0.34),color=(0.04,0.04,0.04,1),roughness=0.08)
        self._add_box((0,sy-0.9,0.006),(0.80,0.60,0.006),
                      color=_hsv(float(r.uniform(0,1)),float(r.uniform(0.3,0.7)),float(r.uniform(0.35,0.7))),roughness=1.0)
        px=float(r.choice([-1,1]))*float(r.uniform(1.5,2.5))
        self._add_cyl((px,2.5,0.22),0.12,0.44,color=(0.30,0.20,0.12,1),roughness=0.9)
        self._add_cyl((px,2.5,0.55),0.16,0.44,color=_hsv(0.30,0.6,0.35),roughness=1.0)

    # ------------------------------------------------------------------
    # Armazém
    # ------------------------------------------------------------------

    def _build_warehouse(self):
        r = self.rng
        self._set_floor("concrete")
        self._build_room(wall_color=_hsv(0.0,0.0,float(r.uniform(0.48,0.72))),roughness=0.9)
        metal=(0.58,0.60,0.62,1)
        for side_x in [-2.8,2.8]:
            for sh_z in [0.40,0.90,1.40,1.90]:
                self._add_box((side_x,0,sh_z),(0.15,1.60,0.012),color=metal,roughness=0.25,metallic=0.9)
            for py in [-1.5,-0.75,0,0.75,1.5]:
                self._add_box((side_x,py,1.15),(0.018,0.018,1.15),color=metal,roughness=0.2,metallic=0.9)
            for sh_z in [0.40,0.90,1.40]:
                for _ in range(int(r.integers(2,6))):
                    bh=float(r.uniform(0.08,0.20)); bw=float(r.uniform(0.08,0.22))
                    self._add_box((side_x,float(r.uniform(-1.4,1.4)),sh_z+bh/2+0.012),(bw/2,bw/2,bh/2),
                                  color=_hsv(float(r.uniform(0,1)),float(r.uniform(0.2,0.6)),float(r.uniform(0.3,0.7))),roughness=0.85)
        for px,py in [(1.5,2.2),(-1.5,-2.0),(0.2,2.8),(-0.8,-2.5)]:
            self._add_box((px,py,0.04),(0.30,0.25,0.04),color=(0.42,0.28,0.12,1),roughness=0.9)
            for stack in range(int(r.integers(1,4))):
                bh=float(r.uniform(0.10,0.22))
                self._add_box((px+float(r.uniform(-0.05,0.05)),py+float(r.uniform(-0.05,0.05)),0.08+stack*bh+bh/2),
                              (0.18,0.15,bh/2),color=_hsv(float(r.uniform(0,1)),0.4,0.5),roughness=0.85)
        self._add_cyl((float(r.uniform(-2.0,2.0)),float(r.uniform(1.5,2.5)),0.44),
                      0.18,0.88,color=_hsv(float(r.uniform(0,1)),0.5,0.4),roughness=0.4,metallic=0.5)

    # ------------------------------------------------------------------
    # Genérico (primitivas)
    # ------------------------------------------------------------------

    def _build_plain(self, cam_pos):
        r = self.rng
        self._set_floor()
        cam_angle = math.atan2(float(cam_pos[1]), float(cam_pos[0]))
        for _ in range(int(r.integers(2,7))):
            for __ in range(20):
                angle = float(r.uniform(0,2*math.pi))
                if abs((angle-cam_angle+math.pi)%(2*math.pi)-math.pi) > 0.70: break
            dist=float(r.uniform(0.5,2.2)); x,y=math.cos(angle)*dist,math.sin(angle)*dist
            sx=float(r.uniform(0.08,0.40)); sz=float(r.uniform(0.08,0.70))
            shape=str(r.choice(["BOX","BOX","CYL","CONE"]))
            if shape=="BOX":
                self._add_box((x,y,sz/2),(sx,float(r.uniform(0.08,0.40)),sz),rot_z=float(r.uniform(0,math.pi*2)))
            elif shape=="CYL":
                self._add_cyl((x,y,sz/2),sx/2,sz)
            else:
                bpy.ops.mesh.primitive_cone_add(radius1=sx/2,depth=sz,location=(x,y,sz/2))
                obj=bpy.context.active_object; self._mat(obj,None,0.8); self._scene_objs.append(obj)
