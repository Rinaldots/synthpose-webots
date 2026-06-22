"""Domain randomization: pose do robô, câmera e ambiente.

A amostragem (sample_*) é pura/testável; a aplicação acontece no controlador.
Toda aleatoriedade vem de um numpy.random.Generator com semente controlada,
para o dataset ser reprodutível.
"""
from __future__ import annotations

import numpy as np


class DomainRandomizer:
    def __init__(self, rng: np.random.Generator, cfg):
        self.rng = rng
        self.cfg = cfg   # RandomizationConfig

    # ----- pose das juntas -----
    def sample_pose(self) -> dict:
        """{nome_junta: ângulo_rad} dentro dos limites, com escala opcional.

        pose_range_scale em (0, 1] reduz a amplitude em torno do meio da faixa,
        o que tende a gerar poses mais naturais do que uniforme no range cheio.
        """
        angles = {}
        scale = float(self.cfg.pose_range_scale)
        for joint, (lo, hi) in self.cfg.joint_limits.items():
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * scale
            angles[joint] = float(self.rng.uniform(mid - half, mid + half))
        return angles

    # ----- câmera (esférica, olhando para o alvo) -----
    def sample_camera_pose(self, target):
        """Amostra posição da câmera numa casca esférica ao redor do alvo.

        Returns (pos, aim_point): pos é a posição da câmera; aim_point é o
        ponto de mira (tronco do robô, com offset vertical configurável).
        A esfera é centrada na base do robô (target), mas a câmera aponta
        para o tronco, garantindo que o corpo inteiro fique no enquadramento.
        """
        c = self.cfg.camera
        r = self.rng.uniform(c["radius_min"], c["radius_max"])
        az = self.rng.uniform(*np.deg2rad(c["azimuth_deg"]))
        el = self.rng.uniform(*np.deg2rad(c["elevation_deg"]))
        target = np.asarray(target, dtype=float)
        offset = r * np.array([np.cos(el) * np.cos(az),
                               np.cos(el) * np.sin(az),
                               np.sin(el)])
        aim_point = target + np.array([0.0, 0.0, float(c.get("target_offset_z", 0.0))])
        return target + offset, aim_point

    # ----- ambiente -----
    def sample_lighting(self) -> dict:
        l = self.cfg.lighting
        return {
            "intensity": float(self.rng.uniform(*l["intensity"])),
            "color": [float(self.rng.uniform(*l["color_range"])) for _ in range(3)],
        }

    def sample_background(self):
        bgs = self.cfg.backgrounds
        return str(self.rng.choice(bgs)) if bgs else None
