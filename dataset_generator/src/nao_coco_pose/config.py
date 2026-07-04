"""Carregamento de configs (YAML) e reprodutibilidade (semente)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass
class CameraConfig:
    width: int
    height: int
    fov_deg: float


@dataclass
class RandomizationConfig:
    joint_limits: dict   # {nome_junta: (lo, hi)} em radianos
    pose_range_scale: float
    camera: dict         # radius_min/max, azimuth_deg, elevation_deg, settle_steps
    lighting: dict       # intensity:[lo,hi], color_range:[lo,hi]
    backgrounds: list
    # Variedade explícita de poses. {nome: {weight, joint_overrides, torso_tilt_deg}}.
    # Vazio -> comportamento antigo (sample_pose, sem inclinação de tronco).
    pose_categories: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    num_samples: int
    splits: dict
    output_dir: str
    image_prefix: str
    image_format: str
    seed: int


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_camera(path) -> CameraConfig:
    return CameraConfig(**_load_yaml(path))


def load_randomization(path) -> RandomizationConfig:
    d = _load_yaml(path)
    d["joint_limits"] = {k: tuple(v) for k, v in d["joint_limits"].items()}
    return RandomizationConfig(**d)


def load_dataset(path) -> DatasetConfig:
    return DatasetConfig(**_load_yaml(path))


def make_rng(seed: int) -> np.random.Generator:
    """Generator do numpy + seeds globais.

    Nota: a física do Webots tem determinismo próprio; semear aqui torna
    reprodutível apenas a AMOSTRAGEM (poses, câmera, luz).
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
