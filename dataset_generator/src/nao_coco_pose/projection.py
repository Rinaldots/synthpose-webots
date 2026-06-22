"""Projeção 3D -> 2D (modelo pinhole) e geometria de câmera.

Tudo aqui é NumPy puro, independente do Webots, portanto testável isolado.
Esta é a etapa mais propensa a bugs do pipeline: sempre confirme o resultado
sobrepondo os pontos na imagem (scripts/visualize_sample.py).
"""
from __future__ import annotations

import numpy as np

# Converte o referencial da câmera (Webots +Z forward) para o padrão CV
# (x: direita, y: baixo, z: para frente).
# Com look_at_axis_angle aplicando 180°Y, a câmera olha ao longo de +Z local:
#   z_cv = +z_cam  (frente é +Z; sem flip)
#   y_cv = -y_cam  (Webots y↑ → CV y↓)
#   x_cv = -x_cam  (o 180°Y inverteu o eixo X local; compensar para não espelhar)
# Se o overlay sair espelhado ou de cabeça p/ baixo, ajuste os sinais aqui.
AXIS_REMAP = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0],
])


def build_intrinsics(width: int, height: int, fov_h_rad: float) -> np.ndarray:
    """Matriz intrínseca K a partir do FOV horizontal (pixels quadrados)."""
    f = (width / 2.0) / np.tan(fov_h_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]])


def pose_list_to_matrix(pose16) -> np.ndarray:
    """Converte os 16 valores (row-major) de Node.getPose() em 4x4 (câmera->mundo)."""
    return np.asarray(pose16, dtype=float).reshape(4, 4)


def world_to_pixels(points_world, cam_to_world, K, axis_remap=AXIS_REMAP):
    """Projeta pontos do mundo para pixels.

    Args:
        points_world: (N, 3) coordenadas no referencial do mundo.
        cam_to_world: 4x4, pose da câmera no mundo.
        K: 3x3, intrínseca.
    Returns:
        (uv, depth): uv (N, 2) em pixels; depth (N,) em metros no eixo óptico.
        depth <= 0 indica ponto atrás da câmera (deve ser descartado).
    """
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    world_to_cam = np.linalg.inv(np.asarray(cam_to_world, dtype=float))
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (world_to_cam @ hom.T).T[:, :3]        # referencial da câmera (Webots)
    cam_cv = (np.asarray(axis_remap) @ cam.T).T  # referencial CV
    depth = cam_cv[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    proj = (np.asarray(K) @ (cam_cv / safe[:, None]).T).T
    return proj[:, :2], depth


def look_at_rotation(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotação 3x3 (câmera->mundo) que faz a câmera (olhando p/ -Z) mirar o alvo."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    z = eye - target                       # +Z aponta p/ longe do alvo (olha -Z)
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:           # up paralelo a z -> escolhe outro up
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x) + 1e-12

    y = np.cross(z, x)
    return np.column_stack([x, y, z])

def look_at_axis_angle(eye, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """[x,y,z,angle] (formato Webots) p/ a câmera em `eye` mirar `target`.

    No Webots R2025a a Camera olha ao longo do +Z local (não -Z).
    look_at_rotation cria local -Z → alvo; giramos 180° em torno de Y local
    (nega colunas X e Z) para que local +Z fique apontando para o alvo.
    """
    R = look_at_rotation(eye, target, up)
    R[:, 0] *= -1.0   # 180° em Y: nega eixo X local
    R[:, 2] *= -1.0   # 180° em Y: nega eixo Z local → +Z agora aponta pro alvo
    return rotation_to_axis_angle(R)

def rotation_to_axis_angle(R) -> list[float]:
    """Converte rotação 3x3 em [x, y, z, angle] (formato 'rotation' do Webots).
    Via quaternion (Shepperd) — estável inclusive para angle ~ pi."""
    R = np.asarray(R, dtype=float)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0          # S = 4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0   # S = 4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0   # S = 4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0   # S = 4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    qw = float(q[0])
    angle = 2.0 * np.arccos(np.clip(qw, -1.0, 1.0))
    s = np.sqrt(max(1.0 - qw * qw, 0.0))
    if s < 1e-9:                              # rotação ~ identidade
        return [0.0, 0.0, 1.0, 0.0]
    axis = q[1:] / s
    return [float(axis[0]), float(axis[1]), float(axis[2]), float(angle)]
