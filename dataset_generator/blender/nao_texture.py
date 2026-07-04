"""Randomizador de cor de time do NAO (domain randomization).

O modelo Phobos carrega a cor de time *assada* em `textureNAO.png` — um azul
chapado exato (RGB 8,38,224, sem anti-aliasing). Variantes recoloridas foram
pré-geradas em `meshes_V40_obj/textures/textureNAO_<nome>.png` (paleta do proto
Webots Nao.proto: azul V4/V5, vermelho V5, laranja V4, cinza).

Todos os 39 nós `Image Texture` do material `nao` compartilham um único datablock
de imagem; trocar a imagem atribuída a esses nós recolore o robô inteiro. Aqui as
variantes são carregadas como datablocks e sorteadas por amostra.
"""
from pathlib import Path

import bpy

_TEX_DIR = Path(__file__).parent / "meshes_V40_obj" / "textures"

# Variantes disponíveis (arquivos textureNAO_<nome>.png). Cores do proto Webots.
TEAM_VARIANTS = [
    "blue_bright",  # azul original da textura
    "blue_v5",      # 0.114 0.282 0.655
    "blue_v4",      # 0.0   0.36  0.524
    "red_v5",       # 0.88  0.01  0.14
    "orange_v4",    # 0.914 0.361 0.067
    "grey",         # 0.5   0.5   0.5
]


class NaoTextureRandomizer:
    """Carrega as variantes de textura e as atribui aos nós do material do NAO."""

    def __init__(self, variants=None):
        names = list(variants) if variants else list(TEAM_VARIANTS)
        self._images = {}
        for name in names:
            path = _TEX_DIR / f"textureNAO_{name}.png"
            if not path.exists():
                raise FileNotFoundError(f"variante de textura ausente: {path}")
            img = bpy.data.images.load(str(path), check_existing=True)
            img.colorspace_settings.name = "sRGB"
            self._images[name] = img
        self.names = list(self._images)

        # Nós Image Texture que usam a textura do NAO (todos apontam para o mesmo
        # datablock 'textureNAO.png' no repouso; capturados uma vez).
        self._nodes = [
            n
            for m in bpy.data.materials if m.use_nodes
            for n in m.node_tree.nodes
            if n.type == "TEX_IMAGE" and n.image and n.image.name.startswith("textureNAO")
        ]
        if not self._nodes:
            raise RuntimeError("nenhum nó Image Texture do NAO encontrado")

    def apply(self, name: str) -> str:
        """Atribui a variante `name` a todos os nós de textura do NAO."""
        img = self._images[name]
        for n in self._nodes:
            n.image = img
        return name

    def randomize(self, rng) -> str:
        """Sorteia uma variante (uniforme) usando o RNG do gerador e a aplica."""
        return self.apply(self.names[int(rng.integers(len(self.names)))])
