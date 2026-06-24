# dataset_generator

Gerador de dataset sintético COCO-pose para o NAO via **Blender 5.1 headless**.

## Comandos

```bash
# Gerar dataset completo (num_samples do config/dataset.yaml)
cd blender
blender --background nao_full.blend --python dataset_generator_blender.py

# Continuar a partir de um índice (append no JSON existente)
blender --background nao_full.blend --python dataset_generator_blender.py -- --start 1000

# Gerar quantidade específica
blender --background nao_full.blend --python dataset_generator_blender.py -- --start 2000 --num 500

# Overlay de calibração (valida projeção + keypoints faciais)
blender --background nao_full.blend --python overlay_test.py
blender --background nao_full.blend --python overlay_multi.py

# Validar JSON gerado
python3 scripts/validate_dataset.py output/annotations/person_keypoints_train.json

# Treinar
python3 scripts/train.py --backbone resnet18 --epochs 50 --batch 8
```

## Configuração

**`config/dataset.yaml`** — parâmetros gerais:
```yaml
num_samples: 1000
splits: {train: 0.8, val: 0.1, test: 0.1}
seed: 42
```

**`config/randomization.yaml`** — domínio de randomização:
```yaml
pose_range_scale: 0.6        # amplitude das poses (0=neutro, 1=limites máximos)
backgrounds:                  # HDRIs (deixar [] para céu sólido)
  - assets/backgrounds/cayley_interior_1k.hdr
  - ...
```

## Saída

```
output/
├── images/
│   ├── train/  nao_00000.png … nao_00799.png
│   ├── val/    nao_00800.png … nao_00899.png
│   └── test/   nao_00900.png … nao_00999.png
└── annotations/
    ├── person_keypoints_train.json
    ├── person_keypoints_val.json
    └── person_keypoints_test.json
```

Formato COCO-pose padrão — compatível com MMPose, ViTPose, pycocotools etc.

## Arquitetura dos módulos

| Arquivo | Responsabilidade |
|---|---|
| `blender/dataset_generator_blender.py` | loop principal, ray cast, salva JSON |
| `blender/nao_poser_blender.py` | FK no armature, keypoints 3D, offsets faciais |
| `blender/blender_camera.py` | matriz K, cam_to_world, BLENDER_AXIS_REMAP |
| `blender/blender_scene_randomizer.py` | cenas procedurais, texturas, HDRIs, oclusores |
| `src/nao_coco_pose/projection.py` | world_to_pixels() — NumPy puro |
| `src/nao_coco_pose/coco_writer.py` | CocoDatasetBuilder, load_existing() |
| `src/nao_coco_pose/visibility.py` | flags COCO (0/1/2), bbox |
| `src/nao_coco_pose/randomization.py` | DomainRandomizer (pose, câmera, iluminação) |
| `scripts/train.py` | SimpleBaseline — ResNet + heatmaps, PCK@0.2 |

## Convenções fixas (não alterar sem re-validar)

- **NAO→Blender**: `(-v.y, v.x, v.z)` (rotação -90° em Z)
- **BLENDER_AXIS_REMAP**: `diag(1,-1,-1)` — validado pelo overlay
- **Keypoints faciais**: offsets em `_HEAD_OFFSETS_NAO`, rotados por `R_delta = R_pose @ R_rest.inverted()`
- **Ordem dos 17 keypoints**: COCO oficial — não reordenar

## Adicionando backgrounds

1. Baixe HDRIs em 1k de [Poly Haven](https://polyhaven.com/hdris) (CC0)
2. Coloque em `assets/backgrounds/`
3. Liste em `config/randomization.yaml` sob `backgrounds:`
