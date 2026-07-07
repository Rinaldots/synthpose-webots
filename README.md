# synthpose-webots

> Geração de **datasets sintéticos de pose** (17 keypoints, formato **COCO-pose**) para o robô **NAO** via **Blender 5.1 headless**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Blender](https://img.shields.io/badge/Blender-5.1%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-funcional-brightgreen)

O pipeline aplica poses aleatórias no armature do NAO, posiciona câmera e iluminação, renderiza com Cycles, projeta os keypoints 3D→2D, calcula visibilidade via ray cast e salva anotações COCO. Todo o processo roda headless, sem interface gráfica.

> Dataset com 20 mil imagens:

https://www.kaggle.com/datasets/rinaldotavares/nao-robot-syntetic-20k

## Início rápido

```bash
# Gerar dataset (padrão: 1000 imagens, config/dataset.yaml)
cd dataset_generator/blender
blender --background nao_full.blend --python dataset_generator_blender.py

# Continuar a partir de um índice (ex: já tem 1000, gerar mais 1000)
blender --background nao_full.blend --python dataset_generator_blender.py -- --start 1000

# Validar o JSON gerado
python3 dataset_generator/scripts/validate_dataset.py \
    dataset_generator/output/annotations/person_keypoints_train.json

# Treinar (requer PyTorch)
python3 dataset_generator/scripts/train.py --backbone resnet18 --epochs 50 --batch 8
```

## Estrutura

```
synthpose-webots/
└── dataset_generator/
    ├── blender/
    │   ├── nao_full.blend               — cena principal (NAO_Armature + 36 meshes)
    │   ├── dataset_generator_blender.py — loop de geração (roda dentro do Blender)
    │   ├── nao_poser_blender.py         — FK + keypoints 3D com offsets faciais
    │   ├── blender_camera.py            — K, cam_to_world, BLENDER_AXIS_REMAP
    │   └── blender_scene_randomizer.py  — cenas procedurais + HDRIs + oclusores
    ├── assets/
    │   └── backgrounds/                 — HDRIs CC0 (Poly Haven, 1k)
    ├── config/
    │   ├── dataset.yaml                 — num_samples, splits, seed
    │   └── randomization.yaml           — limites de juntas, câmera, backgrounds
    ├── src/nao_coco_pose/               — módulos puros (NumPy), testáveis sem Blender
    │   ├── projection.py                — world_to_pixels()
    │   ├── coco_writer.py               — CocoDatasetBuilder (+ load_existing)
    │   ├── visibility.py                — visibility_flag(), bbox_from_keypoints()
    │   ├── randomization.py             — DomainRandomizer
    │   └── config.py                    — carrega YAMLs
    ├── scripts/
    │   ├── train.py                     — SimpleBaseline (ResNet + deconvs)
    │   ├── validate_dataset.py          — QA com pycocotools
    │   └── visualize_sample.py          — overlay de keypoints
    └── output/                          — imagens e JSONs gerados (ignorado pelo git)
```

## Pipeline

```
nao_full.blend
    │
    ▼
[por frame]
  apply_pose()  ←── DomainRandomizer.sample_pose()
  place_camera() ←── sample_camera_pose()
  apply_dof()
  apply_lighting() ←── sample_lighting()
  randomize_scene() ←── cenas procedurais (office/living_room/warehouse/plain)
  set_background() ←── HDRI aleatório com rotação aleatória
    │
    ▼
  render (Cycles 64spp + OIDN denoiser)
    │
    ▼
  project_keypoints 3D→2D  (BLENDER_AXIS_REMAP = diag(1,-1,-1))
  ray_cast_visibility()     (flag 0=ausente / 1=ocluído / 2=visível)
  bbox_from_keypoints()
    │
    ▼
  CocoDatasetBuilder.add_sample()
    │
    ▼
person_keypoints_{train,val,test}.json
```

## Randomização

| Parâmetro | Faixa |
|---|---|
| Pose (22 juntas) | limites NAO × `pose_range_scale=0.6` |
| Câmera — raio | 1.2 m – 2.5 m |
| Câmera — azimute | 0° – 360° |
| Câmera — elevação | 5° – 50° |
| Profundidade de campo | f/2.8 – f/11 |
| Iluminação (intensidade) | 0.6 – 1.4 |
| Cenário | office / living_room / warehouse / plain |
| Textura do chão | wood / tile / concrete / carpet |
| HDRI background | 11 arquivos CC0, rotação aleatória em Z |
| Oclusores | 0 – 2 objetos entre câmera e robô |

## Treino

O script `scripts/train.py` implementa o **SimpleBaseline** (ResNet backbone + 3 camadas deconv → 17 heatmaps).

```bash
# Instalar PyTorch (CPU)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar PyTorch (GPU NVIDIA — requer CUDA 12.8 / driver ≥ 525)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Treinar
python3 dataset_generator/scripts/train.py \
    --backbone resnet50 \
    --epochs 100 \
    --batch 32
```

Métricas reportadas: loss (MSE mascarado por visibilidade) e **PCK@0.2** na validação.  
Checkpoints salvos em `dataset_generator/checkpoints/best.pt`.

## Dependências

| Ferramenta | Uso |
|---|---|
| Blender 5.1+ | renderização e FK headless |
| numpy | projeção, randomização |
| pyyaml | leitura de configs |
| opencv-python | leitura de imagens no treino |
| torch + torchvision | treino do modelo |
| pycocotools | validação do JSON |

## Resultado

https://github.com/user-attachments/assets/e09e9f48-f1a5-45a6-9c40-baad2a0cc1f6

## Licença

MIT
