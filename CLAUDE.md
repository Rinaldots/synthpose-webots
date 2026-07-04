# CLAUDE.md — synthpose-webots

Geração de dataset sintético COCO-pose (17 keypoints) para o NAO via **Blender 5.1.2 headless**.
Modelo: **NAO de alta qualidade do fabricante, exportado via Phobos** (`nao_blender.blend`).
Pipeline: aplica pose aleatória nas juntas → posiciona câmera → renderiza →
projeta 3D→2D → calcula visibilidade/bbox → grava anotação COCO.

## Comandos
- Gerar dataset:
  `cd dataset_generator/blender && blender --background nao_blender.blend --python dataset_generator_phobos.py`
  - Flags (após `--`): `--start N` (índice inicial), `--num N` (quantas amostras),
    `--out DIR` (redireciona a saída — **use sempre em testes**; nunca escreva no `output/` real).
- Overlay de validação (5 poses, projeta os keypoints sobre o render):
  `blender --background nao_blender.blend --python overlay_phobos.py`
- Validar JSON gerado:
  `python scripts/validate_dataset.py output/annotations/person_keypoints_train.json`
- Testar módulos puros fora do Blender: `PYTHONPATH=src python -c "import nao_coco_pose"`

## Arquitetura (decisões que não devem ser quebradas)

### Modelo Phobos (`nao_blender.blend`)
- Export Phobos/URDF do NAO do fabricante: **um armature-objeto por link** (~80),
  encadeados por parenting de bone. NÃO é um único armature com cadeia de bones.
- Cada objeto-link carrega metadata de junta em custom properties:
  `joint/name`, `joint/axis`, `joint/type`, `joint/limits/*`.
- Coleções `collision`/`inertial` e objetos `resource::*` são metadata Phobos —
  ocultados no render por `hide_metadata_collections()`. **Não há simulação física
  ativa** (só metadata inercial/colisão); a pose é FK determinística.

### Blender scripts
- `blender/nao_poser_phobos.py` — `NaoPoserPhobos`: posa cada link e lê os keypoints 3D.
- `blender/blender_camera.py` — extrai K e cam_to_world; define `BLENDER_AXIS_REMAP = diag(1,-1,-1)`.
- `blender/dataset_generator_phobos.py` — loop de geração; roda DENTRO do Blender.
- `blender/overlay_phobos.py` — validação visual (calibração de eixos/offsets).
- `blender/blender_scene_randomizer.py` — cenas procedurais + HDRIs + oclusores.

### Convenções de eixos
- O modelo Phobos está em **frame NAO/URDF nativo**: X=frente, Y=esquerda, Z=cima.
  O **mundo Blender coincide** com o frame NAO — NÃO existe a rotação -90° em Z do rig
  antigo, nem o remap `(-y, x, z)`.
- `BLENDER_AXIS_REMAP = diag(1,-1,-1)`: converte cam-space Blender (Y cima, -Z frente)
  para CV-space (Y baixo, +Z frente). Validado pelo `overlay_phobos.py` — não alterar sem re-validar.

### Posing (FK)
- Cada junta revoluta é um objeto-link com rotação identidade em repouso e eixo em
  `joint/axis` (no frame local, que em repouso coincide com o mundo).
- Posar = aplicar `Quaternion(joint_axis, angle)` ao objeto (pivô na origem do objeto,
  que fica sobre a junta). O parenting de bone acumula a FK — um único `view_layer.update()`
  no fim basta.
- Mapa junta→objeto e keypoint→objeto (`COCO_TO_LINK`) em `nao_poser_phobos.py`.
  Keypoints 3D = `matrix_world.translation` do objeto-link correspondente.

### Keypoints faciais
- Offsets fixos no frame NAO (`_HEAD_OFFSETS_NAO` em `nao_poser_phobos.py`), relativos à
  origem do objeto `Head`, rotacionados pela orientação world da cabeça (`Head.matrix_world`).

### Grounding
- A origem do modelo (`base_link`) fica no tronco; `_ground_robot()` no gerador levanta o
  robô para o ponto mais baixo ficar em z=0 e usa a altura do tronco como centro de órbita da câmera.

### Módulos puros (NumPy, testáveis sem Blender)
- `src/nao_coco_pose/projection.py` — `world_to_pixels()`
- `src/nao_coco_pose/coco_writer.py` — `CocoDatasetBuilder`
- `src/nao_coco_pose/visibility.py` — `visibility_flag()`, `bbox_from_keypoints()`
- `src/nao_coco_pose/randomization.py` — `DomainRandomizer`
- `src/nao_coco_pose/config.py` — carrega YAMLs

## Convenções
- Ordem dos 17 keypoints é a COCO oficial — NÃO reordenar.
- Código com identificadores em inglês; docstrings/comentários em português.
- Não adicionar dependências sem necessidade (numpy, pyyaml, opencv, pycocotools).

## Pontos sensíveis / TODO
- Offsets faciais em `_HEAD_OFFSETS_NAO`: calibrar pelo overlay (orelhas podem precisar de ajuste fino).
- `BLENDER_AXIS_REMAP`: validado pelo `overlay_phobos.py` — não alterar sem re-validar.
- Visibilidade por ray cast já implementada no gerador (flag 0=ausente / 1=ocluído / 2=visível).

## Limites de arquivos
- Não editar: `output/` (gerado), `blender/nao_blender.blend` (salvar só via Blender).
- Editar à vontade: `blender/*.py`, `src/`, `scripts/`, `config/`, `docs/`.
