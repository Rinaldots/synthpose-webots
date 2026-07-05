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
  - Multi-máquina: `--rank R --world-size W [--total N]` deriva `--start`/`--num` de uma
    fatia disjunta do job (via `nao_coco_pose.sharding.shard_range`). Tem precedência sobre
    `--start`/`--num`. Como a semente é `seed + start`, fatias disjuntas → poses e nomes
    de arquivo disjuntos automaticamente.
  - **Coordenação dinâmica (notebook Kaggle, `DYNAMIC=True`)**: em vez de `--rank`
    manual, o notebook usa `scripts/coordinator.py` p/ reivindicar o próximo LOTE livre
    de `_claims/` no Drive (registro `chunk_{i}.json` com machine+ts+done), gerar em
    `chunk_{i}/`, marcar done e repetir. Ligar/desligar máquinas à vontade; lotes sem
    heartbeat há `stale_s` (máquina caiu) voltam à fila e são retomados pelo JSON parcial.
    Índices globais por lote → nomes únicos → o merge só concatena. Exige backup (o Drive
    é a coordenação). Protocolo: escreve→propaga→relê (last-writer-wins resolve empates).
- Juntar datasets de várias máquinas (fora do Blender, sem deps além da stdlib):
  `python scripts/merge_datasets.py runA.zip runB/output --out output_merged`
  - Aceita `.zip` (baixados do Kaggle) ou pastas `output/`. Renumera `image_id`/`ann_id`
    do COCO e usa `file_name` como chave; **aborta** se dois `--start` se sobrepuseram.
- Monitorar GPU + progresso (roda DENTRO de cada máquina Kaggle; `nvidia-smi` só vê a GPU local):
  `python scripts/monitor.py --out output --total N --world-size W --rank R [--interval S]`
  - No notebook, `monitor.run_with_monitor(cmd, out_dir, num)` roda o monitor em paralelo
    com a geração. Sem deps novas — lê `nvidia-smi` e conta PNGs em `<out>/images/**`.
  - Backup incremental do dataset pro PC: `--backup-remote gdrive:pasta [--backup-every 100]`
    a cada N imagens **MOVE** as imagens pra nuvem (`rclone move`, apaga a cópia local →
    libera o disco do Kaggle, poupando renders do último minuto via `--min-age`) e **COPIA**
    os JSON (ficam locais p/ resume + progresso). `--no-backup-prune` volta ao copy puro.
    rclone precisa estar configurado; single-flight e não-bloqueante (backup final é síncrono).
  - **Backend de storage é plugável (é tudo rclone)**: o notebook tem `STORAGE`:
    `"drive"` (Google Drive; cuidado com a cota 403 'Queries per minute' da API
    compartilhada — use client_id próprio) ou `"tailscale"` (grava no **PC do usuário**
    via `rclone serve webdav` + Tailscale userspace no Kaggle; sem cota, espaço = disco
    do PC, mas o PC precisa ficar ligado). Ver `docs/storage_tailscale.md`. Trocar de
    backend = mudar `STORAGE`/`BACKUP_REMOTE`; nenhum código do pipeline muda.
- Migrar um backup ANTIGO (formato por-rank) para o modo dinâmico (reaproveitar trabalho):
  `python scripts/import_drive_backup.py --src ~/synthpose-old --dst ~/synthpose-backup/kaggle --total N --chunk 500`
  - Agrupa as imagens por lote (`c = índice // chunk`); lotes **completos** viram
    `chunk_{c}/imported/` + `_claims/chunk_{c}.json` (done) → o coordenador os pula.
    Lotes incompletos (pontas) são ignorados e o novo job os regenera (sem colisão).
    Casa os índices globais dos nomes `nao_{N}.png`; não mistura poses dos dois modos.
- Baixar do Drive e juntar tudo (no PC, no fim do job):
  `python scripts/download_and_merge.py --remote gdrive:pasta --out output_merged`
  - Baixa todas as pastas de dataset do topo do remote (`chunk_*`/`rank*`, pulando
    `_claims/`) e chama o `merge_datasets`. Como o backup move as imagens, o dataset
    completo vive na nuvem.
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
