# Roteiro para o beta — synthpose-webots

Pipeline funcional (3 imagens + JSON COCO gerados). Dois bugs confirmados pela
inspeção do `output/annotations/person_keypoints.json`:

1. `left_eye` / `right_eye` → flag=0 em todos os frames (depth ≤ 0, projetam
   atrás da câmera).
2. `nose`, `left_ear`, `right_ear` → coordenadas idênticas (mesmo offset
   `(0.06, 0, 0.02)` nos três).

---

## Fase 1 — Overlay: validar a projeção (sem Webots)

**Objetivo:** confirmar se os keypoints visíveis (flag=2) caem nas articulações
corretas nas 3 imagens já geradas.

```bash
cd dataset_generator
python scripts/visualize_sample.py output/annotations/person_keypoints.json output/images 0 overlay_0.png
python scripts/visualize_sample.py output/annotations/person_keypoints.json output/images 1 overlay_1.png
python scripts/visualize_sample.py output/annotations/person_keypoints.json output/images 2 overlay_2.png
```

### O que checar

| Sintoma no overlay | Causa provável | Correção em `src/nao_coco_pose/projection.py` |
|---|---|---|
| Pontos nas articulações certas | OK | Nenhuma |
| Espelhado horizontalmente | Sinal de X invertido | `AXIS_REMAP[0,0]`: +1 → -1 |
| Invertido verticalmente | Sinal de Y invertido | `AXIS_REMAP[1,1]`: -1 → +1 |
| Pontos todos no centro da imagem | Z negado errado | `AXIS_REMAP[2,2]`: -1 → +1 |

`AXIS_REMAP` atual (`projection.py:15-19`):
```python
AXIS_REMAP = np.array([
    [1.0,  0.0,  0.0],
    [0.0, 1.0,  0.0],
    [0.0,  0.0, -1.0],
])
```

### Critério de saída da Fase 1
- [ ] Ombros, cotovelos, pulsos, quadris, joelhos e tornozelos sobre as juntas
  em pelo menos 2 dos 3 overlays.
- [ ] Lateralidade correta (esquerdo/direito não trocados).

---

## Fase 2 — Consertar keypoints faciais (`keypoints.py`)

### Bug 1 — `left_ear` e `right_ear` com offset idêntico ao `nose`

Em `src/nao_coco_pose/keypoints.py:65-66` os três pontos usam `(0.06, 0.000, 0.02)`.
As orelhas precisam de deslocamento lateral.

Offsets sugeridos para primeira tentativa (eixo Y = lateral no referencial da
cabeça do NAO, positivo para a esquerda do robô):

```python
"left_ear":  KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.00, 0.07, 0.01)),
"right_ear": KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.00, -0.07, 0.01)),
```

### Bug 2 — `left_eye` / `right_eye` com depth ≤ 0

O offset aponta para dentro ou atrás da cabeça. Checar a orientação real do
referencial com o print abaixo (rodar uma vez no controlador, após o `resolve()`):

```python
head = resolver._head
print("head pos:", head.getPosition())
print("head R:\n", np.array(head.getOrientation()).reshape(3,3))
```

Depois de identificar qual coluna de R aponta para frente (+X do mundo quando
o NAO está em pé olhando para +X), ajustar o offset dos olhos para que o
componente "para frente da cabeça" seja positivo.

Offsets iniciais a testar:
```python
"left_eye":  KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.05, 0.03, 0.03)),
"right_eye": KeypointSource(SourceKind.HEAD_OFFSET, "HeadPitch", (0.05, -0.03, 0.03)),
```

### Critério de saída da Fase 2
- [ ] `left_eye` e `right_eye` com flag=2 em pelo menos um frame de teste.
- [ ] As 5 orelhas/olhos/nariz não colapsam no mesmo pixel no overlay.

---

## Fase 3 — Escalar a geração

Com projeção e keypoints corretos, gerar o batch real.

1. Editar `config/dataset.yaml`:
   ```yaml
   num_samples: 500   # beta mínimo; 200 para teste rápido
   seed: 42
   ```

2. Abrir `worlds/dataset.wbt` no Webots R2025a → rodar a simulação.

**Estimativa:** com `settle_steps: 3` e timestep de 32 ms, cada frame leva
≈ 100 ms de simulação → 500 frames ≈ 1–2 minutos de clock.

---

## Fase 4 — QA do dataset

```bash
# Valida estrutura COCO (contagens, bboxes, keypoints)
python scripts/validate_dataset.py output/annotations/person_keypoints.json

# Overlay em 5 amostras espaçadas para inspeção visual
for i in 0 100 200 350 499; do
  python scripts/visualize_sample.py \
    output/annotations/person_keypoints.json output/images $i overlay_qa_${i}.png
done
```

### Critérios de aceitação do beta

- [ ] `validate_dataset.py` retorna 0 erros.
- [ ] Keypoints nas articulações corretas em ≥ 4/5 overlays de QA.
- [ ] Média de `num_keypoints` ≥ 12 nas anotações.
- [ ] Bbox envolve o robô visivelmente em todos os overlays.
- [ ] Nenhuma imagem em branco (erro de captura).

---

## Fase 5 — Melhorias pós-beta (não bloqueantes)

| Item | Arquivo(s) | Esforço |
|---|---|---|
| Oclusão real (flag=1) | `worlds/dataset.wbt` + `visibility.py:23` | Médio — adicionar RangeFinder ao DEF RIG |
| Randomização de iluminação | `dataset_generator.py:97` (TODO) | Baixo — chamar `sample_lighting()` e escrever campo `PointLight` via Supervisor |
| Randomização de fundo/textura | `config/randomization.yaml` + controlador | Médio |
| Splits treino/val/teste | `coco_writer.py` | Baixo — config já existe |
| Aumentar para 5 000+ amostras | `config/dataset.yaml` | Trivial |
| Bbox por segmentação | `worlds/dataset.wbt` + controlador | Alto — habilitar `recognitionSegmentation` na Camera |

---

## Estado atual

| Item | Status |
|---|---|
| Pipeline roda end-to-end | OK |
| AXIS_REMAP validado | **OK — overlay confirma projeção correta** |
| Keypoints de membros projetam certo | OK (overlay) |
| NAO visível nas imagens | **Corrigido** — `resetPhysics()` + re-âncora no settle |
| Keypoints faciais corretos | **Bugado — Fase 2** |
| Batch de 500 amostras | Pendente |
| validate_dataset.py limpo | Pendente |

### Bug resolvido: NAO caía durante o settle

O NAO tem `controller "<none>"` e sem fixação de raiz a física o tombava a cada
frame. Fix em `controllers/dataset_generator/dataset_generator.py`:
- `nao_node.resetPhysics()` zera velocidades antes do settle
- Re-ancora `translation` e `rotation` a cada passo do settle loop
