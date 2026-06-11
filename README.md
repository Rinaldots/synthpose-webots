# synthpose-webots

> Geração de **datasets sintéticos de pose** (17 keypoints, formato **COCO-pose**) para robôs humanoides no **Webots** — começando pelo **NAO**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Webots](https://img.shields.io/badge/Webots-R2025a-informational)
![License](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/status-esqueleto%20funcional-orange)

A cada frame, o pipeline renderiza uma imagem do robô e lê a *ground truth* 3D
das articulações via Supervisor, projeta os pontos para 2D, calcula visibilidade
e bounding box, e grava a anotação no formato COCO. Com randomização de pose,
câmera e ambiente, produz um dataset variado, padronizado e reprodutível —
pronto para treinar estimadores de pose (YOLO-pose, HRNet etc.).

## Destaques

- **Padrão COCO-pose** de ponta a ponta (17 keypoints, esqueleto oficial).
- **Ground truth exata**, vinda da cinemática do simulador (sem rotulagem manual).
- **Domain randomization** de pose, câmera e ambiente, com **semente controlada**.
- **Núcleo testável** fora do Webots: projeção, escrita COCO e visibilidade são NumPy puro.
- **Documentação organizada** em `docs/`, incluindo a convenção de keypoints e um datasheet.

## Como funciona

```
configurar cena ─► [loop ×N] randomizar ─► capturar RGB + juntas 3D
                              ─► projetar 3D→2D ─► visibilidade + bbox ─► anotar
                ─► montar JSON COCO ─► validar / QA
```

O ponto central: a cada frame há **duas fontes** — a imagem renderizada (input)
e as juntas 3D do Supervisor (ground truth). Detalhes em
[`docs/pipeline.md`](docs/pipeline.md) e [`docs/keypoint_mapping.md`](docs/keypoint_mapping.md).

## Estrutura

```
synthpose-webots/
├── docs/            # padronização, pipeline e datasheet
├── config/          # câmera, randomização e dataset (YAML)
├── protos/          # MyNao.proto
├── worlds/          # cena de exemplo (NAO + rig da câmera)
├── controllers/     # dataset_generator: o laço de captura
├── src/             # pacote: projeção, juntas, câmera, COCO, ...
└── scripts/         # validação (pycocotools) e overlay de keypoints
```

## Início rápido

Instale as dependências no **mesmo Python que o Webots usa**:

```bash
pip install -e .          # ou: pip install -r requirements.txt
```

1. Abra `worlds/dataset.wbt` no Webots R2025a.
2. Rode a simulação — as saídas vão para `output/images/` e
   `output/annotations/person_keypoints.json`.
3. Valide e inspecione:

```bash
python scripts/validate_dataset.py output/annotations/person_keypoints.json
python scripts/visualize_sample.py \
    output/annotations/person_keypoints.json output/images 0 overlay.png
```

O `overlay.png` é o teste decisivo: se os pontos caem nas juntas, projeção e
mapeamento estão corretos.

## Status e roadmap

Esqueleto funcional. Pontos a ajustar/implementar conforme a sua cena:

- [ ] Validar a convenção de eixos da câmera (`AXIS_REMAP`) pelo overlay
- [ ] Calibrar os offsets faciais (nariz/olhos/orelhas) no referencial da cabeça
- [ ] Oclusão real via RangeFinder (flag de visibilidade 1)
- [ ] Randomização visual de iluminação e fundo
- [ ] Suporte a outros humanoides além do NAO

## Licença

MIT — sinta-se à vontade para ajustar.
