"""Junta datasets COCO-pose gerados em várias máquinas num único dataset.

Cada máquina produz uma pasta ``output/`` independente::

    output/
      images/{train,val,test}/nao_NNNNN.png
      annotations/person_keypoints_{split}.json

Como cada máquina roda uma fatia disjunta (``--rank``/``--world-size``, ver
nao_coco_pose.sharding), os NOMES de arquivo já não colidem. O que colide é o
``image_id``/``ann_id`` do COCO — cada máquina numera a partir de 1. Este script
concatena as imagens e **renumera os IDs sequencialmente**, usando ``file_name``
como identidade estável de cada imagem.

Aceita tanto pastas ``output/`` já extraídas quanto os ``.zip`` baixados do
Kaggle (extraídos para um diretório temporário automaticamente).

Uso:
    python scripts/merge_datasets.py \
        machineA.zip machineB.zip \
        --out output_merged

    # ou pastas já extraídas:
    python scripts/merge_datasets.py runA/output runB/output --out merged
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ANN_GLOB = "person_keypoints_*.json"


def _resolve_sources(src: Path, tmp_root: Path) -> list[Path]:
    """Devolve as raízes de dataset (pastas com images/ + annotations/) em `src`.

    Extrai .zip para tmp_root. Retorna:
      - [src]                      se o próprio dir já tem 'annotations/';
      - [gpu0, gpu1, ...]          se tem vários sub-datasets (várias GPUs numa
                                   máquina — zip de 'output' contendo gpu0/gpu1).
    """
    if src.is_file() and src.suffix.lower() == ".zip":
        dest = tmp_root / src.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
        src = dest
    if not src.is_dir():
        sys.exit(f"[merge] fonte inválida: {src}")
    if (src / "annotations").is_dir():
        return [src]
    roots = [c for c in sorted(src.iterdir())
             if c.is_dir() and (c / "annotations").is_dir()]
    if roots:
        return roots
    sys.exit(f"[merge] não achei 'annotations/' em {src} (nem em subpastas)")


def _split_of(ann_path: Path) -> str:
    """'person_keypoints_train.json' -> 'train'."""
    return ann_path.stem[len("person_keypoints_"):]


def merge(sources: list[Path], out_dir: Path) -> None:
    out_ann = out_dir / "annotations"
    out_ann.mkdir(parents=True, exist_ok=True)

    # Acumuladores por split.
    merged: dict[str, dict] = {}          # split -> {images, annotations, categories, info}
    next_img_id: Counter[str] = Counter()
    next_ann_id: Counter[str] = Counter()
    seen_names: dict[str, set[str]] = {}  # split -> {file_name} p/ detectar colisão

    total_imgs = total_anns = copied = 0

    for src in sources:
        root = src
        for ann_path in sorted((root / "annotations").glob(ANN_GLOB)):
            split = _split_of(ann_path)
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            acc = merged.setdefault(split, {
                "info": data.get("info", {}),
                "images": [],
                "annotations": [],
                "categories": data.get("categories", []),
            })
            if not acc["categories"]:
                acc["categories"] = data.get("categories", [])
            names = seen_names.setdefault(split, set())

            # Remapeia image_id antigo -> novo (por arquivo, dentro deste json).
            id_map: dict[int, int] = {}
            for img in data.get("images", []):
                fn = img["file_name"]
                if fn in names:
                    sys.exit(f"[merge] COLISÃO de file_name '{fn}' no split '{split}'.\n"
                             f"        As máquinas usaram intervalos --start sobrepostos. "
                             f"Regenere com --rank/--world-size disjuntos.")
                names.add(fn)
                next_img_id[split] += 1
                new_id = next_img_id[split]
                id_map[img["id"]] = new_id
                new_img = dict(img); new_img["id"] = new_id
                acc["images"].append(new_img)
                total_imgs += 1

                # Copia o arquivo de imagem (file_name é relativo a images/).
                src_img = root / "images" / fn
                dst_img = out_dir / "images" / fn
                if src_img.is_file():
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_img)
                    copied += 1
                else:
                    print(f"[merge] AVISO: imagem ausente no disco: {src_img}")

            for ann in data.get("annotations", []):
                old = ann["image_id"]
                if old not in id_map:
                    print(f"[merge] AVISO: anotação órfã (image_id={old}) em {ann_path.name}")
                    continue
                next_ann_id[split] += 1
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id[split]
                new_ann["image_id"] = id_map[old]
                acc["annotations"].append(new_ann)
                total_anns += 1

            print(f"[merge] {src.name}:{split}  +{len(data.get('images', []))} imgs")

    # Grava os JSONs mergeados.
    for split, acc in merged.items():
        out_json = out_ann / f"person_keypoints_{split}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(acc, f, ensure_ascii=False, indent=2)
        print(f"[merge] -> {out_json}  ({len(acc['images'])} imgs, {len(acc['annotations'])} anns)")

    print(f"\n[merge] TOTAL: {total_imgs} imagens, {total_anns} anotações, "
          f"{copied} arquivos copiados -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path,
                    help="Pastas output/ ou .zip de cada máquina.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Diretório de saída do dataset mergeado.")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="merge_ds_") as tmp:
        tmp_root = Path(tmp)
        resolved = [r for s in args.sources for r in _resolve_sources(s, tmp_root)]
        merge(resolved, args.out)


if __name__ == "__main__":
    main()
