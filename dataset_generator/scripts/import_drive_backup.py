"""Importa um backup ANTIGO (formato por-rank) para o layout do modo DINÂMICO.

Contexto: o backup feito no modo manual está em ``rankR/gX_pY/{images,annotations}``
e usa índices globais nos nomes (``nao_{N}.png``). O modo dinâmico organiza por LOTE
(``chunk_{c}/``) e decide o que já foi feito pelo registro ``_claims/chunk_{c}.json``.
Este script casa os dois: agrupa as imagens do backup por lote (``c = N // chunk``) e,
para cada lote **completo** (todos os ``chunk`` índices presentes), copia as imagens +
anotações para ``dst/chunk_{c}/imported/`` e marca ``_claims/chunk_{c}.json`` como
``done`` — assim o coordenador **pula** esse lote. Lotes incompletos (as pontas) são
ignorados aqui e o novo job os regenera inteiros (evita colisão de ``file_name``).

Rode NO SEU PC, com caminhos locais. Se o backup ainda está no Drive, baixe antes::

    rclone copy gdrive:synthpose-backup ~/synthpose-old --transfers=8

Uso::

    python scripts/import_drive_backup.py \
        --src ~/synthpose-old \
        --dst ~/synthpose-backup/kaggle \
        --total 10000 --chunk 500

``--dst`` é a MESMA pasta que o Kaggle acessa (ex.: a servida no ``rclone serve
webdav``; ``BACKUP_REMOTE="pc:kaggle"`` → ``~/synthpose-backup/kaggle``). ``--total``
e ``--chunk`` devem bater com o notebook.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Reusa utilitários do merge (mesma pasta scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_datasets import _split_of, ANN_GLOB  # noqa: E402


def _find_roots(src: Path, tmp_root: Path) -> list[Path]:
    """Acha TODAS as raízes de dataset sob ``src`` (qualquer pasta com annotations/
    contendo person_keypoints_*.json), em qualquer profundidade. Extrai .zip antes."""
    import zipfile
    if src.is_file() and src.suffix.lower() == ".zip":
        dest = tmp_root / src.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
        src = dest
    if not src.is_dir():
        sys.exit(f"[import] fonte inválida: {src}")
    roots = {ann.parent.parent for ann in src.rglob("annotations/" + ANN_GLOB)}
    if not roots:
        sys.exit(f"[import] nenhum 'annotations/person_keypoints_*.json' sob {src}")
    return sorted(roots)


def _index_of(file_name: str) -> int | None:
    """'train/nao_00042.png' -> 42. None se não casar o padrão."""
    stem = Path(file_name).stem            # nao_00042
    tail = stem.split("_")[-1]
    return int(tail) if tail.isdigit() else None


def _scan(roots: list[Path]):
    """Varre as fontes. Retorna:
      present[idx]            -> (split, caminho_da_imagem)  (por arquivo em disco)
      img_meta[(split, idx)]  -> dict de imagem do COCO
      anns[(split, idx)]      -> lista de anotações
      cats / info             -> categorias/info (primeiras vistas)
    """
    present: dict[int, tuple[str, Path]] = {}
    img_meta: dict[tuple[str, int], dict] = {}
    anns: dict[tuple[str, int], list] = defaultdict(list)
    cats: list = []
    info: dict = {}

    for root in roots:
        ann_dir = root / "annotations"
        for ann_path in sorted(ann_dir.glob(ANN_GLOB)):
            split = _split_of(ann_path)
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            cats = cats or data.get("categories", [])
            info = info or data.get("info", {})
            id2idx: dict[int, tuple[str, int]] = {}
            for img in data.get("images", []):
                idx = _index_of(img["file_name"])
                if idx is None:
                    continue
                key = (split, idx)
                img_meta[key] = img
                id2idx[img["id"]] = key
            for ann in data.get("annotations", []):
                key = id2idx.get(ann["image_id"])
                if key is not None:
                    anns[key].append(ann)
        # Presença por ARQUIVO em disco (cobre imagens órfãs sem anotação também).
        img_dir = root / "images"
        for p in img_dir.rglob("*.png"):
            idx = _index_of(p.name)
            if idx is None:
                continue
            split = p.parent.name          # images/<split>/nao_N.png
            present.setdefault(idx, (split, p))
    return present, img_meta, anns, cats, info


def _chunk_size(c: int, total: int, chunk: int) -> int:
    start = c * chunk
    return max(0, min(chunk, total - start))


def import_backup(src_roots: list[Path], dst: Path, total: int, chunk: int,
                  dry_run: bool = False) -> None:
    present, img_meta, anns, cats, info = _scan(src_roots)
    present = {i: v for i, v in present.items() if i < total}
    print(f"[import] {len(present)} imagens no backup (índices < {total})")

    # Índices presentes por lote.
    by_chunk: dict[int, list[int]] = defaultdict(list)
    for idx in present:
        by_chunk[idx // chunk].append(idx)

    claims_dir = dst / "_claims"
    n_full = n_imgs = 0
    partial = []
    for c in sorted(by_chunk):
        need = _chunk_size(c, total, chunk)
        have = len(by_chunk[c])
        if have < need:
            partial.append((c, have, need))
            continue                        # incompleto → o novo job regenera
        n_full += 1
        base = dst / f"chunk_{c:05d}" / "imported"
        # Agrupa anotações/imagens deste lote por split.
        per_split_imgs: dict[str, list] = defaultdict(list)
        per_split_anns: dict[str, list] = defaultdict(list)
        for idx in sorted(by_chunk[c]):
            split, img_path = present[idx]
            per_split_imgs[split].append((idx, img_path))
            key = (split, idx)
            if key in img_meta:
                per_split_anns[split].append((img_meta[key], anns.get(key, [])))
        if dry_run:
            n_imgs += have
            continue
        for split, items in per_split_imgs.items():
            dst_img_dir = base / "images" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            for _idx, img_path in items:
                shutil.copy2(img_path, dst_img_dir / img_path.name)
                n_imgs += 1
        # JSON COCO por split (ids renumerados localmente; o merge renumera de novo).
        ann_out = base / "annotations"
        ann_out.mkdir(parents=True, exist_ok=True)
        for split, pairs in per_split_anns.items():
            images, annotations = [], []
            next_id = next_ann = 0
            for img, alist in pairs:
                next_id += 1
                new_img = dict(img); new_img["id"] = next_id
                images.append(new_img)
                for a in alist:
                    next_ann += 1
                    na = dict(a); na["id"] = next_ann; na["image_id"] = next_id
                    annotations.append(na)
            (ann_out / f"person_keypoints_{split}.json").write_text(
                json.dumps({"info": info, "images": images,
                            "annotations": annotations, "categories": cats},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        # Marca o lote como concluído p/ o coordenador pular.
        claims_dir.mkdir(parents=True, exist_ok=True)
        (claims_dir / f"chunk_{c:05d}.json").write_text(
            json.dumps({"machine": "import", "ts": time.time(), "done": True}))

    verb = "(dry-run) " if dry_run else ""
    print(f"[import] {verb}{n_full} lote(s) COMPLETO(s) importado(s) "
          f"({n_imgs} imagens) -> {dst}")
    if partial:
        idxs = ", ".join(f"{c}({h}/{n})" for c, h, n in partial)
        print(f"[import] {len(partial)} lote(s) incompleto(s) IGNORADO(s) "
              f"(serão regenerados): {idxs}")
    n_chunks = -(-total // chunk)
    print(f"[import] resultado: {n_full}/{n_chunks} lotes já prontos; o job dinâmico "
          f"fará os {n_chunks - n_full} restantes.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="Backup antigo (pasta local ou .zip; formato por-rank).")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Pasta de storage do modo dinâmico (a mesma que o Kaggle acessa).")
    ap.add_argument("--total", type=int, required=True, help="TOTAL_SAMPLES do notebook.")
    ap.add_argument("--chunk", type=int, default=500, help="CHUNK do notebook (default 500).")
    ap.add_argument("--dry-run", action="store_true", help="Só relata; não copia nada.")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="import_bkp_") as tmp:
        roots = _find_roots(args.src, Path(tmp))
        import_backup(roots, args.dst, args.total, args.chunk, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
