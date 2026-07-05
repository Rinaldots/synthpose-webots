"""Baixa o dataset completo do Drive (backup por-rank) e junta tudo num só COCO.

Fecha o fluxo do backup incremental do Kaggle: durante a geração, cada máquina
MOVE suas imagens p/ ``{remote}/rank{R}`` (liberando o disco do Kaggle) e mantém só
os JSON locais. No fim, o dataset inteiro vive na nuvem. Este script, rodando NO SEU
PC, puxa cada ``rank{R}`` e chama o ``merge_datasets`` p/ renumerar os IDs COCO num
único dataset.

Layout na nuvem (o que o Kaggle sobe)::

    # modo dinamico (DYNAMIC=True): por LOTE
    {remote}/chunk_00000/g0_p0/{images,annotations}/...
    {remote}/chunk_00001/g0_p0/{images,annotations}/...
    # modo manual (DYNAMIC=False): por RANK
    {remote}/rank0/g0_p0/{images,annotations}/...
    {remote}/rank1/g0_p0/{images,annotations}/...

Baixa e junta TODAS as pastas de dataset do topo do remote (``chunk_*``/``rank*``),
ignorando o registro de coordenacao ``_claims/``.

Pré-requisito: ``rclone`` configurado no seu PC (o MESMO remote do notebook).

Uso:
    python scripts/download_and_merge.py \
        --remote gdrive:synthpose-backup --out output_merged
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Reusa a lógica de merge (mesma pasta scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_datasets import _resolve_sources, merge  # noqa: E402


def _rclone() -> str:
    exe = shutil.which("rclone")
    if not exe:
        sys.exit("[dl-merge] rclone não encontrado no PATH — instale/configure o rclone.")
    return exe


def _list_datasets(exe: str, remote: str) -> list[str]:
    """Lista as pastas de dataset no topo do remote (chunk_*/rank*), pulando _claims."""
    r = subprocess.run([exe, "lsf", remote, "--dirs-only"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[dl-merge] falha ao listar {remote}:\n{r.stderr[:400]}")
    names = []
    for line in r.stdout.splitlines():
        name = line.strip().rstrip("/")
        if name and name != "_claims":
            names.append(name)
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--remote", required=True,
                    help="Destino rclone do backup (ex.: 'gdrive:synthpose-backup').")
    ap.add_argument("--out", type=Path, required=True,
                    help="Diretório de saída do dataset mergeado.")
    ap.add_argument("--stage", type=Path, default=None,
                    help="Onde baixar antes de juntar (default: pasta temporária, apagada no fim).")
    ap.add_argument("--keep-stage", action="store_true",
                    help="Não apagar a pasta de staging (útil p/ inspecionar/reusar o download).")
    args = ap.parse_args()

    exe = _rclone()
    remote = args.remote.rstrip("/")
    names = _list_datasets(exe, remote)
    if not names:
        sys.exit(f"[dl-merge] nenhuma pasta de dataset em {remote} (o backup já rodou?).")
    print(f"[dl-merge] {len(names)} pasta(s) a baixar: {names}")

    tmp_ctx = (tempfile.TemporaryDirectory(prefix="dl_merge_")
               if not args.stage else None)
    stage = Path(tmp_ctx.name) if tmp_ctx else args.stage
    stage.mkdir(parents=True, exist_ok=True)
    try:
        ds_dirs = []
        for name in names:
            src = f"{remote}/{name}"
            dst = stage / name
            print(f"[dl-merge] baixando {src} -> {dst} ...")
            rc = subprocess.run([exe, "copy", src, str(dst),
                                 "--transfers=8", "--checkers=8", "--progress"]).returncode
            if rc != 0:
                sys.exit(f"[dl-merge] rclone copy falhou (rc={rc}) em {src}")
            if not (dst / "annotations").is_dir() and not any(dst.glob("*/annotations")):
                print(f"[dl-merge] AVISO: {dst} sem 'annotations/' — vazio? pulando.")
                continue
            ds_dirs.append(dst)

        with tempfile.TemporaryDirectory(prefix="dl_merge_zips_") as ztmp:
            resolved = [root for d in ds_dirs
                        for root in _resolve_sources(d, Path(ztmp))]
            if not resolved:
                sys.exit("[dl-merge] nada p/ juntar (nenhum dataset resolvido).")
            merge(resolved, args.out)
    finally:
        if tmp_ctx and not args.keep_stage:
            tmp_ctx.cleanup()
        elif args.keep_stage:
            print(f"[dl-merge] staging mantido em: {stage}")


if __name__ == "__main__":
    main()
