"""Coordenação dinâmica de várias máquinas via Drive (sem RANK manual).

Cada máquina, ao ligar, reivindica o próximo LOTE (chunk) ainda livre de um job de
``total`` amostras dividido em lotes de ``chunk`` — gera aquele lote, faz backup e
marca como concluído; repete até acabar. Ligar/desligar máquinas à vontade: quem
liga depois vê os lotes já tomados e pega o próximo. Como o gerador nomeia arquivos
por índice global (``nao_{start+i}``) e semeia por ``start``, lotes disjuntos →
imagens e poses globalmente únicas (o merge no fim só concatena).

O único estado compartilhado é o Drive (via rclone). Não há lock atômico, então a
reivindicação é "escreve → propaga → relê → confirma": last-writer-wins resolve
empates de forma determinística; quem perde tenta o próximo lote. Lotes reivindicados
mas SEM heartbeat há ``stale_s`` (máquina caiu) voltam a ficar disponíveis e são
retomados pelo JSON parcial já no Drive (o ``--resume`` do gerador pula o que já foi).

Registro de claims na nuvem: ``{remote}/_claims/chunk_{i:05d}.json`` contendo
``{"machine": id, "ts": epoch, "done": bool}``.

Testável sem Drive real: passe um caminho LOCAL como ``remote`` (o rclone trata
qualquer pasta como um "remote").
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


def num_chunks(total: int, chunk: int) -> int:
    """Quantos lotes de tamanho ``chunk`` cobrem ``total`` amostras."""
    if total < 0 or chunk < 1:
        raise ValueError("total>=0 e chunk>=1")
    return math.ceil(total / chunk)


def chunk_span(idx: int, total: int, chunk: int) -> tuple[int, int]:
    """Intervalo (start, num) do lote ``idx`` — o último pode ser menor."""
    start = idx * chunk
    return start, max(0, min(chunk, total - start))


def _claims_dir(remote: str) -> str:
    return f"{remote.rstrip('/')}/_claims"


def _claim_path(remote: str, idx: int) -> str:
    return f"{_claims_dir(remote)}/chunk_{idx:05d}.json"


def _rclone(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True, **kw)


def read_claims(remote: str) -> dict[int, dict]:
    """Lê TODOS os claims de uma vez (1 rclone copy do _claims p/ um tmp local)."""
    claims: dict[int, dict] = {}
    with tempfile.TemporaryDirectory(prefix="claims_") as tmp:
        r = _rclone(["copy", _claims_dir(remote), tmp,
                     "--transfers=8", "--checkers=8"])
        if r.returncode != 0:
            return claims                     # _claims ainda não existe → vazio
        for jf in Path(tmp).glob("chunk_*.json"):
            try:
                idx = int(jf.stem.split("_")[1])
                claims[idx] = json.loads(jf.read_text())
            except (ValueError, IndexError, json.JSONDecodeError):
                continue
    return claims


def write_claim(remote: str, idx: int, machine: str, done: bool = False) -> bool:
    """Escreve/atualiza o claim do lote ``idx`` (rcat: stdin → objeto na nuvem)."""
    payload = json.dumps({"machine": machine, "ts": time.time(), "done": done})
    r = _rclone(["rcat", _claim_path(remote, idx)], input=payload)
    return r.returncode == 0


def read_claim(remote: str, idx: int) -> dict | None:
    """Lê um claim específico. None se não existir."""
    r = _rclone(["cat", _claim_path(remote, idx)])
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _available(idx: int, claims: dict[int, dict], now: float, stale_s: float) -> bool:
    """Lote livre: sem claim, ou claim não-concluído e sem heartbeat há stale_s."""
    c = claims.get(idx)
    if c is None:
        return True
    if c.get("done"):
        return False
    return (now - c.get("ts", 0)) > stale_s


@dataclass(frozen=True)
class Claim:
    """Lote reivindicado com sucesso por esta máquina."""
    idx: int
    start: int
    num: int


def claim_next(remote: str, total: int, chunk: int, machine: str,
               stale_s: float = 2700.0, settle_s: float = 5.0,
               max_attempts: int | None = None) -> Claim | None:
    """Reivindica o menor lote livre. None se não há mais nada a fazer.

    Protocolo por tentativa: acha o menor lote livre → escreve o claim → espera
    ``settle_s`` (propagação no Drive) → relê. Se o claim ainda é meu, ganhei; se
    outra máquina sobrescreveu (last-writer-wins), perdi e tento o próximo.
    """
    n = num_chunks(total, chunk)
    attempts = max_attempts if max_attempts is not None else 2 * n + 10
    for _ in range(attempts):
        claims = read_claims(remote)
        now = time.time()
        idx = next((i for i in range(n) if _available(i, claims, now, stale_s)), None)
        if idx is None:
            return None                       # tudo concluído ou em andamento
        if not write_claim(remote, idx, machine):
            time.sleep(settle_s)
            continue                          # falha de rede; tenta de novo
        time.sleep(settle_s)
        winner = read_claim(remote, idx)
        if winner and winner.get("machine") == machine and not winner.get("done"):
            start, num = chunk_span(idx, total, chunk)
            return Claim(idx=idx, start=start, num=num)
        # Outra máquina venceu o empate — tenta o próximo lote livre.
    return None


def heartbeat(remote: str, idx: int, machine: str) -> bool:
    """Renova o timestamp do claim (sinal de vida enquanto gera o lote)."""
    return write_claim(remote, idx, machine, done=False)


def mark_done(remote: str, idx: int, machine: str) -> bool:
    """Marca o lote como concluído (não será mais reivindicado)."""
    return write_claim(remote, idx, machine, done=True)


def progress(remote: str, total: int, chunk: int) -> tuple[int, int, int]:
    """(concluídos, em_andamento, total_de_lotes) segundo os claims na nuvem."""
    n = num_chunks(total, chunk)
    claims = read_claims(remote)
    done = sum(1 for c in claims.values() if c.get("done"))
    active = sum(1 for i, c in claims.items()
                 if not c.get("done") and i < n)
    return done, active, n


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Inspeciona o quadro de claims na nuvem.")
    ap.add_argument("--remote", required=True, help="Raiz do backup (ex.: gdrive:synthpose-backup).")
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()
    done, active, n = progress(args.remote, args.total, args.chunk)
    print(f"lotes: {done}/{n} concluídos, {active} em andamento "
          f"({done * args.chunk}/{args.total} imgs aprox.)")
    for i, c in sorted(read_claims(args.remote).items()):
        age = int(time.time() - c.get("ts", 0))
        state = "done" if c.get("done") else f"ativo(há {age}s)"
        print(f"  chunk {i:05d}  {state:16s} machine={c.get('machine')}")


if __name__ == "__main__":
    main()
