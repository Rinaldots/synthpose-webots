"""Monitor de GPU + progresso da geração, com canal remoto opcional via ntfy.sh.

Mostra ao vivo, para a sessão onde roda:
  - GPU(s) via ``nvidia-smi``: nome, utilização %, VRAM usada/total, temperatura, watts.
  - Progresso da geração: imagens feitas / alvo, taxa (img/s) e ETA — contando os PNGs
    em ``<out>/images/**`` (não depende do stdout do Blender).

``nvidia-smi`` só enxerga a GPU da própria máquina, então cada notebook Kaggle
monitora a sua. Para acompanhar VÁRIAS máquinas de fora, cada uma **publica** seu
snapshot num tópico ntfy.sh e o seu PC **assina** o mesmo tópico, agregando tudo
num painel único. Kaggle só tem internet de saída — ntfy (Kaggle publica → PC lê)
contorna isso sem servidor próprio. Sem dependências além da stdlib.

Uso (dentro de cada máquina Kaggle, publicando):
    python scripts/monitor.py --out /kaggle/working/output \
        --total 10000 --world-size 2 --rank 0 --ntfy meu-topico-secreto-1234

Uso (no seu PC, agregando as máquinas):
    python scripts/monitor.py --subscribe meu-topico-secreto-1234 --world-size 2

Uso (embutido, em paralelo com a geração — ver run_with_monitor):
    from scripts.monitor import run_with_monitor
    run_with_monitor(cmd, cwd=BLENDER_DIR, out_dir=OUT_DIR, num=SHARD_NUM,
                     rank=RANK, ntfy_topic="meu-topico-secreto-1234")

Nota de privacidade: qualquer um que saiba o nome do tópico consegue ler. Use um
nome longo e aleatório (é a sua "senha").
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

_NVIDIA_QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
NTFY_DEFAULT_SERVER = "https://ntfy.sh"


# ---------------------------------------------------------------------------
# Coleta local
# ---------------------------------------------------------------------------
def gpu_stats() -> list[dict]:
    """Lê estado das GPUs via nvidia-smi. Retorna [] se não houver GPU/nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, util, mem_u, mem_t, temp, power = parts[:7]
        gpus.append({
            "index": idx, "name": name, "util": util,
            "mem_used": mem_u, "mem_total": mem_t, "temp": temp, "power": power,
        })
    return gpus


def count_done(out_dir: Path, image_format: str = "png") -> int:
    """Conta PNGs sob qualquer subárvore ``images/`` dentro de ``out_dir``.

    Cobre tanto o layout de 1 processo (``output/images/**``) quanto o de várias
    GPUs numa máquina (``output/gpu0/images/**`` + ``output/gpu1/images/**``).
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return 0
    return sum(1 for p in out_dir.rglob(f"*.{image_format}") if "images" in p.parts)


def snapshot_dict(out_dir: Path, num: int | None, t0: float, done0: int,
                  rank: int | None = None, image_format: str = "png") -> dict:
    """Estado estruturado do instante atual (serializável p/ ntfy)."""
    now = time.time()
    done = count_done(out_dir, image_format)
    elapsed = max(now - t0, 1e-6)
    rate = (done - done0) / elapsed
    eta_s = (num - done) / rate if (num and rate > 1e-9) else None
    return {
        "rank": rank,
        "ts": now,
        "done": done,
        "num": num,
        "rate": rate,
        "eta_s": eta_s,
        "gpus": gpu_stats(),
    }


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------
def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt_gpu(g: dict) -> str:
    return (f"GPU{g['index']} {g['name'][:18]:18s} "
            f"util {g['util']:>3s}%  "
            f"vram {g['mem_used']:>6s}/{g['mem_total']:<6s}MB  "
            f"{g['temp']:>2s}C  {g['power']:>5s}W")


def _fmt_progress(d: dict) -> str:
    done, num, rate = d["done"], d["num"], d["rate"]
    if num:
        pct = 100.0 * done / num
        return f"{done:>5d}/{num:<5d} ({pct:5.1f}%)  {rate:4.2f} img/s  ETA {_fmt_eta(d['eta_s'])}"
    return f"{done:>5d} imgs  {rate:4.2f} img/s"


def _format_line(d: dict) -> str:
    """Linha de status local (uma máquina)."""
    ts = time.strftime("%H:%M:%S", time.localtime(d["ts"]))
    gpus = d.get("gpus") or []
    gpu_txt = "  ||  ".join(_fmt_gpu(g) for g in gpus) if gpus else "sem GPU (nvidia-smi ausente)"
    return f"[{ts}] {_fmt_progress(d)}  |  {gpu_txt}"


def snapshot_line(out_dir: Path, num: int | None, t0: float, done0: int,
                  image_format: str = "png") -> str:
    """Compat: linha de status local a partir de um snapshot novo."""
    return _format_line(snapshot_dict(out_dir, num, t0, done0, image_format=image_format))


# ---------------------------------------------------------------------------
# Canal ntfy (publicar)
# ---------------------------------------------------------------------------
def publish_ntfy(topic: str, payload: dict, server: str = NTFY_DEFAULT_SERVER) -> None:
    """Publica `payload` (JSON) no tópico ntfy. Falha silenciosa (não derruba a geração)."""
    url = f"{server.rstrip('/')}/{topic}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Priority": "min",                       # não vira notificação barulhenta
        "Title": f"rank {payload.get('rank')}",
        "Tags": "robot",
    })
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:                        # noqa: BLE001 — monitor nunca deve quebrar a geração
        print(f"[monitor] aviso: falha ao publicar no ntfy ({e})", flush=True)


# ---------------------------------------------------------------------------
# Loop de monitoramento (dentro da máquina que gera)
# ---------------------------------------------------------------------------
def watch(out_dir: Path, num: int | None, interval: float = 5.0,
          stop: threading.Event | None = None, image_format: str = "png",
          rank: int | None = None, ntfy_topic: str | None = None,
          ntfy_server: str = NTFY_DEFAULT_SERVER) -> None:
    """Imprime (e opcionalmente publica no ntfy) um snapshot a cada `interval` s.

    Para quando `stop` é setado (modo embutido) ou o alvo `num` é atingido.
    """
    out_dir = Path(out_dir)
    t0 = time.time()
    done0 = count_done(out_dir, image_format)
    dest = f"  ->ntfy:{ntfy_topic}" if ntfy_topic else ""
    print(f"[monitor] observando {out_dir}  alvo={num or '?'}  intervalo={interval}s{dest}", flush=True)
    while True:
        d = snapshot_dict(out_dir, num, t0, done0, rank=rank, image_format=image_format)
        print(_format_line(d), flush=True)
        if ntfy_topic:
            publish_ntfy(ntfy_topic, d, ntfy_server)
        if num and d["done"] >= num:
            print("[monitor] alvo atingido.", flush=True)
            return
        if stop is not None and stop.wait(interval):
            print("[monitor] encerrado.", flush=True)
            return
        if stop is None:
            time.sleep(interval)


def _shard_num(total: int | None, world_size: int | None, rank: int | None,
               num: int | None) -> int | None:
    """Deriva o alvo desta máquina: --num direto, ou a fatia de --total/--world-size/--rank."""
    if num is not None:
        return num
    if total is not None and world_size is not None and rank is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from nao_coco_pose.sharding import shard_range
        return shard_range(total, world_size, rank).num
    return None


def run_with_monitor(cmd: list[str], out_dir: str | Path, num: int | None = None,
                     cwd: str | Path | None = None, interval: float = 5.0,
                     rank: int | None = None, ntfy_topic: str | None = None,
                     ntfy_server: str = NTFY_DEFAULT_SERVER) -> int:
    """Roda `cmd` (a geração) e, em paralelo, um monitor até o processo terminar.

    Se `ntfy_topic` for dado, cada snapshot também é publicado no tópico (para o
    painel agregador rodando no seu PC). Retorna o returncode do processo.
    """
    stop = threading.Event()
    th = threading.Thread(
        target=watch,
        kwargs=dict(out_dir=Path(out_dir), num=num, interval=interval, stop=stop,
                    rank=rank, ntfy_topic=ntfy_topic, ntfy_server=ntfy_server),
        daemon=True,
    )
    th.start()
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True)
        return proc.returncode
    finally:
        stop.set()
        th.join(timeout=interval + 2)
        d = snapshot_dict(Path(out_dir), num, time.time() - interval,
                          max(count_done(Path(out_dir)) - 1, 0), rank=rank)
        print(_format_line(d), flush=True)
        if ntfy_topic:
            publish_ntfy(ntfy_topic, d, ntfy_server)


# ---------------------------------------------------------------------------
# Painel agregador (no seu PC) — assina o tópico e junta todas as máquinas
# ---------------------------------------------------------------------------
def _render_merged(state: dict[object, dict], world_size: int | None,
                   stale_after: float) -> None:
    """Redesenha o painel com o último snapshot de cada rank."""
    now = time.time()
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")         # limpa a tela
    ranks = sorted(state, key=lambda r: (r is None, r))
    header = time.strftime("%H:%M:%S", time.localtime(now))
    conn = f"{len(ranks)}" + (f"/{world_size}" if world_size else "")
    print(f"=== painel ntfy — {conn} máquina(s) ===  {header}")

    tot_done = tot_num = 0
    for r in ranks:
        d = state[r]
        age = now - d.get("_recv", now)
        stale = "  [STALE]" if age > stale_after else ""
        gpus = d.get("gpus") or []
        gpu_txt = "  ".join(_fmt_gpu(g) for g in gpus) if gpus else "sem GPU"
        print(f"rank {str(r):>2} | {_fmt_progress(d)} | {gpu_txt} | há {int(age):>3d}s{stale}")
        tot_done += d.get("done", 0)
        if d.get("num"):
            tot_num += d["num"]
    if tot_num:
        pct = 100.0 * tot_done / tot_num
        print(f"{'TOTAL':>6} | {tot_done}/{tot_num} ({pct:.1f}%)")
    print(flush=True)


def subscribe(topic: str, server: str = NTFY_DEFAULT_SERVER,
              world_size: int | None = None, stale_after: float = 60.0) -> None:
    """Assina o tópico ntfy e mostra um painel com o estado de todas as máquinas.

    Reconecta sozinho se a conexão cair. Ctrl+C para sair.
    """
    url = f"{server.rstrip('/')}/{topic}/json?since=5m"
    state: dict[object, dict] = {}                # rank -> último payload (+ _recv)
    print(f"[monitor] assinando {url}\n", flush=True)
    while True:
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                for raw in resp:                  # stream: uma linha JSON por evento
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("event") != "message":
                        continue                  # 'open'/'keepalive' — ignora
                    try:
                        payload = json.loads(evt["message"])
                    except (KeyError, json.JSONDecodeError):
                        continue
                    payload["_recv"] = time.time()
                    state[payload.get("rank")] = payload
                    _render_merged(state, world_size, stale_after)
        except KeyboardInterrupt:
            print("\n[monitor] painel encerrado.")
            return
        except Exception as e:                    # noqa: BLE001
            print(f"[monitor] conexão caiu ({e}); reconectando em 3s...", flush=True)
            time.sleep(3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="Diretório de saída da geração.")
    ap.add_argument("--num", type=int, default=None, help="Alvo desta máquina (imagens).")
    ap.add_argument("--total", type=int, default=None, help="Total do job (com --world-size/--rank).")
    ap.add_argument("--world-size", type=int, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--interval", type=float, default=5.0, help="Segundos entre snapshots.")
    ap.add_argument("--once", action="store_true", help="Imprime um snapshot e sai.")
    # Canal ntfy.
    ap.add_argument("--ntfy", type=str, default=None,
                    help="Tópico ntfy para PUBLICAR os snapshots (modo Kaggle).")
    ap.add_argument("--subscribe", type=str, default=None,
                    help="Tópico ntfy para ASSINAR e agregar as máquinas (modo PC).")
    ap.add_argument("--ntfy-server", type=str, default=NTFY_DEFAULT_SERVER,
                    help=f"Servidor ntfy (default: {NTFY_DEFAULT_SERVER}).")
    args = ap.parse_args()

    if args.subscribe:
        subscribe(args.subscribe, args.ntfy_server, args.world_size,
                  stale_after=max(2 * args.interval, 30.0))
        return

    if args.out is None:
        ap.error("--out é obrigatório (ou use --subscribe para o painel).")

    num = _shard_num(args.total, args.world_size, args.rank, args.num)
    if args.once:
        d = snapshot_dict(args.out, num, time.time(), count_done(args.out), rank=args.rank)
        print(_format_line(d))
        if args.ntfy:
            publish_ntfy(args.ntfy, d, args.ntfy_server)
        return
    try:
        watch(args.out, num, args.interval, rank=args.rank,
              ntfy_topic=args.ntfy, ntfy_server=args.ntfy_server)
    except KeyboardInterrupt:
        print("\n[monitor] interrompido pelo usuário.")


if __name__ == "__main__":
    main()
