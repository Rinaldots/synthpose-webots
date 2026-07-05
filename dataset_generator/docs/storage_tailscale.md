# Armazenar o dataset no SEU PC (via Tailscale + rclone WebDAV)

Alternativa ao Google Drive: em vez de subir pra nuvem (que tem cota de API e limite
de espaço), o Kaggle grava direto no disco do seu PC. Como **todo o pipeline é rclone**
(backup, coordenação de lotes, download), o PC vira só mais um "remote" — nenhuma
mudança de código, só de configuração (`STORAGE="tailscale"` no notebook).

O Kaggle só tem internet de **saída** e não expõe `/dev/net/tun`, então: o PC roda um
servidor WebDAV (`rclone serve webdav`) e ambos entram numa **Tailscale** (VPN P2P
grátis). O Kaggle sobe a Tailscale em **modo userspace** e roteia o rclone por ela.

```
Kaggle  --(HTTP via proxy userspace)-->  Tailscale  -->  PC: rclone serve webdav  -->  disco
```

## Pré-requisitos

- Conta Tailscale (grátis): https://login.tailscale.com — use o mesmo login no PC e
  gere uma **auth key** pro Kaggle.
- Seu PC **ligado durante toda a geração** (o Kaggle grava nele em tempo real).
- Upload de casa razoável: a 1280², ~1–2,5 MB/img. Se o uplink for lento, ele vira o
  gargalo — nesse caso reduza a resolução ou o `BACKUP_EVERY`.

## Setup do PC (uma vez)

```bash
# 1. Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                 # autentica no navegador
tailscale ip -4                   # anote o IP (100.x.y.z) -> vai no notebook (PC_TSIP)

# 2. rclone recente (se ainda nao tiver)
curl -fsSL https://rclone.org/install.sh | sudo bash

# 3. Pasta de armazenamento + servidor WebDAV (deixe rodando; use tmux/screen)
mkdir -p ~/synthpose-backup
rclone serve webdav ~/synthpose-backup \
    --addr 0.0.0.0:8080 \
    --user rinaldo --pass ESCOLHA_UMA_SENHA_FORTE
```

- `--addr 0.0.0.0:8080` escuta em todas as interfaces (inclui a `tailscale0`). O
  acesso fica restrito a quem está na sua tailnet.
- Se usar firewall (ufw), libere a interface do Tailscale: `sudo ufw allow in on tailscale0`.
- Deixe esse comando rodando o tempo todo. Ao acabar, `Ctrl-C` encerra.

## Auth key pro Kaggle

No admin console → **Settings → Keys → Generate auth key**. Marque **Reusable** e
**Ephemeral** (some sozinha quando a sessão Kaggle acaba). Copie a `tskey-auth-...`.

## No notebook (Kaggle)

Na célula de config, deixe `STORAGE = "tailscale"`. Na célula **"Storage: Tailscale"**,
preencha:

- `TS_AUTHKEY`      = a `tskey-auth-...` gerada acima
- `PC_TSIP`         = o IP do `tailscale ip -4` do seu PC
- `PC_WEBDAV_USER`  = o `--user` do serve (ex.: `rinaldo`)
- `PC_WEBDAV_PASS`  = o `--pass` do serve

Rode a célula: ela instala o Tailscale, sobe em modo userspace, roteia o rclone e
valida com `rclone lsd pc:`. Se der `conexao PC OK`, siga a geração normalmente —
o `BACKUP_REMOTE="pc:kaggle"` já aponta pro seu PC, e o coordenador/merge funcionam
igual (os lotes viram `~/synthpose-backup/kaggle/chunk_*`).

No fim, o dataset **já está no seu PC** — não precisa baixar. Junte tudo com:

```bash
python dataset_generator/scripts/download_and_merge.py \
    --remote ~/synthpose-backup/kaggle --out output_merged
```

(`--remote` aceita um caminho local; o rclone trata a pasta como um "remote".)

## Solução de problemas

- **`rclone lsd pc:` falha com timeout**: confirme que o `rclone serve webdav` está
  rodando no PC e que `PC_TSIP` está certo (`tailscale status` mostra os peers).
- **401/403 no WebDAV**: `PC_WEBDAV_USER/PASS` diferentes do `--user/--pass` do serve.
- **`tailscaled` não sobe**: cheque se está em `/usr/sbin` (adicione ao PATH). O modo
  `--tun=userspace-networking` é obrigatório no Kaggle (sem `/dev/net/tun`).
