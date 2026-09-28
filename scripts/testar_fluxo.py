"""
Cliente de linha de comando que faz o mesmo que o app Flutter: envia o vídeo,
recebe o job_id e fica em polling até o resultado sair.

    python scripts/testar_fluxo.py caminho/para/video.mp4 --user-id joao

Requer: pip install requests
"""

import argparse
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--intervalo", type=float, default=1.5)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Arquivo não encontrado: {args.video}")
        return 1

    with args.video.open("rb") as f:
        dados = {"user_id": args.user_id} if args.user_id else {}
        resp = requests.post(
            f"{args.api}/auth/login",
            files={"video": (args.video.name, f, "video/mp4")},
            data=dados, timeout=60,
        )

    if resp.status_code != 202:
        print(f"Login recusado ({resp.status_code}): {resp.text}")
        return 1

    job_id = resp.json()["job_id"]
    print(f"job_id={job_id} -> aguardando...")

    limite = time.time() + args.timeout
    while time.time() < limite:
        r = requests.get(f"{args.api}/auth/result/{job_id}", timeout=30).json()
        if r["status"] == "processando":
            print(f"  [{r['decorrido_s']:>5.1f}s] {r['etapa']}")
            time.sleep(args.intervalo)
            continue

        if r["status"] == "falhou":
            print(f"\nFALHOU ({r['erro_codigo']}): {r['erro']}")
            return 1

        res = r["resultado"]
        print(f"\nVeredicto: {res['veredicto'].upper()}  [{res['codigo']}]")
        print(f"Mensagem : {res['mensagem']}")
        if res.get("ia"):
            ia = res["ia"]
            print(f"Logits                  : {ia['logits']:.4f}")
            print(f"Probabilidade deepfake  : {ia['probabilidade_deepfake_pct']:.2f}%")
            print(f"Liveness score          : {ia['liveness_score_pct']:.2f}%")
        if res.get("qualidade"):
            q = res["qualidade"]
            print(f"Qualidade: deteccao={q['quality_ratio']:.2f} snr={q['signal_snr']:.2f}")
        if res.get("access_token"):
            print(f"Token    : {res['access_token'][:48]}...")
        return 0 if res["aprovado"] else 2

    print("Tempo limite excedido.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
