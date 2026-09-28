"""
Gera um checkpoint NÃO TREINADO só para testar o fluxo da API ponta a ponta
antes de o modelo real estar pronto.

    python scripts/gerar_checkpoint_dummy.py --n-aux 2

ATENÇÃO: os percentuais produzidos por este checkpoint são ruído. Ele existe
para validar plumbing (upload -> ffmpeg -> rPPG -> tensores -> resposta HTTP),
nunca para medir acurácia. Não suba isso em produção.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from app.config import settings  # noqa: E402
from app.ml.model import TruePulseDiscriminator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-aux", type=int, default=2, choices=[2, 4],
                        help="2 = só atrasos; 4 = atrasos + quality_ratio + snr")
    parser.add_argument("--use-freq", action="store_true",
                        help="inclui o ramo de domínio de frequência")
    parser.add_argument("--saida", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    modelo = TruePulseDiscriminator(n_aux=args.n_aux, use_freq=args.use_freq)

    saida = args.saida or settings.checkpoint_path
    saida.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": modelo.state_dict(),
            "config": {"n_aux": args.n_aux, "use_freq": args.use_freq,
                       "seq_len": 150},
            "treinado": False,
            "aviso": "CHECKPOINT DUMMY - pesos aleatorios, sem valor preditivo",
        },
        saida,
    )
    print(f"Checkpoint dummy salvo em: {saida}")
    print("Lembre-se de substituí-lo pelo modelo treinado de verdade.")


if __name__ == "__main__":
    main()
