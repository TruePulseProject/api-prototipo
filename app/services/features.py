"""
Ponte entre rppg_pipeline.build_feature_vector() e os tensores que o
TruePulseDiscriminator espera.

Responsabilidades:
  1. Chamar o pipeline rPPG sobre o vídeo já normalizado a 30fps.
  2. Aplicar os limiares de qualidade IMPORTADOS do pipeline (nunca copiados).
  3. Montar os tensores signals (1,3,150), aux (1,n_aux) e freq (1,3*14).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.ml.model import FREQ_BAND, N_FREQ_BINS
from app.ml.rppg_pipeline import (
    MIN_QUALITY_RATIO,
    MIN_SIGNAL_SNR,
    TARGET_LEN_PADRAO,
    build_feature_vector,
)


class QualidadeInsuficiente(Exception):
    """Vídeo válido, mas fora da faixa de qualidade vista no treino.

    O rppg_pipeline.py é explícito: uma captura abaixo desses limiares
    representa uma condição que o modelo nunca viu, e deve ser rejeitada
    (pedir nova captura) em vez de classificada com confiança enganosa. Por
    isso é exceção própria, e não uma reprovação de login — o usuário não
    "falhou" na biometria, a amostra é que não serve.
    """

    def __init__(self, mensagem: str, detalhes: dict) -> None:
        super().__init__(mensagem)
        self.detalhes = detalhes
        self.codigo = "QUALIDADE_INSUFICIENTE"


class FalhaExtracao(Exception):
    def __init__(self, mensagem: str, codigo: str = "FALHA_RPPG") -> None:
        super().__init__(mensagem)
        self.codigo = codigo


@dataclass
class Features:
    signals: torch.Tensor          # (1, 3, 150)
    aux: torch.Tensor              # (1, n_aux)
    freq: torch.Tensor | None      # (1, 3*N_FREQ_BINS) ou None
    bruto: dict                    # dict original do build_feature_vector


def _espectro_faixa_cardiaca(sinal: np.ndarray, fps: float,
                             n_bins: int = N_FREQ_BINS,
                             banda: tuple[float, float] = FREQ_BAND) -> np.ndarray:
    """Magnitude da FFT restrita à faixa cardíaca, normalizada, com n_bins.

    ATENÇÃO: esta função precisa ser IDÊNTICA à usada no train.py quando o
    modelo foi treinado com --use-freq. Se o treino normalizou diferente (por
    exemplo, sem dividir pela soma), a inferência vira lixo silenciosamente —
    o modelo não erra alto, ele só erra. Se você usa use_freq=True, copie a
    implementação do treino para cá em vez de confiar nesta reconstrução.

    Para seq_len=150 a 30fps, rfftfreq dá passo de 0.2Hz, e a faixa
    0.7-3.5Hz cai exatamente em 14 bins (0.8Hz a 3.4Hz) = N_FREQ_BINS.
    """
    espectro = np.abs(np.fft.rfft(sinal))
    freqs = np.fft.rfftfreq(len(sinal), d=1.0 / fps)
    mascara = (freqs >= banda[0]) & (freqs <= banda[1])
    valores = espectro[mascara]

    if valores.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    if valores.size != n_bins:
        # Reamostra por interpolação se o FPS real fizer a faixa cair num
        # número diferente de bins (não deveria acontecer após o ffmpeg,
        # mas é barato garantir que o shape nunca quebre o forward).
        valores = np.interp(np.linspace(0, 1, n_bins),
                            np.linspace(0, 1, valores.size), valores)
    valores = valores / (valores.sum() + 1e-8)
    return valores.astype(np.float32)


def extrair_features(video_path: Path, extractor, n_aux: int = 2,
                     use_freq: bool = False) -> Features:
    try:
        bruto = build_feature_vector(str(video_path), extractor,
                                     target_len=TARGET_LEN_PADRAO)
    except ValueError as exc:
        # build_feature_vector levanta ValueError quando o vídeo não tem
        # amostras suficientes de rosto ou é curto demais para filtrar.
        raise FalhaExtracao(str(exc), "SINAL_INSUFICIENTE") from exc

    qualidade = {
        "quality_ratio": float(bruto["quality_ratio"]),
        "quality_ratio_min": MIN_QUALITY_RATIO,
        "signal_snr": float(bruto["signal_snr"]),
        "signal_snr_min": MIN_SIGNAL_SNR,
        "fps_detectado": float(bruto["fps"]),
    }
    if (qualidade["quality_ratio"] < MIN_QUALITY_RATIO
            or qualidade["signal_snr"] < MIN_SIGNAL_SNR):
        raise QualidadeInsuficiente(
            "A qualidade do vídeo está abaixo do mínimo exigido "
            f"(detecção facial {qualidade['quality_ratio'] * 100:.0f}% / mínimo "
            f"{MIN_QUALITY_RATIO * 100:.0f}%; qualidade do sinal "
            f"{qualidade['signal_snr']:.2f} / mínimo {MIN_SIGNAL_SNR:.2f}). "
            "Melhore a iluminação, fique parado e refaça a captura.",
            qualidade)

    sinais = np.stack([
        bruto["signal_testa"],
        bruto["signal_bochecha_d"],
        bruto["signal_bochecha_e"],
    ]).astype(np.float32)                       # (3, 150)
    signals = torch.from_numpy(sinais).unsqueeze(0)  # (1, 3, 150)

    # aux: as 2 primeiras colunas são SEMPRE os atrasos em ms (o model.py
    # normaliza por DELAY_NORM_MS internamente). n_aux=4 acrescenta
    # quality_ratio e signal_snr, que já vêm em escala ~0-1.
    valores_aux = [float(bruto["delay_d_ms"]), float(bruto["delay_e_ms"])]
    if n_aux >= 4:
        valores_aux += [qualidade["quality_ratio"], qualidade["signal_snr"]]
    valores_aux = valores_aux[:n_aux]
    while len(valores_aux) < n_aux:
        valores_aux.append(0.0)
    aux = torch.tensor([valores_aux], dtype=torch.float32)

    freq = None
    if use_freq:
        fps = qualidade["fps_detectado"] or 30.0
        vetor = np.concatenate([
            _espectro_faixa_cardiaca(bruto["signal_testa"], fps),
            _espectro_faixa_cardiaca(bruto["signal_bochecha_d"], fps),
            _espectro_faixa_cardiaca(bruto["signal_bochecha_e"], fps),
        ])
        freq = torch.from_numpy(vetor).unsqueeze(0)  # (1, 3*N_FREQ_BINS)

    bruto["_qualidade"] = qualidade
    return Features(signals=signals, aux=aux, freq=freq, bruto=bruto)
