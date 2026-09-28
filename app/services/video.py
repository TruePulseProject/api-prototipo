"""
Normalização do vídeo com FFmpeg (garantia de 30 FPS).

Por que isso é obrigatório e não "opcional": o rppg_pipeline.py reamostra os
sinais para 150 amostras assumindo captura de 5s a 30fps, e usa o FPS lido do
container (cv2.CAP_PROP_FPS) para converter atraso em milissegundos em
compute_sync_delay_ms(). Webcam de navegador/celular frequentemente entrega
VFR (frame rate variável) com metadado mentiroso — nesse caso o cálculo de ms
sai errado silenciosamente. Reencodar com CFR 30fps elimina essa classe de bug.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


class VideoError(Exception):
    """Erro de validação/transcodificação do vídeo enviado."""

    def __init__(self, mensagem: str, codigo: str = "VIDEO_INVALIDO") -> None:
        super().__init__(mensagem)
        self.codigo = codigo


@dataclass
class VideoInfo:
    duracao_s: float
    fps: float
    largura: int
    altura: int
    codec: str


def ffmpeg_disponivel() -> bool:
    return shutil.which(settings.ffmpeg_bin) is not None and \
        shutil.which(settings.ffprobe_bin) is not None


def _parse_fraction(valor: str | None) -> float:
    """ffprobe devolve fps como fração ('30000/1001'). Converte para float."""
    if not valor:
        return 0.0
    if "/" in valor:
        num, _, den = valor.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(valor)
    except ValueError:
        return 0.0


def probe(path: Path) -> VideoInfo:
    """Lê metadados do vídeo com ffprobe."""
    if not shutil.which(settings.ffprobe_bin):
        raise VideoError("ffprobe não encontrado no PATH.", "FFMPEG_AUSENTE")

    cmd = [
        settings.ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,width,height,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise VideoError(
            "Não foi possível ler o vídeo enviado (arquivo corrompido ou "
            "formato não suportado).", "VIDEO_ILEGIVEL")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise VideoError("O arquivo enviado não contém trilha de vídeo.", "SEM_TRILHA_VIDEO")

    s = streams[0]
    duracao = float(s.get("duration") or data.get("format", {}).get("duration") or 0.0)
    fps = _parse_fraction(s.get("avg_frame_rate")) or _parse_fraction(s.get("r_frame_rate"))
    return VideoInfo(
        duracao_s=duracao,
        fps=fps,
        largura=int(s.get("width") or 0),
        altura=int(s.get("height") or 0),
        codec=str(s.get("codec_name") or "desconhecido"),
    )


def validar_duracao(info: VideoInfo) -> None:
    esperado = settings.expected_duration_s
    tol = settings.duration_tolerance_s
    if info.duracao_s <= 0:
        raise VideoError("Duração do vídeo não pôde ser determinada.", "DURACAO_INVALIDA")
    if abs(info.duracao_s - esperado) > tol:
        raise VideoError(
            f"O vídeo tem {info.duracao_s:.1f}s; a captura precisa ter "
            f"~{esperado:.0f}s (tolerância de {tol:.1f}s). Refaça a captura.",
            "DURACAO_INVALIDA")


def normalizar_para_30fps(origem: Path, destino: Path) -> VideoInfo:
    """Reencoda o vídeo para CFR (constant frame rate) no FPS alvo.

    -vsync cfr + -r garantem frame rate constante: o ffmpeg duplica ou
    descarta frames conforme necessário. O áudio é descartado (-an): o
    pipeline rPPG usa só o canal verde da imagem, e guardar áudio de um dado
    biométrico sem necessidade é risco desnecessário.
    """
    if not shutil.which(settings.ffmpeg_bin):
        raise VideoError("ffmpeg não encontrado no PATH.", "FFMPEG_AUSENTE")

    destino.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.ffmpeg_bin, "-y",
        "-i", str(origem),
        "-vf", f"fps={settings.target_fps},format=yuv420p",
        "-vsync", "cfr",
        "-r", str(settings.target_fps),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",          # qualidade alta: compressão agressiva destrói
                               # as micro-variações de cor que o rPPG mede
        "-an",
        "-movflags", "+faststart",
        str(destino),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=settings.ffmpeg_timeout_s)
    if proc.returncode != 0 or not destino.exists():
        erro = (proc.stderr or "").strip().splitlines()[-5:]
        raise VideoError(
            "Falha ao normalizar o vídeo para 30 FPS: " + " | ".join(erro),
            "FALHA_FFMPEG")

    info = probe(destino)
    if abs(info.fps - settings.target_fps) > 1.0:
        raise VideoError(
            f"Vídeo normalizado ficou com {info.fps:.2f} FPS em vez de "
            f"{settings.target_fps}.", "FPS_INCORRETO")
    return info
