"""
Detecção facial + desenho do retângulo de reconhecimento.

Reutiliza o MESMO FaceMeshROIExtractor do rppg_pipeline.py em vez de trazer
outro detector (Haar cascade, por exemplo). Motivo: se a API aprovasse o rosto
com um detector e o pipeline rPPG não encontrasse rosto com outro, o usuário
receberia "rosto detectado" seguido de falha inexplicável. Um detector só =
uma verdade só.

Esta passada também funciona como gate barato: se quase nenhum frame tem
rosto, dá para recusar antes de pagar o custo do rPPG completo (detrend +
bandpass + CWT em 3 ROIs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from app.config import settings


class FaceError(Exception):
    def __init__(self, mensagem: str, codigo: str = "NENHUM_ROSTO") -> None:
        super().__init__(mensagem)
        self.codigo = codigo


@dataclass
class RelatorioFacial:
    frames_totais: int
    frames_com_rosto: int
    video_anotado: Path | None

    @property
    def ratio(self) -> float:
        return self.frames_com_rosto / self.frames_totais if self.frames_totais else 0.0


def _detectar_bbox(extractor, frame):
    """Devolve (left, right, top, bottom) do rosto ou None.

    Usa os helpers internos do extractor (_mp, landmarker, _landmarks_px,
    _bbox) de propósito: são exatamente os mesmos passos que get_rois() faz
    para achar o rosto, e replicá-los aqui garante consistência total com o
    pipeline. Se a assinatura desses helpers mudar no rppg_pipeline.py, esta
    função precisa acompanhar.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = extractor._mp.Image(
        image_format=extractor._mp.ImageFormat.SRGB, data=rgb)
    resultado = extractor.landmarker.detect(mp_image)
    if not resultado.face_landmarks:
        return None
    pts = extractor._landmarks_px(frame, resultado.face_landmarks[0])
    return extractor._bbox(pts)


def anotar_video(origem: Path, destino: Path, extractor,
                 desenhar_rois: bool | None = None) -> RelatorioFacial:
    """Percorre o vídeo, desenha um retângulo verde sobre o rosto detectado e
    grava a cópia anotada. Devolve estatísticas de detecção.

    O vídeo anotado NÃO é o que vai para o pipeline rPPG — os retângulos
    desenhados alterariam os pixels e contaminariam a média do canal verde.
    Ele serve para auditoria/depuração e para o app exibir o feedback visual.
    """
    desenhar_rois = settings.draw_rois if desenhar_rois is None else desenhar_rois

    cap = cv2.VideoCapture(str(origem))
    if not cap.isOpened():
        raise FaceError("Não foi possível abrir o vídeo normalizado.", "VIDEO_ILEGIVEL")

    fps = cap.get(cv2.CAP_PROP_FPS) or settings.target_fps
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if settings.annotate_video:
        destino.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(destino), fourcc, fps, (largura, altura))

    total = detectados = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1

            bbox = _detectar_bbox(extractor, frame)
            if bbox is not None:
                detectados += 1

            if writer is None:
                continue

            saida = extractor.debug_draw(frame) if (desenhar_rois and bbox) else frame.copy()
            if bbox is not None:
                left, right, top, bottom = bbox
                cv2.rectangle(saida, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(saida, "ROSTO DETECTADO", (left, max(20, top - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(saida, "SEM ROSTO", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(saida)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if total == 0:
        raise FaceError("O vídeo não possui frames legíveis.", "VIDEO_ILEGIVEL")

    relatorio = RelatorioFacial(
        frames_totais=total,
        frames_com_rosto=detectados,
        video_anotado=destino if writer is not None else None,
    )

    if relatorio.ratio < settings.min_face_frames_ratio:
        raise FaceError(
            f"Rosto detectado em apenas {relatorio.ratio * 100:.0f}% dos frames "
            f"(mínimo {settings.min_face_frames_ratio * 100:.0f}%). Posicione o "
            f"rosto no centro da câmera, com boa iluminação, e tente novamente.",
            "NENHUM_ROSTO")

    return relatorio
