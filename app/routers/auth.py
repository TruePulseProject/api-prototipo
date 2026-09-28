"""Endpoints de autenticação biométrica."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.config import settings
from app.jobs import JobStatus, get_store
from app.schemas import JobResposta, LoginAceito
from app.services.runtime import get_executor
from app.services.verification import processar_verificacao

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["autenticação"])

CONTENT_TYPES_ACEITOS = {
    "video/mp4", "video/quicktime", "video/webm", "video/x-matroska",
    "video/x-msvideo", "application/octet-stream",
}
EXTENSOES_ACEITAS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
CHUNK = 1024 * 1024  # 1 MiB


async def _salvar_upload(arquivo: UploadFile, destino: Path) -> int:
    """Grava o upload em disco em blocos, respeitando o limite de tamanho.

    Ler em blocos (em vez de await arquivo.read()) evita carregar um vídeo
    inteiro na memória e permite abortar assim que o limite é estourado, em
    vez de depois de já ter consumido tudo.
    """
    limite = settings.max_upload_mb * 1024 * 1024
    total = 0
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("wb") as saida:
        while bloco := await arquivo.read(CHUNK):
            total += len(bloco)
            if total > limite:
                saida.close()
                destino.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Vídeo maior que o limite de {settings.max_upload_mb} MB.")
            saida.write(bloco)
    if total == 0:
        destino.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Arquivo de vídeo vazio.")
    return total


@router.post("/login", response_model=LoginAceito,
             status_code=status.HTTP_202_ACCEPTED,
             summary="Envia o vídeo de 5s e inicia a verificação")
async def login(
    request: Request,
    video: UploadFile = File(..., description="Vídeo de ~5s capturado pela webcam"),
    user_id: str | None = Form(None, description="Identificador do usuário (opcional)"),
) -> LoginAceito:
    """Recebe o vídeo e devolve um job_id imediatamente (HTTP 202).

    A verificação NÃO acontece nesta requisição: ffmpeg + MediaPipe + rPPG +
    inferência levam alguns segundos, e segurar a conexão HTTP todo esse
    tempo convida timeout de proxy e de cliente. O Flutter faz polling em
    /auth/result/{job_id}.
    """
    sufixo = Path(video.filename or "").suffix.lower()
    if video.content_type not in CONTENT_TYPES_ACEITOS and sufixo not in EXTENSOES_ACEITAS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Formato não suportado ({video.content_type or 'desconhecido'}). "
            f"Envie um vídeo {', '.join(sorted(EXTENSOES_ACEITAS))}.")

    store = get_store()
    job = store.create(user_id=user_id)
    destino = settings.uploads_dir / f"{job.id}_bruto{sufixo or '.mp4'}"

    try:
        tamanho = await _salvar_upload(video, destino)
    except HTTPException:
        store.fail(job.id, "Upload rejeitado.", "UPLOAD_INVALIDO")
        raise
    finally:
        await video.close()

    logger.info("[%s] Upload recebido: %s (%.1f MB), user_id=%s",
                job.id, destino.name, tamanho / 1024 / 1024, user_id)

    get_executor().submit(processar_verificacao, job.id, destino, user_id)

    return LoginAceito(
        job_id=job.id,
        poll_url=str(request.url_for("resultado", job_id=job.id)),
    )


@router.get("/result/{job_id}", response_model=JobResposta, name="resultado",
            summary="Consulta o status/resultado da verificação")
async def resultado(job_id: str) -> JobResposta:
    """Enquanto o job roda, devolve status="processando".

    Ao terminar, status vira "concluido" e o campo `resultado` traz o
    veredicto — incluindo `resultado.ia.probabilidade_deepfake_pct`, a
    porcentagem derivada do logit do model.py.

    status="falhou" significa erro do servidor (ex.: checkpoint ausente), não
    reprovação do usuário. Reprovação vem como status="concluido" com
    resultado.aprovado=false.
    """
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Job não encontrado ou expirado. Refaça a autenticação.")

    return JobResposta(
        job_id=job.id,
        status=job.status.value,
        etapa=job.stage.value,
        decorrido_s=job.elapsed_s,
        resultado=job.result if job.status is JobStatus.CONCLUIDO else None,
        erro=job.error,
        erro_codigo=job.error_code,
    )


@router.get("/result/{job_id}/video", summary="Baixa o vídeo com o rosto marcado",
            response_class=FileResponse)
async def video_anotado(job_id: str) -> FileResponse:
    """Devolve a cópia do vídeo com o retângulo de reconhecimento facial
    desenhado. Útil para depuração e para o app mostrar o que foi analisado."""
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job não encontrado ou expirado.")

    caminho = settings.annotated_dir / f"{job_id}_anotado.mp4"
    if not caminho.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Vídeo anotado indisponível (job ainda processando ou "
            "TRUEPULSE_ANNOTATE_VIDEO=false).")
    return FileResponse(caminho, media_type="video/mp4",
                        filename=f"{job_id}_anotado.mp4")
