import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.config import ALLOWED_VIDEO_TYPES, FRAMES_DIR, MAX_VIDEO_SIZE_MB, VIDEOS_DIR
from app.models import JobStatus, LoginResponse, VerificationResult
from app.services import ai_verification, extrair_frames, facerec_frames
from app.storage import create_job, get_job, update_job

router = APIRouter()


def _process_video(job_id: str, video_path: Path) -> None:
    """
    Roda em background depois que o endpoint /auth/login já respondeu.

    Pipeline:
        1. script1 (extrair_frames)  -> vídeo bruto vira 150 frames (.jpg),
           convertendo para 30fps antes se necessário.
        2. script2 (facerec_frames)  -> roda detecção facial em cada frame
           extraído e informa quais frames têm rosto.
        3. Só os frames COM rosto detectado são enviados para a IA de
           deepfake/identidade.
    """
    try:
        frames_brutos_dir = FRAMES_DIR / job_id / "brutos"
        extrair_frames.processar_video(
            str(video_path),
            str(frames_brutos_dir),
            quantidade_frames=150,
        )

        frames_detectados_dir = FRAMES_DIR / job_id / "detectados"
        resultados = facerec_frames.processar_frames(
            str(frames_brutos_dir),
            str(frames_detectados_dir),
        )

        frames_com_rosto = [Path(caminho) for caminho, n_rostos in resultados if n_rostos > 0]

        if not frames_com_rosto:
            update_job(
                job_id,
                status=JobStatus.REJECTED,
                message="Nenhum rosto detectado nos frames extraídos do vídeo.",
            )
            return

        is_deepfake, is_correct_person, confidence = ai_verification.verify_frames(frames_com_rosto)
        approved = (not is_deepfake) and is_correct_person

        update_job(
            job_id,
            status=JobStatus.APPROVED if approved else JobStatus.REJECTED,
            is_deepfake=is_deepfake,
            is_correct_person=is_correct_person,
            confidence=confidence,
            message="Login aprovado." if approved else "Login recusado.",
        )
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, status=JobStatus.ERROR, message=f"Erro no processamento: {exc}")


@router.post("/login", response_model=LoginResponse)
async def login(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """
    Recebe o vídeo de ~5s capturado pela webcam no Flutter.
    Salva o arquivo, dispara o pipeline em background e responde na hora
    com um job_id que o Flutter usa para consultar o resultado depois.
    """
    if video.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Formato de vídeo não suportado.")

    job_id = str(uuid.uuid4())
    extension = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = VIDEOS_DIR / f"{job_id}{extension}"

    size = 0
    max_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    with open(video_path, "wb") as buffer:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                buffer.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"Vídeo excede o tamanho máximo permitido ({MAX_VIDEO_SIZE_MB}MB).",
                )
            buffer.write(chunk)

    create_job(job_id)
    background_tasks.add_task(_process_video, job_id, video_path)

    return LoginResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message="Vídeo recebido, processando verificação.",
    )


@router.get("/result/{job_id}", response_model=VerificationResult)
async def get_result(job_id: str):
    """
    O Flutter chama esse endpoint (via polling, ex: a cada 1s) até o status
    deixar de ser 'processing' e virar 'approved', 'rejected' ou 'error'.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return job
