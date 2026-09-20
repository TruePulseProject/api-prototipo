"""
TruePulse Auth API — ponto de entrada.

Subir em desenvolvimento:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Docs interativas: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.jobs import get_store
from app.routers import auth
from app.schemas import HealthResposta
from app.services import video as video_svc
from app.services.runtime import get_extractor, get_model, modelo_carregado, shutdown_executor
from app.services.verification import limpar_artefatos

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("truepulse")


async def _faxina_periodica() -> None:
    """Remove jobs expirados e seus vídeos.

    Dado biométrico não deve ficar no disco indefinidamente: o TTL padrão é
    30 minutos, tempo de sobra para o app buscar o resultado.
    """
    store = get_store()
    while True:
        try:
            await asyncio.sleep(300)
            for job_id in store.purge_expired():
                limpar_artefatos(job_id)
                logger.info("Job %s expirado: registro e vídeos removidos.", job_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Falha na rotina de limpeza.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()

    if not video_svc.ffmpeg_disponivel():
        logger.warning("ffmpeg/ffprobe NÃO encontrados no PATH — todo login vai "
                       "falhar com FFMPEG_AUSENTE. Instale com: "
                       "sudo apt install ffmpeg")

    # Warm-up: carregar MediaPipe e Torch aqui (e não na primeira requisição)
    # evita que o primeiro usuário pague ~5-10s de inicialização e ache que a
    # autenticação travou. Falha aqui não derruba a API — o /health mostra.
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, get_extractor)
        await loop.run_in_executor(None, get_model)
        logger.info("Warm-up concluído: MediaPipe e modelo prontos.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up incompleto (%s). Verifique /health.", exc)

    faxina = asyncio.create_task(_faxina_periodica())
    logger.info("%s pronta. Armazenamento em %s", settings.app_name, settings.storage_dir)
    try:
        yield
    finally:
        faxina.cancel()
        shutdown_executor()
        logger.info("Encerrando %s.", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Autenticação biométrica por liveness rPPG. Recebe um vídeo de 5s, "
        "normaliza para 30 FPS, detecta o rosto, extrai os sinais rPPG "
        "multi-ROI e classifica com o TruePulseDiscriminator."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)


@app.get("/health", response_model=HealthResposta, tags=["infra"])
async def health() -> HealthResposta:
    """Checagem de dependências — use no readiness probe do deploy."""
    ffmpeg_ok = video_svc.ffmpeg_disponivel()
    checkpoint_ok = settings.checkpoint_path.exists()
    saudavel = ffmpeg_ok and (checkpoint_ok or settings.allow_untrained_model)
    return HealthResposta(
        status="ok" if saudavel else "degradado",
        ffmpeg=ffmpeg_ok,
        checkpoint_carregado=modelo_carregado(),
        detalhes={
            "checkpoint_existe": checkpoint_ok,
            "checkpoint_path": str(settings.checkpoint_path),
            "modelo_nao_treinado_permitido": settings.allow_untrained_model,
            "fps_alvo": settings.target_fps,
            "limiar_deepfake_pct": settings.deepfake_threshold_pct,
        },
    )


@app.get("/", tags=["infra"])
async def raiz() -> dict:
    return {
        "servico": settings.app_name,
        "docs": "/docs",
        "endpoints": ["POST /auth/login", "GET /auth/result/{job_id}"],
    }
