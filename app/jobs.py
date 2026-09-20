"""
Registro de jobs de verificação (em memória).

Fluxo: POST /auth/login cria um job com status "processando" e devolve o
job_id na hora; o processamento pesado (ffmpeg + MediaPipe + rPPG + modelo)
roda numa thread separada e vai atualizando esse registro. O Flutter faz
polling em GET /auth/result/{job_id} até o status sair de "processando".

Limitação conhecida: o dicionário vive no processo. Com múltiplos workers do
uvicorn (--workers > 1) um job criado no worker A não é visível no worker B.
Para produção com mais de um worker, troque JobStore por Redis mantendo a
mesma interface (create/get/update). Ver README, seção "Limitações".
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PROCESSANDO = "processando"
    CONCLUIDO = "concluido"
    FALHOU = "falhou"


class JobStage(str, Enum):
    """Etapa corrente — útil para o app mostrar um texto de progresso."""
    NA_FILA = "na_fila"
    NORMALIZANDO_VIDEO = "normalizando_video"
    DETECTANDO_ROSTO = "detectando_rosto"
    EXTRAINDO_RPPG = "extraindo_rppg"
    CLASSIFICANDO = "classificando"
    FINALIZADO = "finalizado"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    user_id: str | None = None
    status: JobStatus = JobStatus.PROCESSANDO
    stage: JobStage = JobStage.NA_FILA
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or _now()
        return round((end - self.created_at).total_seconds(), 2)


class JobStore:
    def __init__(self, ttl_minutes: int = 30) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(minutes=ttl_minutes)

    def create(self, user_id: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex, user_id=user_id)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_stage(self, job_id: str, stage: JobStage) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage = stage
            job.updated_at = _now()

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.CONCLUIDO
            job.stage = JobStage.FINALIZADO
            job.result = result
            job.updated_at = job.finished_at = _now()

    def fail(self, job_id: str, message: str, code: str = "ERRO_INTERNO") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FALHOU
            job.error = message
            job.error_code = code
            job.updated_at = job.finished_at = _now()

    def purge_expired(self) -> list[str]:
        """Remove jobs antigos. Devolve os ids removidos para que o chamador
        apague os arquivos de vídeo correspondentes (dado biométrico não deve
        ficar no disco além do necessário)."""
        limite = _now() - self._ttl
        with self._lock:
            expirados = [jid for jid, j in self._jobs.items()
                         if (j.finished_at or j.created_at) < limite]
            for jid in expirados:
                self._jobs.pop(jid, None)
        return expirados


store: JobStore | None = None


def get_store() -> JobStore:
    global store
    if store is None:
        from app.config import settings
        store = JobStore(ttl_minutes=settings.job_ttl_minutes)
    return store
