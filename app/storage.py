"""
Guarda o estado de cada verificação (job) em memória.

Isso é suficiente para rodar em uma única instância/processo (bom para
desenvolvimento e para provas de conceito). Se um dia você escalar para
múltiplos workers/instâncias, troque isso por Redis ou um banco de dados,
porque um dict em memória não é compartilhado entre processos.
"""
import threading
from typing import Dict, Optional

from app.models import JobStatus, VerificationResult

_lock = threading.Lock()
_jobs: Dict[str, VerificationResult] = {}


def create_job(job_id: str) -> None:
    with _lock:
        _jobs[job_id] = VerificationResult(job_id=job_id, status=JobStatus.PROCESSING)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            current = _jobs[job_id].model_dump()
            current.update(kwargs)
            _jobs[job_id] = VerificationResult(**current)


def get_job(job_id: str) -> Optional[VerificationResult]:
    with _lock:
        return _jobs.get(job_id)
