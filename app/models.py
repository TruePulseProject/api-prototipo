from enum import Enum
from typing import Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    ERROR = "error"


class LoginResponse(BaseModel):
    """Retornado imediatamente após o upload do vídeo."""
    job_id: str
    status: JobStatus
    message: str


class VerificationResult(BaseModel):
    """Retornado pelo endpoint de resultado (polling)."""
    job_id: str
    status: JobStatus
    is_deepfake: Optional[bool] = None
    is_correct_person: Optional[bool] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
