"""Contratos de entrada/saída da API (o que o Flutter consome)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LoginAceito(BaseModel):
    """Resposta imediata do POST /auth/login (HTTP 202)."""
    job_id: str = Field(..., description="Use em GET /auth/result/{job_id}")
    status: Literal["processando"] = "processando"
    mensagem: str = "Vídeo recebido. Verificação em andamento."
    poll_url: str
    poll_intervalo_ms: int = 1500


class Qualidade(BaseModel):
    quality_ratio: float = Field(..., description="Fração de frames com rosto detectado (0-1)")
    quality_ratio_min: float
    signal_snr: float = Field(..., description="Energia na faixa cardíaca / energia total")
    signal_snr_min: float
    fps_detectado: float
    frames_com_rosto: int
    frames_totais: int
    aprovada: bool


class Sincronia(BaseModel):
    delay_d_ms: float = Field(..., description="Atraso testa -> bochecha direita")
    delay_e_ms: float = Field(..., description="Atraso testa -> bochecha esquerda")


class VeredictoIA(BaseModel):
    """Saída do model.py convertida para porcentagem."""
    logits: float = Field(..., description="Logit cru devolvido por TruePulseDiscriminator.forward()")
    probabilidade_deepfake_pct: float = Field(..., description="sigmoid(logits) * 100")
    liveness_score_pct: float = Field(..., description="100 - probabilidade_deepfake_pct")
    limiar_pct: float
    deepfake: bool


class ResultadoVerificacao(BaseModel):
    aprovado: bool
    veredicto: Literal["aprovado", "reprovado", "inconclusivo"]
    codigo: str = Field(..., description="Código estável para o app tratar (não use a mensagem)")
    mensagem: str
    ia: VeredictoIA | None = None
    qualidade: Qualidade | None = None
    sincronia: Sincronia | None = None
    access_token: str | None = Field(None, description="Emitido apenas quando aprovado=true")
    token_expira_em_s: int | None = None
    arquivos: dict[str, str] | None = None


class JobResposta(BaseModel):
    """Resposta do GET /auth/result/{job_id}."""
    job_id: str
    status: Literal["processando", "concluido", "falhou"]
    etapa: str
    decorrido_s: float
    resultado: ResultadoVerificacao | None = None
    erro: str | None = None
    erro_codigo: str | None = None

    model_config = {"json_schema_extra": {"examples": [{
        "job_id": "9f1c2b...",
        "status": "processando",
        "etapa": "extraindo_rppg",
        "decorrido_s": 3.4,
        "resultado": None,
    }]}}


class HealthResposta(BaseModel):
    status: str
    ffmpeg: bool
    checkpoint_carregado: bool
    detalhes: dict[str, Any] = {}
