"""
Configuração central da API TruePulse.

Todos os valores podem ser sobrescritos por variáveis de ambiente com o
prefixo TRUEPULSE_ (ou por um arquivo .env na raiz do projeto).
Ex.: TRUEPULSE_DEEPFAKE_THRESHOLD_PCT=60

IMPORTANTE: os limiares de qualidade (MIN_QUALITY_RATIO / MIN_SIGNAL_SNR)
NÃO são redefinidos aqui de propósito. O rppg_pipeline.py declara que ele é a
fonte única desses valores, então a API os importa de lá (ver
app/services/features.py). Duplicá-los aqui criaria o risco de produção
aceitar um vídeo com qualidade abaixo da que o modelo viu no treino.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRUEPULSE_",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- Aplicação -------------------------------------------------------
    app_name: str = "TruePulse Auth API"
    debug: bool = False
    cors_origins: str = "*"  # separe por vírgula em produção

    # --- Armazenamento local ---------------------------------------------
    storage_dir: Path = BASE_DIR / "storage"
    models_dir: Path = BASE_DIR / "models"
    checkpoint_name: str = "truepulse_discriminator.pt"

    # --- FFmpeg / vídeo ---------------------------------------------------
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    target_fps: int = 30
    expected_duration_s: float = 5.0
    duration_tolerance_s: float = 1.5  # aceita entre 3.5s e 6.5s
    max_upload_mb: int = 60
    ffmpeg_timeout_s: int = 120

    # --- Detecção facial / anotação ---------------------------------------
    annotate_video: bool = True   # gera cópia com o retângulo do rosto desenhado
    draw_rois: bool = False       # desenha também as 3 ROIs (custa 1 passada extra)
    min_face_frames_ratio: float = 0.30  # gate rápido: sem rosto suficiente, nem roda o rPPG

    # --- Modelo (RF05) ----------------------------------------------------
    deepfake_threshold_pct: float = 50.0
    allow_untrained_model: bool = False  # True = sobe sem checkpoint (SÓ para dev)

    # --- Jobs -------------------------------------------------------------
    job_ttl_minutes: int = 30
    max_concurrent_jobs: int = 1  # MediaPipe/Torch: serializado de propósito

    # --- Token de sessão emitido quando o login é aprovado -----------------
    jwt_secret: str = "troque-este-segredo-em-producao"
    jwt_ttl_minutes: int = 15
    jwt_issuer: str = "truepulse-api"

    # --- Caminhos derivados ------------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        """Vídeo cru, exatamente como chegou do Flutter."""
        return self.storage_dir / "uploads"

    @property
    def pipeline_input_dir(self) -> Path:
        """Pasta local que o rppg_pipeline.py consome (vídeo já a 30fps)."""
        return self.storage_dir / "pipeline_input"

    @property
    def annotated_dir(self) -> Path:
        """Vídeo com o retângulo de reconhecimento facial desenhado."""
        return self.storage_dir / "annotated"

    @property
    def checkpoint_path(self) -> Path:
        return self.models_dir / self.checkpoint_name

    @property
    def face_landmarker_path(self) -> Path:
        """Onde o modelo do MediaPipe fica em cache (evita baixar na CWD)."""
        return self.models_dir / "face_landmarker.task"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for path in (self.storage_dir, self.uploads_dir, self.pipeline_input_dir,
                     self.annotated_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
