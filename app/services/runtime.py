"""
Recursos caros e compartilhados: o FaceLandmarker do MediaPipe, o modelo
PyTorch e o pool de threads que executa os jobs.

Todos são criados uma única vez (lazy) e protegidos por lock. O pool tem
max_workers=1 por padrão porque nem o FaceLandmarker nem uma inferência de
Torch em CPU ganham com concorrência aqui — e o FaceLandmarker do MediaPipe
não é thread-safe. Serializar é mais previsível do que descobrir isso em
produção. Se precisar de throughput, aumente TRUEPULSE_MAX_CONCURRENT_JOBS e
passe a criar um extractor por thread (threading.local).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_extractor = None
_model = None
_model_meta: dict = {}
_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=settings.max_concurrent_jobs,
                thread_name_prefix="truepulse-job",
            )
        return _executor


def shutdown_executor() -> None:
    global _executor
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None


def get_extractor():
    """FaceMeshROIExtractor do rppg_pipeline.py, com o .task em cache local."""
    global _extractor
    with _lock:
        if _extractor is None:
            from app.ml import rppg_pipeline

            # Sem isso, ensure_model() baixaria face_landmarker.task no
            # diretório de trabalho corrente (que varia conforme como o
            # uvicorn foi iniciado). Apontar para models/ deixa o cache
            # estável e fácil de versionar no deploy.
            settings.models_dir.mkdir(parents=True, exist_ok=True)
            rppg_pipeline.MODEL_PATH = str(settings.face_landmarker_path)

            logger.info("Inicializando FaceMeshROIExtractor (MediaPipe)...")
            _extractor = rppg_pipeline.FaceMeshROIExtractor()
            logger.info("FaceMeshROIExtractor pronto.")
        return _extractor


def _instanciar_modelo(n_aux: int, use_freq: bool):
    from app.ml.model import TruePulseDiscriminator
    return TruePulseDiscriminator(n_aux=n_aux, use_freq=use_freq)


def get_model():
    """Carrega TruePulseDiscriminator com os pesos treinados.

    Aceita três formatos de checkpoint:
      1. state_dict puro (torch.save(model.state_dict(), ...))
      2. {"model_state_dict": ..., "config": {"n_aux": 4, "use_freq": true}}
      3. {"state_dict": ..., "n_aux": ..., "use_freq": ...}

    n_aux e use_freq PRECISAM bater com o treino. Se o checkpoint não trouxer
    essa informação, ela é inferida pelo shape dos pesos salvos (aux_fc.weight
    tem shape (16, n_aux); a presença de freq_fc indica use_freq=True) — mais
    confiável do que assumir o padrão e quebrar no forward.
    """
    global _model, _model_meta
    with _lock:
        if _model is not None:
            return _model, dict(_model_meta)

        caminho = settings.checkpoint_path
        if not caminho.exists():
            if not settings.allow_untrained_model:
                raise FileNotFoundError(
                    f"Checkpoint do modelo não encontrado em {caminho}. "
                    f"Copie o arquivo treinado para lá ou defina "
                    f"TRUEPULSE_ALLOW_UNTRAINED_MODEL=true para rodar com pesos "
                    f"aleatórios (apenas desenvolvimento)."
                )
            logger.warning(
                "ATENÇÃO: subindo com modelo NÃO TREINADO (pesos aleatórios). "
                "Os percentuais retornados não têm valor algum.")
            _model = _instanciar_modelo(n_aux=2, use_freq=False)
            _model.eval()
            _model_meta = {"treinado": False, "n_aux": 2, "use_freq": False,
                           "checkpoint": None}
            return _model, dict(_model_meta)

        bruto = torch.load(caminho, map_location="cpu", weights_only=False)
        config = {}
        if isinstance(bruto, dict) and not any(
                k.startswith(("conv1", "lstm", "classifier")) for k in bruto):
            state = (bruto.get("model_state_dict") or bruto.get("state_dict")
                     or bruto.get("model") or {})
            config = bruto.get("config") or bruto.get("hparams") or {}
            for chave in ("n_aux", "use_freq", "seq_len"):
                if chave in bruto:
                    config.setdefault(chave, bruto[chave])
        else:
            state = bruto

        if not state:
            raise ValueError(f"Checkpoint {caminho} não contém pesos reconhecíveis.")

        aux_w = state.get("aux_fc.weight")
        n_aux = int(config.get("n_aux") or (aux_w.shape[1] if aux_w is not None else 2))
        use_freq = bool(config.get("use_freq",
                                   any(k.startswith("freq_fc") for k in state)))

        modelo = _instanciar_modelo(n_aux=n_aux, use_freq=use_freq)
        faltando, sobrando = modelo.load_state_dict(state, strict=False)
        if faltando or sobrando:
            # strict=False + log explícito: falha silenciosa aqui viraria um
            # modelo parcialmente aleatório respondendo com confiança.
            raise ValueError(
                f"Checkpoint incompatível com a arquitetura atual. "
                f"Ausentes: {list(faltando)} | Inesperados: {list(sobrando)}")

        modelo.eval()
        _model = modelo
        _model_meta = {"treinado": True, "n_aux": n_aux, "use_freq": use_freq,
                       "checkpoint": caminho.name}
        logger.info("Modelo carregado de %s (n_aux=%d, use_freq=%s).",
                    caminho.name, n_aux, use_freq)
        return _model, dict(_model_meta)


def modelo_carregado() -> bool:
    return _model is not None
