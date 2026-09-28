"""
Orquestração do job de verificação biométrica.

Sequência executada para cada login:

  1. ffmpeg  -> normaliza o vídeo cru para CFR 30fps           (video.py)
  2. MediaPipe -> detecta rosto, desenha retângulo, gate rápido (face.py)
  3. copia o vídeo normalizado para a pasta que os scripts consomem
  4. rppg_pipeline.build_feature_vector() -> 3 sinais + 2 atrasos (features.py)
  5. TruePulseDiscriminator.forward() -> logits                 (model.py)
  6. sigmoid(logits)*100 -> porcentagem -> decisão de login

Esta função roda numa thread do pool (ver runtime.get_executor), nunca no
event loop do FastAPI: MediaPipe e Torch são CPU-bound e bloqueariam todas as
outras requisições.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from pathlib import Path

import torch

from app.config import settings
from app.jobs import JobStage, get_store
from app.services import face as face_svc
from app.services import video as video_svc
from app.services.features import (
    FalhaExtracao,
    QualidadeInsuficiente,
    extrair_features,
)
from app.services.runtime import get_extractor, get_model
from app.services.token import emitir_token

logger = logging.getLogger(__name__)


def logits_para_porcentagem(logits: torch.Tensor) -> tuple[float, float, float]:
    """Converte o logit cru do model.py em porcentagens legíveis.

    O forward() devolve logits porque o treino usa BCEWithLogitsLoss (mais
    estável numericamente que sigmoid + BCELoss). Quem consome o resultado
    precisa da probabilidade, então a sigmoid é aplicada aqui:

        p_fake        = sigmoid(logits)            -> 0.0 a 1.0
        deepfake_pct  = p_fake * 100               -> 0% a 100%
        liveness_pct  = 100 - deepfake_pct

    Devolve (logit_float, probabilidade_deepfake_pct, liveness_score_pct).
    """
    logit_float = float(logits.detach().reshape(-1)[0].item())
    p_fake = float(torch.sigmoid(torch.tensor(logit_float)).item())
    deepfake_pct = round(p_fake * 100.0, 2)
    liveness_pct = round(100.0 - deepfake_pct, 2)
    return logit_float, deepfake_pct, liveness_pct


def _resultado_reprovado(codigo: str, mensagem: str, **extra) -> dict:
    return {
        "aprovado": False,
        "veredicto": "inconclusivo",
        "codigo": codigo,
        "mensagem": mensagem,
        **extra,
    }


def processar_verificacao(job_id: str, video_bruto: Path,
                          user_id: str | None = None) -> None:
    """Executa o pipeline inteiro e grava o resultado no JobStore.

    Nunca levanta exceção para fora: qualquer falha vira job.status="falhou"
    com um código estável, porque quem chama é um executor em background e
    uma exceção perdida deixaria o job travado em "processando" para sempre.
    """
    store = get_store()
    inicio = time.perf_counter()
    video_30fps = settings.pipeline_input_dir / f"{job_id}_30fps.mp4"
    video_anotado = settings.annotated_dir / f"{job_id}_anotado.mp4"

    try:
        # --- 1. Normalização para 30 FPS ---------------------------------
        store.set_stage(job_id, JobStage.NORMALIZANDO_VIDEO)
        info_bruto = video_svc.probe(video_bruto)
        video_svc.validar_duracao(info_bruto)
        info = video_svc.normalizar_para_30fps(video_bruto, video_30fps)
        logger.info("[%s] Vídeo normalizado: %.1fs @ %.2ffps (%dx%d)",
                    job_id, info.duracao_s, info.fps, info.largura, info.altura)

        # --- 2. Detecção facial + retângulo ------------------------------
        store.set_stage(job_id, JobStage.DETECTANDO_ROSTO)
        extractor = get_extractor()
        relatorio = face_svc.anotar_video(video_30fps, video_anotado, extractor)
        logger.info("[%s] Rosto em %d/%d frames (%.0f%%).", job_id,
                    relatorio.frames_com_rosto, relatorio.frames_totais,
                    relatorio.ratio * 100)

        # --- 3. Vídeo já está na pasta que os scripts consomem -----------
        # (normalizar_para_30fps escreveu direto em storage/pipeline_input/)

        # --- 4. rPPG: vídeo -> 3 sinais + 2 atrasos ----------------------
        store.set_stage(job_id, JobStage.EXTRAINDO_RPPG)
        modelo, meta = get_model()
        features = extrair_features(
            video_30fps, extractor,
            n_aux=meta["n_aux"], use_freq=meta["use_freq"],
        )

        # --- 5. Classificação (RF05) -------------------------------------
        store.set_stage(job_id, JobStage.CLASSIFICANDO)
        with torch.no_grad():
            logits = modelo(features.signals, features.aux, features.freq)
        logit_val, deepfake_pct, liveness_pct = logits_para_porcentagem(logits)

        limiar = settings.deepfake_threshold_pct
        eh_deepfake = deepfake_pct >= limiar
        aprovado = not eh_deepfake

        qualidade = features.bruto["_qualidade"]
        resultado = {
            "aprovado": aprovado,
            "veredicto": "aprovado" if aprovado else "reprovado",
            "codigo": "LOGIN_APROVADO" if aprovado else "DEEPFAKE_DETECTADO",
            "mensagem": (
                f"Pessoa real detectada (liveness {liveness_pct:.1f}%). Login aprovado."
                if aprovado else
                f"Indício de manipulação: {deepfake_pct:.1f}% de probabilidade de "
                f"deepfake (limiar {limiar:.0f}%). Login negado."
            ),
            "ia": {
                "logits": round(logit_val, 6),
                "probabilidade_deepfake_pct": deepfake_pct,
                "liveness_score_pct": liveness_pct,
                "limiar_pct": limiar,
                "deepfake": eh_deepfake,
            },
            "qualidade": {
                **qualidade,
                "frames_com_rosto": relatorio.frames_com_rosto,
                "frames_totais": relatorio.frames_totais,
                "aprovada": True,
            },
            "sincronia": {
                "delay_d_ms": round(float(features.bruto["delay_d_ms"]), 2),
                "delay_e_ms": round(float(features.bruto["delay_e_ms"]), 2),
            },
            "arquivos": {
                "video_30fps": str(video_30fps),
                "video_anotado": str(video_anotado) if relatorio.video_anotado else "",
            },
        }

        if not meta["treinado"]:
            resultado["mensagem"] += (
                " [AVISO: modelo sem checkpoint treinado — resultado sem validade]")
            resultado["codigo"] = "MODELO_NAO_TREINADO"

        if aprovado and meta["treinado"]:
            token, ttl = emitir_token(user_id, job_id, liveness_pct)
            resultado["access_token"] = token
            resultado["token_expira_em_s"] = ttl

        store.finish(job_id, resultado)
        logger.info("[%s] Concluído em %.2fs: %s (%.2f%% deepfake).",
                    job_id, time.perf_counter() - inicio,
                    resultado["veredicto"], deepfake_pct)

    except video_svc.VideoError as exc:
        store.finish(job_id, _resultado_reprovado(exc.codigo, str(exc)))
    except face_svc.FaceError as exc:
        store.finish(job_id, _resultado_reprovado(exc.codigo, str(exc)))
    except QualidadeInsuficiente as exc:
        store.finish(job_id, _resultado_reprovado(
            exc.codigo, str(exc),
            qualidade={**exc.detalhes, "frames_com_rosto": 0,
                       "frames_totais": 0, "aprovada": False}))
    except FalhaExtracao as exc:
        store.finish(job_id, _resultado_reprovado(exc.codigo, str(exc)))
    except FileNotFoundError as exc:
        # Checkpoint ausente: erro de operação, não do usuário.
        logger.error("[%s] %s", job_id, exc)
        store.fail(job_id, str(exc), "MODELO_INDISPONIVEL")
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Falha inesperada:\n%s", job_id, traceback.format_exc())
        detalhe = str(exc) if settings.debug else "Erro interno ao processar o vídeo."
        store.fail(job_id, detalhe, "ERRO_INTERNO")
    finally:
        # O vídeo cru já cumpriu seu papel; manter dado biométrico bruto no
        # disco sem necessidade é risco. Os derivados (30fps/anotado) ficam
        # até o TTL do job expirar, para auditoria.
        try:
            video_bruto.unlink(missing_ok=True)
        except OSError:
            pass


def limpar_artefatos(job_id: str) -> None:
    """Apaga os vídeos derivados de um job (chamado pelo purge de TTL)."""
    for caminho in (settings.pipeline_input_dir / f"{job_id}_30fps.mp4",
                    settings.annotated_dir / f"{job_id}_anotado.mp4",
                    settings.uploads_dir / f"{job_id}_bruto"):
        try:
            Path(caminho).unlink(missing_ok=True)
        except OSError:
            pass
    for extra in settings.uploads_dir.glob(f"{job_id}_bruto*"):
        try:
            extra.unlink(missing_ok=True)
        except OSError:
            pass


def copiar_para_pasta_scripts(origem: Path, destino: Path) -> Path:
    """Helper explícito, caso você queira que os scripts consumam de outra
    pasta que não a padrão (ex.: um volume compartilhado com outro serviço)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origem, destino)
    return destino
