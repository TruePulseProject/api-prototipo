"""
SCRIPT 1 — extração de frames.

Este arquivo é o seu `extrair_frames.py` original, com uma única mudança:
as funções continuam exatamente como você escreveu, e foi adicionada uma
função `processar_video()` no final, que é a versão "chamável pela API" do
que o antigo `main()` fazia lendo `sys.argv`.

Fluxo (igual ao original):
1. Lê o vídeo.
2. Se o fps não for ~30, converte com ffmpeg.
3. Extrai N frames igualmente espaçados (padrão: 150) e salva como .jpg.

Requisitos:
    pip install opencv-python
    ffmpeg instalado no sistema e disponível no PATH.
"""

import os
import subprocess

import cv2


def obter_fps(caminho_video: str) -> float:
    """Retorna o fps do vídeo."""
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def converter_para_30fps(caminho_entrada: str, caminho_saida: str) -> None:
    """Usa o ffmpeg para converter o vídeo para 30fps."""
    comando = [
        "ffmpeg",
        "-y",  # sobrescreve o arquivo de saída se já existir
        "-i", caminho_entrada,
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "copy",
        caminho_saida,
    ]

    print(f"Convertendo vídeo para 30fps...\n  {' '.join(comando)}")
    resultado = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if resultado.returncode != 0:
        print(resultado.stderr)
        raise RuntimeError("Falha ao converter o vídeo com ffmpeg.")

    print(f"Vídeo convertido salvo em: {caminho_saida}")


def extrair_frames_igualmente_espacados(
    caminho_video: str,
    pasta_saida: str,
    quantidade_frames: int = 150,
) -> None:
    """Extrai `quantidade_frames` frames igualmente distribuídos ao longo do vídeo."""
    os.makedirs(pasta_saida, exist_ok=True)

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duracao = total_frames / fps if fps else 0

    print(f"Total de frames no vídeo: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duração aproximada: {duracao:.2f}s")

    if total_frames < quantidade_frames:
        print(
            f"Aviso: o vídeo tem apenas {total_frames} frames, "
            f"menos que os {quantidade_frames} solicitados. "
            f"Serão extraídos todos os {total_frames} frames disponíveis."
        )
        quantidade_frames = total_frames

    # Índices dos frames igualmente espaçados (0 a total_frames - 1)
    if quantidade_frames == 1:
        indices = [0]
    else:
        indices = [
            round(i * (total_frames - 1) / (quantidade_frames - 1))
            for i in range(quantidade_frames)
        ]

    indices = sorted(set(indices))  # remove duplicados, caso existam

    frame_atual = 0
    indice_pos = 0
    salvos = 0

    while cap.isOpened() and indice_pos < len(indices):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_atual == indices[indice_pos]:
            nome_arquivo = os.path.join(
                pasta_saida, f"frame_{salvos:04d}.jpg"
            )
            cv2.imwrite(nome_arquivo, frame)
            salvos += 1
            indice_pos += 1

        frame_atual += 1

    cap.release()
    print(f"{salvos} frames salvos em: {pasta_saida}")


def processar_video(
    caminho_video_original: str,
    pasta_saida: str,
    quantidade_frames: int = 150,
) -> str:
    """
    Orquestra o que o antigo `main()` fazia via linha de comando, mas como
    função chamável pela API: recebe o vídeo bruto, converte fps se
    necessário e extrai os frames igualmente espaçados.

    Retorna o caminho da pasta onde os frames foram salvos.
    """
    if not os.path.isfile(caminho_video_original):
        raise RuntimeError(f"Arquivo não encontrado: {caminho_video_original}")

    fps_original = obter_fps(caminho_video_original)
    print(f"FPS detectado no vídeo original: {fps_original}")

    caminho_para_processar = caminho_video_original

    # Considera "fps errado" qualquer coisa diferente de 30 (com margem de tolerância)
    if abs(fps_original - 30) > 0.5:
        nome_base, ext = os.path.splitext(caminho_video_original)
        caminho_convertido = f"{nome_base}_30fps{ext}"
        converter_para_30fps(caminho_video_original, caminho_convertido)
        caminho_para_processar = caminho_convertido
    else:
        print("Vídeo já está em 30fps. Pulando conversão.")

    extrair_frames_igualmente_espacados(
        caminho_para_processar, pasta_saida, quantidade_frames=quantidade_frames
    )

    return pasta_saida
