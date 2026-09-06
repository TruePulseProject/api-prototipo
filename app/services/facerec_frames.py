"""
SCRIPT 2 — reconhecimento facial nos frames extraídos pelo script 1.

Este arquivo é o seu `facerec_frames.py` original, com UMA mudança de
comportamento (o resto — detect, draw_rects, carregar_cascades — está
idêntico ao que você escreveu):

    A função `processar_frames()` agora RETORNA, além de continuar salvando
    os frames com os retângulos desenhados em disco, a lista de frames em
    que pelo menos 1 rosto foi detectado. Isso é necessário porque, na API,
    só faz sentido mandar para a IA de deepfake os frames que realmente
    contêm um rosto — mandar frames vazios/sem rosto só desperdiçaria
    chamada de IA e poderia confundir o modelo.

Fluxo (igual ao original):
    1. extrair_frames.py -> converte o vídeo para 30fps e salva os frames
       (frame_0000.jpg, frame_0001.jpg, ...) em uma pasta.
    2. facerec_frames.py -> lê cada frame dessa pasta, detecta rostos (e
       opcionalmente olhos dentro dos rostos), desenha os retângulos e
       salva o resultado em outra pasta.

Requisitos:
    pip install opencv-python
"""

import os
import glob
import time
from typing import List, Tuple

import cv2 as cv
import numpy as np


def clock() -> float:
    """Substitui o clock() do módulo common.py dos exemplos do OpenCV."""
    return time.time()


def draw_str(dst, target, s: str) -> None:
    """Substitui o draw_str() do módulo common.py dos exemplos do OpenCV.
    Desenha um texto com contorno preto para ficar legível em qualquer fundo."""
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def detect(img, cascade):
    """Detecta rostos (ou olhos) na imagem usando o classificador em cascata."""
    rects = cascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE,
    )
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    """Desenha retângulos ao redor das detecções."""
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def carregar_cascades():
    """Carrega os classificadores de rosto e olho.

    Tenta primeiro uma pasta local 'haarcascades/' (caso você tenha seus
    próprios arquivos XML, como no script original). Se não encontrar,
    usa os arquivos que já vêm embutidos no pacote opencv-python.
    """
    cascade_local = "haarcascades/haarcascade_frontalface_alt.xml"
    nested_local = "haarcascades/haarcascade_eye.xml"

    if os.path.isfile(cascade_local) and os.path.isfile(nested_local):
        cascade_fn = cascade_local
        nested_fn = nested_local
    else:
        cascade_fn = os.path.join(cv.data.haarcascades, "haarcascade_frontalface_alt.xml")
        nested_fn = os.path.join(cv.data.haarcascades, "haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cascade_fn)
    nested = cv.CascadeClassifier(nested_fn)

    if cascade.empty():
        raise RuntimeError(f"Não foi possível carregar o cascade de rosto: {cascade_fn}")
    if nested.empty():
        print(f"Aviso: não foi possível carregar o cascade de olhos ({nested_fn}). "
              f"A detecção de olhos será pulada.")

    return cascade, nested


def processar_frames(
    pasta_frames: str,
    pasta_saida: str,
    mostrar_janela: bool = False,
) -> List[Tuple[str, int]]:
    """
    Roda a detecção facial em todos os frames de `pasta_frames`, salva as
    versões com retângulo desenhado em `pasta_saida` (igual ao script
    original) e retorna uma lista de tuplas (caminho_frame_salvo, n_rostos)
    — uma entrada por frame processado, na mesma ordem dos arquivos.
    """
    os.makedirs(pasta_saida, exist_ok=True)

    cascade, nested = carregar_cascades()

    # Pega todos os frames extraídos, ordenados (frame_0000.jpg, frame_0001.jpg, ...)
    caminhos_frames = sorted(
        glob.glob(os.path.join(pasta_frames, "*.jpg"))
        + glob.glob(os.path.join(pasta_frames, "*.png"))
    )

    if not caminhos_frames:
        raise RuntimeError(f"Nenhum frame encontrado em: {pasta_frames}")

    print(f"{len(caminhos_frames)} frames encontrados em '{pasta_frames}'. Processando...")

    total_rostos_detectados = 0
    frames_com_rosto = 0
    resultados: List[Tuple[str, int]] = []

    for i, caminho in enumerate(caminhos_frames):
        img = cv.imread(caminho)
        if img is None:
            print(f"Aviso: não foi possível ler {caminho}, pulando.")
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))

        # Detecta olhos dentro de cada rosto encontrado
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))

        dt = clock() - t
        draw_str(vis, (20, 20), "time: %.1f ms" % (dt * 1000))
        draw_str(vis, (20, 40), "faces: %d" % len(rects))

        n_rostos = len(rects)
        if n_rostos > 0:
            frames_com_rosto += 1
            total_rostos_detectados += n_rostos

        nome_arquivo = os.path.basename(caminho)
        caminho_saida = os.path.join(pasta_saida, nome_arquivo)
        cv.imwrite(caminho_saida, vis)
        resultados.append((caminho_saida, n_rostos))

        if mostrar_janela:
            cv.imshow("facedetect", vis)
            # 30ms entre frames ~ preview em 30fps; ESC interrompe
            if cv.waitKey(30) == 27:
                break

    if mostrar_janela:
        cv.destroyAllWindows()

    print("\nProcessamento concluído.")
    print(f"Frames processados: {len(caminhos_frames)}")
    print(f"Frames com pelo menos 1 rosto detectado: {frames_com_rosto}")
    print(f"Total de rostos detectados (somando todos os frames): {total_rostos_detectados}")
    print(f"Resultados salvos em: {pasta_saida}")

    return resultados
