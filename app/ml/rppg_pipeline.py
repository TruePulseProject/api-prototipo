"""
TruePulse - Pipeline de extração rPPG multi-ROI
=================================================
Implementa RF01, RF02, RF03 e RF04 do documento de escopo:
  - RF01: captura + Face Mesh (468 landmarks)
  - RF02: isolamento de 3 ROIs (testa, bochecha direita, bochecha esquerda)
  - RF03: extração do canal verde + filtragem (detrend, bandpass, CWT)
  - RF04: cálculo da sincronia espacial (atraso fisiológico entre ROIs)

Requer: mediapipe, opencv-python, numpy, scipy, PyWavelets

O modelo do Face Mesh (face_landmarker.task) é baixado automaticamente na
primeira execução. Isso exige acesso a storage.googleapis.com (funciona
normalmente no Google Colab / máquina local com internet).
"""

import os
import urllib.request
import numpy as np
import cv2
from scipy.signal import butter, filtfilt, detrend, correlate, welch
import pywt

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DO FACE MESH
# ---------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

# Landmarks-âncora estáveis do MediaPipe Face Mesh (468 pontos), usados como
# referência geométrica para construir as ROIs por percentuais relativos ao
# rosto (e não por bounding-box bruto da câmera, que é sensível a rotação).
#   10  -> topo da testa (nascimento do cabelo)
#   152 -> ponta do queixo
#   234 -> lateral direita do rosto (altura da bochecha)
#   454 -> lateral esquerda do rosto (altura da bochecha)
#   9   -> glabela (entre as sobrancelhas)
#   1   -> ponta do nariz
ANCHOR = {"top": 10, "chin": 152, "right": 234, "left": 454, "glabella": 9, "nose": 1}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Baixando modelo Face Landmarker de {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceMeshROIExtractor:
    """Detecta o Face Mesh e recorta as 3 ROIs fisiológicas do rosto."""

    def __init__(self, model_path=None):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        self._mp = mp
        model_path = model_path or ensure_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _landmarks_px(self, frame, landmarks):
        h, w = frame.shape[:2]
        return {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}

    def _bbox(self, pts):
        """Caixa delimitadora (min/max x,y) de todos os 468 pontos —
        abordagem robusta que não depende de acertar qual índice individual
        corresponde à lateral do rosto (índices 234/454 se mostraram
        incorretos em teste visual: caíam perto do centro do rosto, não das
        laterais)."""
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        return min(xs), max(xs), min(ys), max(ys)  # left, right, top, bottom

    def get_rois(self, frame):
        """Retorna dict {nome_roi: recorte_bgr} ou None se nenhum rosto detectado."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        pts = self._landmarks_px(frame, result.face_landmarks[0])
        left_x, right_x, top_y, bottom_y = self._bbox(pts)
        face_w = max(1, right_x - left_x)
        face_h = max(1, bottom_y - top_y)
        cx_face = (left_x + right_x) // 2

        def box(cx, cy, w_frac, h_frac):
            bw, bh = int(face_w * w_frac), int(face_h * h_frac)
            x0, y0 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            return frame[y0:y0 + bh, x0:x0 + bw]

        # Testa: banda fina logo no topo da malha rastreada (o Face Mesh do
        # MediaPipe não cobre bem a testa acima da altura das sobrancelhas,
        # então a ROI fica deliberadamente próxima dali, não no meio da testa)
        testa_cy = int(top_y + face_h * 0.06)
        testa = box(cx_face, testa_cy, 0.30, 0.10)

        # Bochechas: ~58% da altura do rosto — abaixo do olho (50%), acima
        # da boca (75%), calibrado com a regra clássica dos terços faciais
        # (meio-rosto) e ajustado por feedback visual direto nos frames de
        # teste.
        cy_bochecha = int(top_y + face_h * 0.58)
        bochecha_d_cx = int(cx_face - face_w * 0.32)
        bochecha_e_cx = int(cx_face + face_w * 0.32)
        bochecha_d = box(bochecha_d_cx, cy_bochecha, 0.18, 0.12)
        bochecha_e = box(bochecha_e_cx, cy_bochecha, 0.18, 0.12)

        return {"testa": testa, "bochecha_d": bochecha_d, "bochecha_e": bochecha_e}

    def debug_draw(self, frame):
        """Desenha as 3 caixas de ROI sobre o frame, para conferência visual.
        IMPORTANTE: rode isso em um frame de exemplo antes de processar o
        dataset inteiro, para checar se as caixas caem sobre pele e não
        sobre cabelo/orelha/fundo."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        out = frame.copy()
        if not result.face_landmarks:
            return out
        pts = self._landmarks_px(frame, result.face_landmarks[0])
        left_x, right_x, top_y, bottom_y = self._bbox(pts)
        face_w = max(1, right_x - left_x)
        face_h = max(1, bottom_y - top_y)
        cx_face = (left_x + right_x) // 2

        def draw_box(cx, cy, w_frac, h_frac, color):
            bw, bh = int(face_w * w_frac), int(face_h * h_frac)
            x0, y0 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), color, 2)

        testa_cy = int(top_y + face_h * 0.06)
        draw_box(cx_face, testa_cy, 0.30, 0.10, (0, 255, 0))

        cy_bochecha = int(top_y + face_h * 0.58)
        draw_box(int(cx_face - face_w * 0.32), cy_bochecha, 0.18, 0.12, (255, 0, 0))
        draw_box(int(cx_face + face_w * 0.32), cy_bochecha, 0.18, 0.12, (0, 0, 255))

        # DIAGNÓSTICO: contorno branco = a caixa delimitadora (bbox) que o
        # código está usando como "o rosto inteiro" para calcular as
        # porcentagens acima. Se esse retângulo não cobrir do topo da cabeça
        # até o queixo, e de orelha a orelha, o problema está aqui, não nas
        # frações.
        cv2.rectangle(out, (left_x, top_y), (right_x, bottom_y), (255, 255, 255), 1)

        # DIAGNÓSTICO: linhas-guia de porcentagem (25/50/75% da altura e
        # largura do bbox), para calibrar visualmente qual fração corresponde
        # a qual altura real do rosto (sobrancelha, nariz, boca, etc.)
        for frac in (0.25, 0.50, 0.75):
            y = int(top_y + face_h * frac)
            cv2.line(out, (left_x, y), (right_x, y), (0, 255, 255), 1)
            cv2.putText(out, f"{int(frac*100)}%", (right_x + 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            x = int(left_x + face_w * frac)
            cv2.line(out, (x, top_y), (x, bottom_y), (255, 0, 255), 1)
        return out

    def debug_landmarks(self, frame):
        """DIAGNÓSTICO: desenha TODOS os 468 pontos do Face Mesh (pequenos,
        cinza) e destaca com círculo grande + número os 6 pontos-âncora
        usados em get_rois(). Use isso se debug_draw() mostrar caixas de ROI
        no lugar errado — a imagem resultante mostra exatamente onde cada
        âncora caiu de verdade, permitindo corrigir os índices/percentuais
        com precisão em vez de tentativa e erro."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        out = frame.copy()
        if not result.face_landmarks:
            cv2.putText(out, "ROSTO NAO DETECTADO", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return out
        pts = self._landmarks_px(frame, result.face_landmarks[0])

        for i, (x, y) in pts.items():
            cv2.circle(out, (x, y), 1, (180, 180, 180), -1)

        for name, idx in ANCHOR.items():
            x, y = pts[idx]
            cv2.circle(out, (x, y), 6, (0, 255, 255), 2)
            cv2.putText(out, f"{name}({idx})", (x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return out


# ---------------------------------------------------------------------------
# 2. EXTRAÇÃO DO SINAL BRUTO (canal verde por ROI)
# ---------------------------------------------------------------------------

def extract_raw_signals(video_path, extractor, max_frames=None):
    """Percorre o vídeo e retorna sinais brutos (canal G médio) por ROI.

    Retorna: (dict roi->np.array, fps, quality_ratio)
    quality_ratio = fração de frames em que o rosto foi detectado (RNF05 -
    útil pra descartar vídeos com iluminação ruim demais).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    signals = {"testa": [], "bochecha_d": [], "bochecha_e": []}
    total, detected = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and total >= max_frames:
            break
        total += 1
        rois = extractor.get_rois(frame)
        if rois is None:
            continue
        detected += 1
        for name, crop in rois.items():
            if crop.size == 0:
                continue
            signals[name].append(float(np.mean(crop[:, :, 1])))  # canal G (BGR)
    cap.release()

    quality_ratio = detected / total if total else 0.0
    return {k: np.array(v) for k, v in signals.items()}, fps, quality_ratio


# ---------------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO: DETREND + BANDPASS + CWT (RF03)
# ---------------------------------------------------------------------------

def butter_bandpass_filter(data, lowcut=0.7, highcut=3.5, fs=30.0, order=3):
    """Faixa 0.7-3.5 Hz cobre ~42-210 bpm, conforme já usado no código original."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


def apply_cwt_filter(signal, fps, lowcut=0.7, highcut=3.5, wavelet="morl"):
    """Aplica Transformada de Wavelet Contínua e reconstrói uma versão
    filtrada do sinal usando apenas as escalas correspondentes à faixa de
    frequência cardíaca. Isso ataca ruído de movimento não-periódico que o
    Butterworth sozinho deixa passar (picos bruscos, por exemplo).

    Retorna: (sinal_filtrado, scalogram, frequencias_das_escalas)
    """
    # Gera escalas cobrindo a faixa de frequência desejada
    freqs_target = np.linspace(lowcut, highcut, 40)
    scales = pywt.frequency2scale(wavelet, freqs_target / fps)
    scales = scales[scales > 0]

    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1.0 / fps)
    # Reconstrução aproximada: soma ponderada das partes reais das escalas
    # dentro da banda de interesse (técnica comum de "band reconstruction").
    filtered = np.sum(np.real(coeffs), axis=0)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    return filtered, np.abs(coeffs), freqs


def signal_band_snr(signal, fps, band=(0.7, 3.5)):
    """Proporção de energia do sinal concentrada na faixa de frequência
    cardíaca (band) em relação à energia total do espectro. Sinal fisiológico
    limpo tem essa energia concentrada (SNR alto); sinal ruidoso/sem pulsação
    reconhecível tem energia espalhada (SNR baixo). Usado como filtro extra
    de qualidade, complementar ao quality_ratio (taxa de detecção facial).

    Usa o método de Welch (média de múltiplas janelas sobrepostas) em vez de
    um periodograma único (FFT direta): testado empiricamente e a FFT direta
    mostrou variância alta demais em sinais curtos (~150 amostras) - ruído
    puro podia ocasionalmente "parecer" sinal bom dependendo da amostra.
    Welch reduz drasticamente essa variância."""
    freqs, psd = welch(signal, fs=fps, nperseg=min(len(signal), 64))
    total_power = psd.sum() + 1e-8
    band_power = psd[(freqs >= band[0]) & (freqs <= band[1])].sum()
    return float(band_power / total_power)


def preprocess_roi_signal(raw_signal, fps, return_intermediate=False):
    """Pipeline completo RF03: detrend -> bandpass -> CWT.
    Se return_intermediate=True, também devolve o sinal apenas detrended
    (antes de qualquer filtro de banda) - necessário para medir SNR de forma
    não-circular (ver signal_band_snr)."""
    if len(raw_signal) < 30:
        raise ValueError("Sinal muito curto para filtrar (menos de 1s de vídeo útil).")
    detrended = detrend(raw_signal)
    bandpassed = butter_bandpass_filter(detrended, fs=fps)
    cwt_filtered, scalogram, freqs = apply_cwt_filter(bandpassed, fps)
    if return_intermediate:
        return cwt_filtered, detrended
    return cwt_filtered


# ---------------------------------------------------------------------------
# 4. SINCRONIA ESPACIAL / ATRASO FISIOLÓGICO (RF04)
# ---------------------------------------------------------------------------

def compute_sync_delay_ms(signal_a, signal_b, fps, max_lag_ms=200):
    """Calcula o atraso temporal (em ms) entre dois sinais rPPG via
    correlação cruzada. Positivo = signal_b atrasado em relação a signal_a.

    max_lag_ms restringe a busca a uma faixa fisiologicamente plausível
    (atraso real de propagação sanguínea no rosto é de poucos ms a poucas
    dezenas de ms). Sem essa restrição, em janelas curtas (5s) e sinais
    ruidosos, a correlação cruzada pode "travar" em picos de coincidência
    aleatória distantes do zero (observado empiricamente: atrasos de mais
    de 1s em dados reais, fisiologicamente impossíveis) - restringir a busca
    evita esses falsos positivos, à custa de não conseguir detectar um
    atraso genuíno maior que max_lag_ms (perda aceitável, já que tal atraso
    não seria fisiologicamente plausível de qualquer forma)."""
    n = min(len(signal_a), len(signal_b))
    a, b = signal_a[:n], signal_b[:n]
    corr = correlate(a - a.mean(), b - b.mean(), mode="full")
    lag_samples = np.arange(-n + 1, n)

    max_lag_samples = int(max_lag_ms / 1000 * fps)
    valid = np.abs(lag_samples) <= max_lag_samples
    corr, lag_samples = corr[valid], lag_samples[valid]

    best_lag = lag_samples[np.argmax(corr)]
    # scipy.signal.correlate(a, b) usa a convenção onde um pico de b que
    # ocorre DEPOIS de a produz lag NEGATIVO; invertemos o sinal aqui para
    # que o retorno siga a convenção documentada (positivo = b atrasado
    # em relação a a), verificada empiricamente com sinais de teste.
    return -float(best_lag / fps * 1000.0)


# ---------------------------------------------------------------------------
# 5. ORQUESTRAÇÃO: VÍDEO -> VETOR DE FEATURES (pronto pra treino)
# ---------------------------------------------------------------------------

CAPTURA_DURACAO_S = 5      # janela de captura definida pelo app (UC01: 5s @ 30fps)
CAPTURA_FPS_ESPERADO = 30  # conforme UC01: "O vídeo é capturado a 30 FPS"
TARGET_LEN_PADRAO = CAPTURA_DURACAO_S * CAPTURA_FPS_ESPERADO  # 150 amostras

# Limiares de qualidade mínima - FONTE ÚNICA usada tanto na montagem do
# dataset de treino (dataset_builder.py) quanto na inferência em produção
# (ai_verification_v2.py, no repositório da API). Nunca duplicar esses
# valores em outro arquivo - o modelo é treinado só com vídeos que passam
# nesses limiares, então uma captura em produção abaixo deles representa
# uma condição que o modelo nunca viu durante o treino, e deve ser
# rejeitada (pedir nova captura) em vez de classificada com confiança
# enganosa.
MIN_QUALITY_RATIO = 0.5   # % mínima de frames com rosto detectado
MIN_SIGNAL_SNR = 0.35     # qualidade mínima do sinal (método de Welch, pré-filtro de banda)


def build_feature_vector(video_path, extractor, target_len=TARGET_LEN_PADRAO,
                          max_frames=CAPTURA_DURACAO_S * CAPTURA_FPS_ESPERADO * 2,
                          min_duration_fraction=0.80):
    """Processa um vídeo inteiro e devolve as features prontas para o
    classificador (RF05): 3 sinais rPPG filtrados (reamostrados para
    comprimento fixo) + 2 métricas de sincronia espacial.

    min_duration_fraction: exige que cada ROI tenha pelo menos essa fração
    de target_len amostras REAIS detectadas antes de reamostrar (padrão:
    80% de 150 = 120 amostras = 4s de 5s). Isso evita treinar com vídeos que
    só tiveram 1-2s de rosto detectado "esticados" artificialmente para
    parecer 5s via interpolação - o que não representaria fielmente a
    captura de 5s contínuos que a aplicação em produção sempre vai enviar."""
    raw_signals, fps, quality = extract_raw_signals(video_path, extractor, max_frames)

    min_amostras = int(target_len * min_duration_fraction)
    for roi, sig in raw_signals.items():
        if len(sig) < min_amostras:
            raise ValueError(
                f"ROI '{roi}' teve apenas {len(sig)} amostras detectadas "
                f"(mínimo exigido: {min_amostras}, ~{min_duration_fraction*100:.0f}% "
                f"de {CAPTURA_DURACAO_S}s) - vídeo não representa fielmente uma "
                f"captura de {CAPTURA_DURACAO_S}s contínuos.")

    processed = {}
    pre_bandpass = {}
    for roi, sig in raw_signals.items():
        processed[roi], pre_bandpass[roi] = preprocess_roi_signal(sig, fps, return_intermediate=True)

    # Reamostra todos os sinais para o mesmo comprimento fixo (target_len),
    # necessário porque vídeos podem ter durações/FPS levemente diferentes.
    def resample(sig, n):
        x_old = np.linspace(0, 1, len(sig))
        x_new = np.linspace(0, 1, n)
        return np.interp(x_new, x_old, sig)

    signal_testa = resample(processed["testa"], target_len)
    signal_bd = resample(processed["bochecha_d"], target_len)
    signal_be = resample(processed["bochecha_e"], target_len)

    delay_d = compute_sync_delay_ms(processed["testa"], processed["bochecha_d"], fps)
    delay_e = compute_sync_delay_ms(processed["testa"], processed["bochecha_e"], fps)

    # SNR médio dos 3 sinais - métrica de qualidade complementar ao
    # quality_ratio (detecção facial), usada para descartar vídeos com
    # rosto detectado mas sinal fisiologicamente pouco confiável.
    # IMPORTANTE: medido no sinal apenas detrended (pre_bandpass), não no
    # sinal já filtrado - medir depois do filtro de banda seria circular
    # (o filtro já força a energia pra dentro da própria faixa que estamos
    # "verificando", inflando o valor artificialmente para quase qualquer
    # entrada, inclusive ruído puro - confirmado em teste sintético).
    snr_medio = float(np.mean([
        signal_band_snr(pre_bandpass["testa"], fps),
        signal_band_snr(pre_bandpass["bochecha_d"], fps),
        signal_band_snr(pre_bandpass["bochecha_e"], fps),
    ]))

    return {
        "signal_testa": signal_testa,
        "signal_bochecha_d": signal_bd,
        "signal_bochecha_e": signal_be,
        "delay_d_ms": delay_d,
        "delay_e_ms": delay_e,
        "fps": fps,
        "quality_ratio": quality,
        "signal_snr": snr_medio,
    }
