# TruePulse Auth API

Backend FastAPI para autenticação biométrica com detecção de deepfake por rPPG
(*remote photoplethysmography*). Recebe um vídeo de ~5 s capturado pela webcam
no app Flutter, garante 30 FPS com FFmpeg, detecta o rosto, roda o pipeline de
sinais (`rppg_pipeline.py`), classifica com o modelo (`model.py`) e devolve a
probabilidade de deepfake **em porcentagem** como fator definitivo do login.

---

## Índice

1. [Como funciona](#como-funciona)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Instalação](#instalação)
4. [Colocando o modelo treinado no lugar](#colocando-o-modelo-treinado-no-lugar)
5. [Executando](#executando)
6. [Endpoints](#endpoints)
7. [De `logits` para porcentagem](#de-logits-para-porcentagem)
8. [Integração com o Flutter](#integração-com-o-flutter)
9. [Códigos de resposta](#códigos-de-resposta)
10. [Configuração](#configuração)
11. [Testando sem o modelo treinado](#testando-sem-o-modelo-treinado)
12. [Docker](#docker)
13. [Limitações conhecidas e próximos passos](#limitações-conhecidas-e-próximos-passos)

---

## Como funciona

```
Flutter ──POST /auth/login (vídeo 5s)──▶ API
                                          │  cria job (status: "processando")
                                          └──▶ devolve job_id na hora (HTTP 202)

Thread de background:
  1. ffmpeg      vídeo cru ──▶ CFR 30 FPS, H.264, sem áudio   storage/pipeline_input/
  2. MediaPipe   detecta rosto, desenha retângulo verde       storage/annotated/
                 (gate rápido: sem rosto suficiente, recusa aqui)
  3. rppg_pipeline.build_feature_vector()
                 3 ROIs (testa, bochecha D, bochecha E) ──▶ canal verde
                 detrend ──▶ bandpass 0.7–3.5 Hz ──▶ CWT
                 ──▶ 3 sinais de 150 amostras + 2 atrasos de sincronia
  4. model.TruePulseDiscriminator.forward()  ──▶ logits
  5. sigmoid(logits) × 100  ──▶ probabilidade_deepfake_pct  ──▶ decisão

Flutter ──GET /auth/result/{job_id}──▶ "processando" ... "concluido" + veredicto
```

Por que o processamento é assíncrono: FFmpeg + MediaPipe em 150 frames + CWT em
3 sinais + inferência leva alguns segundos. Segurar a conexão HTTP durante todo
esse tempo convida timeout de proxy e de cliente móvel, e não dá ao app como
mostrar progresso. Com job + polling, o `POST` responde em milissegundos.

### Decisões que valem registrar

- **Um detector facial só.** A detecção do retângulo reutiliza o mesmo
  `FaceMeshROIExtractor` do `rppg_pipeline.py`, em vez de um Haar cascade
  separado. Dois detectores diferentes poderiam discordar — a API diria "rosto
  detectado" e o pipeline falharia depois, sem explicação para o usuário.
- **Limiares de qualidade não são duplicados.** `MIN_QUALITY_RATIO` e
  `MIN_SIGNAL_SNR` são importados do `rppg_pipeline.py`, que se declara fonte
  única deles. O próprio arquivo avisa: um vídeo abaixo desses limiares
  representa uma condição que o modelo nunca viu no treino e deve ser
  rejeitado, não classificado com confiança enganosa.
- **Qualidade insuficiente ≠ reprovação.** Vídeo escuro ou tremido devolve
  `veredicto: "inconclusivo"` com código `QUALIDADE_INSUFICIENTE`, pedindo nova
  captura. Reprovar o usuário por uma amostra ruim seria um falso positivo de
  segurança.
- **O vídeo anotado não vai para o rPPG.** Os retângulos desenhados alterariam
  os pixels e contaminariam a média do canal verde. A cópia anotada existe só
  para auditoria e feedback visual.
- **CRF 18 no reencode.** Compressão agressiva destrói justamente as
  micro-variações de cor que o rPPG mede.

---

## Estrutura do projeto

```
truepulse-api/
├── app/
│   ├── main.py                 # FastAPI, CORS, /health, warm-up, faxina de TTL
│   ├── config.py               # configurações (env TRUEPULSE_*)
│   ├── jobs.py                 # JobStore em memória (status/etapa/resultado)
│   ├── schemas.py              # contratos Pydantic de entrada/saída
│   ├── routers/
│   │   └── auth.py             # POST /auth/login, GET /auth/result/{job_id}
│   ├── services/
│   │   ├── video.py            # ffmpeg/ffprobe: 30 FPS, duração, validação
│   │   ├── face.py             # detecção facial + retângulo desenhado
│   │   ├── features.py         # ponte rppg_pipeline ──▶ tensores do modelo
│   │   ├── runtime.py          # singletons: MediaPipe, modelo, thread pool
│   │   ├── verification.py     # orquestra o pipeline; converte logits em %
│   │   └── token.py            # JWT HS256 emitido no login aprovado
│   └── ml/
│       ├── rppg_pipeline.py    # SEU script, sem modificação
│       └── model.py            # SEU script, sem modificação
├── models/                     # checkpoint .pt + cache do face_landmarker.task
├── storage/
│   ├── uploads/                # vídeo cru (apagado assim que é normalizado)
│   ├── pipeline_input/         # ◀── pasta local que os scripts consomem
│   └── annotated/              # vídeo com o retângulo do rosto
├── scripts/
│   ├── gerar_checkpoint_dummy.py
│   └── testar_fluxo.py
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── .env.example
```

Os dois scripts de IA foram copiados **sem nenhuma alteração** para `app/ml/`.
Quando o modelo for retreinado, substitua os arquivos inteiros — não edite
trechos soltos, para não divergir do que foi usado no treino.

---

## Instalação

**Pré-requisito obrigatório: FFmpeg no PATH.**

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# macOS
brew install ffmpeg

# Windows (PowerShell, com winget)
winget install Gyan.FFmpeg
```

Confirme com `ffmpeg -version` e `ffprobe -version`.

```bash
git clone <seu-repo> && cd truepulse-api

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # ajuste, principalmente TRUEPULSE_JWT_SECRET
```

> **Python 3.10 a 3.12.** O `mediapipe` costuma demorar a publicar wheels para
> a versão mais nova do Python; se o `pip install` falhar nele, use 3.11.

Na primeira execução o MediaPipe baixa `face_landmarker.task` (~3 MB) de
`storage.googleapis.com` para `models/`. Em servidor sem internet de saída,
copie o arquivo para `models/face_landmarker.task` antes de subir.

---

## Colocando o modelo treinado no lugar

Copie o checkpoint do seu treino para:

```
models/truepulse_discriminator.pt
```

A API aceita três formatos e detecta a configuração sozinha:

| Formato salvo                                              | O que a API faz                        |
|------------------------------------------------------------|----------------------------------------|
| `torch.save(model.state_dict(), ...)`                        | infere `n_aux` pelo shape de `aux_fc.weight` e `use_freq` pela presença de `freq_fc` |
| `{"model_state_dict": ..., "config": {"n_aux": 4, "use_freq": true}}` | usa o `config` |
| `{"state_dict": ..., "n_aux": ..., "use_freq": ...}`         | usa as chaves de topo |

Se os pesos não baterem com a arquitetura, a API **falha ao carregar** em vez de
completar com `strict=False` silenciosamente — um modelo parcialmente aleatório
respondendo com confiança é pior que um erro explícito.

> ⚠️ **Se você treinou com `--use-freq`**, abra
> `app/services/features.py::_espectro_faixa_cardiaca` e confirme que a
> normalização do espectro é idêntica à do seu `train.py`. Uma diferença ali não
> gera exceção: o modelo simplesmente passa a errar. Se houver qualquer dúvida,
> copie a função do treino para lá.

---

## Executando

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Docs interativas (Swagger): <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

No startup a API faz *warm-up* do MediaPipe e do modelo, para que o primeiro
usuário não pague ~5–10 s de inicialização e ache que a autenticação travou.

---

## Endpoints

### `POST /auth/login`

`multipart/form-data`:

| Campo     | Tipo   | Obrigatório | Descrição                            |
|-----------|--------|-------------|--------------------------------------|
| `video`   | file   | sim         | Vídeo de ~5 s (`.mp4`, `.mov`, `.webm`, `.mkv`, `.avi`) |
| `user_id` | string | não         | Identificador do usuário, vai no token |

**Resposta `202 Accepted`:**

```json
{
  "job_id": "c42b6e193cc9487996f9d5b3177ae430",
  "status": "processando",
  "mensagem": "Vídeo recebido. Verificação em andamento.",
  "poll_url": "http://localhost:8000/auth/result/c42b6e19...",
  "poll_intervalo_ms": 1500
}
```

Erros: `415` formato não suportado · `413` vídeo acima de `MAX_UPLOAD_MB` ·
`400` arquivo vazio.

---

### `GET /auth/result/{job_id}`

**Enquanto processa:**

```json
{
  "job_id": "c42b6e19...",
  "status": "processando",
  "etapa": "extraindo_rppg",
  "decorrido_s": 3.4,
  "resultado": null
}
```

Etapas possíveis: `na_fila` → `normalizando_video` → `detectando_rosto` →
`extraindo_rppg` → `classificando` → `finalizado`.

**Aprovado:**

```json
{
  "job_id": "c42b6e19...",
  "status": "concluido",
  "etapa": "finalizado",
  "decorrido_s": 7.8,
  "resultado": {
    "aprovado": true,
    "veredicto": "aprovado",
    "codigo": "LOGIN_APROVADO",
    "mensagem": "Pessoa real detectada (liveness 88.1%). Login aprovado.",
    "ia": {
      "logits": -2.001,
      "probabilidade_deepfake_pct": 11.92,
      "liveness_score_pct": 88.08,
      "limiar_pct": 50.0,
      "deepfake": false
    },
    "qualidade": {
      "quality_ratio": 0.97, "quality_ratio_min": 0.5,
      "signal_snr": 0.52,   "signal_snr_min": 0.35,
      "fps_detectado": 30.0,
      "frames_com_rosto": 146, "frames_totais": 150,
      "aprovada": true
    },
    "sincronia": { "delay_d_ms": 33.33, "delay_e_ms": -16.67 },
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_expira_em_s": 900
  }
}
```

**Reprovado (deepfake):** mesma estrutura, com `aprovado: false`,
`codigo: "DEEPFAKE_DETECTADO"`, sem `access_token`.

**Falha do servidor:** `status: "falhou"` com `erro` e `erro_codigo` — isso é
erro de infraestrutura (ex.: checkpoint ausente), **não** reprovação do usuário.

---

### `GET /auth/result/{job_id}/video`

Baixa o vídeo com o retângulo verde de reconhecimento facial desenhado. Útil
para depuração e para o app mostrar o que foi analisado.

### `GET /health`

`status: "ok"` só quando FFmpeg está disponível **e** o checkpoint existe.

---

## De `logits` para porcentagem

O `forward()` do `model.py` devolve **logits crus** — de propósito, porque o
treino usa `BCEWithLogitsLoss`, numericamente mais estável que
`sigmoid` + `BCELoss` separados. A conversão acontece num único lugar,
`app/services/verification.py::logits_para_porcentagem`:

```python
p_fake       = sigmoid(logits)        # 0.0 – 1.0
deepfake_pct = p_fake * 100           # 0% – 100%
liveness_pct = 100 - deepfake_pct
```

Referência rápida:

| `logits` | `probabilidade_deepfake_pct` | `liveness_score_pct` | Decisão (limiar 50%) |
|----------|------------------------------|----------------------|----------------------|
| −4.0     | 1.80 %                       | 98.20 %              | aprovado             |
| −2.0     | 11.92 %                      | 88.08 %              | aprovado             |
| 0.0      | 50.00 %                      | 50.00 %              | reprovado (empate cai para reprovar) |
| +2.2     | 90.02 %                      | 9.98 %               | reprovado            |

O limiar é configurável por `TRUEPULSE_DEEPFAKE_THRESHOLD_PCT`. **Calibre-o com
a curva ROC do seu conjunto de validação**, não no olho: 50 % é apenas o ponto
neutro matemático, e o custo de um falso negativo (deepfake aprovado) é bem
maior que o de um falso positivo (pedir nova captura) num sistema de login.

---

## Integração com o Flutter

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

const apiBase = 'http://10.0.2.2:8000'; // emulador Android; iOS: localhost

Future<Map<String, dynamic>> autenticar(File video, {String? userId}) async {
  // 1. envia o vídeo
  final req = http.MultipartRequest('POST', Uri.parse('$apiBase/auth/login'))
    ..files.add(await http.MultipartFile.fromPath('video', video.path));
  if (userId != null) req.fields['user_id'] = userId;

  final resp = await http.Response.fromStream(await req.send());
  if (resp.statusCode != 202) {
    throw Exception('Falha no envio: ${resp.statusCode} ${resp.body}');
  }
  final jobId = jsonDecode(resp.body)['job_id'] as String;

  // 2. polling até sair de "processando"
  final limite = DateTime.now().add(const Duration(seconds: 90));
  while (DateTime.now().isBefore(limite)) {
    await Future.delayed(const Duration(milliseconds: 1500));

    final r = jsonDecode(
      (await http.get(Uri.parse('$apiBase/auth/result/$jobId'))).body,
    ) as Map<String, dynamic>;

    if (r['status'] == 'processando') {
      debugPrint('etapa: ${r['etapa']}');   // mostre no indicador de progresso
      continue;
    }
    if (r['status'] == 'falhou') {
      throw Exception('Erro no servidor: ${r['erro_codigo']}');
    }
    return r['resultado'] as Map<String, dynamic>;
  }
  throw Exception('Tempo limite da verificação excedido.');
}
```

Uso do resultado:

```dart
final res = await autenticar(arquivoDeVideo, userId: 'joao');

final pct = res['ia']?['probabilidade_deepfake_pct'] as num?;

if (res['aprovado'] == true) {
  await salvarToken(res['access_token'] as String);
  irParaHome();
} else if (res['codigo'] == 'QUALIDADE_INSUFICIENTE' ||
           res['codigo'] == 'NENHUM_ROSTO') {
  mostrarDialogo(res['mensagem'] as String);   // pedir nova captura
} else {
  mostrarErro('Verificação negada (${pct?.toStringAsFixed(1)}% de deepfake).');
}
```

Ramifique sempre pelo campo `codigo`, nunca pelo texto de `mensagem` — as
mensagens podem ser reescritas sem aviso; os códigos são estáveis.

**Na captura, garanta:** ~5 s de duração e resolução mínima de 640×480. O FPS é
corrigido no servidor, mas duração fora da faixa de 3,5 s a 6,5 s é rejeitada.

---

## Códigos de resposta

| Código | Onde aparece | Significado | O que o app deve fazer |
|--------|--------------|-------------|------------------------|
| `LOGIN_APROVADO` | `resultado.codigo` | Pessoa real, abaixo do limiar | Guardar o token e entrar |
| `DEEPFAKE_DETECTADO` | `resultado.codigo` | Acima do limiar | Negar o login |
| `QUALIDADE_INSUFICIENTE` | `resultado.codigo` | Abaixo de `MIN_QUALITY_RATIO`/`MIN_SIGNAL_SNR` | Pedir nova captura com mais luz |
| `NENHUM_ROSTO` | `resultado.codigo` | Poucos frames com rosto | Pedir para centralizar o rosto |
| `SINAL_INSUFICIENTE` | `resultado.codigo` | Rosto perdido no meio dos 5 s | Pedir para ficar parado e repetir |
| `DURACAO_INVALIDA` | `resultado.codigo` | Vídeo fora de ~5 s | Corrigir a captura no app |
| `FALHA_FFMPEG` / `FFMPEG_AUSENTE` | `resultado.codigo` | Problema de transcodificação/ambiente | Erro genérico; alertar a equipe |
| `MODELO_INDISPONIVEL` | `erro_codigo` | Checkpoint ausente | Erro 500 do lado do servidor |
| `MODELO_NAO_TREINADO` | `resultado.codigo` | Rodando com pesos aleatórios | Nunca deve aparecer em produção |
| `ERRO_INTERNO` | `erro_codigo` | Exceção não prevista | Erro genérico; checar os logs |

---

## Configuração

Tudo via variáveis de ambiente com prefixo `TRUEPULSE_` (ou arquivo `.env`).
Lista completa em `.env.example`. As mais relevantes:

| Variável | Padrão | Para que serve |
|----------|--------|----------------|
| `TRUEPULSE_DEEPFAKE_THRESHOLD_PCT` | `50` | Limiar de decisão, em % |
| `TRUEPULSE_MAX_UPLOAD_MB` | `60` | Tamanho máximo do vídeo |
| `TRUEPULSE_EXPECTED_DURATION_S` / `_DURATION_TOLERANCE_S` | `5` / `1.5` | Faixa de duração aceita |
| `TRUEPULSE_ANNOTATE_VIDEO` | `true` | Gera a cópia com o retângulo |
| `TRUEPULSE_DRAW_ROIS` | `false` | Desenha também as 3 ROIs (uma passada extra de detecção) |
| `TRUEPULSE_JOB_TTL_MINUTES` | `30` | Depois disso, job e vídeos são apagados |
| `TRUEPULSE_MAX_CONCURRENT_JOBS` | `1` | Jobs simultâneos (leia as limitações antes de subir) |
| `TRUEPULSE_JWT_SECRET` | — | **Troque obrigatoriamente em produção** |
| `TRUEPULSE_CORS_ORIGINS` | `*` | Restrinja em produção |

Os limiares de qualidade (`MIN_QUALITY_RATIO = 0.5`, `MIN_SIGNAL_SNR = 0.35`)
**não** são configuráveis aqui de propósito: eles definem a faixa em que o
modelo foi treinado e vivem no `rppg_pipeline.py`.

---

## Testando sem o modelo treinado

Para validar o encanamento (upload → ffmpeg → rPPG → resposta) antes de o
checkpoint real existir:

```bash
python scripts/gerar_checkpoint_dummy.py --n-aux 2
TRUEPULSE_ALLOW_UNTRAINED_MODEL=true uvicorn app.main:app --reload
```

Com o servidor de pé, em outro terminal:

```bash
pip install requests
python scripts/testar_fluxo.py /caminho/do/video.mp4 --user-id joao
```

Saída típica:

```
job_id=c42b6e19... -> aguardando...
  [  1.5s] normalizando_video
  [  3.0s] extraindo_rppg
Veredicto: APROVADO  [MODELO_NAO_TREINADO]
Logits                  : -0.1183
Probabilidade deepfake  : 47.05%
Liveness score          : 52.95%
```

Nesse modo a resposta vem com o código `MODELO_NAO_TREINADO` e um aviso na
mensagem, e **nenhum token é emitido** — os percentuais são ruído estatístico e
não devem ser confundidos com um veredicto.

---

## Docker

```bash
docker compose up --build
```

A imagem já traz `ffmpeg`, `libgl1` e `libglib2.0-0` (dependências do OpenCV).
`storage/` e `models/` são volumes: o checkpoint não entra na imagem e os
vídeos biométricos não sobrevivem ao container.

---

## Limitações conhecidas e próximos passos

1. **`JobStore` é em memória.** Com `uvicorn --workers > 1`, um job criado no
   worker A não é visível no worker B, e o polling devolve 404 de forma
   intermitente. Por isso o Dockerfile fixa `--workers 1`. Para escalar,
   substitua `JobStore` por Redis mantendo a mesma interface
   (`create`/`get`/`set_stage`/`finish`/`fail`) e troque o `ThreadPoolExecutor`
   por uma fila Celery/RQ.
2. **Um job por vez.** `max_concurrent_jobs=1` é deliberado: o `FaceLandmarker`
   do MediaPipe não é thread-safe. Para aumentar, crie um extractor por thread
   com `threading.local()` antes de subir o valor.
3. **Sem rate limiting.** Um endpoint de login que aceita upload de vídeo é alvo
   óbvio de abuso. Coloque limite por IP/usuário (nginx, `slowapi` ou API
   gateway) antes de expor à internet.
4. **Não há prova de origem da câmera.** A API verifica se o *conteúdo* parece
   uma pessoa viva, não se o vídeo veio mesmo da câmera do dispositivo — um
   atacante pode enviar um arquivo gravado de uma pessoa real por HTTP direto.
   Em produção, combine com *attestation* do app (Play Integrity / App Attest) e
   um desafio por sessão (nonce que o app precisa exibir ou responder durante a
   captura).
5. **O limiar de 50 % é um ponto de partida.** Calibre pela curva ROC do seu
   conjunto de validação, escolhendo o ponto de operação pelo custo real de cada
   tipo de erro no seu contexto.
6. **Retenção de dado biométrico.** O vídeo cru é apagado logo após a
   normalização; os derivados vivem até o TTL do job (30 min). Verifique se essa
   janela atende à sua política de LGPD e ajuste `TRUEPULSE_JOB_TTL_MINUTES`.
7. **O token é mínimo de propósito.** Ele atesta apenas "esta pessoa passou pela
   verificação biométrica no instante X", com validade de 15 min. Ele não
   substitui o seu controle de sessão/refresh.
