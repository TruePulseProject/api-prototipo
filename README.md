# Biometric Auth API

Backend em FastAPI para autenticação facial com verificação anti-deepfake.

## Estrutura

```
backend/
├── app/
│   ├── main.py                    # App FastAPI + rotas
│   ├── config.py                  # Paths, limites, tipos de vídeo aceitos
│   ├── models.py                  # Schemas Pydantic (request/response)
│   ├── storage.py                 # Estado dos jobs (em memória)
│   ├── routers/
│   │   └── auth.py                # Endpoints /auth/login e /auth/result/{job_id}
│   └── services/
│       ├── extrair_frames.py      # SCRIPT 1: vídeo -> 150 frames (30fps)
│       ├── facerec_frames.py      # SCRIPT 2: detecção facial nos frames
│       └── ai_verification.py     # >>> plugue sua IA de deepfake aqui <<<
├── storage/
│   ├── videos/                    # vídeos recebidos (e convertidos p/ 30fps, se preciso)
│   └── frames/{job_id}/
│       ├── brutos/                # saída do script1: 150 frames extraídos
│       └── detectados/            # saída do script2: frames com retângulo de rosto/olhos
└── requirements.txt
```

## Como rodar

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Documentação interativa automática: `http://localhost:8000/docs`

### Requisito extra: ffmpeg no sistema

O `extrair_frames.py` (script1) usa `ffmpeg` via `subprocess` para
converter o vídeo para 30fps quando o vídeo enviado não estiver nesse fps.
Isso **não é instalado via pip** — precisa estar no sistema operacional:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows: baixar em https://ffmpeg.org/download.html e adicionar ao PATH
```

Sem o ffmpeg instalado, vídeos que não estiverem em 30fps vão falhar no
processamento (o job cai em `status: "error"`; vídeos já gravados a 30fps
funcionam mesmo sem ffmpeg, porque a conversão é pulada).

## Fluxo dos endpoints

### 1. `POST /auth/login`
Recebe o vídeo de ~5s (multipart/form-data, campo `video`) capturado pela
webcam no Flutter.

- Salva o vídeo em `storage/videos/`.
- Responde **imediatamente** com um `job_id` (o processamento roda em
  background, então o app não fica travado esperando).
- Em background, executa o pipeline:
  1. `extrair_frames.processar_video()` (script1) — converte para 30fps se
     necessário e extrai 150 frames igualmente espaçados.
  2. `facerec_frames.processar_frames()` (script2) — roda detecção de
     rosto/olhos em cada frame extraído.
  3. Só os frames em que um rosto foi detectado são enviados para
     `ai_verification.verify_frames()` (deepfake + identidade).

Resposta:
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "processing",
  "message": "Vídeo recebido, processando verificação."
}
```

### 2. `GET /auth/result/{job_id}`
O Flutter consulta esse endpoint (polling, ex: a cada 1 segundo) até o
`status` deixar de ser `"processing"`.

Resposta quando aprovado:
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "approved",
  "is_deepfake": false,
  "is_correct_person": true,
  "confidence": 0.99,
  "message": "Login aprovado."
}
```

Possíveis valores de `status`: `processing`, `approved`, `rejected`, `error`.

## Scripts integrados

Os scripts 1 e 2 são exatamente os que você já tinha, só que "empacotados"
como módulos importáveis em vez de scripts de linha de comando:

1. **`app/services/extrair_frames.py`** — idêntico ao seu original
   (`obter_fps`, `converter_para_30fps`, `extrair_frames_igualmente_espacados`),
   com uma função nova, `processar_video(video, pasta_saida, quantidade_frames)`,
   que é o que o antigo `main()` fazia lendo `sys.argv`, só que chamável
   pela API em vez de rodar pelo terminal.

2. **`app/services/facerec_frames.py`** — idêntico ao seu original
   (`detect`, `draw_rects`, `carregar_cascades`, etc.), com uma mudança na
   função `processar_frames()`: além de continuar salvando os frames com
   os retângulos desenhados (igual antes), ela agora **retorna** uma lista
   `[(caminho_do_frame, quantidade_de_rostos), ...]`. Isso é necessário
   porque o router usa essa informação para filtrar e mandar pra IA
   **só** os frames em que um rosto foi realmente detectado — sem essa
   mudança, o router não teria como saber quais frames têm rosto sem
   reabrir e reprocessar tudo de novo.

O único arquivo que ainda é placeholder (porque você disse que a parte de
IA já está pronta em outro lugar) é:

3. **`app/services/ai_verification.py`** → função `verify_frames(frames)`.
   Troque o corpo pela chamada à sua IA: recebe a lista de frames com
   rosto detectado, retorna `(is_deepfake, is_correct_person, confidence)`.

Nenhuma outra parte do backend precisa mudar — o pipeline em
`app/routers/auth.py` já chama essas três etapas nessa ordem.

## Exemplo de chamada a partir do Flutter

```dart
import 'package:http/http.dart' as http;

Future<String> enviarVideo(File videoFile) async {
  final uri = Uri.parse('http://SEU_HOST:8000/auth/login');
  final request = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('video', videoFile.path));

  final response = await request.send();
  final body = await response.stream.bytesToString();
  final data = jsonDecode(body);
  return data['job_id'];
}

Future<Map<String, dynamic>> consultarResultado(String jobId) async {
  final uri = Uri.parse('http://SEU_HOST:8000/auth/result/$jobId');
  final response = await http.get(uri);
  return jsonDecode(response.body);
}
```

No Flutter, depois de enviar o vídeo, faça polling em `consultarResultado`
(por exemplo com `Timer.periodic` a cada 1s) até `status` sair de
`"processing"`.

## Observações importantes

- **Estado em memória**: `app/storage.py` guarda os jobs num dict em
  memória. Funciona bem para 1 processo/instância. Se for rodar com
  múltiplos workers (`--workers > 1`) ou em produção com escala, troque
  por Redis ou banco de dados — um dict local não é compartilhado entre
  processos.
- **Limite de tamanho de vídeo**: configurável via variável de ambiente
  `MAX_VIDEO_SIZE_MB` (padrão 25MB), em `app/config.py`.
- **CORS**: liberado para `*` para facilitar testes. Restrinja ao domínio
  do seu app antes de ir para produção.
- **Limpeza de arquivos**: vídeos e frames ficam salvos em `storage/`. Vale
  adicionar uma rotina (cron ou background task) para apagar arquivos
  antigos periodicamente.
- **Segurança**: como é um endpoint de autenticação, considere adicionar
  HTTPS, rate limiting e, futuramente, um token de sessão temporário
  emitido só depois do `status: approved`.
