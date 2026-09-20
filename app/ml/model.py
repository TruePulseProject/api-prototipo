"""
TruePulse - Modelo Discriminador (RF05)
=========================================
CNN1D + Bi-LSTM que recebe os 3 sinais rPPG filtrados (testa, bochecha D,
bochecha E) + os 2 valores de atraso de sincronia espacial, e devolve a
probabilidade do vídeo ser uma manipulação (Liveness Score = 1 - p_fake).

Entrada:
  signals: tensor (batch, 3, 150)  - os 3 sinais empilhados como canais
  delays:  tensor (batch, 2)       - delay_d_ms, delay_e_ms (em ms, não normalizados)

Saída:
  logits: tensor (batch,) - use sigmoid (ou o método predict_proba) para
  converter em probabilidade; treino usa BCEWithLogitsLoss diretamente sobre
  os logits (mais estável numericamente que sigmoid+BCELoss separados).

Por que essa arquitetura (ver LOG_DECISOES.md seção 10): a sincronia espacial
isolada (um único número por vídeo) não se mostrou um discriminador confiável
sozinha, dado o nível de ruído do sinal. A CNN1D captura padrões locais na
forma da onda (formato do pulso) que uma estatística resumida não captura; a
Bi-LSTM captura dependência temporal ao longo da janela de 5s. Os atrasos
continuam entrando como informação auxiliar, não descartados.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DELAY_NORM_MS = 200.0  # mesmo valor de max_lag_ms em compute_sync_delay_ms
FREQ_BAND = (0.7, 3.5)  # faixa cardíaca, mesma usada no rppg_pipeline.py
N_FREQ_BINS = 14  # bins de FFT dentro da faixa cardíaca, para seq_len=150 @ 30fps


class TruePulseDiscriminator(nn.Module):
    def __init__(self, seq_len=150, cnn_channels=(32, 64), lstm_hidden=64,
                 dropout=0.3, n_aux=2, use_freq=False, n_freq_bins=N_FREQ_BINS):
        super().__init__()
        c1, c2 = cnn_channels

        self.conv1 = nn.Conv1d(3, c1, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.dropout_cnn = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=c2, hidden_size=lstm_hidden,
                             batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout)

        lstm_out_dim = 2 * lstm_hidden  # bidirecional

        # Embute os escalares auxiliares num vetor pequeno antes de combinar
        # com a representação temporal - evita que poucos números "dominem"
        # ou "sumam" frente a um vetor de 128 dimensões vindo da LSTM.
        # n_aux=2 (padrão/compatibilidade): só os atrasos de sincronia.
        # n_aux=4: atrasos + quality_ratio + signal_snr (ver train.py --extra-features).
        self.n_aux = n_aux
        self.aux_fc = nn.Linear(n_aux, 16)

        # Ramo de domínio de frequência (opcional, ver train.py --use-freq):
        # recebe a magnitude do espectro de FFT de cada um dos 3 sinais,
        # restrito à faixa cardíaca (0.7-3.5Hz) - representação
        # complementar à forma de onda no tempo, que o ramo CNN+LSTM já
        # captura. Literatura de sinais fisiológicos (ECG/PPG) reporta
        # ganho ao combinar domínio de tempo e frequência.
        self.use_freq = use_freq
        if use_freq:
            freq_in = 3 * n_freq_bins
            self.freq_fc = nn.Sequential(
                nn.Linear(freq_in, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_dim = lstm_out_dim + 16 + 32
        else:
            fusion_dim = lstm_out_dim + 16

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, signals, aux, freq=None):
        # signals: (batch, 3, seq_len)
        # aux: (batch, n_aux) - as 2 primeiras colunas são sempre os atrasos
        # em ms (normalizados aqui por DELAY_NORM_MS); colunas extras (se
        # n_aux>2) já vêm em escala ~0-1 (quality_ratio, signal_snr) e não
        # são renormalizadas.
        # freq: (batch, 3*n_freq_bins) - só usado se use_freq=True; ignorado
        # (pode vir None) caso contrário, para manter compatibilidade.
        x = F.relu(self.bn1(self.conv1(signals)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout_cnn(x)

        x = x.permute(0, 2, 1)  # (batch, seq_len, canais) - formato que a LSTM espera
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 2*hidden)
        lstm_out = self.dropout_lstm(lstm_out)

        # Pooling médio ao longo do tempo: resume a sequência inteira num
        # único vetor, robusto a variações de qual parte dos 5s foi mais
        # "informativa" (não assume que só o último instante importa).
        pooled = lstm_out.mean(dim=1)  # (batch, 2*hidden)

        aux_delays = aux[:, :2] / DELAY_NORM_MS
        aux_extra = aux[:, 2:]  # já em escala ~0-1, sem normalização extra
        aux_norm = torch.cat([aux_delays, aux_extra], dim=1)
        aux_out = F.relu(self.aux_fc(aux_norm))

        pieces = [pooled, aux_out]
        if self.use_freq:
            freq_out = self.freq_fc(freq)
            pieces.append(freq_out)

        combined = torch.cat(pieces, dim=1)
        logits = self.classifier(combined).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, signals, aux, freq=None):
        """Retorna a probabilidade (0-1) de o vídeo ser fake. Use em
        inferência; para treino, use forward() + BCEWithLogitsLoss."""
        self.eval()
        logits = self.forward(signals, aux, freq)
        return torch.sigmoid(logits)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: valida que o forward/backward roda sem erro de dimensão,
    # com tensores fictícios (não é treino de verdade).
    torch.manual_seed(0)
    model = TruePulseDiscriminator()
    print(f"Parâmetros treináveis: {count_parameters(model):,}")

    batch_size = 8
    dummy_signals = torch.randn(batch_size, 3, 150)
    dummy_delays = torch.randn(batch_size, 2) * 100  # simula ms na faixa +-200
    dummy_labels = torch.randint(0, 2, (batch_size,)).float()

    logits = model(dummy_signals, dummy_delays)
    print(f"Shape da saída (esperado [{batch_size}]): {tuple(logits.shape)}")

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, dummy_labels)
    print(f"Loss inicial (deve ser um número finito, ~0.6-0.8 tipicamente): {loss.item():.4f}")

    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradientes fluíram para todos os parâmetros: {grad_ok}")

    probs = model.predict_proba(dummy_signals, dummy_delays)
    print(f"predict_proba shape: {tuple(probs.shape)}, "
          f"faixa: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
