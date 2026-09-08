# RL Multi MARL Competition

Arena 3D com 9 agentes (3 equipes × 3) treinados com **PPO + PyTorch** para comparar paradigmas de **Aprendizado por Reforço Multi-Agente (MARL)**.

O repositório contém **dois experimentos autocontidos**, com a mesma engine de simulação, mas comparações distintas:

| Experimento | Localização | Comparado | Treino | Tracking |
|-------------|-------------|-----------|--------|----------|
| **1** | raiz do repositório | CTE × DTE × CTDE | `scripts/train_rl.py` | `data/…` (JSON/CSV/PNG) |
| **2** | `ctde_arena/` | CTDE-VD × CTDE-CAC × CTDE-Comm | `ctde_arena/scripts/train_rl.py` | **MLflow** + `ctde_arena/data/…` |

Cada experimento é independente (pacote próprio `marl_arena`, `.env` e dados), então todos os comandos devem ser executados **a partir do diretório do experimento** (raiz do repositório ou `ctde_arena/`).

---

## Índice

- [Experimento 1 — CTE × DTE × CTDE](#experimento-1--cte--dte--ctde-raiz)
- [Experimento 2 — CTDE-VD × CTDE-CAC × CTDE-Comm](#experimento-2--variantes-ctde-ctde_arena)
- [Instalação](#instalação)
- [Como usar](#como-usar)
- [Resultados](#resultados)
- [Configuração (.env)](#configuração-env)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Arquivos de saída](#arquivos-de-saída)
- [Métricas e gráficos](#métricas-e-gráficos)
- [Testes](#testes)
- [Próximos passos](#próximos-passos)

---

## Experimento 1 — CTE × DTE × CTDE (raiz)

Compara três paradigmas MARL diferentes em execução/decisão.

| Equipe | Paradigma | Descrição |
|--------|-----------|-----------|
| Equipe 1 | **CTE** | Centralized Training & Execution — ator e crítico centralizados (decisão única para a equipe) |
| Equipe 2 | **DTE** | Decentraled Training & Execution — ator e crítico locais por agente |
| Equipe 3 | **CTDE** | Centralized Training, Decentralized Execution — ator local, crítico global só no treino |

### Treinar

```bash
python scripts/train_rl.py
```

Gera checkpoints e log de treino e atualiza o resumo de métricas ao final.

### Executar a arena 3D (visual)

```bash
python main.py
```

Carrega os checkpoints e roda a simulação interativa. No fim de cada partida grava métricas e exporta o dashboard.

---

## Experimento 2 — Variantes CTDE (`ctde_arena/`)

Extensão autocontida que aprofunda dentro do paradigma **CTDE**, comparando três sub-abordagens:

| Equipe | Paradigma | Descrição |
|--------|-----------|-----------|
| Equipe 1 | **CTDE-VD** | *Value Decomposition* (estilo VDN). O valor do time é decomposto na soma dos valores locais: \(V_{tot}(s)=\sum_{i=1}^{3} V_i(o_i)\). Ajuda na atribuição de crédito e estabiliza o gradiente. |
| Equipe 2 | **CTDE-CAC** | Ator-Crítico Centralizado padrão (estilo **MAPPO**). Atores locais decidem; crítico estima valor do estado global completo. |
| Equipe 3 | **CTDE-Comm** | Comunicação explícita (estilo **CommNet**). Atores trocam mensagens diferenciáveis durante a execução e usam crítico centralizado no treino. |

### Treinar (com **MLflow**)

```bash
cd ctde_arena
python scripts/train_rl.py
```

O treino registra hiperparâmetros, métricas por equipe ao longo dos steps, artefatos (plots) e checkpoints no **MLflow**.

Para inspecionar no MLflow UI (runs, win-rate, gráficos e Model Registry):

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Executar a arena 3D (visual)

```bash
cd ctde_arena
python main.py
```

Carrega os checkpoints de `ctde_arena/data/checkpoints` e roda a simulação interativa.

### Docker

O sub-projeto contém um `Dockerfile` para rodar o treino de forma isolada.

```bash
cd ctde_arena
docker build -t ctde-arena .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena
```

---

## Instalação

Pré-requisito: **Python 3.10+**. Execute a partir do diretório de cada experimento (raiz ou `ctde_arena/`).

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

Em seguida, copie `.env.example` para `.env` e ajuste conforme necessário.

Dependências principais:

- raiz: `ursina 6.1.2`, `numpy 2.2.6`, `matplotlib 3.10.3`, `python-dotenv 1.0.1`, `torch 2.6.0`, `pytest 8.3.5`.
- `ctde_arena/`: as mesmas + **`mlflow 2.17.2`**.

> Observação: o script auxiliar `scripts/plot_metrics.py` (na raiz) usa **pandas**, que não está em `requirements.txt`. Instale com `pip install pandas` para usá-lo ou gere gráficos via `src/marl_arena/systems/plotting.py`.

---

## Como usar

Fluxo típico (para cada experimento, a partir do seu diretório):

1. **Instalar** dependências e configurar `.env`.
2. **Treinar** (opcional — o repo inclui checkpoints pré-treinados de 100.000 steps): `python scripts/train_rl.py`.
3. **Rodar a arena 3D para avaliar/visualizar**: `python main.py`.

Os dois experimentos expõem as mesmas entradas:

| Comando | Função |
|---------|--------|
| `python scripts/train_rl.py` | Treina as políticas PPO (padrão configurado para 3M steps em `.env`, mas os checkpoints versionados usaram 100k) |
| `python main.py` | Executa a arena 3D visual usando os checkpoints |

---

## Resultados

### Experimento 1 — CTE × DTE × CTDE

Valores extraídos de `data/metrics/summary.json` (treino versionado: 100.000 steps, 463 partidas).

| Equipe | Paradigma | Win rate | Elim./partida | Sobrevivência média (s) | Precisão de tiro |
|--------|-----------|---------:|--------------:|-------------------------:|-----------------:|
| Equipe 1 | CTE  | 27.39% | 1.72 | 9.63  | 8.10%  |
| Equipe 2 | DTE  | 25.43% | 1.84 | 8.59  | 8.13%  |
| Equipe 3 | CTDE | 47.17% | 3.42 | 9.93  | 12.84% |

**Interpretação**

- **Equipe 3 (CTDE)** tem o melhor desempenho geral: maior win rate, mais eliminações e melhor precisão de tiro.
- Equipes 1 (CTE) e 2 (DTE) têm desempenho parecido, com win rates ~25–27% e precisão ~8%.

### Experimento 2 — Variantes CTDE

Valores extraídos de `ctde_arena/data/metrics/summary.json` (treino versionado: 100.000 steps, 456 partidas).

| Equipe | Paradigma | Win rate | Elim./partida | Sobrevivência média (s) | Precisão de tiro |
|--------|-----------|---------:|--------------:|-------------------------:|-----------------:|
| Equipe 1 | CTDE-VD   | 40.67% | 2.50 | 10.20 | 11.29% |
| Equipe 2 | CTDE-CAC  | 42.89% | 2.63 | 9.83  | 12.33% |
| Equipe 3 | CTDE-Comm | 16.44% | 1.96 | 7.47  | 12.14% |

**Análise e diagnóstico**

O comportamento após 100.000 steps reflete as características arquiteturais internas:

1. **Ator-Crítico Centralizado (CTDE-CAC / MAPPO)** — **42,89%**. O crítico centralizado vê o estado global completo de todos os 9 agentes, produzindo estimativas de valor com baixa variância e sinais de vantagem precisos para as políticas locais. Isso acelera o aprendizado nas fases iniciais.
2. **Value Decomposition (CTDE-VD / VDN)** — **40,67%**. A decomposição do valor conjunto em soma de valores locais facilita a atribuição de crédito multi-agente e restringe a função de valor a observações locais, estabilizando o gradiente e reduzindo overfitting no início.
3. **Comunicação Explícita (CTDE-Comm / CommNet)** — **16,44%**. A política depende de \([o_i, c_i]\) (observação local + mensagens recebidas). No início as mensagens são ruído; a política precisa aprender simultaneamente a agir e a desenvolver um protocolo de comunicação. Esse problema de aprendizado duplo exige muito mais steps (ex.: >500k–1M) para que as mensagens se tornem úteis.

> Observação: o treino padrão em `.env` é 3.000.000 steps para os dois experimentos; os checkpoints e métricas versionados correspondem a execuções mais curtas (100.000 steps). Para treinar por mais tempo, ajuste `RL_TRAIN_TOTAL_STEPS` no `.env`.

---

## Configuração (.env)

Todas as variáveis (idênticas nos dois experimentos). Copie de `.env.example`.

### Ambiente de simulação

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `ARENA_SIZE` | 32 | Tamanho do lado da arena |
| `MATCH_DURATION_SECONDS` | 90 | Duração máxima da partida (s) |
| `RESPAWN_ENABLED` | false | Respawn de agentes (atualmente não implementado na simulação) |
| `AGENT_MOVE_SPEED` | 4.5 | Velocidade de movimento dos agentes |
| `AGENT_TURN_SPEED` | 110 | Velocidade de rotação (graus/s) |
| `JUMP_SPEED` | 6.3 | Velocidade vertical de pulo |
| `GRAVITY` | 14.0 | Gravidade |
| `SHOOT_RANGE` | 20.0 | Alcance máximo de tiro |
| `SHOOT_COOLDOWN` | 0.45 | Cooldown entre tiros (s) |
| `RANDOM_SEED` | 7 | Semente de aleatoriedade |
| `SIM_STEP_DT` | 0.1 | Passo de integração da simulação (s) |
| `PLOT_UPDATE_INTERVAL` | 1.0 | (config auxiliar) Intervalo de atualização de gráficos |
| `METRICS_FLUSH_INTERVAL` | 1.5 | (config auxiliar) Intervalo de escrita de métricas |
| `DOMAIN_RANDOMIZATION` | true | Habilita randomização de domínio no treino |

### Treino / Algoritmo (PPO / RL)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `RL_TRAIN_TOTAL_STEPS` | 3000000 | Total de env steps de treino |
| `RL_SAVE_EVERY_STEPS` | 100000 | Salvar checkpoints a cada N steps |
| `RL_LOG_EVERY_STEPS` | 50000 | Logar progresso a cada N steps |
| `RL_METRICS_EVERY_MATCHES` | 10 | Registrar métricas a cada N partidas |
| `RL_LEARNING_RATE` | 0.0003 | Taxa de aprendizado |
| `RL_GAMMA` | 0.99 | Fator de desconto |
| `RL_GAE_LAMBDA` | 0.95 | Lambda do GAE |
| `RL_CLIP_EPS` | 0.2 | Clipping do PPO |
| `RL_VALUE_COEF` | 0.5 | Coeficiente da perda de valor |
| `RL_ENTROPY_COEF` | 0.01 | Coeficiente do bônus de entropia |
| `RL_MAX_GRAD_NORM` | 0.5 | Clipping de gradiente |
| `RL_PPO_EPOCHS` | 4 | Épocas de atualização por rollout |
| `RL_BATCH_SIZE` | 256 | Tamanho do batch |
| `RL_HIDDEN_DIM` | 128 | Largura das camadas ocultas |
| `RL_DEVICE` | cpu | Dispositivo (`cpu`, `cuda`, `mps`) |

### Domain Randomization (intervalos) — usado no treino

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `DR_ARENA_SIZE_MIN` / `_MAX` | 28 / 36 | Tamanho da arena |
| `DR_MATCH_DURATION_MIN` / `_MAX` | 60 / 120 | Duração da partida (s) |
| `DR_MOVE_SPEED_MIN` / `_MAX` | 3.5 / 5.5 | Velocidade de movimento |
| `DR_TURN_SPEED_MIN` / `_MAX` | 90 / 130 | Velocidade de rotação |
| `DR_SHOOT_RANGE_MIN` / `_MAX` | 16 / 24 | Alcance de tiro |
| `DR_SHOOT_COOLDOWN_MIN` / `_MAX` | 0.35 / 0.6 | Cooldown de tiro |
| `DR_OBSTACLE_COUNT_MIN` / `_MAX` | 5 / 10 | Quantidade de obstáculos |

---

## Estrutura do repositório

```text
.
├── main.py                     # Arena 3D (experimento 1)
├── scripts/
│   ├── train_rl.py             # Treino (experimento 1)
│   └── plot_metrics.py         # Gera gráficos/CSV a partir de data/metrics/summary.json (usa pandas)
├── src/marl_arena/
│   ├── config.py               # Configuração (env)
│   ├── models.py               # Dataclasses (snapshots, métricas, etc.)
│   ├── controllers/
│   │   ├── base.py             # BaseTeamController, features
│   │   └── rl_controller.py    # Paradigmas CTE/DTE/CTDE (experimento 1)
│   ├── rl/
│   │   ├── actions.py          # Espaço de ações e checkpoint I/O
│   │   ├── buffer.py           # RolloutBuffer + GAE
│   │   ├── networks.py         # ActorNetwork, Centralized_Critic etc.
│   │   └── ppo.py              # PPOTrainer (updates actor-critic, ctde, cte)
│   ├── systems/
│   │   ├── match_variant.py    # Variantes de partida e randomização de domínio
│   │   ├── simulation.py       # Lógica de física/combate
│   │   ├── metrics.py          # MetricsStore (CSV/JSON + dashboard PNG)
│   │   └── plotting.py         # Dashboard matplotlib de métricas acumuladas
│   └── ui/
│       └── dashboard.py        # Overlay in-game
├── tests/
│   ├── test_match_variant.py
│   ├── test_rl_training.py
│   └── test_rl_networks.py
├── data/
│   ├── checkpoints/            # *.pt por equipe (experimento 1)
│   ├── metrics/                # *.csv e summary.json
│   └── exports/                # dashboards PNG
└── ctde_arena/                 # Experimento 2 (variantes CTDE)
    ├── main.py
    ├── Dockerfile
    ├── scripts/train_rl.py     # Treino + MLflow
    ├── src/marl_arena/         # Mesma engine; RL próprio (VD, CAC, Comm)
    ├── tests/test_components.py
    └── data/                   # checkpoints, metrics, exports próprios
```

---

## Arquivos de saída

Cada experimento grava em **seu próprio** diretório `data/`:

| Caminho | Conteúdo |
|---------|----------|
| `data/checkpoints/` | Checkpoints das políticas (`*.pt`) por equipe/paradigma |
| `data/checkpoints/training_log.json` | Histórico de treino (steps, partidas, snapshot de métricas) |
| `data/metrics/team_match_metrics.csv` | Métricas por equipe por partida |
| `data/metrics/agent_match_metrics.csv` | Métricas por agente por partida |
| `data/metrics/trajectory_metrics.csv` | Série temporal (vitorias/eliminações/precisão acumuladas) |
| `data/metrics/summary.json` | **Resumo consolidado** das métricas (mostrado acima) |
| `data/exports/*.png` | Dashboard(s) exportados (ex.: `comparative_dashboard.png`) |

Notas: CSVs/JSON/PNG em `data/` são ignorados pelo git; os arquivos `.legacy*.*` são backups automáticos quando o esquema de CSV muda.

Arquivos de métricas e checkpoint versionados usados neste README:

- Experimento 1: `data/metrics/summary.json`
- Experimento 2: `ctde_arena/data/metrics/summary.json`

---

## Métricas e gráficos

- **Dashboard automático** (ambos os experimentos): ao final de cada partida, `systems/plotting.py` gera `data/exports/comparative_dashboard.png` (win rate, eliminações acumuladas, sobrevivência média e precisão de tiro por partida).
- **Gráficos rápidos (raiz)**: `python scripts/plot_metrics.py` lê `data/metrics/summary.json` e exporta CSV + PNG em `exports/metrics/` (requer **pandas**).
- **MLflow (apenas `ctde_arena/`)**: o treino registra `win_rate`, `shot_accuracy`, `mean_survival_time` e `eliminations_per_match` por equipe ao longo dos steps, além dos plots e checkpoints como artefatos. UI: `mlflow ui --backend-store-uri file:./mlruns`.

---

## Testes

Execute a partir do diretório do experimento correspondente.

```bash
# Experimento 1 (3 testes: variante, treino curto, redes)
python -m pytest tests/ -q

# Experimento 2 (testes de componentes: VD critic e Comm actor)
cd ctde_arena
python -m pytest tests/ -q
```

---

## Próximos passos

Ideias de aprofundamento (aplicáveis a ambos os experimentos):

- Treinar com mais steps (o padrão `.env` é 3M; os resultados versionados usaram 100k), especialmente para o **CTDE-Comm**, que precisa de mais tempo.
- Rodar avaliações com múltiplas seeds para validar a estabilidade estatística dos win-rates.
- Registrar/plotar as estatísticas do PPO (policy/value loss, entropy) ao longo do treino — já calculadas em `PPOStats`, mas ainda não persistidas.
- Expandir a cobertura de testes (PPO, buffer, métricas, colisões de projétil).

---

## Contato

Abra uma issue ou PR neste repositório para discutir experimentos, dúvidas ou melhorias.
