# RL Multi MARL Competition

Arena 3D com 9 agentes (3 equipes × 3), cada equipe com um paradigma MARL treinado via **PPO + PyTorch**:

| Equipe | Paradigma | Rede |
|--------|-----------|------|
| Equipe 1 | CTE | Ator centralizado + crítico global |
| Equipe 2 | DTE | Actor-Critic local |
| Equipe 3 | CTDE | Ator local + crítico global (só no treino) |

## Pontos de entrada

O projeto usa **dois comandos** — não há pasta `scripts/` cheia de utilitários:

| Comando | Arquivo | Função |
|---------|---------|--------|
| `python scripts/train_rl.py` | Treino headless | Roda N partidas, atualiza redes com PPO, salva checkpoints e métricas |
| `python main.py` | Arena 3D | Carrega checkpoints e exibe a competição em tempo real (inferência) |

Isso é intencional: treino e visualização são fluxos separados, mas cada um tem um único script. Testes automatizados ficam em `tests/` (pytest), não em `scripts/`.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e ajuste se necessário.

## Treinar

```bash
python scripts/train_rl.py
```

Durante o treino:

- checkpoints em `data/checkpoints/` (`equipe_1_cte.pt`, `equipe_2_dte.pt`, `equipe_3_ctde.pt`)
- log resumido em `data/checkpoints/training_log.json`
- métricas CSV em `data/metrics/`
- gráfico comparativo em `data/exports/comparative_dashboard.png` (gerado ao registrar cada partida)

## Jogar (visual 3D)

```bash
python main.py
```

Requer checkpoints já treinados. Os agentes usam política gulosa (sem exploração). Use o mouse para orbitar a câmera; o botão **Reiniciar Partida** reseta a rodada.

## Testes

```bash
python -m pytest tests/ -q
```

Cobre redes neurais e um ciclo curto de treino/simulação.

## Estrutura do projeto

```text
.
├── main.py                 # entrada: visualização 3D
├── scripts/
│   └── train_rl.py         # entrada: treino PPO
├── tests/
│   ├── test_rl_networks.py
│   └── test_rl_training.py
├── requirements.txt
├── .env / .env.example
└── src/marl_arena/
    ├── config.py
    ├── models.py
    ├── controllers/
    │   ├── base.py         # features, alvos, movimento
    │   └── rl_controller.py
    ├── rl/
    │   ├── actions.py      # espaço de ação + checkpoints
    │   ├── networks.py
    │   ├── buffer.py
    │   └── ppo.py
    ├── systems/
    │   ├── simulation.py   # física, combate, partidas
    │   ├── metrics.py
    │   └── plotting.py
    └── ui/
        └── dashboard.py
```

## Configuração (`.env`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `RL_TRAIN_MATCHES` | 300 | Partidas por sessão de treino |
| `RL_SAVE_EVERY` | 25 | Frequência de salvamento dos `.pt` |
| `RL_DEVICE` | cpu | `cpu`, `cuda` ou `mps` |
| `RL_LEARNING_RATE` | 0.0003 | Taxa do Adam no PPO |
| `RL_PPO_EPOCHS` | 4 | Épocas PPO por partida |
| `MATCH_DURATION_SECONDS` | 90 | Tempo máximo por partida |
| `SHOOT_RANGE` | 20 | Alcance do disparo |
| `SHOOT_COOLDOWN` | 0.45 | Intervalo entre tiros |
| `RANDOM_SEED` | 7 | Semente da simulação |

## Saídas geradas

| Caminho | Conteúdo |
|---------|----------|
| `data/checkpoints/*.pt` | Pesos treinados por equipe/paradigma |
| `data/checkpoints/training_log.json` | Win rates ao longo do treino |
| `data/metrics/team_match_metrics.csv` | Resultado por partida/equipe |
| `data/metrics/summary.json` | Resumo acumulado |
| `data/exports/comparative_dashboard.png` | Gráficos de vitória, eliminações, sobrevivência, precisão |
