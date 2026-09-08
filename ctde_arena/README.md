# CTDE Paradigm Comparison Arena

Extensão autocontida (sub-projeto) do repositório [RL Multi MARL Competition](../README.md) que compara três **categorias de Aprendizado por Reforço Multi-Agente (MARL)** dentro da arquitetura **CTDE (Centralized Training with Decentralized Execution)**:

1. **Value Decomposition (CTDE-VD)** — estilo VDN. Decompõe o valor do time na soma de valores locais de cada agente: $V_{tot}(s) = \sum_{i=1}^{3} V_i(o_i)$.
2. **Centralized Actor-Critic (CTDE-CAC)** — estilo MAPPO. Atores tomam ações com base em observações locais, e o crítico estima o valor do estado global completo.
3. **Explicit Communication (CTDE-Comm)** — estilo CommNet. Atores descentralizados trocam mensagens diferenciáveis durante a execução, com crítico centralizado no treino.

Cada paradigma é representado por uma equipe na simulação (3 equipes × 3 agentes na mesma arena).

> Visão unificada e resultados do outro experimento (CTE × DTE × CTDE) estão no [README raiz](../README.md). Este documento é autocontido para o experimento de variantes CTDE.

---

## Índice

- [Arquitetura (descrição dos paradigmas)](#arquitetura-descrição-dos-paradigmas)
- [Estrutura do sub-projeto](#estrutura-do-sub-projeto)
- [Instalação](#instalação)
- [Treinamento e MLOps com MLflow](#treinamento-e-mlops-com-mlflow)
- [Executar a simulação visual (UI)](#executar-a-simulação-visual-ui)
- [Uso com Docker](#uso-com-docker)
- [Resultados do treinamento](#resultados-do-treinamento)
- [Configuração (.env)](#configuração-env)
- [Testes automatizados](#testes-automatizados)
- [Próximos passos](#próximos-passos)

---

## Arquitetura (descrição dos paradigmas)

| Equipe | Paradigma | Crítico | Ator | Observações principais |
|--------|-----------|---------|------|-------------------------|
| Equipe 1 | **CTDE-VD** | `ValueDecompositionCriticNetwork` (soma de $V_i(o_i)$ por agente) | `ActorNetwork` (local) | Bom équilibrio entre coordenação e atribuição de crédito; gradiente estável. |
| Equipe 2 | **CTDE-CAC** | `CentralizedCriticNetwork` (estado global) | `ActorNetwork` (local) | Variância de valor baixa; aprendizado rápido em fases iniciais. |
| Equipe 3 | **CTDE-Comm** | `CentralizedCriticNetwork` (estado global) | `CommActorNetwork` (local + mensagens) | Requer aprendizado simultâneo de ação e protocolo de comunicação. |

Implementações específicas (diferenças em relação à raiz):

- `src/marl_arena/rl/networks.py` — adiciona `ValueDecompositionCriticNetwork` e `CommActorNetwork`.
- `src/marl_arena/rl/ppo.py` — adiciona `update_ctde_vd` e `update_ctde_comm`.
- `src/marl_arena/controllers/rl_controller.py` — seleção de paradigmas `CTDE-VD`/`CTDE-CAC`/`CTDE-Comm` e lógica de decisão para o caso de comunicação.

---

## Estrutura do sub-projeto

```text
ctde_arena/
├── main.py                  # Arena 3D visual (Ursina)
├── Dockerfile               # Build do container de treino
├── README.md                # Este arquivo
├── requirements.txt         # Inclui mlflow 2.17.2
├── scripts/
│   ├── train_rl.py          # Treino PPO (integrado ao MLflow)
│   └── plot_metrics.py      # Gera CSV/PNG a partir de data/metrics/summary.json
├── src/marl_arena/
│   ├── config.py            # Mesma config da raiz (.env)
│   ├── controllers/         # base.py, rl_controller.py (paradigmas CTDE-*)
│   ├── rl/                  # actions, buffer, networks, ppo
│   ├── systems/             # match_variant, simulation, metrics, plotting
│   └── ui/                  # dashboard
└── tests/
    └── test_components.py   # Testes de sanidade das novas redes
```

---

## Instalação

```bash
cd ctde_arena
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e ajuste conforme necessário.

Dependências (fixadas): `ursina==6.1.2`, `numpy==2.2.6`, `matplotlib==3.10.3`, `python-dotenv==1.0.1`, `torch==2.6.0`, `pytest==8.3.5`, `mlflow==2.17.2`.

---

## Treinamento e MLOps com MLflow

O treino é monitorado com **MLflow**: hiperparâmetros, logs de experimentos, métricas por equipe no decorrer dos steps, gráficos e modelos registrados no Model Registry.

```bash
cd ctde_arena
python scripts/train_rl.py
```

A execução registra:

- Parâmetros de configuração (todas as chaves simples de `CONFIG`);
- Por equipe, a cada `RL_LOG_EVERY_STEPS`: `win_rate`, `shot_accuracy`, `mean_survival_time`, `eliminations_per_match`;
- Artefatos de checkpoint e dashboards em `mlruns/` e em `data/…`.

### MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Veja runs, win-rate das equipes, gráficos e modelos no Model Registry (nomes: `CTDE_Arena_*_Actor`).

---

## Executar a simulação visual (UI)

```bash
cd ctde_arena
python main.py
```

Carrega os checkpoints e mostra a arena 3D interativa (botão “Reiniciar Partida” incluído).

---

## Uso com Docker

```bash
cd ctde_arena
docker build -t ctde-arena .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena
```

O container executa `python scripts/train_rl.py` por padrão (headless) e inclui as dependências gráficas (OpenGL/X11) para eventual simulação.

---

## Resultados do treinamento

Valores consolidados em `data/metrics/summary.json` (treino versionado: 100.000 steps, 456 partidas).

| Equipe | Paradigma | Win rate | Elim./partida | Sobrevivência média (s) | Precisão de tiro |
|--------|-----------|---------:|--------------:|-------------------------:|-----------------:|
| Equipe 1 | CTDE-VD   | 40.67% | 2.50 | 10.20 | 11.29% |
| Equipe 2 | CTDE-CAC  | 42.89% | 2.63 | 9.83  | 12.33% |
| Equipe 3 | CTDE-Comm | 16.44% | 1.96 | 7.47  | 12.14% |

### Análise e diagnóstico

O comportamento após 100.000 passos reflete as características arquiteturais internas:

1. **Ator-Crítico Centralizado (CTDE-CAC / MAPPO)** — **42,89%**. O crítico centralizado tem acesso ao estado global completo de todos os 9 agentes, fornecendo estimativas de valor com baixa variância e sinais de vantagem precisos às políticas locais. Isso acelera fortemente o aprendizado nas fases iniciais.
2. **Value Decomposition (CTDE-VD / VDN)** — **40,67%**. A decomposição do valor conjunto em soma de valores locais ajuda na atribuição de crédito multi-agente e restringe a função de valor a observações locais, estabilizando o gradiente e reduzindo overfitting no início.
3. **Comunicação Explícita (CTDE-Comm / CommNet)** — **16,44%**. A política depende de $[o_i, c_i]$ (observação local + mensagens). No início as mensagens são ruído; a política precisa aprender simultaneamente a agir e a desenvolver um protocolo de comunicação. Esse “aprendizado duplo” exige mais steps (ex.: >500k–1M) para produzir mensagens úteis.

> Observação: o alvo padrão em `.env` é `RL_TRAIN_TOTAL_STEPS=3000000`; os resultados versionados correspondem a execuções de 100.000 steps.

---

## Configuração (.env)

Compartilha as mesmas variáveis da raiz. Documentação completa e todos os parâmetros (incluindo intervalos de *domain randomization*) em 

- [README raiz — Configuração](../README.md#configuração-env)

Resumo dos mais usados aqui:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `RL_TRAIN_TOTAL_STEPS` | 3.000.000 | Alvo total de steps de treino |
| `RL_SAVE_EVERY_STEPS` | 100.000 | Frequência de save/checkpoint |
| `RL_LOG_EVERY_STEPS` | 50.000 | Frequência de logging/MLflow |
| `RL_DEVICE` | cpu | `cpu`, `cuda` ou `mps` |
| `RANDOM_SEED` | 7 | Semente |

---

## Testes automatizados

```bash
cd ctde_arena
python -m pytest tests/ -q
```

`tests/test_components.py` valida a forma e sanidade das tensores de `ValueDecompositionCriticNetwork` e `CommActorNetwork`.

---

## Próximos passos

Como o objetivo principal é avaliar se CTDE-Comm ultrapassa os outros métodos com treino prolongado:

- Re-treinar com 3M steps (ou mais) no `.env` e reavaliar os win-rates.
- Rodar múltiplas seeds para estabilidade estatística.
- Registrar as estatísticas do PPO (policy/value loss, entropy) no MLflow.
