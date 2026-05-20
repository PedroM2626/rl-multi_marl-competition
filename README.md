# RL Multi MARL Competition

Arena 3D com 9 agentes (3 equipes × 3), cada equipe com um paradigma MARL treinado via **PPO + PyTorch**:

| Equipe | Paradigma | Rede |
|--------|-----------|------|
| Equipe 1 | CTE | Ator centralizado + crítico global |
| Equipe 2 | DTE | Actor-Critic local |
| Equipe 3 | CTDE | Ator local + crítico global (só no treino) |

## Pontos de entrada

| Comando | Função |
|---------|--------|
| `python scripts/train_rl.py` | Treino até **3M steps** (padrão) com randomização de domínio |
| `python main.py` | Arena 3D com layout padrão e checkpoints carregados |

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copie `.env.example` para `.env`.

## Treinar (3 milhões de steps)

```bash
python scripts/train_rl.py
```

O treino para quando `total_env_steps >= RL_TRAIN_TOTAL_STEPS` (padrão: **3.000.000**).

A cada nova partida, com `DOMAIN_RANDOMIZATION=true`, o ambiente sorteia:

- **Spawns** das 3 equipes (posições e pequenos deslocamentos)
- **Obstáculos** (5–10: barreiras, passagens, blocos móveis com eixo/amplitude/velocidade aleatórios)
- **Tamanho da arena** (28–36)
- **Duração da partida** (60–120 s)
- **Velocidade de movimento e giro**
- **Alcance e cooldown de disparo**

Isso torna as políticas mais robustas a layouts e regras variáveis.

Saídas:

- `data/checkpoints/equipe_*_{cte,dte,ctde}.pt` (salvos a cada `RL_SAVE_EVERY_STEPS`, padrão 100k)
- `data/checkpoints/training_log.json`
- Métricas/gráfico a cada `RL_METRICS_EVERY_MATCHES` partidas

## Jogar (visual 3D)

```bash
python main.py
```

Usa **layout fixo** (`DOMAIN_RANDOMIZATION=false`) para visualização estável.

## Testes

```bash
python -m pytest tests/ -q
```

## Estrutura

```text
main.py
scripts/train_rl.py
src/marl_arena/
  config.py
  models.py
  controllers/base.py, rl_controller.py
  rl/actions.py, networks.py, buffer.py, ppo.py
  systems/simulation.py, match_variant.py, metrics.py, plotting.py
  ui/dashboard.py
tests/
```

## Configuração principal (`.env`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `RL_TRAIN_TOTAL_STEPS` | 3000000 | Total de steps de ambiente no treino |
| `RL_SAVE_EVERY_STEPS` | 100000 | Salvar checkpoints a cada N steps |
| `RL_LOG_EVERY_STEPS` | 50000 | Log de progresso no console |
| `DOMAIN_RANDOMIZATION` | true | Variar mapa/regras a cada partida (treino) |
| `SIM_STEP_DT` | 0.1 | Delta de tempo por step |
| `DR_*` | ver `.env.example` | Intervalos da randomização de domínio |

## Estimativa de tempo

Com `SIM_STEP_DT=0.1` e partidas de ~60–120 s, 3M steps ≈ **3.000–5.000 partidas** (~horas em CPU). Use `RL_DEVICE=cuda` se tiver GPU.
