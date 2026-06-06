# RL Multi MARL Competition

Arena 3D com 9 agentes (3 equipes × 3). Cada equipe foi treinada com **PPO + PyTorch** adotando paradigmas MARL distintos.

Equipe e paradigmas

| Equipe   | Paradigma | Arquitetura (resumo) |
|----------|-----------|----------------------|
| Equipe 1 | CTE       | Ator centralizado + crítico global |
| Equipe 2 | DTE       | Actor-Critic local |
| Equipe 3 | CTDE      | Ator local + crítico global (treino centralizado) |

Pontos de entrada

| Comando | Função |
|---------|--------|
| `python scripts/train_rl.py` | Treino (padrão: 3M steps) |
| `python main.py` | Executa arena 3D (visual) usando checkpoints |

Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e ajuste variáveis conforme necessário.

Treinar (exemplo mínimo)

```bash
python scripts/train_rl.py
```

Arquivos de saída importantes

- `data/checkpoints/` — checkpoints por equipe
- `data/checkpoints/training_log.json` — histórico de treino
- `data/metrics/summary.json` — resumo de métricas de avaliação (usado neste README)

Testes

```bash
python -m pytest tests/ -q
```

Estrutura do repositório (resumo)

```text
main.py
scripts/train_rl.py
src/marl_arena/
  config.py
  models.py
  controllers/
  rl/
  systems/
  ui/
tests/
data/
```

Configuração principal (`.env`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `RL_TRAIN_TOTAL_STEPS` | 3000000 | Total de steps de treino |
| `RL_SAVE_EVERY_STEPS` | 100000 | Frequência de salvamento |
| `DOMAIN_RANDOMIZATION` | true | Randomização de domínio durante treino |

Resultados do experimento

Os resultados abaixo foram extraídos de `data/metrics/summary.json`.

| Equipe   | Paradigma | Win rate | Eliminações/partida | Tempo médio sobrevivência (s) | Precisão de tiro |
|----------|-----------:|---------:|--------------------:|------------------------------:|----------------:|
| Equipe 1 | CTE   | 27.39% | 1.72  | 9.63  | 8.10%  |
| Equipe 2 | DTE   | 25.43% | 1.84  | 8.59  | 8.13%  |
| Equipe 3 | CTDE  | 47.17% | 3.42  | 9.93  | 12.84% |

Interpretação rápida

- `Equipe 3 (CTDE)` obteve o melhor desempenho geral: maior win rate, mais eliminações por partida e melhor precisão de tiro.
- `Equipe 1` e `Equipe 2` têm desempenho parecido, com win rates ~25–27% e baixa precisão (~8%).

Reproduzir/inspecionar métricas

- O arquivo fonte das métricas está em `data/metrics/summary.json`.
- Para gerar gráficos rápidos, veja `src/marl_arena/systems/plotting.py` (se disponível) ou exporte para CSV.

Próximos passos sugeridos

- Gerar gráficos comparativos (win rate, eliminações, precisão).
- Rodar avaliações com seeds diferentes para validar estabilidade dos resultados.

Contato

Abra uma issue ou PR neste repositório para discutir experimentos, dúvidas ou melhorias.

