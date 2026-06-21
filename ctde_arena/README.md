# CTDE Paradigm Comparison Arena

Este projeto é uma extensão auto-contida para comparar três categorias de Aprendizado por Reforço Multi-Agente (MARL) sob a arquitetura **CTDE (Centralized Training with Decentralized Execution)**:

1. **Value Decomposition (CTDE-VD)**: Decompõe a função de valor do time na soma dos valores locais individuais estimadas por observações parciais de cada agente.
   \[V_{tot}(s) = \sum_{i=1}^{3} V_i(o_i)\]
2. **Centralized Actor-Critic (CTDE-CAC)**: Método padrão de Ator-Crítico Centralizado (estilo MAPPO), onde os atores tomam ações baseadas em observações locais, e o crítico estima o valor do estado global.
3. **Explicit Communication (CTDE-Comm)**: Atores descentralizados trocam mensagens diferenciáveis durante a execução (estilo CommNet), e utilizam um crítico centralizado durante o treinamento.

Cada paradigma é representado por uma equipe na simulação do jogo.

---

## Estrutura do Projeto

* `src/marl_arena/`: Módulos de lógica do jogo, modelos RL, redes neurais e controle de simulação.
  * `rl/networks.py`: Contém as novas redes `ValueDecompositionCriticNetwork` e `CommActorNetwork`.
  * `rl/ppo.py`: Contém as implementações de atualização de PPO específicas de cada paradigma.
  * `controllers/rl_controller.py`: Contém a inicialização e decisões de cada equipe.
* `scripts/`:
  * `train_rl.py`: Script principal de treinamento integrado com o **MLflow**.
* `tests/`:
  * `test_components.py`: Testes unitários para validar os modelos e fluxo de tensores das redes.
* `main.py`: Executável com a interface visual em 3D usando Ursina Engine.
* `requirements.txt`: Lista de dependências com versões exatas do ambiente.
* `Dockerfile`: Configuração para empacotar o projeto em containers Docker.

---

## Treinamento e MLOps com MLflow

O treinamento é monitorado usando **MLflow** para registrar hiperparâmetros, logs de experimentos, métricas em tempo real por equipe, gráficos de performance e registrar os modelos treinados de cada equipe.

Para rodar o treinamento:
```bash
python scripts/train_rl.py
```

### MLflow UI
Para visualizar as runs do experimento, win rate das equipes, gráficos e os modelos registrados no Model Registry:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

---

## Como Executar a Simulação Visual (UI)

Para rodar a simulação com a interface 3D interativa da arena (requer ambiente com display gráfico):
```bash
python main.py
```

---

## Uso com Docker

Para construir a imagem Docker do ambiente e rodar o treinamento de forma isolada:

1. Construir a imagem:
   ```bash
   docker build -t ctde-arena .
   ```
2. Rodar o container (headless training):
   ```bash
   docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena
   ```

---

## Testes Automatizados

Para executar os testes unitários de sanidade dos modelos e tensores:
```bash
pytest tests/
```
