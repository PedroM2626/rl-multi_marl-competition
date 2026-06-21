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

## Resultados do Treinamento (100.000 Passos)

Após realizar o treinamento completo de **100.000 passos** de simulação (totalizando 456 partidas), os resultados finais consolidados das três abordagens de CTDE foram:

| Equipe / Paradigma | Taxa de Vitória (Win Rate) | Eliminações por Partida | Tempo Médio de Sobrevivência | Precisão de Disparos (Accuracy) |
| :--- | :---: | :---: | :---: | :---: |
| **Equipe 1 (CTDE-VD)** | 40,67% | 2,50 | 10,20s | 11,29% |
| **Equipe 2 (CTDE-CAC)** | 42,89% | 2,63 | 9,83s | 12,33% |
| **Equipe 3 (CTDE-Comm)** | 16,44% | 1,96 | 7,47s | 12,14% |

### Análise e Diagnóstico de Desempenho

O comportamento observado de cada uma das abordagens após 100.000 passos reflete suas características arquiteturais internas:

1. **Ator-Crítico Centralizado (CTDE-CAC / MAPPO)**
   * **Desempenho**: Melhor taxa de vitória (**42,89%**) e consistência geral.
   * **Por quê?**: O crítico centralizado tem acesso ao estado global completo de todos os 9 agentes na arena (posição, ângulo, alive/dead). Por ter a visão completa do ambiente, ele fornece estimativas de valor com variância muito baixa e sinais de vantagem precisos para atualizar as políticas locais. Isso acelera significativamente o aprendizado nas fases iniciais do treinamento (100k passos).

2. **Decomposição de Valor (CTDE-VD / VDN para V-value)**
   * **Desempenho**: Altamente competitivo (**40,67%**), muito próximo do MAPPO.
   * **Por quê?**: A decomposição da função de valor conjunta como a soma dos valores locais individuais \(V_{tot} = \sum V_i(o_i)\) ajuda no problema de atribuição de crédito multi-agente (entender quais ações de qual agente levaram à recompensa conjunta do time), ao mesmo tempo em que restringe a função de valor para depender apenas das observações locais do agente. Essa simplificação estrutural estabiliza o gradiente de política e reduz o overfitting nas fases iniciais, atingindo resultados excelentes.

3. **Comunicação Explícita (CTDE-Comm / CommNet)**
   * **Desempenho**: Menor taxa de vitória (**16,44%**).
   * **Por quê?**: Em redes de comunicação, a política do agente passa a depender de um vetor concatenado da observação local e das mensagens recebidas das outras redes dos aliados: \([o_i, c_i]\). No início do treinamento, a rede geradora de mensagens envia vetores sem significado útil (ruído). A política do agente tem que aprender simultaneamente a se movimentar/atirar e a desenvolver um protocolo de mensagens coordenado com os aliados. Esse problema de aprendizado duplo (política + protocolo) gera uma barreira inicial, necessitando de muito mais passos de treinamento (ex: > 500k-1M passos) para que as mensagens se tornem úteis e superem os outros métodos.

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
