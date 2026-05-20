# RL Multi MARL Competition

Projeto completo de simulacao 3D com 9 agentes em formato de capsula, distribuidos em exatamente 3 equipes com 3 agentes cada, competindo em uma arena 3D com movimento, giro, pulo, disparo hit-kill instantaneo e coleta persistente de metricas.

## Equipes e Paradigmas

- `Equipe 1` usa `CTE (Centralized Training with Execution)`.
- `Equipe 2` usa `DTE (Decentralized Training and Execution)`.
- `Equipe 3` usa `CTDE (Centralized Training with Decentralized Execution)`.

## Stack

- `Python 3.13` testado localmente
- `Python 3.11+` suportado pelo projeto
- `PyTorch` para treino PPO das politicas MARL
- `Ursina` para visualizacao 3D em tempo real
- `NumPy` para calculos vetoriais
- `Matplotlib` para exportacao de graficos
- `python-dotenv` para configuracao via `.env`

## Estrutura

```text
.
|-- main.py
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- .env
|-- .env.example
|-- data/
|   |-- exports/
|   `-- metrics/
|-- scripts/
|   |-- train_rl.py
|   |-- headless_batch.py
|   |-- live_metrics_viewer.py
|   |-- test_projectile_collision.py
|   `-- validate_initial_behavior.py
|-- tests/
|   |-- test_rl_networks.py
|   `-- test_rl_training.py
`-- src/
    `-- marl_arena/
        |-- config.py
        |-- models.py
        |-- controllers/
        |   |-- base.py
        |   |-- factory.py
        |   |-- cte_controller.py
        |   |-- dte_controller.py
        |   |-- ctde_controller.py
        |   |-- cte_rl_controller.py
        |   |-- dte_rl_controller.py
        |   `-- ctde_rl_controller.py
        |-- rl/
        |   |-- networks.py
        |   |-- buffer.py
        |   |-- ppo.py
        |   |-- spaces.py
        |   `-- checkpoints.py
        |-- systems/
        |   |-- simulation.py
        |   |-- metrics.py
        |   |-- analytics.py
        |   `-- plotting.py
        `-- ui/
            `-- dashboard.py
```

## Como Instalar

### Opcao 1: ambiente local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Opcao 2: Docker

```bash
docker compose up --build
```

Observacao: o container foi preparado para execucao `headless`, voltada para simulacoes, persistencia de metricas e exportacao de graficos. A visualizacao 3D em janela deve ser executada localmente.

## Como Executar

### Visualizacao 3D em tempo real

```bash
python main.py
```

Esse script abre a arena 3D e mostra:

- os 9 agentes em formato de capsula
- as 3 equipes com cores distintas
- identificadores textuais por agente
- barreiras fixas, obstaculos moveis e zonas de passagem restrita
- competicao automatica em tempo real
- reinicio automatico de partidas
- exportacao de metricas e graficos ao fim de cada partida

### Treino RL real (PPO + PyTorch)

```bash
pip install -r requirements.txt
```

No `.env`, defina:

```env
CONTROLLER_MODE=rl
RL_TRAIN_MATCHES=300
RL_DEVICE=cpu
```

Execute o treino headless:

```bash
python scripts/train_rl.py
```

O treino:

- usa **PPO** com redes neurais por paradigma
- salva checkpoints em `data/checkpoints/` (`equipe_1_cte.pt`, `equipe_2_dte.pt`, `equipe_3_ctde.pt`)
- grava log resumido em `data/checkpoints/training_log.json`
- atualiza as politicas ao final de cada partida com rollouts coletados na simulacao

Paradigmas implementados com redes:

| Paradigma | Equipe | Arquitetura |
|-----------|--------|-------------|
| **CTE** | Equipe 1 | ator centralizado (estado global + slot do agente) + critico global |
| **DTE** | Equipe 2 | ator-critico local (somente observacao local na execucao) |
| **CTDE** | Equipe 3 | ator local na execucao + critico global no treino PPO |

Espaco de acao discreto: `8` acoes (`4` alvos taticos x `2` estados de disparo).

Para voltar ao modo heuristico legado:

```env
CONTROLLER_MODE=heuristic
```

### Simulacao headless em lote

```bash
python scripts/headless_batch.py
```

Esse script executa varias partidas sem abrir a janela 3D, ideal para gerar historico estatistico rapidamente. Com `CONTROLLER_MODE=rl`, carrega checkpoints treinados e roda em modo inferencia (sem gradiente).

### Visualizador de metricas em tempo real

```bash
python scripts/live_metrics_viewer.py
```

Esse script acompanha o arquivo de metricas e atualiza os graficos dinamicamente, exportando uma imagem consolidada em `data/exports/live_metrics_snapshot.png`.

### Validacao do comportamento inicial

```bash
python scripts/validate_initial_behavior.py
```

Esse script confirma, sem depender da janela 3D, que:

- os agentes possuem atividade inicial mesmo sem treinamento offline
- os obstaculos moveis realmente se deslocam
- nao ha sobreposicao invalida entre agentes vivos e obstaculos
- o estado "todos completamente parados" nao deve ser interpretado como comportamento normal da simulacao atual

## Controles da Camera

- use o mouse e navegacao da `EditorCamera` para orbitar, aproximar e inspecionar a arena
- o botao `Reiniciar Partida` permite reset manual da rodada atual

## Metricas Persistidas

Os dados sao gravados em:

- `data/metrics/team_match_metrics.csv`
- `data/metrics/agent_match_metrics.csv`
- `data/metrics/trajectory_metrics.csv`
- `data/metrics/summary.json`

As metricas incluem:

- taxa de eliminacoes por equipe
- tempo medio de sobrevivencia dos agentes
- quantidade de disparos acertados
- quantidade de disparos errados
- taxa de vitorias por equipe

## Obstaculos do Mapa

O mapa agora contem elementos variados e distribuidos para alterar a navegacao:

- `barreiras fixas` que quebram linhas retas longas e criam cobertura
- `obstaculos moveis` que varrem corredores horizontais e forçam replanejamento local
- `passagens restritas` que formam gargalos e corredores estreitos

Interacoes implementadas:

- colisao dos agentes com obstaculos sem atravessar a geometria
- deslizamento lateral ao tentar contornar barreiras
- reposicionamento de seguranca caso um obstaculo movel pressione um agente
- bloqueio fisico de projeteis por obstaculos fixos, moveis e paredes da arena
- deteccao continua de intersecao entre o segmento percorrido pelo projetil e as hitboxes 3D do cenario
- resolucao do primeiro impacto valido no trajeto do projetil, evitando falsos positivos por amostragem grosseira
- acerto em agentes baseado na trajetoria real do projetil, em vez de eliminacao instantanea desacoplada do disparo

## Graficos Exportados

Os graficos sao gerados automaticamente em:

- `data/exports/comparative_dashboard.png`
- `data/exports/live_metrics_snapshot.png`

O dashboard consolidado plota ao longo do tempo:

- taxa de vitorias
- eliminacoes acumuladas
- sobrevivencia media
- precisao de disparos

## Diferencas Entre as Equipes

### Modo heuristico (`CONTROLLER_MODE=heuristic`)

- **CTE**: pesos heuristicos com visao global na decisao
- **DTE**: pesos locais independentes por agente
- **CTDE**: ator heuristico local + ajuste de critico heuristico

### Modo RL (`CONTROLLER_MODE=rl`)

- **CTE**: `CentralizedActorNetwork` + `CentralizedCriticNetwork`, treinados com PPO usando estado global da arena
- **DTE**: `ActorNetwork` com observacao local de 8 dimensoes e valor local
- **CTDE**: ator local + critico global; o critico nao entra na execucao, apenas no bootstrap e no loss do PPO

## Configuracao

Edite `.env` para alterar parametros como:

- `CONTROLLER_MODE` (`heuristic` ou `rl`)
- duracao da partida
- velocidade de movimento
- velocidade de giro
- alcance de tiro
- cooldown de disparo
- quantidade de partidas headless
- hiperparametros PPO (`RL_LEARNING_RATE`, `RL_GAMMA`, `RL_PPO_EPOCHS`, ...)
- seed aleatoria

## Comportamento Inicial Sem Treinamento

Nesta versao, os agentes `nao` ficam necessariamente imoveis antes de treinamento, porque os controladores possuem heuristicas iniciais de decisao:

- `CTE` parte de pesos coordenados de perseguicao, flanqueamento e suporte
- `DTE` parte de pesos locais por agente
- `CTDE` parte de pesos iniciais de ator e critico

Resultado pratico:

- ja existe movimentacao automatica basica antes de qualquer ajuste online
- portanto, ver todos os agentes completamente parados no modo visual nao era o comportamento esperado da logica
- a validacao automatica mostrou deslocamento logo no primeiro passo e ao longo de multiplos passos sem treinamento

Resumo do teste executado:

- no primeiro passo sem treinamento, `6 de 9` agentes se deslocaram
- em `30` passos sem treinamento, `8 de 9` agentes acumularam deslocamento relevante
- `2` obstaculos moveis se deslocaram como esperado
- houve `0` sobreposicoes invalidas entre agentes vivos e obstaculos

### Validacao pratica da colisao de projeteis

```bash
python scripts/test_projectile_collision.py
```

Esse script executa cenarios deterministas e um smoke test headless para validar:

- impacto do projetil na parede central antes de atingir um alvo atras dela
- acerto direto em inimigo sem falso bloqueio
- interrupcao do projetil ao atingir o limite fisico da arena
- limpeza correta dos eventos de colisao a cada passo da simulacao
- permanencia dos projeteis dentro dos limites da arena em execucao continua

## Boas Praticas e DevOps

- configuracao externa via `.env`
- containerizacao com `Dockerfile`
- orquestracao simples com `docker-compose.yml`
- saidas persistidas em arquivos estruturados
- simulacao visual e headless sobre a mesma logica central

## Testes

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Limitacoes Conhecidas

- o modo Docker foca em execucao headless, nao em renderizacao 3D com janela
- o treino RL usa PPO tabular por partida (sem replay buffer persistente entre sessoes)
- convergencia depende de `RL_TRAIN_MATCHES` e do balanceamento do mapa; para resultados fortes, aumente partidas e ajuste hiperparametros

## Correcao da Colisao de Projetil

Problemas corrigidos nesta versao:

- remocao da antiga verificacao por amostragem que podia perder colisores finos ou detectar impactos em pontos incorretos
- eliminacao do erro de alcance por frame que verificava ate `10x` alem do deslocamento real do projetil
- alinhamento entre arena visual e arena logica, evitando divergencia entre parede renderizada e limite de colisao
- substituicao do acerto instantaneo por resolucao da colisao ao longo da trajetoria do projetil
- controle consistente do ciclo de vida dos eventos de impacto para nao acumular explosoes antigas em passos futuros
