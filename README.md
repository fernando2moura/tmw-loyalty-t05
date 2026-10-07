# Projeto de Fidelidade Téo Me Why

# 🎯 TMW Loyalty — Predição de Ativação de Clientes

Projeto de Ciência de Dados end-to-end para prever a **probabilidade de ativação de clientes** na plataforma TMW (Trampar de Casa / TMW Education), utilizando dados de transações do programa de loyalty e de consumo de cursos.

O objetivo é identificar, ao final de cada safra mensal, quais clientes possuem maior propensão a se ativarem nos próximos 28 dias, permitindo ações de marketing direcionadas.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura do Projeto](#-arquitetura-do-projeto)
- [Estrutura de Pastas](#-estrutura-de-pastas)
- [Pré-requisitos](#-pré-requisitos)
- [Passo a Passo](#-passo-a-passo)
  - [1. TMW Discovery](#1-tmw-discovery)
  - [2. Explorar TMW Loyalty](#2-explorar-tmw-loyalty)
  - [3. Análise de Usuários](#3-análise-de-usuários)
  - [4. Features](#4-features)
  - [5. fs_cursos](#5-fs_cursos)
  - [6. fs_pontos](#6-fs_pontos)
  - [7. ABT](#7-abt)
  - [8. Train](#8-train)
  - [9. Train (Meu)](#9-train-meu)
  - [10. Predict](#10-predict)
- [Métricas do Modelo](#-métricas-do-modelo)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

---

## 🔍 Visão Geral

O projeto segue o fluxo clássico de um pipeline de Machine Learning:

> **Descoberta → Análise Exploratória → Feature Engineering → ABT → Treino → Predição**

**Definição do problema:**

- **Target:** `flAtivacao` — 1 se o cliente realizar alguma transação nos próximos 28 dias, 0 caso contrário.
- **Safra (dtRef):** Mensal (ex: 2025-01-01, 2025-02-01, ... 2026-06-01).
- **Base OOT (Out-of-Time):** Última safra disponível (2026-06-01).

---

## 🏗 Arquitetura do Projeto

| Camada | Descrição |
|--------|-----------|
| **Fonte de Dados** | `workspace.tmw_loyalty` (transações, clientes, produtos) e `workspace.tmw_education` (cursos, episódios) |
| **Feature Store** | Tabelas Delta em `workspace.analytics`: `fs_pontos` e `fs_cursos` |
| **ABT** | Tabela `workspace.analytics.abt_ativacao` |
| **Modelo** | MLflow Experiment ID `4259626519550072`, registrado como `workspace.analytics.modelo_asn_tmw` |
| **Predição** | Notebook `predict` gera ranking dos top 50 clientes com maior probabilidade de ativação |

---

## 📁 Estrutura de Pastas

```text
tmw-loyalty/
├── 1. TMW Discovery.ipynb            # Análise exploratória inicial
├── 2. explorar_tmw_loyalty.ipynb     # Exploração detalhada das tabelas
├── 3. Analise Usuarios.ipynb         # Clusterização de usuários (KMeans)
├── 4. Features.ipynb                 # Orquestração da criação das feature stores
├── 5. fs_cursos.sql                  # Query da feature store de cursos
├── 6. fs_pontos.sql                  # Query da feature store de pontos
├── 7. ABT.ipynb                      # Construção da Analytical Base Table
├── 8. train.ipynb                    # Treino baseline (XGBoost + Random Forest)
├── 9. (Clone) train_meu.ipynb        # Treino com comparação de múltiplos modelos
└── 10. predict.ipynb                 # Carregamento do modelo e predição


## ⚙️ Pré-requisitos

- **Databricks** (Runtime 14+ com Unity Catalog)
- **Python 3.10+**

**Bibliotecas:**

```bash
pip install feature_engine xgboost scikit-learn mlflow pandas numpy matplotlib
```

**Acesso às tabelas:**

- `workspace.tmw_loyalty.transacoes`
- `workspace.tmw_loyalty.transacao_produto`
- `workspace.tmw_loyalty.produtos`
- `workspace.tmw_loyalty.clientes`
- `workspace.tmw_education.usuarios_tmw`
- `workspace.tmw_education.cursos_episodios`
- `workspace.tmw_education.cursos_episodios_completos`

**Permissão de escrita** em `workspace.analytics`.

---

### 📋 Detalhamento das Tabelas

| Schema | Tabela | Descrição |
|--------|--------|-----------|
| `workspace.tmw_loyalty` | `transacoes` | Histórico de transações do programa de loyalty |
| `workspace.tmw_loyalty` | `transacao_produto` | Itens/produtos vinculados a cada transação |
| `workspace.tmw_loyalty` | `produtos` | Cadastro de produtos (nome, ID, descrição) |
| `workspace.tmw_loyalty` | `clientes` | Cadastro de clientes e flags de engajamento |
| `workspace.tmw_education` | `usuarios_tmw` | Mapeamento usuário ↔ cliente TMW |
| `workspace.tmw_education` | `cursos_episodios` | Cadastro de episódios por curso |
| `workspace.tmw_education` | `cursos_episodios_completos` | Episódios concluídos por usuário |


## 🚀 Passo a Passo

## 1. TMW Discovery

**Arquivo:** `1. TMW Discovery.ipynb`

Notebook inicial de **descoberta da base**. Contém as primeiras consultas SQL para entender o volume e o comportamento dos dados.

**Principais análises:**

**a) Histórico de transações diárias:**

```sql
SELECT date(DtCriacao) AS dtDia,
       count(*) AS qtdeTransacoes,
       count(DISTINCT idCliente) AS qtdeCliente
FROM workspace.tmw_loyalty.transacoes
GROUP BY ALL
ORDER BY dtDia;
```
**b) Histórico mensal** (com `date_trunc('MONTH', DtCriacao)`).

**c) Cálculo de métricas de engajamento:**

- **DAU** (Daily Active Users) — clientes únicos por dia.
- **WAU** (Weekly Active Users) — janela de 7 dias.
- **MAU** (Monthly Active Users) — janela de 28 dias.
- **ARPU** — pontos positivos / clientes únicos na janela de 28 dias.
- **Churn** — clientes que estavam no MAU e não retornaram 28 dias depois.

**d) Curvas analíticas:**

- **Curva de recorrência** — média de dias entre transações consecutivas por cliente.
- **Curva de sobrevivência** — `1 - (qtdeAcum / max(qtdeAcum))`.

> 💡 Este notebook é o "caderno de rascunho" do projeto: serve para validar hipóteses antes de virar feature.


## 2. Explorar TMW Loyalty

**Arquivo:** 2. explorar_tmw_loyalty.ipynb

Exploração mais objetiva, respondendo perguntas de negócio:

| Pergunta | Query |
| :--- | :--- |
| Quantos usuários existem no loyalty? | SELECT count(DISTINCT idCliente) FROM clientes |
| Histórico diário / mensal / anual de transações | date_trunc + count(DISTINCT IdTransacao) |
| Quantos novos usuários em 2026? | Filtro year(DtCriacao) = 2026 |
| Qual produto mais transacionado? | Join transacao_produto + produtos |
| Quais produtos têm maiores valores acumulados? | sum(qtdeProduto) por produto |
| Quando temos picos de transação? | Ordenação por count(DISTINCT IdCliente) DESC |
| Pico de novos usuários | Mesma lógica em clientes |

**Reuso:** as queries de DAU/WAU/MAU/ARPU foram refinadas aqui e depois replicadas no notebook 1.


## 3. Análise de Usuários

**Arquivo:** 3. Analise Usuarios.ipynb

Primeira tentativa de segmentação de clientes com técnicas não supervisionadas.

**Etapas:**

**1) Foto da base em 2026-06-01 — últimos 28 dias de transações:**

```sql
SELECT IdCliente,
       count(DISTINCT date(DtCriacao)) AS frequencia,
       count(*) AS frequenciaTransacoes,
       sum(CASE WHEN QtdePontos > 0 THEN qtdePontos ELSE 0 END) AS valor,
       min(date_diff('2026-06-01', DtCriacao)) AS recencia
FROM workspace.tmw_loyalty.transacoes
WHERE DtCriacao < '2026-06-01'
  AND DtCriacao >= '2026-06-01' - INTERVAL 28 DAYS
GROUP BY ALL;
```

**2) Clusterização KMeans** (`n_clusters=8`) com features `frequencia` e `valor` normalizadas por `MinMaxScaler`.

**3) Z-Score** dos pontos acumulados positivos para identificar outliers.

> ⚠️ **Observação:** este notebook é exploratório. A clusterização não entrou no pipeline final de produção — serviu para entender a distribuição dos clientes.



## 4. Features

**Arquivo:** 4. Features.ipynb

Orquestrador da criação das feature stores. Executa as queries `.sql` para cada safra mensal e persiste em tabelas Delta.

**Lógica:**

```python
import pandas as pd

dates = [i.strftime("%Y-%m-%d") for i in pd.date_range('2025-01-01', '2026-06-01', freq='MS')]
feature_name = 'fs_cursos'  # ou 'fs_pontos'

with open(f"{feature_name}.sql") as f:
    query = f.read()

for i in dates:
    df = spark.sql(query.format(date=i))
    (df.write
       .format("delta")
       .mode("overwrite")
       .option("replaceWhere", f"dtRef = '{i}'")
       .saveAsTable(f"workspace.analytics.{feature_name}"))
```

**Safras geradas:**

| Feature Store | Período |
| :--- | :--- |
| `fs_cursos` | 2025-01-01 até 2026-06-01 |
| `fs_pontos` | 2025-01-01 até 2026-06-01 |
| **Produção** | 2026-06-01 e 2026-07-01 |



## 5. fs_cursos

**Arquivo:** 5. fs_cursos.sql

Feature store com informações de **consumo de cursos** na plataforma TMW Education.

**Principais features geradas:**

| Feature | Descrição |
| :--- | :--- |
| `qtdCursosIniciados` | Quantidade de cursos distintos iniciados |
| `qtdCursosFinalizados` | Cursos com 100% dos episódios assistidos |
| `python_2025`, `mlflow_2025`, `sql_2025`, ... | % de conclusão por curso específico |
| `avgTempoInicioUltimo` | Média de dias entre o início e o último episódio |
| `avgTempoInicioFim` | Média de dias entre início e conclusão (cursos finalizados) |

**Lógica central:**

```sql
WITH tb_usuario_curso AS (
    SELECT t1.idTMWCliente,
           t2.descSlugCurso,
           min(dtCriacao) AS dtInicioCurso,
           max(dtCriacao) AS dtUltimoEp,
           date_diff(max(dtCriacao), min(dtCriacao)) AS diasEntrePrimeiroUltimo,
           count(*) AS qtEpCurso
    FROM workspace.tmw_education.usuarios_tmw AS t1
    LEFT JOIN workspace.tmw_education.cursos_episodios_completos AS t2
      ON t1.idUsuario = t2.idUsuario
    WHERE t2.dtCriacao < '{date}'
    GROUP BY ALL
)
-- ... cálculo de pctCursoCompleto e pivot por curso

--Filtro temporal: WHERE t2.dtCriacao < '{date}' — garante que só entram dados anteriores à safra (evita data leakage).
```


## 6. fs_pontos

**Arquivo:** 6. fs_pontos.sql

Feature store com informações de transações de loyalty (pontos).

**Janela:** 28 dias anteriores à safra (`dtRef`).

**Principais features:**

| Feature | Descrição |
| :--- | :--- |
| `qtFrequencia` | Dias distintos com transação |
| `qtPontos` | Soma total de pontos |
| `qtPontosPositivos` | Soma apenas de pontos positivos |
| `recencia` | Dias desde a última transação |
| `QtdeTransacoes` | Total de transações |
| `zScore` | Z-score dos pontos positivos |
| `pctTransacaoChatMessage` | % de transações de chat |
| `pctTransacaoListaPresenca` | % de transações de lista de presença |
| `pctTransacaoPresencaStreak` | % de transações de streak |
| `pctTransacaoResgatarPonei` | % de resgates de ponei |
| `pctTransacaoTrocaPontosStreamElements` | % de trocas de pontos |
| `flStreak` | Flag se participou de streak |
| `DiasUltimoStreak` | Dias desde o último streak |
| `qtdeProdutoDistintos` | Quantidade de produtos distintos |
| `saldoDia` | Saldo total de pontos na vida |
| `diasPrimeiraTransacao` | Dias desde a primeira transação |
| `freqVida` | Frequência total na vida |


## 7. ABT

**Arquivo:** 7. ABT.ipynb

Construção da **Analytical Base Table** — união de todas as features + target.

**Estrutura da ABT:**

```sql
WITH tb_ativacao AS (
    SELECT DISTINCT IdCliente, date(DtCriacao) AS dtAtivacao
    FROM workspace.tmw_loyalty.transacoes
),
tb_resposta AS (
    SELECT t1.IdCliente,
           t1.dtRef,
           count(t2.dtAtivacao) AS qtdAtivacoes,
           max(CASE WHEN t2.dtAtivacao IS NOT NULL THEN 1 ELSE 0 END) AS flAtivacao
    FROM workspace.analytics.fs_pontos AS t1
    LEFT JOIN tb_ativacao AS t2
      ON t1.IdCliente = t2.idcliente
     AND t1.dtRef <= t2.dtAtivacao
     AND t2.dtAtivacao <= (t1.dtRef + INTERVAL 28 DAYS)
    GROUP BY ALL
)
-- ... join com fs_pontos e fs_cursos
```
Target: flAtivacao = 1 se houve alguma transação nos 28 dias seguintes à safra.

Amostragem:

* Safras <= 2026-05-01: 1 registro por cliente (amostra aleatória).

* Safras > 2026-05-01: mantidas todas (para produção).

Tabela final: workspace.analytics.abt_ativacao



## 8. Train

**Arquivo:** 8. train.ipynb

Treino **baseline** com foco em XGBoost e Random Forest + tracking no MLflow.

**Pipeline:**

```python
from sklearn import pipeline, ensemble, linear_model
from xgboost import XGBClassifier

clf = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_estimators=200,
    max_depth=5
)

model_pipeline = pipeline.Pipeline(steps=[
    ('imputacao_0', imput_0),        # ArbitraryNumberImputer(0)
    ('imput_max', imput_max_tail),   # EndTailImputer(max, fold=1)
    ('classificador', clf)
])

model_pipeline.fit(X_train, y_train)
```

**Tratamento de missing:**

| Tipo de feature | Estratégia |
| :--- | :--- |
| Features de cursos (não fez o curso) | ArbitraryNumberImputer(0) |
| `DiasUltimoStreak`, `avgTempoInicioUltimo`, `avgTempoInicioFim` | EndTailImputer(max, fold=1) |

**Split:**

*   Treino: 80% | Teste: 20% (estratificado)
*   OOT: última safra (`dtRef.max()`)

**Métricas registradas no MLflow:**

*   `auc_train`, `auc_test`, `auc_oot`
*   `accuracy_test`, `precision_test`, `recall_test`, `f1_test`
*   `threshold` (baseado na proporção de positivos)

**Registro do modelo:** `workspace.analytics.modelo_asn_tmw`

**Curvas geradas:**

*   ROC (Teste e OOT)
*   Matriz de Confusão (Teste e OOT)

## 9. Train (Meu)

**Arquivo:** 9. (Clone) train_meu.ipynb

Versão aprimorada do treino, com:

*   **Comparação de 5 modelos** via `GridSearchCV`:
    1.  Logistic Regression
    2.  Decision Tree
    3.  SVM (RBF)
    4.  Random Forest
    5.  KNN
*   **RandomizedSearchCV** específico para Random Forest (20 iterações).
*   **Balanceamento** via `class_weight='balanced'` quando aplicável.
*   **Threshold** calculado pela proporção de positivos.

**Tabela comparativa de resultados:**

| Modelo | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest | 0.7843 | 0.7958 | 0.7077 | 0.4000 | 0.5111 | **0.7574** |
| Logistic Regression | 0.7735 | 0.7842 | 0.5932 | 0.6087 | 0.6009 | 0.7488 |
| Decision Tree | 0.7608 | 0.7935 | 0.7407 | 0.3478 | 0.4734 | 0.7266 |
| SVM | 0.7202 | 0.7703 | 0.7353 | 0.2174 | 0.3356 | 0.7155 |
| KNN | 0.7200 | 0.7517 | 0.5769 | 0.2609 | 0.3593 | 0.6919 |

🏆 **Melhor modelo:** Random Forest (`max_depth=10`, `n_estimators=200`)


## 10. Predict

**Arquivo:** 10. predict.ipynb

Notebook de **inferência em produção**. Carrega o modelo registrado e gera o ranking dos clientes mais propensos a se ativarem.

**Etapas:**

**1) Carrega o modelo:**

```python
import mlflow
model = mlflow.sklearn.load_model("models:/workspace.analytics.modelo_asn_tmw/2")
```
**2) Busca a safra mais recente:**

```sql
SELECT max(dtRef) FROM workspace.analytics.fs_pontos;
```
**3) Monta o dataset de inferência:**

```python
df = spark.sql("""
    SELECT * EXCEPT (t2.dtRef, t2.idCliente)
    FROM workspace.analytics.fs_pontos AS t1
    LEFT JOIN workspace.analytics.fs_cursos AS t2
      ON t1.IdCliente = t2.idCliente
     AND t1.dtRef = t2.dtRef
    WHERE t1.dtRef = (SELECT max(dtRef) FROM workspace.analytics.fs_pontos)
""").toPandas()
```

**4) Seleciona apenas as features do modelo:**

```python
X = df[model.feature_names_in_]
```

**5) Gera probabilidades e ranking:**

```python
predict = model.predict_proba(X)[:, 0]
df['proba'] = predict
df_list_50 = df[['IdCliente', 'proba']].head(50).sort_values('proba', ascending=False)
```
📤 Output: lista dos top 50 clientes com maior probabilidade de ativação, pronta para ação de marketing.



## 📊 **Métricas do Modelo**

**Modelo campeão:** Random Forest

| Métrica | Valor |
| :--- | :--- |
| CV ROC-AUC | 0.784 |
| Test Accuracy | 0.796 |
| Test ROC-AUC | 0.757 |
| Test Precision | 0.708 |
| Test Recall | 0.400 |
| Test F1 | 0.511 |


## 🛠️ **Tecnologias Utilizadas**

| Categoria | Ferramentas |
| :--- | :--- |
| **Plataforma** | Databricks, Unity Catalog, Delta Lake |
| **Linguagens** | Python 3.10+, SQL (Spark SQL) |
| **Manipulação** | Pandas, NumPy, PySpark |
| **Machine Learning** | Scikit-learn, XGBoost, Feature-engine |
| **Experiment Tracking** | MLflow (Experiment ID `4259626519550072`) |
| **Visualização** | Matplotlib |
| **Model Registry** | Databricks Model Registry |
