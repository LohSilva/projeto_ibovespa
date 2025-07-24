# Justificativa Técnica

**Introdução:**

Este projeto propõe o desenvolvimento de um modelo preditivo para identificar a tendência de alta ou baixa do índice IBOVESPA com base em dados históricos. A iniciativa visa apoiar decisões estratégicas de um fundo de investimentos, combinando análise estatística e inteligência de dados. A seguir, apresentaremos as escolhas técnicas realizadas e os critérios adotados para garantir a confiabilidade do modelo.

**Seção 1: Aquisição e Pré-processamento de Dados:**

A análise exploratória (EDA) é essencial para compreender e preparar o conjunto de dados antes do treinamento. Nesta etapa, aplicamos diversas ações para garantir a consistência e qualidade dos dados:

* **Data:**
    * Conversão para formato `datetime` com `pd.to_datetime()`.
    * Definição como índice via `.set_index()`.
    * Preenchimento de lacunas (fins de semana/feriados) com `.asfreq('D').ffill()`, garantindo a continuidade da série temporal.
    * Ordenação cronológica usando `.sort_values()`.

* **Variação:**
    * Substituição de ',' por '.' e remoção de '%' com `.str.replace()`.
    * Conversão para `float` via `pd.to_numeric()`.

* **Volume:**
    * Criação de função personalizada para tratar e remover sufixos como 'K', 'M' ou 'B'.
    * Conversão para formato numérico adequado (`int64`).

Outras ações importantes incluíram:
* Verificação de nulos com `.isnull().sum()` e visualização via `missingno.bar()`.
* Sumário estatístico via `.describe()` para insights sobre a distribuição das variáveis.
* **Visualizações:**
    * Correlação com `heatmap` e `pair plot` (incluindo `hue='tendencia'`) para identificar relações lineares e não lineares entre as features e com o target.
    * Histogramas para analisar a distribuição das variáveis.

**Insight Adicional do Pairplot:**

Os gráficos de dispersão revelaram que variáveis de preço apresentam forte correlação entre si, formando padrões lineares bem definidos. Por outro lado, o volume mostrou maior dispersão e baixa correlação com os preços. Um destaque importante foi o par `variação vs volume`, cuja nuvem de pontos oval e densa, embora sem padrão linear, indicou uma tendência: variações extremas — positivas ou negativas — costumam ocorrer em dias com volume elevado. Esse padrão não linear contribuiu para decisões futuras na criação de atributos sensíveis à volatilidade.

**Insight adicional do pairplot**

Os gráficos de dispersão revelaram que variáveis de preço apresentam forte correlação entre si, formando padrões lineares bem definidos. Por outro lado, o volume mostrou maior dispersão e baixa correlação com os preços. Um destaque importante foi o par variação vs volume, cuja nuvem de pontos oval e densa, embora sem padrão linear, indicou uma tendência: variações extremas — positivas ou negativas — costumam ocorrer em dia com volume elevado. Esse padrão não linear contribuiu para decisões futuras na criação de atributos sensíveis à volatilidade.

**Primeiros passos para Engenharia de Atributos**

Duas variáveis derivadas ao final da EDA antecipam a próxima etapa do projeto. A primeira é a variável target tendencia, criada a partir do preço de fechamento, com isso transformamos o desafio de previsão de série temporal em um problema de classificação binária. A segunda variável é a amplitude_diaria, obtida pela diferença entre os preços máximo e mínimo do dia, que representa a volatilidade, representando um indicativo da intensidade dos movimentos do mercado.

**Seção 2: Engenharia de Features**

Nesta etapa, criamos as features necessárias para que nosso projeto alcançasse o objetivo, focando em atributos que capturassem a dinâmica do mercado financeiro:

* **Lag Features:** São valores defasados de uma série temporal, funcionando como a "memória" histórica dos dados. Sua inclusão permite que modelos de Machine Learning compreendam a dependência temporal e utilizem o comportamento passado da série para fazer previsões sobre o futuro.
    * Incluímos lags para `preco_fechamento`, `variacao`, `volume` e `amplitude_diaria`.

* **Retornos:** Medem a mudança percentual no preço de um ativo de um período para o outro (`retorno_pct`). Eles são cruciais por serem mais estacionários que os preços absolutos e por indicarem a força e a direção da mudança de preço que já ocorreu, sendo mais relevantes para a previsão de movimentos futuros.
    * Criamos o `retorno_pct` e seus respectivos lags.

* **Médias Móveis Simples (SMAs) e Diferenças:**
    * `SMA_10`, `SMA_20`, `SMA_50`: Suavizam o ruído do preço e identificam tendências em diferentes horizontes (curto, médio e longo prazo).
    * `SMA_10_vs_20`, `SMA_20_vs_50`: Representam a diferença entre médias móveis, sendo importantes para identificar cruzamentos que podem sinalizar mudanças na relação entre tendências de curto e longo prazo.

* **RSI (Relative Strength Index):** Um oscilador de momentum que mede a velocidade e a mudança dos movimentos de preço. É utilizado para identificar condições de sobrecompra (RSI > 70) e sobrevenda (RSI < 30), indicando potenciais reversões.

* **MACD (Moving Average Convergence Divergence):** Um indicador de momentum que revela a relação entre duas Médias Móveis Exponenciais (EMAs). Ele fornece sinais de força e direção da tendência, bem como possíveis reversões através de cruzamentos (MACD Line vs. Signal Line) e divergências.
    * Calculamos a `MACD Line`, `Signal Line` e `MACD Histograma`.

* **ATR (Average True Range):** Uma medida de volatilidade que quantifica o quanto o preço de um ativo tende a se mover em um dado período, capturando a amplitude total das oscilações, incluindo gaps. É útil para entender o "nível de turbulência" do mercado.

* **Tratamento Final de Nulos:** Após a criação de todas essas features (especialmente aquelas que dependem de janelas de tempo como as médias móveis e indicadores), um `.dropna(inplace=True)` final foi aplicado para remover quaisquer valores nulos resultantes, garantindo a integridade dos dados para o treinamento do modelo.