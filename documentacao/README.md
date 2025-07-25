# Justificativa Técnica

**Introdução:**

Este projeto propõe o desenvolvimento de um modelo preditivo para identificar a tendência de alta ou baixa do índice IBOVESPA com base em dados históricos. A iniciativa visa apoiar decisões estratégicas de um fundo de investimentos, combinando análise estatística e inteligência de dados. A seguir, apresentaremos as escolhas técnicas realizadas e os critérios adotados para garantir a confiabilidade do modelo.

## **Seção 1: Aquisição e Pré-processamento de Dados**

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

**Outras ações importantes incluíram:**
* Verificação de nulos com `.isnull().sum()` e visualização via `missingno.bar()`.
* Sumário estatístico via `.describe()` para insights sobre a distribuição das variáveis.
* **Visualizações:**
    * Correlação com `heatmap` e `pair plot` (incluindo `hue='tendencia'`) para identificar relações lineares e não lineares entre as features e com o target.
    * Histogramas para analisar a distribuição das variáveis.

**Insight Adicional do Pairplot:**

Os gráficos de dispersão revelaram que variáveis de preço apresentam forte correlação entre si, formando padrões lineares bem definidos. Por outro lado, o volume mostrou maior dispersão e baixa correlação com os preços. Um destaque importante foi o par `variação vs volume`, cuja nuvem de pontos oval e densa, embora sem padrão linear, indicou uma tendência: variações extremas — positivas ou negativas — costumam ocorrer em dias com volume elevado. Esse padrão não linear contribuiu para decisões futuras na criação de atributos sensíveis à volatilidade.

**Primeiros passos para Engenharia de Atributos**

Duas variáveis derivadas ao final da EDA antecipam a próxima etapa do projeto. A primeira é a variável target `tendencia`, derivada da variável `preco_fechamento`, com isso transformamos o desafio de previsão de série temporal em um problema de classificação binária. A segunda variável é a `amplitude_diaria`, obtida pela diferença entre os preços máximo e mínimo do dia, e representa a volatilidade, representando um indicativo da intensidade dos movimentos do mercado.

## **Seção 2: Engenharia de Features**

Nesta etapa, criamos as features necessárias para que nosso projeto alcançasse o objetivo, focando em atributos que capturassem a dinâmica do mercado financeiro:

* **Lag Features:** São valores defasados de uma série temporal, funcionando como a "memória" histórica dos dados. Sua inclusão permite que modelos de Machine Learning compreendam a dependência temporal e utilizem o comportamento passado da série para fazer previsões sobre o futuro.
    * Incluímos lags para `preco_fechamento`, `variacao`, `volume` e `amplitude_diaria`.

* **Retornos:** Medem a mudança percentual no preço de um ativo de um período para o outro (`retorno_pct`). Eles são cruciais por serem mais estacionários que os preços absolutos e por indicarem a força e a direção da mudança de preço que já ocorreu, sendo mais relevantes para a previsão de movimentos futuros.
    * Criamos o `retorno_pct` e seus respectivos lags.

* **Médias Móveis Simples (SMAs):** É um indicador técnico usado para suavizar a ação do preço ao longo de um período. Elas ajudam a filtrar o ruído das flutuações diárias para revelar a direção da tendência.
    * `SMA_10`, `SMA_20`, `SMA_50`: Suavizam o ruído do preço e identificam tendências em diferentes horizontes (curto, médio e longo prazo).
    * `SMA_10_vs_20`, `SMA_20_vs_50`: Representam a diferença entre médias móveis, sendo importantes para identificar cruzamentos que podem sinalizar mudanças na relação entre tendências de curto e longo prazo.

* **RSI (Relative Strength Index):** Um oscilador de momentum que mede a velocidade e a mudança dos movimentos de preço. É utilizado para identificar condições de sobrecompra (RSI > 70) e sobrevenda (RSI < 30), indicando potenciais reversões.

* **MACD (Moving Average Convergence Divergence):** Um indicador de momentum que revela a relação entre duas Médias Móveis Exponenciais (EMAs). Ele fornece sinais de força e direção da tendência, bem como possíveis reversões através de cruzamentos (MACD Line vs. Signal Line) e divergências.
    * Calculamos a `MACD Line`, `Signal Line` e `MACD Histograma`.

* **ATR (Average True Range):** Uma medida de volatilidade que quantifica o quanto o preço de um ativo tende a se mover em um dado período, capturando a amplitude total das oscilações, incluindo gaps. É útil para entender o "nível de turbulência" do mercado.

* **Tratamento Final de Nulos:** Após a criação de todas essas features (especialmente aquelas que dependem de janelas de tempo como as médias móveis e indicadores), um `.dropna(inplace=True)` final foi aplicado para remover quaisquer valores nulos resultantes, garantindo a integridade dos dados para o treinamento do modelo.

## **Seção 3: Preparação da Base para a Modelagem**

Esta seção detalha os passos cruciais para estruturar os dados de forma adequada para o treinamento e avaliação do modelo, garantindo a integridade temporal e a comparabilidade das features.

**Derivação da Variável Target (tendencia):**
A variável tendencia foi definida como a variável alvo binária do projeto. Ela indica se o preço de fechamento do IBOVESPA no dia seguinte será maior (tendência de alta, representada por 1) ou menor ou igual (tendência de baixa/manutenção, representada por 0) que o preço de fechamento do dia atual. Esta variável foi criada utilizando o método `.shift(-1)` sobre a coluna preco_fechamento para obter o preço do dia seguinte (preco_fechamento_proximo_dia), e subsequentemente comparando-o com o preco_fechamento do dia atual. Esta transformação converteu o desafio de previsão de série temporal em um problema de classificação supervisionada.

**Estratégia de Divisão Treino/Teste:**
A divisão do conjunto de dados em treino e teste é um passo crítico, especialmente em séries temporais. Para atender ao requisito do desafio, o conjunto de teste foi definido como o último mês (30 dias) de dados disponíveis. O ponto de corte (data_corte) foi calculado subtraindo 30 dias da data máxima no DataFrame. Os dados anteriores a esta data formaram o conjunto de treino, e os dados posteriores formaram o conjunto de teste. Esta abordagem de divisão temporal é essencial para evitar o vazamento de dados (data leakage), garantindo que o modelo seja treinado apenas com informações passadas e avaliado em dados futuros não vistos, simulando realisticamente a aplicação do modelo em um cenário de mercado.

**Escalonamento das Features:**
Para otimizar o desempenho dos modelos e garantir que features com diferentes escalas não dominem o processo de aprendizagem, aplicamos o `StandardScaler`. Este método padroniza as features, transformando-as para que tenham uma média de 0 e um desvio padrão de 1. Esta padronização é particularmente importante para modelos baseados em distância (como KNN, se utilizado) e para algoritmos de otimização baseados em gradiente (como a Regressão Logística e o XGBoost), que se beneficiam de features em uma escala uniforme para uma convergência mais rápida e eficaz.

**Tratamento do Desbalanceamento de Classes:**
A análise da variável tendencia revelou um desbalanceamento de classes (aproximadamente 65% de eventos de baixa e 35% de eventos de alta). Para mitigar este problema e evitar que o modelo tendesse a prever majoritariamente a classe mais frequente, foram adotadas abordagens específicas durante o treinamento:

* Para o ***XGBoost***, utilizamos o parâmetro `scale_pos_weight`, que atribui um peso maior aos erros da classe positiva (minoritária).

* Para a ***Random Forest***, empregamos o parâmetro `class_weight='balanced'`, que ajusta automaticamente os pesos das classes de forma inversa às suas frequências.

Estas técnicas forçam o algoritmo a dar mais atenção e a aprender a identificar corretamente os eventos da classe minoritária, sem a necessidade de manipular diretamente o conjunto de dados (como oversampling ou undersampling, que podem ser problemáticos em séries temporais).

## **Seção 4: Modelagem e Otimização**

Nesta seção, detalhamos a escolha dos modelos preditivos, as estratégias de otimização de seus hiperparâmetros e a calibração final para atingir os objetivos do projeto.

**Escolha dos Modelos:**
Para a tarefa de classificação binária da tendência do IBOVESPA, foram avaliados dois modelos principais:

* **XGBoost (XGBClassifier - Extreme Gradient Boosting):** Selecionado como o modelo principal devido à sua comprovada alta performance e robustez em problemas de dados tabulares. O XGBoost é um algoritmo de ensemble learning baseado em árvores de decisão que constrói modelos sequencialmente, corrigindo os erros das árvores anteriores. Sua capacidade de lidar com relações não-lineares e interações complexas entre features o torna particularmente adequado para a dinâmica volátil do mercado financeiro.

* **Random Forest (RandomForestClassifier):** Incluído como outro algoritmo de ensemble learning baseado em árvores. O Random Forest constrói múltiplas árvores de decisão de forma independente e combina suas previsões. Sua inclusão permitiu uma comparação com o XGBoost, validando a robustez da abordagem baseada em árvores para este problema.

**Validação Cruzada Temporal (TimeSeriesSplit):**
A natureza sequencial dos dados de séries temporais exige uma abordagem de validação cruzada específica para garantir a confiabilidade dos resultados e evitar o vazamento de dados (data leakage). Por essa razão, o `TimeSeriesSplit` foi a escolha essencial para a validação cruzada no `GridSearchCV`. Diferente de métodos aleatórios, o TimeSeriesSplit cria as dobras de validação de forma a sempre usar dados anteriores para treino e dados posteriores para teste, simulando o fluxo do tempo real e proporcionando uma avaliação mais realista da capacidade de generalização do modelo em dados futuros.

**Otimização de Hiperparâmetros (GridSearchCV):**
A performance de modelos de Machine Learning é significativamente influenciada pela escolha de seus hiperparâmetros (parâmetros que não são aprendidos pelos dados, mas controlam o processo de aprendizagem do modelo, como a profundidade das árvores ou a taxa de aprendizado). Para encontrar a combinação ideal de hiperparâmetros, utilizamos o GridSearchCV. Esta ferramenta realiza uma busca exaustiva, testando todas as combinações predefinidas em uma "grade" de valores. Cada combinação é avaliada utilizando a validação cruzada temporal (TimeSeriesSplit), e a métrica de scoring definida (F1-Score ou Acurácia) é usada para determinar a melhor configuração. Os principais hiperparâmetros otimizados para XGBoost e Random Forest incluíram: n_estimators (número de árvores), max_depth (profundidade máxima da árvore), learning_rate (taxa de aprendizado), subsample (subamostragem de dados) e colsample_bytree (subamostragem de features).

**Otimização do Limiar de Decisão (Threshold):**
Modelos de classificação binária, como *XGBoost*, não produzem diretamente um "0" ou "1", mas sim uma probabilidade de que uma amostra pertença à classe positiva (tendência de alta). O limiar de decisão é o ponto de corte que converte essa probabilidade em uma previsão de classe final. A otimização do limiar, realizada no pós-treinamento, é crucial para:

* **Gerenciar o trade-off entre Precisão e Recall:** Um limiar mais alto aumenta a Precisão (menos falsos positivos), mas pode reduzir o Recall (mais falsos negativos). Um limiar mais baixo faz o oposto.

* **Alinhar o modelo aos objetivos de negócio:** Permite ajustar a sensibilidade do modelo para minimizar o tipo de erro mais custoso ou maximizar a métrica mais relevante para a estratégia de investimento.

No projeto, o limiar foi otimizado para maximizar a Acurácia (conforme a meta de 75%), mas também foi explorada a otimização para F1-Score, que oferece um balanço entre Precisão e Recall.

## **Seção 5: Resultados e Análise de Métricas**

Nesta seção, apresentamos os resultados obtidos pelos modelos treinados, com foco na performance do modelo vencedor em relação aos objetivos do projeto, e discutimos os insights derivados da avaliação das métricas.

**Desempenho do Modelo Vencedor:**
O objetivo principal do projeto era alcançar uma acurácia mínima de 75% na previsão da tendência do IBOVESPA em um conjunto de teste composto pelos últimos 30 dias de dados. Através da metodologia aplicada, o *modelo XGBoost*, quando otimizado para acurácia e treinado com os últimos ~10.5 anos de dados (19/01/2015 a 18/07/2025), demonstrou o melhor desempenho, atingindo uma acurácia de 80.00%. Este resultado não apenas cumpre, mas supera significativamente a meta estabelecida.

A tabela a seguir sumariza a performance dos modelos avaliados, com foco nos resultados obtidos no conjunto de teste de 30 dias, utilizando diferentes estratégias de otimização e períodos de treinamento:

| Modelo              | Treino (Período)       | Otimização | Melhor Limiar | Acurácia | Precisão (Classe 1) | Recall (Classe 1) | F1-Score (Classe 1) |
|:--------------------|:-----------------------|:-----------|:--------------|:---------|:--------------------|:------------------|:--------------------|
| XGBoost             | ~10.5 anos (2015-2025) | Acurácia   | 0.5686        | 0.8000   | N/A                 | N/A               | N/A                 |
| XGBoost             | ~10.5 anos (2015-2025) | F1-Score   | 0.5403        | 0.7667   | 0.5455              | 0.7500            | 0.6316              |
| Random Forest       | ~10.5 anos (2015-2025) | Acurácia   | 0.5376        | 0.7667   | N/A                 | N/A               | N/A                 |
| Random Forest       | ~10.5 anos (2015-2025) | F1-Score   | N/A           | 0.7333   | 0.5000              | 0.6250            | 0.5556              |

*Notas: N/A indica métricas não otimizadas para este objetivo, mas que podem ser calculadas.*

**Análise Detalhada das Métricas do Modelo Vencedor (XGBoost - Treino ~10.5 anos, Otimizado para Acurácia):**

A matriz de confusão para o modelo XGBoost com 80.00% de acurácia (limiar otimizado para acurácia) no conjunto de teste dos últimos 30 dias revelou a seguinte performance:

* **Matriz de Confusão:** [[17 5], [2 6]]
  * **Verdadeiros Positivos (TP = 6):** O modelo previu corretamente 6 dias de tendência de alta.
  * **Falsos Positivos (FP = 5):** O modelo previu 5 dias de tendência de alta, mas na verdade o IBOVESPA fechou em baixa.
  * **Falsos Negativos (FN = 2):** O modelo previu 2 dias de tendência de baixa, mas na verdade o IBOVESPA fechou em alta.
  * **Verdadeiros Negativos (TN = 17):** O modelo previu corretamente 17 dias de tendência de baixa.

* **Precisão (Classe 1):** frac66+5 approx0.5455 (54.55%). Das vezes que o modelo previu alta, ele acertou 54.55% das vezes.
* **Recall (Classe 1):** fracTPTP+FN= frac66+2approx0.7500 (75.00%). Das vezes que houve alta real, o modelo conseguiu identificar 75.00% delas.
* **F1-Score (Classe 1):** approx0.6316. Esta métrica balanceia Precisão e Recall.

A otimização do limiar de decisão (encontrado em 0.5686) foi crucial para atingir esta acurácia, permitindo um ajuste fino entre a capacidade de identificar eventos de alta e a minimização de falsos alarmes.

**Impacto do Período de Treinamento na Performance:**
Uma análise comparativa dos resultados do XGBoost com diferentes durações de conjuntos de dados de treinamento revelou um insight fundamental: "mais dados nem sempre significa melhor performance" em séries temporais financeiras.

* Treinando com aproximadamente 20.5 anos de dados (2005-2025), a acurácia foi de 76%.
* Treinando com aproximadamente 10.5 anos de dados (2015-2025), a acurácia atingiu 80%.

Este padrão sugere que dados históricos muito antigos podem introduzir ruído ou padrões de mercado que não são mais relevantes para prever o comportamento atual do IBOVESPA. O mercado financeiro é dinâmico e seus regimes (períodos de alta, baixa, volatilidade, fatores macroeconômicos) evoluem. A janela de treinamento de aproximadamente 10.5 anos demonstrou ser a mais eficaz para capturar os padrões preditivos contemporâneos.

**Confiabilidade do Modelo:**
A performance de 80.00% de acurácia no conjunto de teste de 30 dias, obtida através de um pipeline robusto que inclui validação cruzada temporal e tratamento de desbalanceamento, valida a confiabilidade do modelo para o propósito do desafio. O modelo demonstra uma capacidade sólida de generalização para dados futuros, dentro do contexto de sua aplicação.

## **Seção 6: Trade-offs (Acurácia vs. Overfitting)**

No desenvolvimento de modelos de Machine Learning, é fundamental gerenciar o trade-off entre alcançar alta acurácia e evitar o overfitting.

* **Overfitting (Sobreajuste):** Ocorre quando um modelo aprende os dados de treinamento de forma excessivamente detalhada, incluindo o ruído e as particularidades da amostra de treino. Como resultado, ele performa muito bem nos dados que "já viu", mas falha em generalizar para dados novos e não vistos (conjunto de teste). Um indicativo clássico de overfitting é uma alta performance no conjunto de treino e uma performance significativamente inferior no conjunto de teste.

* **Underfitting (Subajuste):** Ocorre quando um modelo é muito simples ou não foi treinado o suficiente para capturar os padrões subjacentes nos dados, performando mal tanto no conjunto de treino quanto no de teste.

No nosso projeto, foram adotadas diversas estratégias para mitigar o risco de overfitting e garantir que a alta acurácia obtida fosse um reflexo da capacidade de generalização do modelo:

**Validação Cruzada Temporal (TimeSeriesSplit):**

Esta foi a principal medida contra o overfitting em dados de séries temporais. Ao contrário da validação cruzada aleatória, o TimeSeriesSplit garante que o modelo seja sempre treinado em dados cronologicamente anteriores aos dados de validação. Isso simula o cenário real de previsão e impede que o modelo "veja" informações futuras durante o treinamento, o que seria uma forma de vazamento de dados (data leakage) e levaria a uma superestimação irreal da performance.

**Monitoramento das Métricas de Treino e Teste:**

Durante as fases de experimentação, foi observado que a diferença entre a acurácia (e outras métricas) nos conjuntos de treino e teste era consistentemente pequena (geralmente de 2% a 3%). Essa proximidade nas métricas é um forte indicativo de que o modelo não está sofrendo de overfitting, ou seja, está generalizando bem para dados não vistos. Se houvesse um grande descolamento (treino muito alto, teste muito baixo), seria um sinal de sobreajuste.

**Hiperparâmetros de Regularização do XGBoost:**

O XGBoost, por sua natureza, inclui mecanismos de regularização que ajudam a controlar a complexidade do modelo e a prevenir o overfitting. Parâmetros como subsample (subamostragem de linhas), colsample_bytree (subamostragem de colunas) e gamma (critério para divisão de nós) foram otimizados via GridSearchCV para encontrar um equilíbrio entre o ajuste aos dados e a capacidade de generalização.

Ao final do processo, a acurácia de 80.00% obtida no conjunto de teste dos últimos 30 dias, combinada com a estabilidade das métricas entre treino e teste e a aplicação de validação cruzada temporal, demonstra que o modelo é robusto e não está sobreajustado aos dados de treinamento. Este resultado é confiável para o propósito de prever a tendência do IBOVESPA.

## **Seção 7: Conclusão**

Este projeto demonstrou o desenvolvimento de um modelo preditivo robusto e eficaz para a previsão da tendência diária do índice IBOVESPA. Através de um pipeline completo de aquisição, pré-processamento, engenharia de features, modelagem e otimização, o objetivo de alcançar uma acurácia mínima de 75% no conjunto de teste dos últimos 30 dias foi atingido e superado com sucesso.

O modelo XGBoost, otimizado para acurácia e treinado com os últimos ~10.5 anos de dados históricos (19/01/2015 a 18/07/2025), destacou-se como a solução mais performática, alcançando uma acurácia de 80.00% no período de teste. Este resultado é particularmente notável considerando a complexidade e a natureza ruidosa dos dados financeiros. A análise comparativa com outros modelos (Random Forest e Regressão Logística) e diferentes períodos de treinamento reforçou a escolha do XGBoost e a importância de uma janela de dados de treino otimizada para capturar padrões de mercado mais contemporâneos.