## Ensaios em Algoritmos de Machine Learning: Um Guia Detalhado

**Introdução**

A Data Money, empresa de consultoria em análise e ciência de dados, é reconhecida pelo alto retorno financeiro proporcionado aos seus clientes através de algoritmos de machine learning. Para aprimorar ainda mais a expertise da equipe, os cientistas de dados propõem a realização de ensaios em algoritmos de classificação, regressão e clusterização.

**Objetivo**

O objetivo principal deste projeto é realizar 3 ensaios abrangentes com algoritmos de machine learning, extraindo aprendizados sobre seu funcionamento em diferentes cenários e compartilhando esse conhecimento com o restante da equipe.

**Produto Final**

O produto final consistirá tabelas detalhadas que apresentam a performance dos algoritmos em diferentes conjuntos de dados: Treinamento, Validação e Teste. As tabelas serão acompanhadas por análises e insights sobre os resultados obtidos.

**Algoritmos Ensaiados**

* **Classificação:**
    * Algoritmos: KNN, Decision Tree, Random Forest e Logistic Regression
    * Métricas de Performance: Accuracy, Precision, Recall e F1-Score
* **Regressão:**
    * Algoritmos: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polynomial Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net, Polynomial Regression Lasso, Polynomial Regression Ridge e Polynomial Regression Elastic Net
    * Métricas de Performance: R2, MSE, RMSE, MAE e MAPE
* **Agrupamento:**
    * Algoritmos: K-Means e Affinity Propagation
    * Métricas de Performance: Silhouette Score

**Ferramentas Utilizadas**

* Python 3.8
* Scikit-learn

**Desenvolvimento**

**Estratégia da Solução**

Os ensaios serão realizados utilizando a linguagem Python e a biblioteca Scikit-learn. Para cada algoritmo, serão realizados os seguintes passos:

1. **Divisão dos Dados:** Os dados serão divididos em conjuntos de Treinamento, Validação e Teste.
2. **Treinamento com Parâmetros Default:** O algoritmo será treinado com os parâmetros padrão e sua performance será avaliada nos conjuntos de Treinamento e Validação.
3. **Ajuste Fino dos Parâmetros:** Os parâmetros do algoritmo serão ajustados iterativamente para encontrar a combinação que otimiza a performance no conjunto de Validação.
4. **Retreinamento com Parâmetros Ótimos:** O algoritmo será re-treinado com os parâmetros ótimos encontrados na etapa anterior e sua performance será avaliada no conjunto de Teste.
5. **Análise e Insights:** Os resultados dos ensaios serão analisados e os principais insights serão extraídos.

**Passo a Passo Detalhado**

**Passo 1: Divisão dos Dados em Treino, Teste e Validação**

Os dados serão divididos em três conjuntos: Treinamento (70%), Validação (15%) e Teste (15%). Essa divisão garante que o modelo seja treinado com uma quantidade significativa de dados, validado em um conjunto separado e avaliado em um conjunto final que não foi utilizado no treinamento ou validação.

**Passo 2: Treinamento dos Algoritmos com os Dados de Treinamento e Parâmetros Default**

Cada algoritmo de classificação, regressão e agrupamento será treinado utilizando os dados de treinamento e os parâmetros padrão da biblioteca Scikit-learn.

**Passo 3: Medição da Performance com Parâmetros Default**

A performance dos algoritmos treinados na etapa anterior será avaliada nos conjuntos de Treinamento e Validação, utilizando as métricas de performance específicas para cada tipo de algoritmo.

**Passo 4: Ajuste Fino dos Parâmetros**

Para cada algoritmo, os parâmetros que controlam o overfitting serão ajustados iterativamente. Para isso, serão utilizados métodos de busca de hiperparâmetros, como GridSearchCV ou RandomSearchCV, disponíveis na biblioteca Scikit-learn.

**Passo 5: Retreinamento com Parâmetros Ótimos**

O algoritmo com os parâmetros ótimos encontrados na etapa anterior será re-treinado utilizando os dados de Treinamento e Validação combinados.

**Passo 6: Medição da Performance com Parâmetros Ótimos**

A performance do algoritmo re-treinado com os parâmetros ótimos será avaliada no conjunto de Teste, utilizando as métricas de performance específicas para cada tipo de algoritmo.

**Passo 7: Avaliação dos Ensaios e Análise de Insights**

Os resultados dos ensaios serão analisados em conjunto para identificar os algoritmos com melhor performance em cada tipo de tarefa (classificação, regressão e agrupamento). Além disso, serão extraídos os 3 principais insights que se destacaram durante os ensaios, como os parâmetros que


## Análise de Dados de Performance

As tabelas fornecidas resumem as métricas de desempenho de vários modelos de regressão e classificação. Vamos analisar e explicar os dados em cada seção.

#### Algoritmos de Regressão em Dois Conjuntos de Validação Diferentes

**Tabela 1: Conjunto de Validação 1**
| Algorithm                        | R²     | MSE       | RMSE     | MAE      | MAPE      |
|----------------------------------|--------|-----------|----------|----------|-----------|
| Linear Regression                | 0.0523 | 461.428   | 21.481   | 17.1300  | 852.1859  |
| Decision Tree Regressor          | 0.0382 | 468.311   | 21.641   | 16.7906  | 707.0827  |
| Random Forest Regressor          | 0.2320 | 373.939   | 19.338   | 15.4075  | 729.1451  |
| Polynomial Regression            | 0.0901 | 443.041   | 21.049   | 16.7205  | 824.2536  |
| Lasso Regression                 | 0.0323 | 471.176   | 21.707   | 17.2746  | 861.4964  |
| Ridge Regression                 | 0.0523 | 461.429   | 21.481   | 17.1300  | 852.1883  |
| ElasticNet Regression            | 0.0323 | 471.168   | 21.706   | 17.2697  | 864.3283  |
| Polynomial Lasso Regression      | 0.0495 | 462.799   | 21.513   | 17.0930  | 849.8468  |
| Polynomial Ridge Regression      | 0.0901 | 443.042   | 21.049   | 16.7205  | 824.2636  |
| Polynomial ElasticNet Regression | 0.0561 | 459.581   | 21.438   | 17.0416  | 845.1359  |

**Tabela 2: Conjunto de Validação 2**
| Algorithm                        | R²     | MSE       | RMSE     | MAE      | MAPE      |
|----------------------------------|--------|-----------|----------|----------|-----------|
| Linear Regression                | 0.0461 | 455.996   | 21.354   | 16.9982  | 865.3186  |
| Decision Tree Regressor          | 0.3846 | 294.158   | 17.151   | 12.9312  | 486.6387  |
| Random Forest Regressor          | 0.4660 | 255.273   | 15.977   | 12.6981  | 577.2319  |
| Polynomial Regression            | 0.0942 | 432.986   | 20.808   | 16.4580  | 835.0611  |
| Lasso Regression                 | 0.0310 | 463.197   | 21.522   | 17.1327  | 869.8865  |
| Ridge Regression                 | 0.0461 | 455.996   | 21.354   | 16.9983  | 865.3191  |
| ElasticNet Regression            | 0.0309 | 463.245   | 21.523   | 17.1240  | 870.4490  |
| Polynomial Lasso Regression      | 0.0478 | 455.163   | 21.335   | 16.9455  | 860.2302  |
| Polynomial Ridge Regression      | 0.0942 | 432.986   | 20.808   | 16.4580  | 835.0567  |
| Polynomial ElasticNet Regression | 0.0545 | 451.956   | 21.259   | 16.8726  | 856.0976  |

### Principais Observações

1. **R² (Coeficiente de Determinação):**
   - **Validação 1:** O Random Forest Regressor possui o valor de R² mais alto (0,2320), indicando que ele explica a maior parte da variância. O Decision Tree Regressor possui o menor (0,0382).
   - **Validação 2:** O Random Forest Regressor novamente possui o valor de R² mais alto (0,4660), seguido pelo Decision Tree Regressor (0,3846). A Regressão Elástica possui o menor R² (0,0309).

2. **MSE (Erro Médio Quadrático):**
   - **Validação 1:** Random Forest Regressor possui o menor MSE (373,939), enquanto a Regressão Lasso possui o maior (471,176).
   - **Validação 2:** Random Forest Regressor possui o menor MSE (255,273), indicando melhor desempenho, enquanto a Regressão Elástica possui o maior (463,245).

3. **RMSE (Raiz do Erro Médio Quadrático):**
   - **Validação 1:** Random Forest Regressor novamente possui o menor RMSE (19,3375), sugerindo a menor magnitude de erro. A Regressão Lasso possui o maior RMSE (21,7066).
   - **Validação 2:** Random Forest Regressor possui o menor RMSE (15,977), indicando um melhor ajuste em comparação com outros, enquanto a Regressão Elástica possui o maior (21,523).

4. **MAE (Erro Absoluto Médio):**
   - **Validação 1:** Random Forest Regressor possui o menor MAE (15,4075), sugerindo o menor erro médio absoluto. A Regressão Lasso possui o maior MAE (17,2746).
   - **Validação 2:** Random Forest Regressor possui o menor MAE (12,6981), seguido pelo Decision Tree Regressor (12,9312). A Regressão Elástica possui o maior MAE (17,1240).

5. **MAPE (Erro Percentual Médio Absoluto):**
   - **Validação 1:** Random Forest Regressor possui o menor MAPE (729,1451). O Decision Tree Regressor possui o menor MAPE entre todos os modelos nas duas tabelas (707,0827). A Regressão Elástica possui o maior MAPE (864,3283).
   - **Validação 2:** Decision Tree Regressor possui o menor MAPE (486,6387), seguido pelo Random Forest Regressor (577,2319). A Regressão Lasso possui o maior MAPE (870,4490).

### Resumo

- **Melhor Algoritmo:** O Random Forest Regressor supera consistentemente outros modelos em ambos os conjuntos de validação, demonstrando o maior R² e menor MSE, RMSE, MAE e MAPE.
- **Pior Algoritmo:** A Regressão Lasso e a Regressão Elástica geralmente apresentam desempenho inferior com valores maiores de MSE, RMSE, MAE e MAPE.
- **Decision Tree Regressor:** Embora seu desempenho não seja tão alto quanto o Random Forest, ele se sai relativamente bem, especialmente no segundo conjunto de validação, onde atinge um bom equilíbrio de métricas.

### Desempenho de Modelos de Classificação

A tabela abaixo mostra as métricas de desempenho para os modelos Random Forest e Regressão Logística em diferentes conjuntos de dados.
| Metric      | Random Forest 1 | Random Forest 2     | Random Forest 3    | Logistic Regression 1 | Logistic Regression 2 | Logistic Regression 3 |
|-------------|-----------------|---------------------|--------------------|-----------------------|-----------------------|-----------------------|
| Accuracy    | 1.0             | 0.9646              | 0.9632             | 0.5666                | 0.5666                | 0.5611                |
| Precision   | 1.0             | 0.9648              | 0.9634             | 0.3210                | 0.3211                | 0.3148                |
| Recall      | 1.0             | 0.9646              | 0.9632             | 0.5666                | 0.5666                | 0.5611                |
| F1-Score    | 1.0             | 0.9645              | 0.9631             | 0.4099                | 0.4099                | 0.4033                |

Principais Observações
Random Forest:

Apresenta desempenho perfeito (Acurácia, Precisão, Revocação e F1-Score iguais a 1.0) no primeiro conjunto de dados.
Mantém alto desempenho no segundo e terceiro conjuntos de dados, com Acurácia em torno de 0.964 e F1-Score em torno de 0.964.
Regressão Logística:

Possui desempenho muito inferior ao Random Forest.
A acurácia fica em torno de 0.566 em todos os três conjuntos de dados, indicando que é significativamente menos eficaz.
Precisão, Revocação e F1-Score estão todos em torno de 0.32 e 0.41, respectivamente.
Resumo
Algoritmo de Melhor Desempenho: O Random Forest mostra consistentemente alto desempenho em todas as métricas e conjuntos de dados.
Algoritmo de Pior Desempenho: A Regressão Logística tem um desempenho ruim, com Acurácia, Precisão, Revocação e F1-Score muito menores em comparação com o Random Forest.

