# Previsão de Inadimplência com Machine Learning

## 1. Problema

A inadimplência de clientes representa um risco financeiro relevante para instituições de crédito. Antecipar esse comportamento permite reduzir perdas, ajustar políticas de concessão e priorizar estratégias de cobrança.

Este projeto tem como objetivo prever a probabilidade de um cliente se tornar inadimplente em até 2 anos com base em informações cadastrais e comportamentais.

## 2. Justificativa

Antecipar inadimplência possibilita:

- Redução de perdas financeiras
- Melhoria na gestão de risco
- Estratégias proativas de cobrança
- Definição mais assertiva de limites de crédito

## 3. Fonte de Dados

O dataset utilizado foi retirado da base pública **"Give Me Some Credit"**, disponível originalmente no [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit).

O arquivo utilizado foi:

- `cs-training.csv`

Este arquivo contém cerca de 150.000 registros históricos de clientes, incluindo informações sobre idade, renda, histórico de dívidas, entre outros.

## 4. Dicionário de Variáveis

| Coluna                   | Descrição                                               |
|--------------------------|----------------------------------------------------------|
| `SeriousDlqin2yrs`       | 1 se houve inadimplência nos últimos 2 anos, 0 caso contrário |
| `RevolvingUtilizationOfUnsecuredLines` | Percentual de uso do crédito rotativo |
| `age`                    | Idade do cliente                                        |
| `NumberOfTime30-59DaysPastDueNotWorse` | Quantidade de atrasos de 30 a 59 dias |
| `DebtRatio`              | Relação entre dívidas e renda                           |
| `MonthlyIncome`          | Renda mensal declarada                                  |
| `NumberOfOpenCreditLinesAndLoans` | Número de linhas de crédito abertas            |
| `NumberOfTimes90DaysLate` | Atrasos de 90 dias ou mais                             |
| `NumberRealEstateLoansOrLines` | Financiamentos ou linhas de crédito imobiliário  |
| `NumberOfTime60-89DaysPastDueNotWorse` | Atrasos de 60 a 89 dias               |
| `NumberOfDependents`     | Número de dependentes                                   |

## 5. Etapas do Projeto

### 5.1. Carregamento dos Dados

```python
df = pd.read_csv("cs-training.csv", index_col=0)
````

### 5.2. Análise Exploratória (EDA)

* Verificação de valores ausentes
* Estatísticas descritivas
* Visualização da distribuição de idade, renda e target
* Proporção de inadimplentes: cerca de 6.7%

### 5.3. Tratamento de Dados

* `MonthlyIncome`: preenchido com a mediana
* `NumberOfDependents`: preenchido com zero

### 5.4. Modelagem

Modelo: Regressão Logística

* Divisão em treino e teste
* Ajuste do modelo com `LogisticRegression(max_iter=1000)`
* Avaliação com:

  * AUC (área sob a curva ROC)
  * Matriz de confusão
  * Curva ROC
  * Curva precisão-revocação
  * `classification_report` (precisão, recall, f1-score)

### 5.5. Resultados

* AUC: \~0.85
* O modelo é capaz de distinguir inadimplentes com desempenho satisfatório.
* As curvas indicam bom equilíbrio entre sensibilidade e especificidade.

## 6. Visualizações

As principais visualizações geradas foram:

* Distribuição de idade (`histplot`)
* Boxplot da renda mensal
* Curva ROC
* Curva de precisão vs. revocação
* Matriz de confusão

Essas visualizações estão disponíveis no notebook do projeto.

## 7. Conclusão

O modelo desenvolvido demonstrou capacidade de prever inadimplência com métricas robustas. A Regressão Logística mostrou-se adequada para uma primeira versão, com boa interpretabilidade.

Próximos passos possíveis:

* Testar modelos não lineares (Random Forest, XGBoost)
* Realizar tuning de hiperparâmetros
* Implementar balanceamento (oversampling/undersampling)
* Implantar o modelo em ambiente de teste ou produção
