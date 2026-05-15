# modelo_pca_v2.pkl — PCA v2 (XGBoost + Cor + FFT)

## Motivação

O modelo original (`modelo_pca.pkl`) ficou em **67.7% de acurácia** com bias severo para IA (acertava 80.6% das imagens IA mas apenas 54.8% das reais). A v2 foi retreinada com um conjunto de features mais rico e um classificador diferente para corrigir isso.

## O que mudou em relação ao original

| | PCA original | PCA v2 |
|---|---|---|
| Resolução de entrada | 256×256 | 128×128 |
| Componentes PCA | 50 | 30 |
| Features | 57 (só cinza) | 60 (cinza + cor + FFT) |
| Classificador | MLP (128-64-32) | XGBoost (400 árvores) |

## Features extraídas (60 no total)

**PCA sobre canal cinza (37 features):**
- 30 razões de variância explicada (eigenvalue ratios)
- Entropia dos autovalores
- Spectral flatness
- Dominância do 1º componente
- Coeficiente de variação dos autovalores
- Erros de reconstrução em k=5, 10, 20

**Cor por canal RGB (12 features):**
- Média, desvio padrão, skewness e kurtosis para R, G e B

**HSV (6 features):**
- Média e desvio padrão para H, S e V

**FFT — espectro de frequência (5 features):**
- Energia nas bandas baixa, média e alta frequência (normalizadas)
- Razão alta/baixa frequência
- Desvio padrão do log da magnitude

## Dataset de treino

- **Treino:** 48.000 imagens (24k reais + 24k IA) — `D:\TCCDataset\train`
- **Teste (validação early stopping):** 12.000 imagens (6k reais + 6k IA) — `D:\TCCDataset\test`

## Treinamento XGBoost

```
n_estimators=400, max_depth=6, learning_rate=0.05
subsample=0.8, colsample_bytree=0.8
early_stopping_rounds=20
```

Convergência da loss de validação:
```
[0]    logloss: 0.68342
[50]   logloss: 0.54716
[100]  logloss: 0.52365
[200]  logloss: 0.50282
[300]  logloss: 0.49256
[399]  logloss: 0.48707
```

## Resultado no conjunto de teste (12k imagens)

```
              precision    recall  f1-score
Real              0.78      0.73      0.76
IA                0.75      0.80      0.77
accuracy                              0.76
AUC-ROC: 0.8452
```

## Análise de bias (benchmark — 1000 imagens)

Testado com 500 imagens reais e 500 IA do dataset de teste.

| Modelo | Acc Real | Acc IA | Bias |
|---|---|---|---|
| PCA original (MLP) | 54.8% | 80.6% | IA (severo) |
| **PCA v2 (XGBoost)** | **82.0%** | **60.0%** | **Real (moderado)** |
| Luminescência (SVM) | 58.6% | 80.2% | IA |
| Ruído (SRM+DCT) | 78.0% | 73.0% | Real (leve) |

O v2 **inverteu o bias** do original. As features de cor e FFT puxaram o modelo para favorecer imagens reais, provavelmente porque imagens IA tendem a ter distribuição de cor e espectro de frequência mais "uniforme".

## Benchmark individual (1000 imagens)

| Modelo | Acurácia | Tempo médio |
|---|---|---|
| PCA original (MLP) | 67.70% | 84.0 ms |
| Luminescência (SVM) | 69.40% | 105.1 ms |
| PCA v2 (XGBoost+cor+FFT) | 71.00% | 115.5 ms |
| Ruído (SRM+DCT) | 75.50% | 65.2 ms |

## Ensemble (PCA v2 + Luminescência + Ruído)

Combinação por **média ponderada das probabilidades**, com pesos separados por classe baseados no bias de cada modelo — um modelo recebe mais peso quando prevê contra seu bias (sinal mais confiável):

| Modelo | Peso quando diz Real | Peso quando diz IA |
|---|---|---|
| PCA v2 (bias Real) | 0.600 | **0.820** |
| Luminescência (bias IA) | **0.802** | 0.586 |
| Ruído (bias Real) | 0.730 | **0.780** |

**Resultado do ensemble: 77.80% de acurácia — tempo médio: 293 ms/imagem**

O ensemble supera todos os modelos individuais em acurácia, aproveitando que a Luminescência (bias IA) equilibra o PCA v2 e o Ruído (ambos com bias Real).
