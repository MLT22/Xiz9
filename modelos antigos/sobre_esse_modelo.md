# modelo_pca_rf.pkl — Random Forest (versão antiga)

Esse aqui é o primeiro modelo que a gente treinou. Ele usa **Random Forest** com **7 features** extraídas via PCA:

- variância acumulada nos top 5, 10, 20 e 50 componentes
- dominância do 1º componente
- entropia dos autovalores
- spectral flatness

Treinado com 48k imagens (24k reais + 24k de IA) do dataset do Kaggle.

## Por que foi aposentado?

Funcionou, mas ficou em **65% de acurácia** e **AUC-ROC 0.70**. A gente migrou pra uma MLP com 57 features (espectro completo de autovalores + erros de reconstrução), que chegou em **69% de acurácia** e **AUC-ROC 0.75**.

A diferença não é enorme, mas foi suficiente pra justificar a troca — e as features mais ricas dão um argumento melhor na monografia.

## Serve pra alguma coisa ainda?

Sim! Tá guardado aqui como **baseline de comparação**. Na hora de escrever o TCC dá pra mostrar a evolução do modelo e justificar por que a MLP foi a escolha final.
