from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import time

TRAIN_REAL   = r'D:\TCCDataset\train\real'
TRAIN_FAKE   = r'D:\TCCDataset\train\fake'
TEST_REAL    = r'D:\TCCDataset\test\real'
TEST_FAKE    = r'D:\TCCDataset\test\fake'

EXTS         = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
RESIZE_TO    = (256, 256)
N_COMPONENTS = 50

FEATURE_NAMES = (
    [f'eigenvalue_ratio_{i+1}' for i in range(N_COMPONENTS)] +
    ['eigenvalue_entropy', 'spectral_flatness', 'first_pc_dominance', 'coef_variacao'] +
    ['reconstruction_error_k5', 'reconstruction_error_k10', 'reconstruction_error_k20']
)


def reconstruction_error(matrix, pca, scores, k):
    scores_k          = np.zeros_like(scores)
    scores_k[:, :k]   = scores[:, :k]
    reconstructed     = pca.inverse_transform(scores_k)
    return float(np.mean((matrix - reconstructed) ** 2))


def extract_pca_features(image_path):
    try:
        img    = Image.open(image_path).convert("L").resize(RESIZE_TO)
        matrix = np.array(img, dtype=np.float64) / 255.0

        n      = min(N_COMPONENTS, min(matrix.shape))
        pca    = PCA(n_components=n, svd_solver='randomized', random_state=42)
        scores = pca.fit_transform(matrix)

        evr         = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_

        evr_padded       = np.zeros(N_COMPONENTS)
        evr_padded[:n]   = evr

        probs              = eigenvalues / eigenvalues.sum()
        eigen_entropy      = float(-np.sum(probs * np.log(probs + 1e-12)))
        geo_mean           = float(np.exp(np.mean(np.log(eigenvalues + 1e-12))))
        spectral_flatness  = geo_mean / (float(np.mean(eigenvalues)) + 1e-12)
        first_pc_dominance = float(evr[0])
        coef_variacao      = float(np.std(eigenvalues) / (np.mean(eigenvalues) + 1e-12))

        rec_k5  = reconstruction_error(matrix, pca, scores, min(5,  n))
        rec_k10 = reconstruction_error(matrix, pca, scores, min(10, n))
        rec_k20 = reconstruction_error(matrix, pca, scores, min(20, n))

        return (
            list(evr_padded) +
            [eigen_entropy, spectral_flatness, first_pc_dominance, coef_variacao] +
            [rec_k5, rec_k10, rec_k20]
        )
    except Exception as e:
        print(f"Erro: {image_path} — {e}")
        return None


def processar_diretorio(directory, label):
    X, y  = [], []
    paths = [p for p in Path(directory).rglob('*') if p.suffix.lower() in EXTS]
    print(f"  {len(paths)} imagens em '{Path(directory).name}/'")

    start = time.time()
    for i, path in enumerate(paths):
        if i > 0 and i % 1000 == 0:
            elapsed  = time.time() - start
            restante = (elapsed / i) * (len(paths) - i)
            print(f"  {i}/{len(paths)} — ~{restante/60:.1f} min restantes")
        features = extract_pca_features(str(path))
        if features is not None:
            X.append(features)
            y.append(label)
    return X, y


if __name__ == '__main__':
    print("=== Extraindo features — TREINO ===")
    Xr, yr = processar_diretorio(TRAIN_REAL, label=0)
    Xf, yf = processar_diretorio(TRAIN_FAKE, label=1)
    X_train = np.array(Xr + Xf)
    y_train = np.array(yr + yf)
    print(f"Treino: {len(X_train)} imagens ({sum(y_train==0)} reais, {sum(y_train==1)} IA)\n")

    print("=== Extraindo features — TESTE ===")
    Xr_t, yr_t = processar_diretorio(TEST_REAL, label=0)
    Xf_t, yf_t = processar_diretorio(TEST_FAKE, label=1)
    X_test  = np.array(Xr_t + Xf_t)
    y_test  = np.array(yr_t + yf_t)
    print(f"Teste: {len(X_test)} imagens ({sum(y_test==0)} reais, {sum(y_test==1)} IA)\n")

    print("=== Treinando MLP ===")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=True
        ))
    ])
    pipeline.fit(X_train, y_train)
    print("Concluído.\n")

    print("=== Avaliação no conjunto de teste ===")
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=['Real', 'IA']))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    print("\n=== Salvando modelo ===")
    os.makedirs('modelospkl', exist_ok=True)
    joblib.dump(pipeline, 'modelospkl/modelo_pca.pkl')
    print("Modelo salvo: modelospkl/modelo_pca.pkl")
