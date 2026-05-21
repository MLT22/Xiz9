import sys
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import joblib

MODEL_PATH   = 'modelospkl/modelo_pca.pkl'
RESIZE_TO    = (256, 256)
N_COMPONENTS = 50
THRESHOLD    = 0.80


def reconstruction_error(matrix, pca, scores, k):
    scores_k        = np.zeros_like(scores)
    scores_k[:, :k] = scores[:, :k]
    return float(np.mean((matrix - pca.inverse_transform(scores_k)) ** 2))


def extract_pca_features(image_path):
    img    = Image.open(image_path).convert("L").resize(RESIZE_TO)
    matrix = np.array(img, dtype=np.float64) / 255.0

    n      = min(N_COMPONENTS, min(matrix.shape))
    pca    = PCA(n_components=n, svd_solver='randomized', random_state=42)
    scores = pca.fit_transform(matrix)

    evr         = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    evr_padded     = np.zeros(N_COMPONENTS)
    evr_padded[:n] = evr

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


def testar(image_path):
    print(f"\nImagem: {image_path}")

    features   = extract_pca_features(image_path)
    model      = joblib.load(MODEL_PATH)
    prediction = int(model.predict([features])[0])
    proba      = model.predict_proba([features])[0]
    confianca  = max(proba)
    label      = "IA" if prediction == 1 else "REAL"
    status     = "CONCLUSIVO" if confianca >= THRESHOLD else "INCERTO — passar para próximo check"

    print(f"Resultado:    {label}")
    print(f"Confiança:    {confianca*100:.2f}%")
    print(f"Status:       {status}")
    print(f"Prob. Real:   {proba[0]*100:.2f}%")
    print(f"Prob. IA:     {proba[1]*100:.2f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python test_model.py <imagem1> <imagem2> ...")
        print("Exemplo: python test_model.py real.jpg fake.jpg")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            testar(path)
        except Exception as e:
            print(f"\nErro ao processar '{path}': {e}")
        print("-" * 40)
