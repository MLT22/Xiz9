import os
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from scipy.ndimage import convolve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

TRAIN_REAL = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\train\real'
TRAIN_FAKE = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\train\fake'
TEST_REAL  = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\test\real'
TEST_FAKE  = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\test\fake'

EXTS       = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
RESIZE_TO  = (256, 256)
PATCH_SIZE = 32
N_PATCHES  = 3  # top N patches de alta e baixa frequência

# Filtros SRM (extração de ruído)
SRM_FILTERS = [
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -2, 1], [0, 1, 0]], dtype=np.float32),
]


def apply_srm(patch: np.ndarray) -> np.ndarray:
    """Aplica filtros SRM no patch e retorna o padrão de ruído."""
    results = []
    for f in SRM_FILTERS:
        filtered = convolve(patch.astype(np.float32), f)
        results.append(float(np.mean(np.abs(filtered))))
        results.append(float(np.std(filtered)))
    return np.array(results)


def dct_score(patch: np.ndarray) -> float:
    """Calcula score de frequência do patch via DCT."""
    dct = cv2.dct(patch.astype(np.float32))
    return float(np.sum(np.abs(dct)))


def extract_features(image_path: str) -> np.ndarray:
    """Extrai features DCT + SRM de uma imagem."""
    img = Image.open(image_path).convert("L").resize(RESIZE_TO)
    arr = np.array(img, dtype=np.float32) / 255.0

    h, w = arr.shape
    patches = []

    # Divide a imagem em patches e calcula score DCT de cada um
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            score = dct_score(patch)
            patches.append((score, patch))

    if not patches:
        return None

    # Ordena por score DCT
    patches.sort(key=lambda p: p[0])

    # Pega os N de menor e N de maior frequência
    low_patches  = [p for _, p in patches[:N_PATCHES]]
    high_patches = [p for _, p in patches[-N_PATCHES:]]

    # Extrai features SRM de cada patch selecionado
    features = []
    for patch in low_patches + high_patches:
        features.extend(apply_srm(patch))

    laplacian = cv2.Laplacian((arr * 255).astype(np.uint8), cv2.CV_64F)
    features.extend([
        float(np.var(laplacian)),
        float(np.mean(np.abs(laplacian))),
        float(np.std(laplacian)),
        float(np.percentile(np.abs(laplacian), 90)),
    ])

    return np.array(features)


def processar_diretorio(directory: str, label: int):
    """Processa todas as imagens de um diretório e retorna features e labels."""
    X, y  = [], []
    paths = [p for p in Path(directory).rglob('*') if p.suffix.lower() in EXTS]
    print(f"  {len(paths)} imagens em '{Path(directory).name}/'")

    start = time.time()
    for i, path in enumerate(paths):
        if i > 0 and i % 1000 == 0:
            elapsed  = time.time() - start
            restante = (elapsed / i) * (len(paths) - i)
            print(f"  {i}/{len(paths)} — ~{restante/60:.1f} min restantes")
        try:
            features = extract_features(str(path))
            if features is not None:
                X.append(features)
                y.append(label)
        except Exception as e:
            print(f"  Erro: {path.name} — {e}")

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
    X_test = np.array(Xr_t + Xf_t)
    y_test = np.array(yr_t + yf_t)
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
    joblib.dump(pipeline, 'modelospkl/modelo_ruido.pkl')
    print("Modelo salvo: modelospkl/modelo_ruido.pkl")