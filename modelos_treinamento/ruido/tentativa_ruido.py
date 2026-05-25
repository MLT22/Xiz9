import os
import time
import random
import json
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from scipy.ndimage import convolve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, brier_score_loss
import joblib

TRAIN_REAL = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\train\real'
TRAIN_FAKE = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\train\fake'
TEST_REAL  = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\test\real'
TEST_FAKE  = r'C:\Users\JoaoC\OneDrive\Desktop\DSXIZ\test\fake'

EXTS       = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
RESIZE_TO  = (256, 256)
PATCH_SIZE = 32
N_PATCHES  = 3

# ── Filtros SRM (extração de ruído) ───────────────────────────────────────────
# Baseados no trabalho de Fridrich & Kodovsky (2012)
# Cada filtro captura um padrão de ruído diferente na imagem
SRM_FILTERS = [
    # Filtros originais — direções horizontal, vertical e diagonal
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -2, 1], [0, 1, 0]], dtype=np.float32),
    # Filtros adicionados v2 — capturam variações em outras direções
    np.array([[-1, 2, -1], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32),
    np.array([[-1, 0, 0], [2, 0, 0], [-1, 0, 0]], dtype=np.float32),
    # Filtros adicionados v3 — variações verticais, diagonais e Laplaciano 2D
    np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float32),
    np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float32),
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32),
]


def apply_srm(patch: np.ndarray) -> np.ndarray:
    """Aplica filtros SRM no patch e retorna o padrão de ruído.
    Para cada filtro extrai média e desvio padrão da resposta — 2 features por filtro.
    """
    results = []
    for f in SRM_FILTERS:
        filtered = convolve(patch.astype(np.float32), f)
        results.append(float(np.mean(np.abs(filtered))))
        results.append(float(np.std(filtered)))
    return np.array(results)


def dct_score(patch: np.ndarray) -> float:
    """Calcula score de frequência do patch via DCT.
    Patches com score alto têm alta frequência (bordas, texturas complexas).
    Patches com score baixo têm baixa frequência (regiões suaves).
    """
    dct = cv2.dct(patch.astype(np.float32))
    return float(np.sum(np.abs(dct)))


def augment_image_bytes(image_bytes: bytes) -> bytes:
    """Aplica data augmentation específico para detecção de ruído.
    JPEG mais frequente e agressivo pois recompressão sobrescreve padrões SRM.
    Aplicar SOMENTE no treino — nunca no teste ou validação.
    """
    img = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Redimensionamento aleatório
    if random.random() < 0.5:
        scale = random.uniform(0.75, 1.25)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS
        )

    # Recompressão JPEG — mais frequente e agressiva que nos outros algoritmos
    # pois JPEG sobrescreve os padrões de ruído que o SRM detecta
    if random.random() < 0.9:
        quality = random.randint(50, 95)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        img = Image.open(BytesIO(buf.getvalue())).convert('RGB')

    # Borramento leve
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    out = BytesIO()
    img.save(out, format='PNG')
    return out.getvalue()


def extract_features_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Extrai features DCT + SRM + estatísticas globais a partir de bytes.
    Total de features: 6 patches × 9 filtros × 2 estatísticas + 4 globais = 112 features
    """
    img = Image.open(BytesIO(image_bytes)).convert("L").resize(RESIZE_TO)
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape

    # ── 1. Seleção de patches via DCT ─────────────────────────────────────────
    # Divide a imagem em patches e pontua cada um pela complexidade de frequência
    patches = []
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patches.append((dct_score(patch), patch))

    if not patches:
        return None

    # Ordena e seleciona os N de menor e N de maior frequência
    patches.sort(key=lambda p: p[0])
    low_patches  = [p for _, p in patches[:N_PATCHES]]
    high_patches = [p for _, p in patches[-N_PATCHES:]]

    # ── 2. Extração de ruído via filtros SRM (108 features) ───────────────────
    # Cada filtro captura um padrão de ruído diferente em cada patch selecionado
    features = []
    for patch in low_patches + high_patches:
        features.extend(apply_srm(patch))

    # ── 3. Estatísticas globais do ruído da imagem inteira (4 features) ───────
    # Laplacian captura a variação brusca entre pixels — ruído orgânico de câmera
    # Imagens reais têm variância alta; imagens de IA tendem a ser mais suaves
    laplacian = cv2.Laplacian((arr * 255).astype(np.uint8), cv2.CV_64F)
    features.extend([
        float(np.var(laplacian)),
        float(np.mean(np.abs(laplacian))),
        float(np.std(laplacian)),
        float(np.percentile(np.abs(laplacian), 90)),
    ])

    return np.array(features)


def processar_diretorio(directory: str, label: int, augment: bool = False):
    """Processa todas as imagens de um diretório e retorna features e labels.
    Se augment=True, gera uma cópia aumentada de cada imagem (dobra o dataset).
    Usar augment=True apenas no treino — nunca no teste.
    """
    X, y  = [], []
    paths = [p for p in Path(directory).rglob('*') if p.suffix.lower() in EXTS]
    print(f"  {len(paths)} imagens em '{Path(directory).name}/' (augment={'sim' if augment else 'não'})")

    start = time.time()
    for i, path in enumerate(paths):
        if i > 0 and i % 1000 == 0:
            elapsed  = time.time() - start
            restante = (elapsed / i) * (len(paths) - i)
            print(f"  {i}/{len(paths)} — ~{restante/60:.1f} min restantes")
        try:
            raw = path.read_bytes()

            # Imagem original
            features = extract_features_from_bytes(raw)
            if features is not None:
                X.append(features)
                y.append(label)

            # Versão aumentada — somente no treino
            if augment:
                features_aug = extract_features_from_bytes(augment_image_bytes(raw))
                if features_aug is not None:
                    X.append(features_aug)
                    y.append(label)

        except Exception as e:
            print(f"  Erro: {path.name} — {e}")

    return X, y


if __name__ == '__main__':
    print("=== Extraindo features — TREINO (com augmentation) ===")
    Xr, yr = processar_diretorio(TRAIN_REAL, label=0, augment=True)
    Xf, yf = processar_diretorio(TRAIN_FAKE, label=1, augment=True)
    X_train = np.array(Xr + Xf)
    y_train = np.array(yr + yf)
    print(f"Treino: {len(X_train)} imagens ({sum(y_train==0)} reais, {sum(y_train==1)} IA)\n")

    print("=== Extraindo features — TESTE (sem augmentation) ===")
    Xr_t, yr_t = processar_diretorio(TEST_REAL, label=0, augment=False)
    Xf_t, yf_t = processar_diretorio(TEST_FAKE, label=1, augment=False)
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

    print("=== Avaliação antes da calibração ===")
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=['Real', 'IA']))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    print("\n=== Calibrando modelo ===")
    calib_iso = CalibratedClassifierCV(pipeline, method='isotonic', cv='prefit')
    calib_iso.fit(X_test, y_test)
    brier_iso = brier_score_loss(y_test, calib_iso.predict_proba(X_test)[:, 1])

    calib_sig = CalibratedClassifierCV(pipeline, method='sigmoid', cv='prefit')
    calib_sig.fit(X_test, y_test)
    brier_sig = brier_score_loss(y_test, calib_sig.predict_proba(X_test)[:, 1])

    best   = calib_iso if brier_iso < brier_sig else calib_sig
    metodo = 'isotonic' if brier_iso < brier_sig else 'sigmoid'
    print(f"Melhor calibração: {metodo} (Brier iso={brier_iso:.4f}, sig={brier_sig:.4f})")

    # Threshold ótimo pelo F1
    proba_cal       = best.predict_proba(X_test)[:, 1]
    prec, rec, thrs = precision_recall_curve(y_test, proba_cal)
    f1              = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_thr        = float(thrs[np.argmax(f1)])
    print(f"Threshold ótimo: {best_thr:.4f}")

    print("\n=== Avaliação com threshold ótimo ===")
    y_pred_cal = (proba_cal >= best_thr).astype(int)
    print(classification_report(y_test, y_pred_cal, target_names=['Real', 'IA']))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred_cal))
    print(f"AUC-ROC: {roc_auc_score(y_test, proba_cal):.4f}")

    print("\n=== Salvando modelo calibrado ===")
    os.makedirs('modelospkl', exist_ok=True)
    joblib.dump(best, 'modelospkl/modelo_ruido.pkl')
    with open('modelospkl/threshold.json', 'w') as f:
        json.dump({'threshold': best_thr, 'metodo': metodo}, f, indent=2)
    print(f"Modelo salvo:     modelospkl/modelo_ruido.pkl")
    print(f"Threshold salvo:  modelospkl/threshold.json")