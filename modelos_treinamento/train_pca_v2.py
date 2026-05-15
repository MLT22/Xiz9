from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore', message='.*decompression bomb.*', category=UserWarning)
Image.MAX_IMAGE_PIXELS = None
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os
import time

TRAIN_REAL   = r'D:\TCCDataset\train\real'
TRAIN_FAKE   = r'D:\TCCDataset\train\fake'
TEST_REAL    = r'D:\TCCDataset\test\real'
TEST_FAKE    = r'D:\TCCDataset\test\fake'

EXTS         = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
RESIZE_TO    = (128, 128)
N_COMPONENTS = 30


def _reconstruction_error(matrix, pca, scores, k):
    sk = np.zeros_like(scores)
    sk[:, :k] = scores[:, :k]
    return float(np.mean((matrix - pca.inverse_transform(sk)) ** 2))


def extract_features(image_path):
    try:
        img_rgb = Image.open(image_path).convert('RGB').resize(RESIZE_TO)
        img_gray = img_rgb.convert('L')
        arr_rgb  = np.array(img_rgb,  dtype=np.float64) / 255.0
        arr_gray = np.array(img_gray, dtype=np.float64) / 255.0

        # --- PCA no canal cinza (features originais reduzidas) ---
        n      = min(N_COMPONENTS, min(arr_gray.shape))
        pca    = PCA(n_components=n, svd_solver='randomized', random_state=42)
        scores = pca.fit_transform(arr_gray)
        evr    = pca.explained_variance_ratio_
        eig    = pca.explained_variance_

        evr_pad     = np.zeros(N_COMPONENTS)
        evr_pad[:n] = evr
        probs       = eig / eig.sum()

        eigen_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        geo_mean      = float(np.exp(np.mean(np.log(eig + 1e-12))))
        spec_flat     = geo_mean / (float(np.mean(eig)) + 1e-12)
        first_dom     = float(evr[0])
        coef_var      = float(np.std(eig) / (np.mean(eig) + 1e-12))

        rec_k5  = _reconstruction_error(arr_gray, pca, scores, min(5,  n))
        rec_k10 = _reconstruction_error(arr_gray, pca, scores, min(10, n))
        rec_k20 = _reconstruction_error(arr_gray, pca, scores, min(20, n))

        pca_features = (list(evr_pad)
                        + [eigen_entropy, spec_flat, first_dom, coef_var]
                        + [rec_k5, rec_k10, rec_k20])

        # --- Features de cor por canal RGB ---
        color_features = []
        for c in range(3):
            ch = arr_rgb[:, :, c].ravel()
            color_features += [
                float(np.mean(ch)),
                float(np.std(ch)),
                float(np.mean((ch - ch.mean())**3) / (ch.std()**3 + 1e-12)),  # skewness
                float(np.mean((ch - ch.mean())**4) / (ch.std()**4 + 1e-12)),  # kurtosis
            ]

        # --- Features HSV ---
        img_hsv = img_rgb.convert('HSV') if hasattr(img_rgb, 'convert') else None
        try:
            img_hsv  = Image.open(image_path).convert('RGB').resize(RESIZE_TO).convert('HSV')
            arr_hsv  = np.array(img_hsv, dtype=np.float64) / 255.0
            hsv_features = []
            for c in range(3):
                ch = arr_hsv[:, :, c].ravel()
                hsv_features += [float(np.mean(ch)), float(np.std(ch))]
        except Exception:
            hsv_features = [0.0] * 6

        # --- Features FFT (espectro de frequencia) ---
        fft      = np.fft.fft2(arr_gray)
        fft_mag  = np.abs(np.fft.fftshift(fft))
        h, w     = fft_mag.shape
        cy, cx   = h // 2, w // 2
        Y, X     = np.ogrid[:h, :w]
        dist     = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)

        low_mask  = dist <= max_dist * 0.1
        mid_mask  = (dist > max_dist * 0.1) & (dist <= max_dist * 0.4)
        high_mask = dist > max_dist * 0.4

        e_low   = float(np.sum(fft_mag[low_mask]**2))
        e_mid   = float(np.sum(fft_mag[mid_mask]**2))
        e_high  = float(np.sum(fft_mag[high_mask]**2))
        e_total = e_low + e_mid + e_high + 1e-12

        fft_features = [
            e_low  / e_total,
            e_mid  / e_total,
            e_high / e_total,
            e_high / (e_low + 1e-12),       # razao alta/baixa freq
            float(np.std(np.log1p(fft_mag))),
        ]

        return pca_features + color_features + hsv_features + fft_features

    except Exception as e:
        print(f"Erro: {image_path} — {e}")
        return None


def processar_diretorio(directory, label):
    X, y  = [], []
    paths = [p for p in Path(directory).rglob('*') if p.suffix.lower() in EXTS]
    print(f"  {len(paths)} imagens em '{Path(directory).name}/'")

    start = time.time()
    for i, path in enumerate(paths):
        if i > 0 and i % 2000 == 0:
            elapsed  = time.time() - start
            restante = (elapsed / i) * (len(paths) - i)
            print(f"  {i}/{len(paths)} — ~{restante/60:.1f} min restantes")
        features = extract_features(str(path))
        if features is not None:
            X.append(features)
            y.append(label)
    return X, y


if __name__ == '__main__':
    n_pca   = N_COMPONENTS
    n_color = 4 * 3
    n_hsv   = 2 * 3
    n_fft   = 5
    total   = n_pca + 4 + 3 + n_color + n_hsv + n_fft
    print(f"Features por imagem: {total}  "
          f"(PCA={n_pca+7}, cor={n_color}, HSV={n_hsv}, FFT={n_fft})\n")

    print("=== Extraindo features — TREINO ===")
    Xr, yr = processar_diretorio(TRAIN_REAL, label=0)
    Xf, yf = processar_diretorio(TRAIN_FAKE, label=1)
    X_train = np.array(Xr + Xf)
    y_train = np.array(yr + yf)
    print(f"Treino: {len(X_train)} imagens\n")

    print("=== Extraindo features — TESTE ===")
    Xr_t, yr_t = processar_diretorio(TEST_REAL, label=0)
    Xf_t, yf_t = processar_diretorio(TEST_FAKE, label=1)
    X_test = np.array(Xr_t + Xf_t)
    y_test = np.array(yr_t + yf_t)
    print(f"Teste: {len(X_test)} imagens\n")

    print("=== Treinando XGBoost ===")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
        ))
    ])

    X_val = X_test
    y_val = y_test
    pipeline.named_steps['scaler'].fit(X_train)
    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
    X_val_scaled   = pipeline.named_steps['scaler'].transform(X_val)
    pipeline.named_steps['xgb'].fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=50,
    )
    print("Concluido.\n")

    print("=== Avaliacao no conjunto de teste ===")
    y_pred  = pipeline.named_steps['xgb'].predict(X_val_scaled)
    y_proba = pipeline.named_steps['xgb'].predict_proba(X_val_scaled)[:, 1]
    print(classification_report(y_test, y_pred, target_names=['Real', 'IA']))
    print("Matriz de confusao:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    print("\n=== Salvando modelo ===")
    os.makedirs('modelos/PCA_anomaly', exist_ok=True)
    joblib.dump(pipeline, 'modelos/PCA_anomaly/modelo_pca_v2.pkl')
    print("Modelo salvo: modelos/PCA_anomaly/modelo_pca_v2.pkl")
