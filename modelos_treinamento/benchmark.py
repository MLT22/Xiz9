"""
Uso:
    python modelos_treinamento/benchmark.py <pasta_reais> <pasta_ia>

Exemplo:
    python modelos_treinamento/benchmark.py dataset/real dataset/ia
"""

import sys
import time
import io
import numpy as np
import joblib
import cv2
from pathlib import Path
from PIL import Image, PngImagePlugin
from sklearn.decomposition import PCA
from scipy.ndimage import sobel, convolve

MODEL_PCA       = 'modelos/PCA_anomaly/modelo_pca.pkl'
MODEL_PCA_V2    = 'modelos/PCA_anomaly/modelo_pca_v2.pkl'
MODEL_PCA_RF    = 'modelos antigos/modelo_pca_rf.pkl'
MODEL_LUM       = 'modelos/Luminescencia/svm_luminance.pkl'
SCALER_LUM      = 'modelos/Luminescencia/scaler_luminance.pkl'
MODEL_RUIDO     = 'modelos/Ruido/modelo_ruido.pkl'
EXTENSIONS      = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}


# -- Feature extraction --------------------------------------------------------

def _pca_features(image_bytes: bytes):
    N_COMPONENTS = 50
    img    = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256))
    matrix = np.array(img, dtype=np.float64) / 255.0

    def rec_error(matrix, pca, scores, k):
        s = np.zeros_like(scores)
        s[:, :k] = scores[:, :k]
        return float(np.mean((matrix - pca.inverse_transform(s)) ** 2))

    n      = min(N_COMPONENTS, min(matrix.shape))
    pca    = PCA(n_components=n, svd_solver='randomized', random_state=42)
    scores = pca.fit_transform(matrix)
    evr    = pca.explained_variance_ratio_
    eig    = pca.explained_variance_

    evr_pad        = np.zeros(N_COMPONENTS)
    evr_pad[:n]    = evr
    probs          = eig / eig.sum()
    eigen_entropy  = float(-np.sum(probs * np.log(probs + 1e-12)))
    geo_mean       = float(np.exp(np.mean(np.log(eig + 1e-12))))
    spec_flat      = geo_mean / (float(np.mean(eig)) + 1e-12)
    first_dom      = float(evr[0])
    coef_var       = float(np.std(eig) / (np.mean(eig) + 1e-12))

    return (list(evr_pad)
            + [eigen_entropy, spec_flat, first_dom, coef_var]
            + [rec_error(matrix, pca, scores, min(5, n)),
               rec_error(matrix, pca, scores, min(10, n)),
               rec_error(matrix, pca, scores, min(20, n))])


_GRID = 4  # grade 4x4 = 16 blocos — adicionar como atributo de classe


def _lum_features(image_bytes: bytes):
    RESIZE_TO   = (256, 256)
    GRID        = 4
    N_COMP      = 30

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # ── Luminância ───────────────────────────────────────────────────────
    img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    img_r      = cv2.resize(img_rgb, RESIZE_TO)
    luminancia = (0.299 * img_r[:,:,0] +
                    0.587 * img_r[:,:,1] +
                    0.114 * img_r[:,:,2])

    # ── Gradientes Sobel ─────────────────────────────────────────────────
    from scipy.ndimage import sobel as scipy_sobel
    grad_x    = scipy_sobel(luminancia, axis=1)
    grad_y    = scipy_sobel(luminancia, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direcao   = np.arctan2(grad_y, grad_x)

    # ── 1. Features PCA (11 features) ────────────────────────────────────
    from sklearn.decomposition import PCA
    n   = min(N_COMP, min(magnitude.shape))
    pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
    pca.fit(magnitude)
    var = pca.explained_variance_ratio_

    pca_feats = [
        float(var[0]),
        float(np.sum(var[:5])),
        float(np.sum(var[:10])),
        float(np.sum(var[:20])),
        float(-np.sum(var * np.log(var + 1e-10))),
        float(np.std(var)),
        float(np.std(direcao)),
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
        float(np.percentile(magnitude, 90)),
        float(np.mean(magnitude) / (np.std(magnitude) + 1e-10)),
    ]

    # ── 2. FFT da magnitude do gradiente (5 features) ────────────────────
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(magnitude)))
    h, w    = fft_mag.shape
    cy, cx  = h // 2, w // 2
    Y, X    = np.ogrid[:h, :w]
    dist    = np.sqrt((X - cx)**2 + (Y - cy)**2)
    md      = np.sqrt(cx**2 + cy**2)
    e_low   = float(np.sum(fft_mag[dist <= md * 0.1]**2))
    e_mid   = float(np.sum(fft_mag[(dist > md * 0.1) & (dist <= md * 0.4)]**2))
    e_high  = float(np.sum(fft_mag[dist > md * 0.4]**2))
    e_tot   = e_low + e_mid + e_high + 1e-12
    fft_feats = [
        e_low  / e_tot,
        e_mid  / e_tot,
        e_high / e_tot,
        e_high / (e_low + 1e-12),
        float(np.std(np.log1p(fft_mag))),
    ]

    # ── 3. Histograma de direções HOG (8 features) ───────────────────────
    hist_dir, _ = np.histogram(direcao, bins=8, range=(-np.pi, np.pi))
    hist_dir    = hist_dir / (hist_dir.sum() + 1e-10)
    hog_feats   = hist_dir.tolist()

    # ── 4. Coerência local do gradiente (2 features) ─────────────────────
    coer_v     = float(np.mean(np.cos(direcao[:-1, :] - direcao[1:, :])))
    coer_h     = float(np.mean(np.cos(direcao[:, :-1] - direcao[:, 1:])))
    coer_feats = [coer_v, coer_h]

    # ── 5. Normalização local (3 features) ───────────────────────────────
    blur       = cv2.GaussianBlur(magnitude.astype(np.float32), (15, 15), 0)
    mag_norm   = magnitude / (blur.astype(np.float64) + 1e-6)
    norm_feats = [
        float(np.mean(mag_norm)),
        float(np.std(mag_norm)),
        float(np.percentile(mag_norm, 95)),
    ]

    # ── 6. Features por blocos 4x4 (48 features) ─────────────────────────
    bh, bw      = RESIZE_TO[0] // GRID, RESIZE_TO[1] // GRID
    block_feats = []
    for i in range(GRID):
        for j in range(GRID):
            bloco = magnitude[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_feats.append(float(np.mean(bloco)))
            block_feats.append(float(np.std(bloco)))
            block_feats.append(float(np.percentile(bloco, 90)))

    # ── Concatenar e sanitizar ────────────────────────────────────────────
    all_feats = (pca_feats + fft_feats + hog_feats +
                    coer_feats + norm_feats + block_feats)

    result = np.array(all_feats, dtype=np.float64)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result.tolist()


def _pca_rf_features(image_bytes: bytes):
    N_COMPONENTS = 50
    img    = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256))
    matrix = np.array(img, dtype=np.float64) / 255.0
    n      = min(N_COMPONENTS, min(matrix.shape))
    pca    = PCA(n_components=n, svd_solver='randomized', random_state=42)
    pca.fit_transform(matrix)
    evr = pca.explained_variance_ratio_
    eig = pca.explained_variance_

    probs         = eig / eig.sum()
    eigen_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    geo_mean      = float(np.exp(np.mean(np.log(eig + 1e-12))))
    spec_flat     = geo_mean / (float(np.mean(eig)) + 1e-12)
    first_dom     = float(evr[0])

    return [
        float(np.sum(evr[:5])),
        float(np.sum(evr[:10])),
        float(np.sum(evr[:20])),
        float(np.sum(evr[:50])),
        first_dom,
        eigen_entropy,
        spec_flat,
    ]


def _pca_v2_features(image_bytes: bytes):
    N_COMPONENTS = 30
    img_rgb  = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((128, 128))
    img_gray = img_rgb.convert('L')
    arr_rgb  = np.array(img_rgb,  dtype=np.float64) / 255.0
    arr_gray = np.array(img_gray, dtype=np.float64) / 255.0

    n      = min(N_COMPONENTS, min(arr_gray.shape))
    pca    = PCA(n_components=n, svd_solver='randomized', random_state=42)
    scores = pca.fit_transform(arr_gray)
    evr    = pca.explained_variance_ratio_
    eig    = pca.explained_variance_
    evr_pad     = np.zeros(N_COMPONENTS); evr_pad[:n] = evr
    probs       = eig / eig.sum()
    eigen_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    geo_mean      = float(np.exp(np.mean(np.log(eig + 1e-12))))
    spec_flat     = geo_mean / (float(np.mean(eig)) + 1e-12)
    first_dom     = float(evr[0])
    coef_var      = float(np.std(eig) / (np.mean(eig) + 1e-12))
    def rec(k):
        sk = np.zeros_like(scores); sk[:, :k] = scores[:, :k]
        return float(np.mean((arr_gray - pca.inverse_transform(sk)) ** 2))
    pca_feats = list(evr_pad) + [eigen_entropy, spec_flat, first_dom, coef_var, rec(min(5,n)), rec(min(10,n)), rec(min(20,n))]

    color_feats = []
    for c in range(3):
        ch = arr_rgb[:, :, c].ravel()
        color_feats += [float(np.mean(ch)), float(np.std(ch)),
                        float(np.mean((ch-ch.mean())**3) / (ch.std()**3 + 1e-12)),
                        float(np.mean((ch-ch.mean())**4) / (ch.std()**4 + 1e-12))]

    try:
        arr_hsv = np.array(img_rgb.convert('HSV'), dtype=np.float64) / 255.0
        hsv_feats = [float(np.mean(arr_hsv[:,:,c])) for c in range(3)] + \
                    [float(np.std(arr_hsv[:,:,c]))  for c in range(3)]
    except Exception:
        hsv_feats = [0.0] * 6

    fft     = np.fft.fft2(arr_gray)
    fft_mag = np.abs(np.fft.fftshift(fft))
    h, w    = fft_mag.shape; cy, cx = h//2, w//2
    dist    = np.sqrt((np.ogrid[:h,:w][1]-cx)**2 + (np.ogrid[:h,:w][0]-cy)**2)
    md      = np.sqrt(cx**2+cy**2)
    e_low   = float(np.sum(fft_mag[dist <= md*0.1]**2))
    e_mid   = float(np.sum(fft_mag[(dist > md*0.1) & (dist <= md*0.4)]**2))
    e_high  = float(np.sum(fft_mag[dist > md*0.4]**2))
    e_tot   = e_low + e_mid + e_high + 1e-12
    fft_feats = [e_low/e_tot, e_mid/e_tot, e_high/e_tot, e_high/(e_low+1e-12), float(np.std(np.log1p(fft_mag)))]

    return pca_feats + color_feats + hsv_feats + fft_feats


SRM_FILTERS = [
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -2, 1], [0, 1, 0]], dtype=np.float32),
]

def _ruido_features(image_bytes: bytes):
    PATCH_SIZE = 32
    N_PATCHES  = 3
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape
    patches = []
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            dct   = cv2.dct(patch.astype(np.float32))
            patches.append((float(np.sum(np.abs(dct))), patch))
    if not patches:
        return None
    patches.sort(key=lambda p: p[0])
    selected = [p for _, p in patches[:N_PATCHES]] + [p for _, p in patches[-N_PATCHES:]]
    features = []
    for patch in selected:
        for f in SRM_FILTERS:
            filtered = convolve(patch, f)
            features.append(float(np.mean(np.abs(filtered))))
            features.append(float(np.std(filtered)))
    return np.array(features)


# -- Benchmark -----------------------------------------------------------------

def load_images(real_dir: Path, ia_dir: Path, limit_per_class: int = None):
    real = [p for p in real_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
    ia   = [p for p in ia_dir.iterdir()  if p.suffix.lower() in EXTENSIONS]
    if limit_per_class:
        real = real[:limit_per_class]
        ia   = ia[:limit_per_class]
    return [(p, 0) for p in real] + [(p, 1) for p in ia]


def run_benchmark(samples, extract_fn, model, scaler=None, name="Modelo"):
    correct = 0
    times   = []
    erros   = 0

    for path, label_true in samples:
        try:
            image_bytes = path.read_bytes()
            t0          = time.perf_counter()
            features    = extract_fn(image_bytes)
            feat_arr    = [features]
            if scaler:
                feat_arr = scaler.transform(feat_arr)
            pred   = int(model.predict(feat_arr)[0])
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if pred == label_true:
                correct += 1
        except Exception as e:
            print(f"  [ERRO] {path.name}: {e}")
            erros += 1

    total    = len(samples) - erros
    accuracy = correct / total if total else 0
    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'-'*45}")
    print(f"  {name}")
    print(f"{'-'*45}")
    print(f"  Imagens testadas : {total}  ({erros} com erro)")
    print(f"  Acurácia         : {accuracy*100:.2f}%  ({correct}/{total})")
    print(f"  Tempo médio      : {avg_time*1000:.1f} ms/imagem")
    print(f"  Tempo total      : {sum(times)*1000:.1f} ms")


def run_ensemble(samples, members):
    """
    members: lista de (extract_fn, model, scaler, w_real, w_ia)
    Pesos separados por classe: cada modelo pesa mais onde tem melhor acuracia.
    Score final: soma(prob_ia * w_ia) vs soma(prob_real * w_real)
    """
    correct = 0
    times   = []
    erros   = 0

    for path, label_true in samples:
        try:
            image_bytes   = path.read_bytes()
            t0            = time.perf_counter()

            score_ia   = 0.0
            score_real = 0.0
            sum_w_ia   = sum(w_ia   for *_, w_ia in members)
            sum_w_real = sum(w_real for _, __, ___, w_real, ____ in members)

            for extract_fn, model, scaler, w_real, w_ia in members:
                feat  = extract_fn(image_bytes)
                fa    = scaler.transform([feat]) if scaler else [feat]
                proba = model.predict_proba(fa)[0]
                score_real += proba[0] * w_real
                score_ia   += proba[1] * w_ia

            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            pred = 1 if (score_ia / sum_w_ia) >= (score_real / sum_w_real) else 0
            if pred == label_true:
                correct += 1

        except Exception as e:
            print(f"  [ERRO] {path.name}: {e}")
            erros += 1

    total    = len(samples) - erros
    accuracy = correct / total if total else 0
    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'-'*45}")
    print(f"  Ensemble ponderado por classe")
    print(f"{'-'*45}")
    print(f"  Imagens testadas : {total}  ({erros} com erro)")
    print(f"  Acuracia         : {accuracy*100:.2f}%  ({correct}/{total})")
    print(f"  Tempo medio      : {avg_time*1000:.1f} ms/imagem")
    print(f"  Tempo total      : {sum(times)*1000:.1f} ms")


def main():
    if len(sys.argv) != 3:
        print("Uso: python modelos_treinamento/benchmark.py <pasta_reais> <pasta_ia>")
        sys.exit(1)

    real_dir = Path(sys.argv[1])
    ia_dir   = Path(sys.argv[2])

    if not real_dir.is_dir() or not ia_dir.is_dir():
        print("Erro: um dos caminhos fornecidos não é uma pasta válida.")
        sys.exit(1)

    samples = load_images(real_dir, ia_dir, limit_per_class=500)
    if not samples:
        print("Nenhuma imagem encontrada nas pastas.")
        sys.exit(1)

    print(f"\nDataset: {len(samples)} imagens  "
          f"({sum(1 for _,l in samples if l==0)} reais, "
          f"{sum(1 for _,l in samples if l==1)} IA)")

    ensemble_members = []

    # PCA v2 (XGBoost + cor + FFT)
    try:
        model_pca_v2 = joblib.load(MODEL_PCA_V2)
        run_benchmark(samples, _pca_v2_features, model_pca_v2, name="PCA v2 (XGBoost+cor+FFT)")
        ensemble_members.append((_pca_v2_features, model_pca_v2, None, 1.0, 1.0))
    except FileNotFoundError:
        print(f"\n[AVISO] Modelo PCA v2 nao encontrado em: {MODEL_PCA_V2}")

    # Luminescência
    try:
        model_lum  = joblib.load(MODEL_LUM)
        scaler_lum = joblib.load(SCALER_LUM)
        run_benchmark(samples, _lum_features, model_lum, scaler=scaler_lum, name="Luminescencia")
        ensemble_members.append((_lum_features, model_lum, scaler_lum, 1.0, 1.0))
    except FileNotFoundError:
        print(f"\n[AVISO] Modelo Luminescencia nao encontrado em: {MODEL_LUM} / {SCALER_LUM}")

    # Ruído
    try:
        model_ruido = joblib.load(MODEL_RUIDO)
        run_benchmark(samples, _ruido_features, model_ruido, name="Ruido (SRM+DCT)")
        ensemble_members.append((_ruido_features, model_ruido, None, 1.0, 1.0))
    except FileNotFoundError:
        print(f"\n[AVISO] Modelo Ruido nao encontrado em: {MODEL_RUIDO}")

    # Ensemble
    if len(ensemble_members) >= 2:
        run_ensemble(samples, ensemble_members)

    print(f"\n{'-'*45}")


if __name__ == '__main__':
    main()
