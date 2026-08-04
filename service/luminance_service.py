import json

import cv2
import joblib
import numpy as np
from fastapi import UploadFile
from scipy.ndimage import sobel
from sklearn.decomposition import PCA

from service.pca_utils import pca_distribution_features


class LuminanceService:
    """Luminescência: gradiente Sobel + PCA + FFT + HOG + coerência local."""

    _PATHS = {
        "model": "modelos/Luminescencia/svm_luminance.pkl",
        "scaler": "modelos/Luminescencia/scaler_luminance.pkl",
        "calibrator": "modelos/Luminescencia/calibrator_luminance.pkl",
        "threshold": "modelos/Luminescencia/threshold.json",
    }

    @staticmethod
    def _extract_features(image_bytes: bytes):
        RESIZE_TO = (256, 256)
        GRID = 4
        N_COMP = 30

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # ── Luminância ───────────────────────────────────────────────────────
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
        img_r = cv2.resize(img_rgb, RESIZE_TO)
        luminancia = (0.299 * img_r[:,:,0] +
                      0.587 * img_r[:,:,1] +
                      0.114 * img_r[:,:,2])

        # ── Gradientes Sobel ─────────────────────────────────────────────────
        grad_x = sobel(luminancia, axis=1)
        grad_y = sobel(luminancia, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direcao = np.arctan2(grad_y, grad_x)

        # ── 1. PCA invariante (6 features) ───────────────────────────────────
        n = min(N_COMP, min(magnitude.shape))
        pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
        pca.fit(magnitude)
        eig = pca.explained_variance_

        pca_feats = pca_distribution_features(eig) + [
            float(np.std(direcao)),
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.percentile(magnitude, 90)),
            float(np.mean(magnitude) / (np.std(magnitude) + 1e-10)),
        ]

        # ── 2. FFT da magnitude do gradiente (5 features) ────────────────────
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(magnitude)))
        h, w = fft_mag.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        md = np.sqrt(cx**2 + cy**2)
        e_low = float(np.sum(fft_mag[dist <= md * 0.1]**2))
        e_mid = float(np.sum(fft_mag[(dist > md * 0.1) & (dist <= md * 0.4)]**2))
        e_high = float(np.sum(fft_mag[dist > md * 0.4]**2))
        e_tot = e_low + e_mid + e_high + 1e-12
        fft_feats = [
            e_low  / e_tot,
            e_mid  / e_tot,
            e_high / e_tot,
            e_high / (e_low + 1e-12),
            float(np.std(np.log1p(fft_mag))),
        ]

        # ── 3. Histograma de direções HOG (8 features) ───────────────────────
        hist_dir, _ = np.histogram(direcao, bins=8, range=(-np.pi, np.pi))
        hist_dir = hist_dir / (hist_dir.sum() + 1e-10)
        hog_feats = hist_dir.tolist()

        # ── 4. Coerência local do gradiente (2 features) ─────────────────────
        coer_v = float(np.mean(np.cos(direcao[:-1, :] - direcao[1:, :])))
        coer_h = float(np.mean(np.cos(direcao[:, :-1] - direcao[:, 1:])))
        coer_feats = [coer_v, coer_h]

        # ── 5. Normalização local (3 features) ───────────────────────────────
        blur = cv2.GaussianBlur(magnitude.astype(np.float32), (15, 15), 0)
        mag_norm = magnitude / (blur.astype(np.float64) + 1e-6)
        norm_feats = [
            float(np.mean(mag_norm)),
            float(np.std(mag_norm)),
            float(np.percentile(mag_norm, 95)),
        ]

        # ── 6. Features por blocos 4x4 (48 features) ─────────────────────────
        bh, bw = RESIZE_TO[0] // GRID, RESIZE_TO[1] // GRID
        block_feats = []
        for i in range(GRID):
            for j in range(GRID):
                bloco = magnitude[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                block_feats.append(float(np.mean(bloco)))
                block_feats.append(float(np.std(bloco)))
                block_feats.append(float(np.percentile(bloco, 90)))

        # ── Concatenar e sanitizar ────────────────────────────────────────────
        all_feats = (pca_feats + fft_feats + hog_feats + coer_feats + norm_feats + block_feats)
        result = np.array(all_feats, dtype=np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result.tolist()

    @staticmethod
    def predict(contents: bytes):
        paths = LuminanceService._PATHS
        features = LuminanceService._extract_features(contents)

        scaler = joblib.load(paths["scaler"])
        model = joblib.load(paths["model"])
        calibrator = joblib.load(paths["calibrator"])

        with open(paths["threshold"]) as f:
            threshold = json.load(f)["threshold"]

        features_norm = scaler.transform([features])
        proba_raw = model.predict_proba(features_norm)[0]
        proba_cal = calibrator.predict([proba_raw[1]])[0]
        proba = np.array([1 - proba_cal, proba_cal])
        return proba, threshold

    @staticmethod
    async def analyze(file: UploadFile):
        THRESHOLD = 0.80

        contents = await file.read()
        proba, lum_threshold = LuminanceService.predict(contents)

        prediction = int(proba[1] >= lum_threshold)
        confidence = float(max(proba))
        label = "IA" if prediction == 1 else "REAL"
        status = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": confidence, "status": status,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }
