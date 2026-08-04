import io
import json

import joblib
import numpy as np
from fastapi import UploadFile
from PIL import Image
from sklearn.decomposition import PCA

from service.pca_utils import pca_distribution_features


class FreqCorService:
    """Frequência-Cor: FFT + cor RGB/HSV + estatísticas PCA invariantes."""

    _PATHS = {
        "model": "modelos/Frequencia_Cor/modelo_freq_cor_v1.pkl",
        "scaler": "modelos/Frequencia_Cor/scaler_freq_cor.pkl",
        "calibrator": "modelos/Frequencia_Cor/calibrator_freq_cor.pkl",
        "threshold": "modelos/Frequencia_Cor/threshold.json",
    }

    @staticmethod
    def _extract_features(image_bytes: bytes):
        N_PCA_COMP = 30
        RESIZE_TO = (256, 256)

        img_rgb = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(RESIZE_TO)
        img_gray = img_rgb.convert('L')
        arr_rgb = np.array(img_rgb, dtype=np.float64) / 255.0
        arr_gray = np.array(img_gray, dtype=np.float64) / 255.0

        n = min(N_PCA_COMP, min(arr_gray.shape))
        pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
        scores = pca.fit_transform(arr_gray)
        eig = pca.explained_variance_
        evr = pca.explained_variance_ratio_

        pca_inv_feats = pca_distribution_features(eig)

        geo_mean = float(np.exp(np.mean(np.log(eig + 1e-12))))
        spec_flat = geo_mean / (float(np.mean(eig)) + 1e-12)
        first_dom = float(evr[0])
        coef_var = float(np.std(eig) / (np.mean(eig) + 1e-12))

        def _rec(k):
            sk = np.zeros_like(scores); sk[:, :k] = scores[:, :k]
            return float(np.mean((arr_gray - pca.inverse_transform(sk)) ** 2))

        pca_scalar_feats = [spec_flat, first_dom, coef_var,
                            _rec(min(5, n)), _rec(min(10, n)), _rec(min(20, n))]

        color_feats = []
        for c in range(3):
            ch = arr_rgb[:, :, c].ravel()
            mu = ch.mean(); sd = ch.std() + 1e-12
            color_feats += [float(mu), float(sd),
                            float(np.mean(((ch - mu) / sd) ** 3)),
                            float(np.mean(((ch - mu) / sd) ** 4))]

        try:
            arr_hsv = np.array(
                Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(RESIZE_TO).convert('HSV'),
                dtype=np.float64
            ) / 255.0
            hsv_feats = []
            for c in range(3):
                ch = arr_hsv[:, :, c].ravel()
                hsv_feats += [float(np.mean(ch)), float(np.std(ch))]
        except Exception:
            hsv_feats = [0.0] * 6

        fft = np.fft.fft2(arr_gray)
        fft_mag = np.abs(np.fft.fftshift(fft))
        h, w = fft_mag.shape; cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        md = np.sqrt(cx**2 + cy**2)
        e_low = float(np.sum(fft_mag[dist <= md * 0.1]**2))
        e_mid = float(np.sum(fft_mag[(dist > md * 0.1) & (dist <= md * 0.4)]**2))
        e_high = float(np.sum(fft_mag[dist > md * 0.4]**2))
        e_tot = e_low + e_mid + e_high + 1e-12
        fft_feats = [e_low/e_tot, e_mid/e_tot, e_high/e_tot,
                     e_high/(e_low + 1e-12), float(np.std(np.log1p(fft_mag)))]

        result = np.array(pca_inv_feats + pca_scalar_feats + color_feats + hsv_feats + fft_feats,
                          dtype=np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def predict(contents: bytes):
        paths = FreqCorService._PATHS
        features = FreqCorService._extract_features(contents)
        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        calibrator = joblib.load(paths["calibrator"])

        with open(paths["threshold"]) as f:
            threshold = json.load(f)["threshold"]

        feat_scaled = scaler.transform([features])
        raw_proba = model.predict_proba(feat_scaled)[0, 1]
        prob_ia = float(calibrator.transform([raw_proba])[0])
        proba = np.array([1.0 - prob_ia, prob_ia])
        return proba, threshold

    @staticmethod
    async def analyze(file: UploadFile):
        contents = await file.read()
        proba, threshold = FreqCorService.predict(contents)
        prob_real, prob_ia = float(proba[0]), float(proba[1])

        prediction = 1 if prob_ia >= threshold else 0
        confidence = prob_ia if prediction == 1 else prob_real
        label = "IA" if prediction == 1 else "REAL"
        status = "CONCLUSIVO" if confidence >= threshold else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": round(confidence, 4), "status": status,
            "prob_real": round(prob_real, 4), "prob_ia": round(prob_ia, 4),
        }
