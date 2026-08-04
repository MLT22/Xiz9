import io
import json

import cv2
import joblib
import numpy as np
from fastapi import UploadFile
from PIL import Image
from scipy.ndimage import convolve


class RuidoService:
    """Ruído: filtros SRM sobre patches selecionados via DCT + estatísticas globais."""

    _SRM_FILTERS = [
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -2, 1], [0, 1, 0]], dtype=np.float32),
        np.array([[-1, 2, -1], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32),
        np.array([[-1, 0, 0], [2, 0, 0], [-1, 0, 0]], dtype=np.float32),
        np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float32),
        np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float32),
        np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32),
    ]

    _PATHS = {
        "model": "modelos/Ruido/modelo_ruido.pkl",
        "threshold": "modelos/Ruido/threshold.json",
    }

    @staticmethod
    def _extract_features(image_bytes: bytes):
        PATCH_SIZE = 32
        N_PATCHES = 3
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256))
        arr = np.array(img, dtype=np.float32) / 255.0
        h, w = arr.shape

        # ── 1. Seleção de patches via DCT ─────────────────────────────────────
        # Divide a imagem em patches e pontua cada um pela complexidade de frequência
        patches = []
        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patches.append((float(np.sum(np.abs(cv2.dct(patch)))), patch))
        if not patches:
            return None

        # Ordena e seleciona os N de menor e N de maior frequência
        patches.sort(key=lambda p: p[0])
        selected = [p for _, p in patches[:N_PATCHES]] + [p for _, p in patches[-N_PATCHES:]]

        # ── 2. Extração de ruído via filtros SRM (108 features) ───────────────
        # Cada filtro captura um padrão de ruído diferente em cada patch selecionado
        # 6 patches × 9 filtros(na interação atual) × 2 estatísticas (média + desvio) = 108 features
        features = []
        for patch in selected:
            for f in RuidoService._SRM_FILTERS:
                filtered = convolve(patch, f)
                features.extend([float(np.mean(np.abs(filtered))), float(np.std(filtered))])

        # ── 3. Estatísticas globais do ruído da imagem inteira (4 features) ───
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

    @staticmethod
    def predict(contents: bytes):
        paths = RuidoService._PATHS
        features = RuidoService._extract_features(contents)
        model = joblib.load(paths["model"])

        with open(paths["threshold"]) as f:
            threshold = json.load(f)["threshold"]

        proba = model.predict_proba([features])[0]
        return proba, threshold

    @staticmethod
    async def analyze(file: UploadFile):
        contents = await file.read()
        proba, threshold = RuidoService.predict(contents)

        prediction = 1 if proba[1] >= threshold else 0
        confidence = float(max(proba))
        label = "IA" if prediction == 1 else "REAL"
        status = "CONCLUSIVO" if confidence >= threshold else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": confidence, "status": status,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }
