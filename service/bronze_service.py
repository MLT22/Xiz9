import io
import json
import joblib
import piexif
import numpy as np
import cv2
from scipy.ndimage import sobel, convolve
from sklearn.decomposition import PCA
from fastapi import UploadFile
from PIL import Image, PngImagePlugin


_AI_KEYWORDS = [
    "stable diffusion", "stablediffusion", "automatic1111", "a1111",
    "comfyui", "comfy ui", "novelai", "invokeai", "midjourney",
    "dall-e", "dalle", "firefly", "adobe firefly", "imagen",
    "dreamstudio", "openjourney", "kandinsky",
]

_AI_PNG_KEYS = {"parameters", "workflow", "prompt", "invokeai_metadata", "sd-metadata"}


class BronzeService:

    # ── Metadados ─────────────────────────────────────────────────────────────

    @staticmethod
    def extract_metadata(filename: str, content_type: str, contents: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise ValueError("Não foi possível processar o arquivo como imagem.")

        width, height = image.size
        return {
            "filename": filename,
            "content_type": content_type,
            "file_size_bytes": len(contents),
            "format": image.format,
            "mode": image.mode,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 4) if height else None,
            "is_animated": getattr(image, "is_animated", False),
            "frames": getattr(image, "n_frames", 1),
            "exif": BronzeService._extract_all_metadata(image, contents),
        }

    @staticmethod
    def _extract_all_metadata(image: Image.Image, contents: bytes) -> dict:
        all_metadata = {}

        if isinstance(image, PngImagePlugin.PngImageFile):
            png_info = image.info or {}
            for key, value in png_info.items():
                if isinstance(value, bytes):
                    try:
                        all_metadata[key] = value.decode("utf-8", errors="replace")
                    except Exception:
                        all_metadata[key] = repr(value)
                elif isinstance(value, str):
                    try:
                        all_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        all_metadata[key] = value
                else:
                    all_metadata[key] = value

        try:
            exif_bytes = image.info.get("exif")
            if exif_bytes:
                exif_dict = piexif.load(exif_bytes)
                for ifd_name, ifd_data in exif_dict.items():
                    if not isinstance(ifd_data, dict):
                        continue
                    for tag_id, value in ifd_data.items():
                        tag_name = piexif.TAGS[ifd_name].get(tag_id, {}).get("name", str(tag_id))
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace").strip("\x00")
                        all_metadata.setdefault("exif_piexif", {})[tag_name] = value
        except Exception:
            pass

        xmp = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
        if xmp:
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", errors="replace")
            all_metadata["xmp"] = xmp

        return all_metadata

    @staticmethod
    def check_metadata_ai_indicators(image: Image.Image, raw_metadata: dict) -> dict:
        indicators = []

        # Chaves PNG exclusivas de geradores de IA
        if isinstance(image, PngImagePlugin.PngImageFile):
            found_keys = _AI_PNG_KEYS & set(image.info.keys())
            for k in found_keys:
                indicators.append(f"PNG key '{k}' encontrada (comum em SD/ComfyUI)")

        # Varredura por keywords em todos os valores de metadados
        def _search(obj, path=""):
            if isinstance(obj, str):
                low = obj.lower()
                for kw in _AI_KEYWORDS:
                    if kw in low:
                        indicators.append(f"Keyword '{kw}' em {path}")
                        return
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _search(v, path=f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _search(v, path=f"{path}[{i}]")

        _search(raw_metadata)

        return {
            "has_ai_indicators": len(indicators) > 0,
            "indicators": indicators,
        }

    # ── PCA v2 (XGBoost + cor + FFT) ─────────────────────────────────────────

    @staticmethod
    def _extract_pca_v2_features(image_bytes: bytes):
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

        pca_feats = list(evr_pad) + [eigen_entropy, spec_flat, first_dom, coef_var,
                                     rec(min(5, n)), rec(min(10, n)), rec(min(20, n))]

        color_feats = []
        for c in range(3):
            ch = arr_rgb[:, :, c].ravel()
            color_feats += [
                float(np.mean(ch)), float(np.std(ch)),
                float(np.mean((ch - ch.mean())**3) / (ch.std()**3 + 1e-12)),
                float(np.mean((ch - ch.mean())**4) / (ch.std()**4 + 1e-12)),
            ]

        try:
            arr_hsv   = np.array(img_rgb.convert('HSV'), dtype=np.float64) / 255.0
            hsv_feats = [float(np.mean(arr_hsv[:, :, c])) for c in range(3)] + \
                        [float(np.std(arr_hsv[:, :, c]))  for c in range(3)]
        except Exception:
            hsv_feats = [0.0] * 6

        fft     = np.fft.fft2(arr_gray)
        fft_mag = np.abs(np.fft.fftshift(fft))
        h, w    = fft_mag.shape; cy, cx = h // 2, w // 2
        Y, X    = np.ogrid[:h, :w]
        dist    = np.sqrt((X - cx)**2 + (Y - cy)**2)
        md      = np.sqrt(cx**2 + cy**2)
        e_low   = float(np.sum(fft_mag[dist <= md * 0.1]**2))
        e_mid   = float(np.sum(fft_mag[(dist > md * 0.1) & (dist <= md * 0.4)]**2))
        e_high  = float(np.sum(fft_mag[dist > md * 0.4]**2))
        e_tot   = e_low + e_mid + e_high + 1e-12
        fft_feats = [e_low/e_tot, e_mid/e_tot, e_high/e_tot,
                     e_high/(e_low + 1e-12), float(np.std(np.log1p(fft_mag)))]

        return pca_feats + color_feats + hsv_feats + fft_feats

    @staticmethod
    async def pca_anomaly(file: UploadFile):
        THRESHOLD  = 0.80
        MODEL_PATH = 'modelos/PCA_anomaly/modelo_pca_v2.pkl'

        contents = await file.read()
        features = BronzeService._extract_pca_v2_features(contents)

        model      = joblib.load(MODEL_PATH)
        proba      = model.predict_proba([features])[0]
        prediction = int(model.predict([features])[0])
        confidence = float(max(proba))
        label      = "IA" if prediction == 1 else "REAL"
        status     = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": confidence, "status": status,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }

    # ── Luminescência ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_luminance_features(image_bytes: bytes):
        RESIZE_TO    = (256, 256)
        N_COMPONENTS = 30

        img       = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(RESIZE_TO)
        img_array = np.array(img, dtype=np.float64)
        luminancia = (0.299 * img_array[:,:,0] +
                      0.587 * img_array[:,:,1] +
                      0.114 * img_array[:,:,2])

        grad_x    = sobel(luminancia, axis=1)
        grad_y    = sobel(luminancia, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direcao   = np.arctan2(grad_y, grad_x)

        n   = min(N_COMPONENTS, min(magnitude.shape))
        pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
        pca.fit(magnitude)
        var = pca.explained_variance_ratio_

        return [
            float(var[0]), float(np.sum(var[:5])), float(np.sum(var[:10])),
            float(np.sum(var[:20])), float(-np.sum(var * np.log(var + 1e-10))),
            float(np.std(var)), float(np.std(direcao)),
            float(np.mean(magnitude)), float(np.std(magnitude)),
            float(np.percentile(magnitude, 90)),
            float(np.mean(magnitude) / (np.std(magnitude) + 1e-10)),
        ]

    @staticmethod
    async def luminance_analysis(file: UploadFile):
        THRESHOLD  = 0.80
        MODEL_PATH  = 'modelos/Luminescencia/svm_luminance.pkl'
        SCALER_PATH = 'modelos/Luminescencia/scaler_luminance.pkl'

        contents      = await file.read()
        features      = BronzeService._extract_luminance_features(contents)
        scaler        = joblib.load(SCALER_PATH)
        model         = joblib.load(MODEL_PATH)
        features_norm = scaler.transform([features])
        prediction    = int(model.predict(features_norm)[0])
        proba         = model.predict_proba(features_norm)[0]
        confidence    = float(max(proba))
        label         = "IA" if prediction == 1 else "REAL"
        status        = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": confidence, "status": status,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }

    # ── Ruído (SRM + DCT) ─────────────────────────────────────────────────────

    _SRM_FILTERS = [
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, -2, 1], [0, 1, 0]], dtype=np.float32),
    ]

    @staticmethod
    def _extract_ruido_features(image_bytes: bytes):
        PATCH_SIZE = 32
        N_PATCHES  = 3
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256))
        arr = np.array(img, dtype=np.float32) / 255.0
        h, w = arr.shape
        patches = []
        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patches.append((float(np.sum(np.abs(cv2.dct(patch)))), patch))
        if not patches:
            return None
        patches.sort(key=lambda p: p[0])
        selected = [p for _, p in patches[:N_PATCHES]] + [p for _, p in patches[-N_PATCHES:]]
        features = []
        for patch in selected:
            for f in BronzeService._SRM_FILTERS:
                filtered = convolve(patch, f)
                features.extend([float(np.mean(np.abs(filtered))), float(np.std(filtered))])
        return np.array(features)

    @staticmethod
    async def ruido_analysis(file: UploadFile):
        THRESHOLD  = 0.80
        MODEL_PATH = 'modelos/Ruido/modelo_ruido.pkl'

        contents   = await file.read()
        features   = BronzeService._extract_ruido_features(contents)
        model      = joblib.load(MODEL_PATH)
        proba      = model.predict_proba([features])[0]
        prediction = int(model.predict([features])[0])
        confidence = float(max(proba))
        label      = "IA" if prediction == 1 else "REAL"
        status     = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": confidence, "status": status,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }

    # ── Assembly (metadados + ensemble dos 3 modelos) ─────────────────────────

    @staticmethod
    async def assembly(file: UploadFile):
        THRESHOLD = 0.80

        contents = await file.read()

        # Metadados
        try:
            image        = Image.open(io.BytesIO(contents))
            raw_metadata = BronzeService._extract_all_metadata(image, contents)
            meta_check   = BronzeService.check_metadata_ai_indicators(image, raw_metadata)
        except Exception:
            image        = None
            meta_check   = {"has_ai_indicators": False, "indicators": []}

        # PCA v2
        pca_feat  = BronzeService._extract_pca_v2_features(contents)
        pca_model = joblib.load('modelos/PCA_anomaly/modelo_pca_v2.pkl')
        pca_proba = pca_model.predict_proba([pca_feat])[0]
        pca_pred  = int(pca_model.predict([pca_feat])[0])

        # Luminescência
        lum_feat   = BronzeService._extract_luminance_features(contents)
        lum_scaler = joblib.load('modelos/Luminescencia/scaler_luminance.pkl')
        lum_model  = joblib.load('modelos/Luminescencia/svm_luminance.pkl')
        lum_proba  = lum_model.predict_proba(lum_scaler.transform([lum_feat]))[0]
        lum_pred   = int(lum_model.predict(lum_scaler.transform([lum_feat]))[0])

        # Ruído
        ruido_feat  = BronzeService._extract_ruido_features(contents)
        ruido_model = joblib.load('modelos/Ruido/modelo_ruido.pkl')
        ruido_proba = ruido_model.predict_proba([ruido_feat])[0]
        ruido_pred  = int(ruido_model.predict([ruido_feat])[0])

        # Ensemble — média simples das probabilidades
        prob_ia   = float((pca_proba[1] + lum_proba[1] + ruido_proba[1]) / 3)
        prob_real = float((pca_proba[0] + lum_proba[0] + ruido_proba[0]) / 3)
        ensemble_pred = 1 if prob_ia >= 0.5 else 0
        ensemble_conf = float(max(prob_ia, prob_real))
        ensemble_label = "IA" if ensemble_pred == 1 else "REAL"

        def _model_result(pred, proba):
            conf  = float(max(proba))
            label = "IA" if pred == 1 else "REAL"
            return {
                "label": label, "confidence": conf,
                "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
            }

        return {
            "metadata": meta_check,
            "modelos": {
                "pca_v2":       _model_result(pca_pred,   pca_proba),
                "luminescencia": _model_result(lum_pred,   lum_proba),
                "ruido":        _model_result(ruido_pred, ruido_proba),
            },
            "ensemble": {
                "label":      ensemble_label,
                "confidence": ensemble_conf,
                "prob_real":  prob_real,
                "prob_ia":    prob_ia,
                "status":     "CONCLUSIVO" if ensemble_conf >= THRESHOLD else "INCERTO",
            },
        }
    
    # ── Avaliação Geral (Stacking XGBoost) ───────────────────────────────────

    @staticmethod
    def _build_stacking_meta_row(p_pca, p_lum, p_ruido):
        p_ia_pca, p_ia_lum, p_ia_ruido = p_pca[1], p_lum[1], p_ruido[1]

        margin_pca   = p_ia_pca   - p_pca[0]
        margin_lum   = p_ia_lum   - p_lum[0]
        margin_ruido = p_ia_ruido - p_ruido[0]

        vote_ia   = int(p_ia_pca > 0.5) + int(p_ia_lum > 0.5) + int(p_ia_ruido > 0.5)
        all_agree = 1.0 if vote_ia in (0, 3) else 0.0

        mean_p_ia  = float(np.mean([p_ia_pca, p_ia_lum, p_ia_ruido]))
        all_probs  = [p_pca[0], p_ia_pca, p_lum[0], p_ia_lum, p_ruido[0], p_ia_ruido]

        return [
            p_pca[0],   p_ia_pca,
            p_lum[0],   p_ia_lum,
            p_ruido[0], p_ia_ruido,
            margin_pca, margin_lum, margin_ruido,
            float(vote_ia), all_agree,
            mean_p_ia, float(max(all_probs)), float(min(all_probs)),
        ]

    @staticmethod
    async def avaliacao_geral(file: UploadFile):
        STACK_THRESHOLD = 0.51
        contents = await file.read()

        # Metadados
        try:
            image        = Image.open(io.BytesIO(contents))
            raw_metadata = BronzeService._extract_all_metadata(image, contents)
            meta_check   = BronzeService.check_metadata_ai_indicators(image, raw_metadata)
        except Exception:
            meta_check = {"has_ai_indicators": False, "indicators": []}

        # Modelos base — extrai features e probabilidades
        pca_feat  = BronzeService._extract_pca_v2_features(contents)
        pca_model = joblib.load('modelos/PCA_anomaly/modelo_pca_v2.pkl')
        pca_proba = pca_model.predict_proba([pca_feat])[0]

        lum_feat   = BronzeService._extract_luminance_features(contents)
        lum_scaler = joblib.load('modelos/Luminescencia/scaler_luminance.pkl')
        lum_model  = joblib.load('modelos/Luminescencia/svm_luminance.pkl')
        lum_proba  = lum_model.predict_proba(lum_scaler.transform([lum_feat]))[0]

        ruido_feat  = BronzeService._extract_ruido_features(contents)
        ruido_model = joblib.load('modelos/Ruido/modelo_ruido.pkl')
        ruido_proba = ruido_model.predict_proba([ruido_feat])[0]

        # Meta-learner stacking
        meta_row    = BronzeService._build_stacking_meta_row(pca_proba, lum_proba, ruido_proba)
        stack_model = joblib.load('modelos/Stacking/modelo_stacking.pkl')
        threshold   = STACK_THRESHOLD

        stack_proba = stack_model.predict_proba([meta_row])[0]
        prob_ia     = float(stack_proba[1])
        prob_real   = float(stack_proba[0])
        label       = "IA" if prob_ia >= threshold else "REAL"
        confidence  = prob_ia if label == "IA" else prob_real

        def _base_result(proba):
            pred = int(proba[1] > 0.5)
            return {
                "label":      "IA" if pred == 1 else "REAL",
                "prob_real":  float(proba[0]),
                "prob_ia":    float(proba[1]),
            }

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "prob_real":  round(prob_real, 4),
            "prob_ia":    round(prob_ia, 4),
            "metadata":   meta_check,
            "modelos_base": {
                "pca_v2":        _base_result(pca_proba),
                "luminescencia": _base_result(lum_proba),
                "ruido":         _base_result(ruido_proba),
            },
        }

    def __init__():
        return
