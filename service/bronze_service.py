import io
import json
import joblib
import piexif
import numpy as np
from sklearn.decomposition import PCA
from fastapi import UploadFile
from PIL import Image, PngImagePlugin


class BronzeService:
    # Analise de metadados de imagens, incluindo EXIF, XMP e metadados específicos de PNG (comuns em IA)
    @staticmethod
    def extract_metadata(filename: str, content_type: str, contents: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise ValueError("Não foi possível processar o arquivo como imagem.")

        width, height = image.size

        metadata = {
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

        return metadata

    @staticmethod
    def _extract_all_metadata(image: Image.Image, contents: bytes) -> dict:
        all_metadata = {}

        # Metadados PNG (onde IA costuma guardar prompt, seed, etc.)
        if isinstance(image, PngImagePlugin.PngImageFile):
            png_info = image.info or {}
            for key, value in png_info.items():
                if isinstance(value, bytes):
                    try:
                        all_metadata[key] = value.decode("utf-8", errors="replace")
                    except Exception:
                        all_metadata[key] = repr(value)
                elif isinstance(value, str):
                    # Tenta parsear como JSON (comum em ComfyUI)
                    try:
                        all_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        all_metadata[key] = value
                else:
                    all_metadata[key] = value

        # Metadados EXIF via piexif (JPEG/TIFF)
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

        # Metadados XMP
        xmp = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
        if xmp:
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", errors="replace")
            all_metadata["xmp"] = xmp

        return all_metadata
    
    # Detecção de anomalias usando PCA

    @staticmethod
    def _extract_pca_features(image_bytes: bytes):
        N_COMPONENTS = 50
        RESIZE_TO = (256, 256)
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(RESIZE_TO)
        matrix = np.array(img, dtype=np.float64) / 255.0

        def _reconstruction_error(matrix, pca, scores, k):
            scores_k = np.zeros_like(scores)
            scores_k[:, :k] = scores[:, :k]
            return float(np.mean((matrix - pca.inverse_transform(scores_k)) ** 2))

        n = min(N_COMPONENTS, min(matrix.shape))
        pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
        scores = pca.fit_transform(matrix)

        evr = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_

        evr_padded = np.zeros(N_COMPONENTS)
        evr_padded[:n] = evr

        probs = eigenvalues / eigenvalues.sum()
        eigen_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        geo_mean = float(np.exp(np.mean(np.log(eigenvalues + 1e-12))))
        spectral_flatness = geo_mean / (float(np.mean(eigenvalues)) + 1e-12)
        first_pc_dominance = float(evr[0])
        coef_variacao = float(np.std(eigenvalues) / (np.mean(eigenvalues) + 1e-12))

        rec_k5 = _reconstruction_error(matrix, pca, scores, min(5, n))
        rec_k10 = _reconstruction_error(matrix, pca, scores, min(10, n))
        rec_k20 = _reconstruction_error(matrix, pca, scores, min(20, n))

        return (
            list(evr_padded)
            + [eigen_entropy, spectral_flatness, first_pc_dominance, coef_variacao]
            + [rec_k5, rec_k10, rec_k20]
        )

    @staticmethod
    async def pca_anomaly(file: UploadFile):
        THRESHOLD = 0.80
        MODEL_PATH = 'modelos/PCA_anomaly/modelo_pca.pkl'

        contents = await file.read()
        features = BronzeService._extract_pca_features(contents)

        model = joblib.load(MODEL_PATH)
        prediction = int(model.predict([features])[0])
        proba = model.predict_proba([features])[0]
        confidence = float(max(proba))
        label = "IA" if prediction == 1 else "REAL"
        status = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label,
            "confidence": confidence,
            "status": status,
            "prob_real": float(proba[0]),
            "prob_ia": float(proba[1]),
        }

    # Analise da Luminescencia
    @staticmethod
    def _extract_luminance_features(image_bytes: bytes):
        from scipy.ndimage import sobel
        
        RESIZE_TO = (256, 256)
        N_COMPONENTS = 30

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(RESIZE_TO)
        img_array = np.array(img, dtype=np.float64)

        # Calcula luminância
        luminancia = (
            0.299 * img_array[:,:,0] +
            0.587 * img_array[:,:,1] +
            0.114 * img_array[:,:,2]
        )

        # Gradientes com Sobel
        grad_x = sobel(luminancia, axis=1)
        grad_y = sobel(luminancia, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direcao   = np.arctan2(grad_y, grad_x)

        # PCA sobre a magnitude dos gradientes
        n = min(N_COMPONENTS, min(magnitude.shape))
        pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
        pca.fit(magnitude)
        var = pca.explained_variance_ratio_

        return [
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
    
    @staticmethod
    async def luminance_analysis(file: UploadFile):
        THRESHOLD = 0.80
        MODEL_PATH = 'modelos/Luminescencia/svm_luminance.pkl'
        SCALER_PATH = 'modelos/Luminescencia/scaler_luminance.pkl'

        contents = await file.read()
        features = BronzeService._extract_luminance_features(contents)

        scaler = joblib.load(SCALER_PATH)
        model  = joblib.load(MODEL_PATH)

        features_norm = scaler.transform([features])
        prediction    = int(model.predict(features_norm)[0])
        proba         = model.predict_proba(features_norm)[0]
        confidence    = float(max(proba))
        label         = "IA" if prediction == 1 else "REAL"
        status        = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label,
            "confidence": confidence,
            "status": status,
            "prob_real": float(proba[0]),
            "prob_ia":   float(proba[1]),
        }
    
    # Análise de Ruído (DCT + SRM)
    @staticmethod
    def _extract_noise_features(image_bytes: bytes):
        import cv2
        from scipy.ndimage import convolve

        RESIZE_TO  = (256, 256)
        PATCH_SIZE = 32
        N_PATCHES  = 3

        SRM_FILTERS = [
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [0, -2, 1], [0, 1, 0]], dtype=np.float32),
        ]

        def apply_srm(patch):
            results = []
            for f in SRM_FILTERS:
                filtered = convolve(patch.astype(np.float32), f)
                results.append(float(np.mean(np.abs(filtered))))
                results.append(float(np.std(filtered)))
            return np.array(results)

        def dct_score(patch):
            dct = cv2.dct(patch.astype(np.float32))
            return float(np.sum(np.abs(dct)))

        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(RESIZE_TO)
        arr = np.array(img, dtype=np.float32) / 255.0

        h, w = arr.shape
        patches = []
        for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                patch = arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patches.append((dct_score(patch), patch))

        patches.sort(key=lambda p: p[0])
        selected = [p for _, p in patches[:N_PATCHES]] + [p for _, p in patches[-N_PATCHES:]]

        features = []
        for patch in selected:
            features.extend(apply_srm(patch))

        return np.array(features)

    @staticmethod
    async def noise_analysis(file: UploadFile):
        THRESHOLD  = 0.80
        MODEL_PATH = 'modelos/Ruido/modelo_ruido.pkl'

        contents   = await file.read()
        features   = BronzeService._extract_noise_features(contents)

        model      = joblib.load(MODEL_PATH)
        prediction = int(model.predict([features])[0])
        proba      = model.predict_proba([features])[0]
        confidence = float(max(proba))
        label      = "IA" if prediction == 1 else "REAL"
        status     = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label":      label,
            "confidence": confidence,
            "status":     status,
            "prob_real":  float(proba[0]),
            "prob_ia":    float(proba[1]),
        }
    def __init__():
        return