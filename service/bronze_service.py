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

class BronzeService:

    # ── Frequencia_Cor (FFT + cor RGB/HSV + estatísticas PCA invariantes) ────────

    @staticmethod
    def _pca_distribution_features(eig: np.ndarray) -> list:
        eig_sorted = np.sort(eig)[::-1]
        norm_eig   = eig_sorted / (eig_sorted.sum() + 1e-12)
        eigen_entropy = float(-np.sum(norm_eig * np.log(norm_eig + 1e-12)))
        top5_ratio    = float(eig_sorted[:5].sum() / (eig_sorted[-5:].sum() + 1e-12))
        cumsum = np.cumsum(norm_eig)
        n_50   = int(np.searchsorted(cumsum, 0.50) + 1)
        n_90   = int(np.searchsorted(cumsum, 0.90) + 1)
        n_95   = int(np.searchsorted(cumsum, 0.95) + 1)
        k      = np.arange(1, len(eig_sorted) + 1)
        slope  = float(np.polyfit(np.log(k), np.log(eig_sorted + 1e-12), 1)[0])
        return [eigen_entropy, top5_ratio, n_50, n_90, n_95, slope]

    @staticmethod
    def _extract_freq_cor_features(image_bytes: bytes):
        from io import BytesIO as _BytesIO
        N_PCA_COMP = 30
        RESIZE_TO  = (256, 256)

        img_rgb  = Image.open(_BytesIO(image_bytes)).convert('RGB').resize(RESIZE_TO)
        img_gray = img_rgb.convert('L')
        arr_rgb  = np.array(img_rgb,  dtype=np.float64) / 255.0
        arr_gray = np.array(img_gray, dtype=np.float64) / 255.0

        n      = min(N_PCA_COMP, min(arr_gray.shape))
        pca    = PCA(n_components=n, svd_solver='randomized', random_state=42)
        scores = pca.fit_transform(arr_gray)
        eig    = pca.explained_variance_
        evr    = pca.explained_variance_ratio_

        pca_inv_feats = BronzeService._pca_distribution_features(eig)

        geo_mean  = float(np.exp(np.mean(np.log(eig + 1e-12))))
        spec_flat = geo_mean / (float(np.mean(eig)) + 1e-12)
        first_dom = float(evr[0])
        coef_var  = float(np.std(eig) / (np.mean(eig) + 1e-12))

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
                Image.open(_BytesIO(image_bytes)).convert('RGB').resize(RESIZE_TO).convert('HSV'),
                dtype=np.float64
            ) / 255.0
            hsv_feats = []
            for c in range(3):
                ch = arr_hsv[:, :, c].ravel()
                hsv_feats += [float(np.mean(ch)), float(np.std(ch))]
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

        result = np.array(pca_inv_feats + pca_scalar_feats + color_feats + hsv_feats + fft_feats,
                          dtype=np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    async def freq_cor(file: UploadFile):
        MODEL_PATH      = 'modelos/Frequencia_Cor/modelo_freq_cor_v1.pkl'
        SCALER_PATH     = 'modelos/Frequencia_Cor/scaler_freq_cor.pkl'
        CALIBRATOR_PATH = 'modelos/Frequencia_Cor/calibrator_freq_cor.pkl'
        THRESHOLD_PATH  = 'modelos/Frequencia_Cor/threshold.json'

        contents  = await file.read()
        features  = BronzeService._extract_freq_cor_features(contents)
        model     = joblib.load(MODEL_PATH)
        scaler    = joblib.load(SCALER_PATH)
        calibrator = joblib.load(CALIBRATOR_PATH)

        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f)['threshold']

        feat_scaled = scaler.transform([features])
        raw_proba   = model.predict_proba(feat_scaled)[0, 1]
        prob_ia     = float(calibrator.transform([raw_proba])[0])
        prob_real   = float(1.0 - prob_ia)
        prediction  = 1 if prob_ia >= threshold else 0
        confidence  = prob_ia if prediction == 1 else prob_real
        label       = "IA" if prediction == 1 else "REAL"
        status      = "CONCLUSIVO" if confidence >= threshold else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": round(confidence, 4), "status": status,
            "prob_real": round(prob_real, 4), "prob_ia": round(prob_ia, 4),
        }

    # ── Luminescência ─────────────────────────────────────────────────────────

    @staticmethod
    def _pca_distribution_features(eig: np.ndarray) -> list:
        """Features invariantes à base — comparáveis entre imagens."""
        eig_sorted    = np.sort(eig)[::-1]
        norm_eig      = eig_sorted / (eig_sorted.sum() + 1e-12)
        eigen_entropy = float(-np.sum(norm_eig * np.log(norm_eig + 1e-12)))
        top5_ratio    = float(eig_sorted[:5].sum() / (eig_sorted[-5:].sum() + 1e-12))
        cumsum        = np.cumsum(norm_eig)
        n_50          = int(np.searchsorted(cumsum, 0.50) + 1)
        n_90          = int(np.searchsorted(cumsum, 0.90) + 1)
        n_95          = int(np.searchsorted(cumsum, 0.95) + 1)
        k             = np.arange(1, len(eig_sorted) + 1)
        slope         = float(np.polyfit(np.log(k), np.log(eig_sorted + 1e-12), 1)[0])
        return [eigen_entropy, top5_ratio, n_50, n_90, n_95, slope]
 
    @staticmethod
    def _extract_luminance_features(image_bytes: bytes):
        RESIZE_TO = (256, 256)
        GRID      = 4
        N_COMP    = 30
 
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
 
        # ── 1. PCA invariante (6 features) ───────────────────────────────────
        from sklearn.decomposition import PCA
        n   = min(N_COMP, min(magnitude.shape))
        pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
        pca.fit(magnitude)
        eig = pca.explained_variance_
 
        pca_feats = BronzeService._pca_distribution_features(eig) + [
            float(np.std(direcao)),
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.percentile(magnitude, 90)),
            float(np.mean(magnitude) / (np.std(magnitude) + 1e-10)),
        ]
 
        # ── 2. FFT da magnitude do gradiente (5 features) ────────────────────
        fft_mag   = np.abs(np.fft.fftshift(np.fft.fft2(magnitude)))
        h, w      = fft_mag.shape
        cy, cx    = h // 2, w // 2
        Y, X      = np.ogrid[:h, :w]
        dist      = np.sqrt((X - cx)**2 + (Y - cy)**2)
        md        = np.sqrt(cx**2 + cy**2)
        e_low     = float(np.sum(fft_mag[dist <= md * 0.1]**2))
        e_mid     = float(np.sum(fft_mag[(dist > md * 0.1) & (dist <= md * 0.4)]**2))
        e_high    = float(np.sum(fft_mag[dist > md * 0.4]**2))
        e_tot     = e_low + e_mid + e_high + 1e-12
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
        all_feats = (pca_feats + fft_feats + hog_feats + coer_feats + norm_feats + block_feats)
        result = np.array(all_feats, dtype=np.float64)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result.tolist()



    @staticmethod
    async def luminance_analysis(file: UploadFile):
        THRESHOLD       = 0.80
        MODEL_PATH      = 'modelos/Luminescencia/svm_luminance.pkl'
        SCALER_PATH     = 'modelos/Luminescencia/scaler_luminance.pkl'
        CALIBRATOR_PATH = 'modelos/Luminescencia/calibrator_luminance.pkl'
        THRESHOLD_PATH  = 'modelos/Luminescencia/threshold.json'

        contents  = await file.read()
        features  = BronzeService._extract_luminance_features(contents)

        scaler     = joblib.load(SCALER_PATH)
        model      = joblib.load(MODEL_PATH)
        calibrator = joblib.load(CALIBRATOR_PATH)

        with open(THRESHOLD_PATH) as f:
            lum_threshold = json.load(f)['threshold']

        features_norm = scaler.transform([features])
        proba_raw     = model.predict_proba(features_norm)[0]
        proba_cal     = calibrator.predict([proba_raw[1]])[0]
        proba         = np.array([1 - proba_cal, proba_cal])

        prediction = int(proba[1] >= lum_threshold)
        confidence = float(max(proba))
        label      = "IA" if prediction == 1 else "REAL"
        status     = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": confidence, "status": status,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }
    # ── Ruído (SRM + DCT) ─────────────────────────────────────────────────────

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

    @staticmethod
    def _extract_ruido_features(image_bytes: bytes):
        PATCH_SIZE = 32
        N_PATCHES  = 3
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
            for f in BronzeService._SRM_FILTERS:
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
    async def ruido_analysis(file: UploadFile):
        import json
        MODEL_PATH     = 'modelos/Ruido/modelo_ruido.pkl'
        THRESHOLD_PATH = 'modelos/Ruido/threshold.json'

        contents = await file.read()
        features = BronzeService._extract_ruido_features(contents)
        model    = joblib.load(MODEL_PATH)

        with open(THRESHOLD_PATH) as f:
            THRESHOLD = json.load(f)['threshold']

        proba      = model.predict_proba([features])[0]
        prediction = 1 if proba[1] >= THRESHOLD else 0
        confidence = float(max(proba))
        label      = "IA" if prediction == 1 else "REAL"
        status     = "CONCLUSIVO" if confidence >= THRESHOLD else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": confidence, "status": status,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }

    # ── Ensamble (ensemble dos 3 modelos) ─────────────────────────

    def _model_result(pred, proba):
                conf  = float(max(proba))
                label = "IA" if pred == 1 else "REAL"
                return {
                    "label": label, "confidence": conf,
                    "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
                }

    @staticmethod
    async def ensamble(file: UploadFile):
        THRESHOLD = 0.80

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Frequencia_Cor
        fc_feat = BronzeService._extract_freq_cor_features(contents)
        fc_model = joblib.load('modelos/Frequencia_Cor/modelo_freq_cor_v1.pkl')
        fc_scaler = joblib.load('modelos/Frequencia_Cor/scaler_freq_cor.pkl')
        fc_calibrator = joblib.load('modelos/Frequencia_Cor/calibrator_freq_cor.pkl')
        with open('modelos/Frequencia_Cor/threshold.json') as _f:
            fc_threshold = json.load(_f)['threshold']
            fc_scaled = fc_scaler.transform([fc_feat])
            fc_raw = fc_model.predict_proba(fc_scaled)[0, 1]
            fc_prob_ia = float(fc_calibrator.transform([fc_raw])[0])
            fc_proba = np.array([1.0 - fc_prob_ia, fc_prob_ia])
            fc_pred = int(fc_prob_ia >= fc_threshold)

        # Luminescência
        lum_feat = BronzeService._extract_luminance_features(contents)
        lum_scaler = joblib.load('modelos/Luminescencia/scaler_luminance.pkl')
        lum_model = joblib.load('modelos/Luminescencia/svm_luminance.pkl')
        lum_calibrator = joblib.load('modelos/Luminescencia/calibrator_luminance.pkl')
        with open('modelos/Luminescencia/threshold.json') as f:
            lum_threshold = json.load(f)['threshold']

        lum_feat_s = lum_scaler.transform([lum_feat])
        lum_proba_raw = lum_model.predict_proba(lum_feat_s)[0]
        lum_proba_cal = lum_calibrator.predict(lum_proba_raw[1:2])
        lum_proba = np.array([1 - lum_proba_cal[0], lum_proba_cal[0]])
        lum_pred = int(lum_proba[1] >= lum_threshold)

        # Ruído
        ruido_feat = BronzeService._extract_ruido_features(contents)
        ruido_model = joblib.load('modelos/Ruido/modelo_ruido.pkl')
        ruido_proba = ruido_model.predict_proba([ruido_feat])[0]
        ruido_pred = int(ruido_model.predict([ruido_feat])[0])

        # Ensemble — média simples das probabilidades
        prob_ia = float((fc_proba[1] + lum_proba[1] + ruido_proba[1]) / 3)
        prob_real = float((fc_proba[0] + lum_proba[0] + ruido_proba[0]) / 3)
        ensemble_pred = 1 if prob_ia >= 0.5 else 0
        ensemble_conf = float(max(prob_ia, prob_real))
        ensemble_label = "IA" if ensemble_pred == 1 else "REAL"

        return {
            "modelos": {
                "frequencia_cor": _model_result(fc_pred,    fc_proba),
                "luminescencia": _model_result(lum_pred,   lum_proba),
                "ruido": _model_result(ruido_pred, ruido_proba),
            },
            "ensemble": {
                "label": ensemble_label,
                "confidence": ensemble_conf,
                "prob_real": prob_real,
                "prob_ia": prob_ia,
                "status": "CONCLUSIVO" if ensemble_conf >= THRESHOLD else "INCERTO",
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
        CONCLUSIVO_THRESHOLD = 0.75
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))


        # Modelos base — extrai features e probabilidades
        fc_feat = BronzeService._extract_freq_cor_features(contents)
        fc_model = joblib.load('modelos/Frequencia_Cor/modelo_freq_cor_v1.pkl')
        fc_scaler = joblib.load('modelos/Frequencia_Cor/scaler_freq_cor.pkl')
        fc_calibrator = joblib.load('modelos/Frequencia_Cor/calibrator_freq_cor.pkl')
        with open('modelos/Frequencia_Cor/threshold.json') as _f:
            fc_threshold = json.load(_f)['threshold']
        fc_scaled = fc_scaler.transform([fc_feat])
        fc_raw = fc_model.predict_proba(fc_scaled)[0, 1]
        fc_prob_ia = float(fc_calibrator.transform([fc_raw])[0])

        fc_proba   = np.array([1.0 - fc_prob_ia, fc_prob_ia])

        lum_feat = BronzeService._extract_luminance_features(contents)
        lum_scaler = joblib.load('modelos/Luminescencia/scaler_luminance.pkl')
        lum_model = joblib.load('modelos/Luminescencia/svm_luminance.pkl')
        lum_calibrator = joblib.load('modelos/Luminescencia/calibrator_luminance.pkl')
        with open('modelos/Luminescencia/threshold.json') as f:
            lum_threshold = json.load(f)['threshold']

        lum_feat_s = lum_scaler.transform([lum_feat])
        lum_proba_raw = lum_model.predict_proba(lum_feat_s)[0]
        lum_proba_cal = lum_calibrator.predict(lum_proba_raw[1:2])
        lum_proba  = np.array([1 - lum_proba_cal[0], lum_proba_cal[0]])

        ruido_feat = BronzeService._extract_ruido_features(contents)
        ruido_model = joblib.load('modelos/Ruido/modelo_ruido.pkl')
        ruido_proba = ruido_model.predict_proba([ruido_feat])[0]

        # Consenso — média das probabilidades dos 3 modelos base
        # Luminescência usa threshold calibrado; PCA e Ruído usam 0.5
        lum_pred = int(lum_proba[1] >= lum_threshold)
        prob_ia = round(float((fc_proba[1] + lum_proba[1] + ruido_proba[1]) / 3), 4)
        prob_real = round(float((fc_proba[0] + lum_proba[0] + ruido_proba[0]) / 3), 4)
        label = "IA" if prob_ia >= 0.5 else "REAL"
        confidence = prob_ia if label == "IA" else prob_real
        escalate = confidence < CONCLUSIVO_THRESHOLD

        def _base_result(proba):
            return {
                "label": "IA" if proba[1] > 0.5 else "REAL",
                "prob_real": round(float(proba[0]), 4),
                "prob_ia": round(float(proba[1]), 4),
            }


        return {
            "label": label,
            "confidence": round(confidence, 4),
            "prob_real": prob_real,
            "prob_ia": prob_ia,
            "escalate": escalate,
            "modelos_base": {
                "frequencia_cor": _base_result(fc_proba),
                "luminescencia":  _base_result(lum_proba),
                "ruido": _base_result(ruido_proba),
            },
        }

    def __init__():
        return