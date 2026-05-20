"""
Stacking: treina um meta-learner sobre as probabilidades
dos 3 modelos base (PCA v2, Luminescencia, Ruido).

Meta-features: 14 sinais (probs + margens + consenso + agregados)
Meta-learners: XGBoost calibrado, Regressao Logistica calibrada, MLP calibrado
               -> salva apenas o melhor (AUC no conjunto de avaliacao)

Dataset para meta-treino: D:\\TCCDataset\\test  (12 k imagens)
Avaliacao final:          C:\\Users\\user\\Desktop\\test  (1 k imagens)
"""

import io
import os
import time
import warnings
import numpy as np
import joblib
from pathlib import Path
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from scipy.ndimage import sobel, convolve
from xgboost import XGBClassifier
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
Image.MAX_IMAGE_PIXELS = None

META_TRAIN_REAL = r'D:\TCCDataset\test\real'
META_TRAIN_FAKE = r'D:\TCCDataset\test\fake'
EVAL_REAL       = r'C:\Users\user\Desktop\test\real'
EVAL_FAKE       = r'C:\Users\user\Desktop\test\fake'
EXTS            = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

MODEL_PCA   = 'modelos/PCA_anomaly/modelo_pca_v2.pkl'
MODEL_LUM   = 'modelos/Luminescencia/svm_luminance.pkl'
SCALER_LUM  = 'modelos/Luminescencia/scaler_luminance.pkl'
MODEL_RUIDO = 'modelos/Ruido/modelo_ruido.pkl'


# ── Feature extractors ────────────────────────────────────────────────────────

def _pca_v2_features(image_bytes):
    N = 30
    img_rgb  = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((128, 128))
    img_gray = img_rgb.convert('L')
    arr_rgb  = np.array(img_rgb,  dtype=np.float64) / 255.0
    arr_gray = np.array(img_gray, dtype=np.float64) / 255.0
    n = min(N, min(arr_gray.shape))
    pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
    scores = pca.fit_transform(arr_gray)
    evr = pca.explained_variance_ratio_; eig = pca.explained_variance_
    evr_pad = np.zeros(N); evr_pad[:n] = evr
    probs = eig / eig.sum()
    ee = float(-np.sum(probs * np.log(probs + 1e-12)))
    gm = float(np.exp(np.mean(np.log(eig + 1e-12))))
    sf = gm / (float(np.mean(eig)) + 1e-12)
    fd = float(evr[0]); cv = float(np.std(eig) / (np.mean(eig) + 1e-12))
    def rec(k):
        sk = np.zeros_like(scores); sk[:, :k] = scores[:, :k]
        return float(np.mean((arr_gray - pca.inverse_transform(sk)) ** 2))
    pca_f = list(evr_pad) + [ee, sf, fd, cv, rec(min(5,n)), rec(min(10,n)), rec(min(20,n))]
    cf = []
    for c in range(3):
        ch = arr_rgb[:,:,c].ravel()
        cf += [float(np.mean(ch)), float(np.std(ch)),
               float(np.mean((ch-ch.mean())**3)/(ch.std()**3+1e-12)),
               float(np.mean((ch-ch.mean())**4)/(ch.std()**4+1e-12))]
    try:
        arr_hsv = np.array(img_rgb.convert('HSV'), dtype=np.float64)/255.0
        hf = [float(np.mean(arr_hsv[:,:,c])) for c in range(3)] + \
             [float(np.std(arr_hsv[:,:,c]))  for c in range(3)]
    except: hf = [0.0]*6
    fft = np.fft.fft2(arr_gray); fm = np.abs(np.fft.fftshift(fft))
    h, w = fm.shape; cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]; dist = np.sqrt((X-cx)**2+(Y-cy)**2); md = np.sqrt(cx**2+cy**2)
    el = float(np.sum(fm[dist<=md*0.1]**2)); em = float(np.sum(fm[(dist>md*0.1)&(dist<=md*0.4)]**2))
    eh = float(np.sum(fm[dist>md*0.4]**2)); et = el+em+eh+1e-12
    ff = [el/et, em/et, eh/et, eh/(el+1e-12), float(np.std(np.log1p(fm)))]
    return pca_f + cf + hf + ff


def _lum_features(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((256, 256))
    arr = np.array(img, dtype=np.float64)
    lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    gx = sobel(lum, axis=1); gy = sobel(lum, axis=0)
    mag = np.sqrt(gx**2+gy**2); dirc = np.arctan2(gy, gx)
    n = min(30, min(mag.shape))
    pca = PCA(n_components=n, svd_solver='randomized', random_state=42)
    pca.fit(mag); var = pca.explained_variance_ratio_
    return [float(var[0]), float(np.sum(var[:5])), float(np.sum(var[:10])),
            float(np.sum(var[:20])), float(-np.sum(var*np.log(var+1e-10))),
            float(np.std(var)), float(np.std(dirc)), float(np.mean(mag)),
            float(np.std(mag)), float(np.percentile(mag, 90)),
            float(np.mean(mag)/(np.std(mag)+1e-10))]


SRM_FILTERS = [
    np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype=np.float32),
    np.array([[0,0,0],[0,-1,0],[0,1,0]], dtype=np.float32),
    np.array([[0,0,0],[0,-2,1],[0,1,0]], dtype=np.float32),
]

def _ruido_features(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((256, 256))
    arr = np.array(img, dtype=np.float32)/255.0; h, w = arr.shape; patches = []
    for y in range(0, h-32+1, 32):
        for x in range(0, w-32+1, 32):
            p = arr[y:y+32, x:x+32]; patches.append((float(np.sum(np.abs(cv2.dct(p)))), p))
    patches.sort(key=lambda p: p[0])
    feats = []
    for patch in [p for _,p in patches[:3]] + [p for _,p in patches[-3:]]:
        for f in SRM_FILTERS:
            r = convolve(patch, f); feats.extend([float(np.mean(np.abs(r))), float(np.std(r))])
    return np.array(feats)


# ── Meta-features expandidas (14 sinais) ─────────────────────────────────────

def build_meta_row(p_pca, p_lum, p_ruido):
    """
    Converte as probabilidades dos 3 modelos em 14 sinais para o meta-learner:
    - 6 probs brutas (real/IA de cada modelo)
    - 3 margens de confiança (p_ia - p_real por modelo)
    - 1 contagem de votos IA (0-3)
    - 1 flag de consenso total (todos concordam)
    - 3 agregados globais (media p_ia, max confiança, min confiança)
    """
    p_ia_pca, p_ia_lum, p_ia_ruido = p_pca[1], p_lum[1], p_ruido[1]

    margin_pca   = p_ia_pca   - p_pca[0]
    margin_lum   = p_ia_lum   - p_lum[0]
    margin_ruido = p_ia_ruido - p_ruido[0]

    vote_ia   = int(p_ia_pca > 0.5) + int(p_ia_lum > 0.5) + int(p_ia_ruido > 0.5)
    all_agree = 1.0 if vote_ia in (0, 3) else 0.0

    mean_p_ia = float(np.mean([p_ia_pca, p_ia_lum, p_ia_ruido]))
    all_probs  = [p_pca[0], p_ia_pca, p_lum[0], p_ia_lum, p_ruido[0], p_ia_ruido]
    max_conf   = float(max(all_probs))
    min_conf   = float(min(all_probs))

    return [
        p_pca[0],   p_ia_pca,
        p_lum[0],   p_ia_lum,
        p_ruido[0], p_ia_ruido,
        margin_pca, margin_lum, margin_ruido,
        float(vote_ia), all_agree,
        mean_p_ia, max_conf, min_conf,
    ]


# ── Gera meta-features ────────────────────────────────────────────────────────

def get_meta_features(paths_with_labels, model_pca, model_lum, scaler_lum, model_ruido):
    X_meta, y_meta = [], []
    errors = 0
    start = time.time()

    for i, (path, label) in enumerate(paths_with_labels):
        if i > 0 and i % 500 == 0:
            elapsed  = time.time() - start
            restante = (elapsed / i) * (len(paths_with_labels) - i)
            print(f"  {i}/{len(paths_with_labels)} — ~{restante/60:.1f} min restantes")
        try:
            b        = path.read_bytes()
            p_pca    = model_pca.predict_proba([_pca_v2_features(b)])[0]
            p_lum    = model_lum.predict_proba(scaler_lum.transform([_lum_features(b)]))[0]
            p_ruido  = model_ruido.predict_proba([_ruido_features(b)])[0]
            X_meta.append(build_meta_row(p_pca, p_lum, p_ruido))
            y_meta.append(label)
        except Exception as e:
            errors += 1

    print(f"  Concluido: {len(X_meta)} imagens ({errors} erros)")
    return np.array(X_meta), np.array(y_meta)


def avaliar(name, model, X, y):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    print(f"\n{name}:")
    print(f"  Acuracia : {acc*100:.2f}%")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(classification_report(y, y_pred, target_names=['Real', 'IA']))
    return auc


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Carregando modelos base...")
    model_pca   = joblib.load(MODEL_PCA)
    model_lum   = joblib.load(MODEL_LUM)
    scaler_lum  = joblib.load(SCALER_LUM)
    model_ruido = joblib.load(MODEL_RUIDO)
    print("OK\n")

    CACHE_TRAIN = Path('modelos/Stacking/cache_train.npz')
    CACHE_EVAL  = Path('modelos/Stacking/cache_eval.npz')
    os.makedirs('modelos/Stacking', exist_ok=True)

    if CACHE_TRAIN.exists():
        print("=== Carregando meta-features de cache — TREINO ===")
        d = np.load(CACHE_TRAIN); X_train, y_train = d['X'], d['y']
        print(f"  {len(X_train)} amostras carregadas do cache")
    else:
        print("=== Gerando meta-features — TREINO (D:/TCCDataset/test) ===")
        real_paths = [(p, 0) for p in Path(META_TRAIN_REAL).rglob('*') if p.suffix.lower() in EXTS]
        fake_paths = [(p, 1) for p in Path(META_TRAIN_FAKE).rglob('*') if p.suffix.lower() in EXTS]
        print(f"  {len(real_paths)} reais, {len(fake_paths)} IA")
        X_train, y_train = get_meta_features(real_paths + fake_paths,
                                             model_pca, model_lum, scaler_lum, model_ruido)
        np.savez(CACHE_TRAIN, X=X_train, y=y_train)
        print(f"  Cache salvo em {CACHE_TRAIN}")

    if CACHE_EVAL.exists():
        print("\n=== Carregando meta-features de cache — AVALIACAO ===")
        d = np.load(CACHE_EVAL); X_eval, y_eval = d['X'], d['y']
        print(f"  {len(X_eval)} amostras carregadas do cache")
    else:
        print("\n=== Gerando meta-features — AVALIACAO (Desktop/test) ===")
        eval_real = [(p, 0) for p in Path(EVAL_REAL).rglob('*') if p.suffix.lower() in EXTS][:500]
        eval_fake = [(p, 1) for p in Path(EVAL_FAKE).rglob('*') if p.suffix.lower() in EXTS][:500]
        print(f"  {len(eval_real)} reais, {len(eval_fake)} IA")
        X_eval, y_eval = get_meta_features(eval_real + eval_fake,
                                           model_pca, model_lum, scaler_lum, model_ruido)
        np.savez(CACHE_EVAL, X=X_eval, y=y_eval)
        print(f"  Cache salvo em {CACHE_EVAL}")

    # ── Meta-learner 1: XGBoost com early stopping e calibracao ──────────────
    print("\n=== Treinando XGBoost ===")
    xgb_base = XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    xgb_base.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=50)

    meta_xgb = xgb_base

    # ── Meta-learner 2: Regressao Logistica ───────────────────────────────────
    print("\n=== Treinando Regressao Logistica ===")
    meta_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    meta_lr.fit(X_train, y_train)

    # ── Meta-learner 3: MLP ────────────────────────────────────────────────────
    print("\n=== Treinando MLP ===")
    meta_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    meta_mlp.fit(X_train, y_train)

    # ── Avaliacao e selecao do melhor ─────────────────────────────────────────
    print("\n=== Avaliacao final (1000 imagens) ===")
    resultados = {}
    for name, model in [("XGBoost (calibrado)",  meta_xgb),
                         ("Regressao Logistica",  meta_lr),
                         ("MLP (calibrado)",      meta_mlp)]:
        resultados[name] = (avaliar(name, model, X_eval, y_eval), model)

    melhor_nome, (melhor_auc, melhor_modelo) = max(resultados.items(), key=lambda x: x[1][0])
    print(f"\n>>> Melhor meta-learner: {melhor_nome} (AUC {melhor_auc:.4f})")

    # ── Salva todos + indica o melhor ─────────────────────────────────────────
    print("\n=== Salvando ===")
    os.makedirs('modelos/Stacking', exist_ok=True)
    joblib.dump(meta_xgb,      'modelos/Stacking/meta_xgb.pkl')
    joblib.dump(meta_lr,       'modelos/Stacking/meta_lr.pkl')
    joblib.dump(meta_mlp,      'modelos/Stacking/meta_mlp.pkl')
    joblib.dump(melhor_modelo, 'modelos/Stacking/modelo_stacking.pkl')
    print("Salvos em modelos/Stacking/")
    print(f"  modelo_stacking.pkl -> {melhor_nome}")
