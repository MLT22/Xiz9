import sys
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import convolve
import joblib

MODEL_PATH = 'modelospkl/modelo_ruido.pkl'
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


def extract_features(image_path):
    img = Image.open(image_path).convert("L").resize(RESIZE_TO)
    arr = np.array(img, dtype=np.float32) / 255.0

    h, w = arr.shape
    patches = []

    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = arr[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            score = dct_score(patch)
            patches.append((score, patch))

    if not patches:
        return None

    patches.sort(key=lambda p: p[0])
    low_patches  = [p for _, p in patches[:N_PATCHES]]
    high_patches = [p for _, p in patches[-N_PATCHES:]]

    features = []
    for patch in low_patches + high_patches:
        features.extend(apply_srm(patch))

    return np.array(features)


def testar(image_path):
    print(f"\nImagem: {image_path}")

    features = extract_features(image_path)
    if features is None:
        print("Erro: não foi possível extrair features.")
        return

    model      = joblib.load(MODEL_PATH)
    prediction = int(model.predict([features])[0])
    proba      = model.predict_proba([features])[0]
    confianca  = float(max(proba))
    label      = "IA" if prediction == 1 else "REAL"
    status     = "CONCLUSIVO" if confianca >= 0.80 else "INCERTO"

    print(f"Resultado:   {label}")
    print(f"Confiança:   {confianca*100:.2f}%")
    print(f"Status:      {status}")
    print(f"Prob. Real:  {proba[0]*100:.2f}%")
    print(f"Prob. IA:    {proba[1]*100:.2f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python testar_imagem.py <imagem1> <imagem2> ...")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            testar(path)
        except Exception as e:
            print(f"\nErro ao processar '{path}': {e}")
        print("-" * 40)