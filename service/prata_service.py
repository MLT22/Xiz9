import io
import json
import threading

import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import UploadFile
from PIL import Image

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class PrataService:

    # ── EfficientNet_V2S (CNN binária: real=0 / IA=1) ───────────────────────────

    _EFFICIENTNET_V2S_DIR = 'modelos/CNN/EfficientNet_V2S'
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _load_lock = threading.Lock()

    _efficientnet_v2s_model = None
    _efficientnet_v2s_transform = None
    _efficientnet_v2s_threshold = None

    @staticmethod
    def _build_efficientnet_v2s(input_size: int) -> nn.Module:
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

        state_dict = torch.load(
            f'{PrataService._EFFICIENTNET_V2S_DIR}/best_model.pt',
            map_location=PrataService._device,
        )
        model.load_state_dict(state_dict)
        model.to(PrataService._device)
        model.eval()
        return model

    @staticmethod
    def _get_efficientnet_v2s():
        """Carrega modelo/transform/threshold uma única vez e cacheia entre requisições."""
        if PrataService._efficientnet_v2s_model is None:
            with PrataService._load_lock:
                if PrataService._efficientnet_v2s_model is None:
                    with open(f'{PrataService._EFFICIENTNET_V2S_DIR}/metadata.json') as f:
                        metadata = json.load(f)
                    input_size = metadata['input_size']

                    PrataService._efficientnet_v2s_model = PrataService._build_efficientnet_v2s(input_size)
                    PrataService._efficientnet_v2s_transform = transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
                    ])
                    PrataService._efficientnet_v2s_threshold = metadata['threshold']

        return (
            PrataService._efficientnet_v2s_model,
            PrataService._efficientnet_v2s_transform,
            PrataService._efficientnet_v2s_threshold,
        )

    @staticmethod
    async def efficientnet_v2s(file: UploadFile):
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception:
            raise ValueError("Não foi possível processar o arquivo como imagem.")

        model, transform, threshold = PrataService._get_efficientnet_v2s()
        tensor = transform(image).unsqueeze(0).to(PrataService._device)

        with torch.no_grad():
            prob_ia = float(model(tensor).squeeze(1).sigmoid().item())

        prob_real = 1.0 - prob_ia
        prediction = 1 if prob_ia >= threshold else 0
        confidence = prob_ia if prediction == 1 else prob_real
        label = "IA" if prediction == 1 else "REAL"
        status = "CONCLUSIVO" if confidence >= threshold else "INCERTO — passar para próximo check"

        return {
            "label": label, "confidence": round(confidence, 4), "status": status,
            "prob_real": round(prob_real, 4), "prob_ia": round(prob_ia, 4),
        }
