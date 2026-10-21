from fastapi import UploadFile

from service.prata_service import PrataService

INCERTO_THRESHOLD = 0.75


class AnaliseService:
    """Classifica a imagem usando a camada Prata (CNN EfficientNet_V2S)."""

    @staticmethod
    async def classificar(file: UploadFile):
        contents = await file.read()
        prata = PrataService.predict_bytes(contents)

        veredito = "INCERTO" if prata["confidence"] < INCERTO_THRESHOLD else prata["label"]

        return {
            "label": prata["label"],
            "confidence": prata["confidence"]
            
        }
