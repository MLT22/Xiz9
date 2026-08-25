from fastapi import UploadFile

from service.bronze_service import BronzeService
from service.prata_service import PrataService


class AnaliseService:
    """Orquestra a cascata bronze -> prata: só escala para a prata quando o bronze fica incerto."""

    @staticmethod
    async def classificar(file: UploadFile):
        contents = await file.read()
        bronze = BronzeService.avaliacao_geral_bytes(contents)

        if bronze["escalate"]:
            prata = PrataService.predict_bytes(contents)
            camada_decisiva = "prata"
            label = prata["label"]
            confidence = prata["confidence"]
        else:
            prata = None
            camada_decisiva = "bronze"
            label = bronze["label"]
            confidence = bronze["confidence"]

        return {
            "label": label,
            "confidence": confidence,
            "camada_decisiva": camada_decisiva,
            "bronze": bronze,
            "prata": prata,
        }
