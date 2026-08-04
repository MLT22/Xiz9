from fastapi import UploadFile

from service.freq_cor_service import FreqCorService
from service.luminance_service import LuminanceService
from service.metadata_service import MetadataService
from service.ruido_service import RuidoService


class BronzeService:
    """Orquestra metadados + os 3 modelos de detecção em respostas combinadas."""

    @staticmethod
    def _model_result(pred, proba):
        conf = float(max(proba))
        label = "IA" if pred == 1 else "REAL"
        return {
            "label": label, "confidence": conf,
            "prob_real": float(proba[0]), "prob_ia": float(proba[1]),
        }

    @staticmethod
    async def assembly(file: UploadFile):
        THRESHOLD = 0.80

        contents = await file.read()
        meta_check = MetadataService.check(contents)

        fc_proba, fc_threshold = FreqCorService.predict(contents)
        fc_pred = int(fc_proba[1] >= fc_threshold)

        lum_proba, lum_threshold = LuminanceService.predict(contents)
        lum_pred = int(lum_proba[1] >= lum_threshold)

        # Ruído usa o corte padrão de 0.5 (equivalente ao model.predict() nativo),
        # não o threshold calibrado em threshold.json usado por RuidoService.analyze.
        ruido_proba, _ = RuidoService.predict(contents)
        ruido_pred = int(ruido_proba[1] >= 0.5)

        # Ensemble — média simples das probabilidades
        prob_ia = float((fc_proba[1] + lum_proba[1] + ruido_proba[1]) / 3)
        prob_real = float((fc_proba[0] + lum_proba[0] + ruido_proba[0]) / 3)
        ensemble_pred = 1 if prob_ia >= 0.5 else 0
        ensemble_conf = float(max(prob_ia, prob_real))
        ensemble_label = "IA" if ensemble_pred == 1 else "REAL"

        return {
            "metadata": meta_check,
            "modelos": {
                "frequencia_cor": BronzeService._model_result(fc_pred, fc_proba),
                "luminescencia": BronzeService._model_result(lum_pred, lum_proba),
                "ruido": BronzeService._model_result(ruido_pred, ruido_proba),
            },
            "ensemble": {
                "label": ensemble_label,
                "confidence": ensemble_conf,
                "prob_real": prob_real,
                "prob_ia": prob_ia,
                "status": "CONCLUSIVO" if ensemble_conf >= THRESHOLD else "INCERTO",
            },
        }

    @staticmethod
    async def avaliacao_geral(file: UploadFile):
        CONCLUSIVO_THRESHOLD = 0.75

        contents = await file.read()
        meta_check = MetadataService.check(contents)

        # Modelos base — probabilidades calibradas
        fc_proba, _ = FreqCorService.predict(contents)
        lum_proba, _ = LuminanceService.predict(contents)
        ruido_proba, _ = RuidoService.predict(contents)

        # Consenso — média das probabilidades dos 3 modelos base
        prob_ia = round(float((fc_proba[1] + lum_proba[1] + ruido_proba[1]) / 3), 4)
        prob_real = round(float((fc_proba[0] + lum_proba[0] + ruido_proba[0]) / 3), 4)
        label = "IA" if prob_ia >= 0.5 else "REAL"
        confidence = prob_ia if label == "IA" else prob_real
        escalate = confidence < CONCLUSIVO_THRESHOLD

        # Metadados com indicadores de IA são sempre conclusivos
        if meta_check.get("has_ai_indicators"):
            label = "IA"
            confidence = 0.99
            escalate = False

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
            "metadata": meta_check,
            "modelos_base": {
                "frequencia_cor": _base_result(fc_proba),
                "luminescencia": _base_result(lum_proba),
                "ruido": _base_result(ruido_proba),
            },
        }
