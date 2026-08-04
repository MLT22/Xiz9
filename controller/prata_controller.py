from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from service.prata_service import PrataService

prata_router = APIRouter(prefix="/prata", tags=["Image"])
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}


def _validate(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo não suportado: '{file.content_type}'. Aceitos: {ALLOWED_TYPES}",
        )


@prata_router.post("/efficientnet_v2s")
async def efficientnet_v2s(file: UploadFile = File(...)):
    _validate(file)
    try:
        result = await PrataService.efficientnet_v2s(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)
