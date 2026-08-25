from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from service.analise_service import AnaliseService

analise_router = APIRouter(prefix="/Xiz9", tags=["Analise"])
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}


def _validate(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo não suportado: '{file.content_type}'. Aceitos: {ALLOWED_TYPES}",
        )


@analise_router.post("/analise")
async def analise(file: UploadFile = File(...)):
    _validate(file)
    try:
        result = await AnaliseService.classificar(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)
