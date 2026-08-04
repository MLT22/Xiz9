from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from service.bronze_service import BronzeService
from service.freq_cor_service import FreqCorService
from service.luminance_service import LuminanceService
from service.metadata_service import MetadataService
from service.ruido_service import RuidoService

bronze_router = APIRouter(prefix="/bronze", tags=["Image"])
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}


def _validate(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo não suportado: '{file.content_type}'. Aceitos: {ALLOWED_TYPES}",
        )


@bronze_router.post("/metadata")
async def get_image_metadata(file: UploadFile = File(...)):
    _validate(file)
    contents = await file.read()
    try:
        metadata = MetadataService.extract(file.filename, file.content_type, contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=metadata)


@bronze_router.post("/frequencia_cor")
async def freq_cor(file: UploadFile = File(...)):
    _validate(file)
    try:
        result = await FreqCorService.analyze(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)


@bronze_router.post("/luminescencia")
async def luminescencia(file: UploadFile = File(...)):
    _validate(file)
    try:
        result = await LuminanceService.analyze(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)

@bronze_router.post("/ruido")
async def ruido(file: UploadFile = File(...)):
    _validate(file)
    try:
        result = await RuidoService.analyze(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)


@bronze_router.post("/assembly")
async def assembly(file: UploadFile = File(...)):
    _validate(file)
    try:
        result = await BronzeService.assembly(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)


@bronze_router.post("/avaliacao_geral")
async def avaliacao_geral(file: UploadFile = File(...)):
    _validate(file)
    try:
        result = await BronzeService.avaliacao_geral(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)
