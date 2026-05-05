from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from service.bronze_service import BronzeService
 
bronze_router = APIRouter(prefix="/bronze", tags=["Image"])
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
 
@bronze_router.post("/metadata")
async def get_image_metadata(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: '{file.content_type}'. Tipos aceitos: {ALLOWED_TYPES}"
        )
 
    contents = await file.read()
 
    try:
        metadata = BronzeService.extract_metadata(file.filename, file.content_type, contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    return JSONResponse(content=metadata)

@bronze_router.post("/PCA_anomaly")
async def pca_anomaly(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: '{file.content_type}'. Tipos aceitos: {ALLOWED_TYPES}"
        )
    try:
        result = await BronzeService.pca_anomaly(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=result)


def __init__():
    return