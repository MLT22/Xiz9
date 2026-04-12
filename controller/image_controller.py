from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from service.image_service import ImageService
 
router = APIRouter(prefix="/image", tags=["Image"])
 
 
@router.post("/metadata")
async def get_image_metadata(file: UploadFile = File(...)):
    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
 
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: '{file.content_type}'. Tipos aceitos: {ALLOWED_TYPES}"
        )
 
    contents = await file.read()
 
    try:
        metadata = ImageService.extract_metadata(file.filename, file.content_type, contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
    return JSONResponse(content=metadata)

def __init__():
    return