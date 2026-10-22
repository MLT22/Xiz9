from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response

from service.drive_service import DriveService

drive_router = APIRouter(prefix="/drive", tags=["Drive"])


@drive_router.get("/imagens")
async def listar_imagens():
    try:
        imagens = DriveService.listar_imagens()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro ao consultar o Drive: {e}")
    return JSONResponse(content={"imagens": imagens, "total": len(imagens)})


@drive_router.get("/imagem/{file_id}")
async def baixar_imagem(file_id: str):
    try:
        conteudo, mime_type = DriveService.baixar_imagem(file_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro ao baixar do Drive: {e}")
    return Response(content=conteudo, media_type=mime_type)