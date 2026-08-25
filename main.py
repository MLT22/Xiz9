from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.bronze_controller import bronze_router
from controller.prata_controller import prata_router
from controller.analise_controller import analise_router

app = FastAPI(title="Image Metadata API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(bronze_router)
app.include_router(prata_router)
app.include_router(analise_router)