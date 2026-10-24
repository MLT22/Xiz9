from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.bronze_controller import bronze_router
from controller.prata_controller import prata_router
from controller.analise_controller import analise_router
from controller.drive_controller import drive_router
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="Image Metadata API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/site", StaticFiles(directory="site", html=True), name="site")
app.include_router(bronze_router)
app.include_router(prata_router)
app.include_router(analise_router)
app.include_router(drive_router)