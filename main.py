from fastapi import FastAPI
from controller.bronze_controller import bronze_router
 
app = FastAPI(title="Image Metadata API")
 
app.include_router(bronze_router)