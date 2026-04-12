from fastapi import FastAPI
from controller.image_controller import router
 
app = FastAPI(title="Image Metadata API")
 
app.include_router(router)