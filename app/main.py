from fastapi import FastAPI
from app.api.v1.routes import router as api_router
from app.core.config import Config

app = FastAPI(
    title=Config.TITLE,
    version=Config.VERSION,
    description=Config.DESCRIPTION
)

app.include_router(api_router, prefix="/api/v1")
