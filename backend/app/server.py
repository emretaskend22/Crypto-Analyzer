from fastapi import FastAPI
from .api.system import system_router as api_router

def create_app() -> FastAPI:
    app = FastAPI(title="Crypto Analyzer")
    app.include_router(api_router)
    return app