"""FastAPI application entry point -- port 8011."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api.protocol import router as protocol_router
from backend.api.analysis import router as analysis_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Target Trial Emulation Framework for TCM using Real-World Evidence",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(protocol_router, prefix="/api/protocols", tags=["protocols"])
    app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.version}

    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=settings.debug)
