from __future__ import annotations

from fastapi import FastAPI

from backend.app.routes.risk import router as risk_router


def create_app() -> FastAPI:
    app = FastAPI(title="EC-CAD API", version="0.1.0")
    app.include_router(risk_router, prefix="/api")
    return app


app = create_app()

