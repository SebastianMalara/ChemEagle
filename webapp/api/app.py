from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .routers import config, experiments, review, runs, uploads


def _frontend_dist() -> Path:
    return Path(__file__).resolve().parents[1] / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="ChemEagle Web App", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8000",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(config.router, prefix="/api")
    app.include_router(uploads.router, prefix="/api")
    app.include_router(experiments.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")
    app.include_router(review.router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        dist = _frontend_dist()
        index = dist / "index.html"
        if not dist.exists() or not index.exists():
            return JSONResponse(
                status_code=503,
                content={
                    "status": "frontend_not_built",
                    "message": "Frontend assets were not found. Run the Vite build or use the dev server for the SPA.",
                },
            )

        requested = (dist / full_path).resolve()
        if full_path and requested.exists() and dist in requested.parents:
            return FileResponse(requested)
        return FileResponse(index)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.api.app:app", host="127.0.0.1", port=8000, reload=True)
