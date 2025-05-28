from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import config


def setup_cors(app: FastAPI) -> None:
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allow_origins,
        allow_credentials=True,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
    )
