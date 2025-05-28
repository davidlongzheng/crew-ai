from fastapi import FastAPI

from backend.app.const import OPEN_API_DESCRIPTION, OPEN_API_TITLE
from backend.app.middleware import setup_cors
from backend.app.routers import game, main

app = FastAPI(
    title=OPEN_API_TITLE,
    description=OPEN_API_DESCRIPTION,
    version="0.0.1",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)
setup_cors(app)

app.include_router(main.router)
app.include_router(game.router)
