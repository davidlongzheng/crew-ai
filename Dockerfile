FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake
COPY ./src ./src
COPY ./build.sh .
COPY CMakeLists.txt .
RUN ./build.sh
COPY ./requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

FROM python:3.12-slim 
WORKDIR /app

COPY ./src ./src
COPY --from=builder /app/build/cpp_game.* ./build/
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ./models ./models
ENV PORT=8080
# Necessary hack :(. Somehow, the cpp_game isn't properly
# installing otherwise.
ENV PYTHONPATH=/app/build
WORKDIR /app/src
RUN python -c "from backend.app.routers import game, main; from backend.app.middleware import *; import cpp_game; cpp_game.Engine; cpp_game.Rng; from ai.tests.test_ai import *; test_ai_batch_rollout()"
CMD exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}