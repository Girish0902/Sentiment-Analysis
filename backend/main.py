"""FastAPI application: text and image sentiment endpoints."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend import engine, image_engine

app = FastAPI(
    title="Sentiment Lab API",
    version=engine.MODEL_VERSION,
    description="Text and image sentiment analysis for the Sentiment Lab project.",
)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(engine.InputError)
async def _input_error(_request: Request, exc: engine.InputError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(engine.ModelUnavailableError)
async def _model_error(_request: Request, exc: engine.ModelUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        raise engine.InputError("The request body is not valid JSON.") from None
    if not isinstance(payload, dict):
        raise engine.InputError("The request body must be a JSON object.")
    return payload


@app.get("/api/health")
def health() -> dict[str, Any]:
    model = engine.load_model()
    return {"status": "ok", "modelVersion": model["version"], "engine": "python"}


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    model = engine.load_model()
    return {
        "version": model["version"],
        "classes": model["classes"],
        "training": model["training"],
        "report": model["report"],
        "limits": {
            "maxTexts": engine.MAX_ITEMS,
            "maxChars": engine.MAX_CHARS,
            "maxCsvBytes": engine.MAX_CSV_BYTES,
        },
    }


@app.post("/api/analyze")
async def analyze(request: Request) -> dict[str, Any]:
    payload = await _json_body(request)
    return {"results": engine.analyze_items(payload.get("items"))}


@app.post("/api/evaluate")
async def evaluate(request: Request) -> dict[str, Any]:
    payload = await _json_body(request)
    return engine.evaluate_items(payload.get("items"))


@app.post("/api/analyze-image")
async def analyze_image(request: Request) -> dict[str, Any]:
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    body = await request.body()
    if content_type not in image_engine.SUPPORTED_TYPES:
        return JSONResponse(status_code=415, content={"detail": "Use a JPG, PNG or WebP image."})
    if len(body) > image_engine.MAX_BYTES:
        return JSONResponse(status_code=413, content={"detail": "The image is larger than 10 MB."})
    if not body:
        return JSONResponse(status_code=422, content={"detail": "The image body is empty."})
    try:
        result = await run_in_threadpool(image_engine.analyze_image_bytes, body, content_type)
    except image_engine.ImageValidationError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})
    except image_engine.ImageModelUnavailable as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    return {"result": result}
