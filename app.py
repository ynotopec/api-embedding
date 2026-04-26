#!/usr/bin/env python3
import os
import time
from typing import Any, List, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from FlagEmbedding import BGEM3FlagModel


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

MODEL_ID = os.getenv("MODEL_ID", "BAAI/bge-m3")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", MODEL_ID)
API_KEY = os.getenv("API_KEY", "")

MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
USE_FP16 = os.getenv("USE_FP16", "1") == "1"
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

RETURN_SPARSE = os.getenv("RETURN_SPARSE", "0") == "1"
RETURN_COLBERT = os.getenv("RETURN_COLBERT", "0") == "1"

app = FastAPI(title="BGE-M3 OpenAI-compatible Embeddings API")

print(f"Loading {MODEL_ID} on {DEVICE}...")
model = BGEM3FlagModel(
    MODEL_ID,
    use_fp16=USE_FP16,
    device=DEVICE,
)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = Field(default=SERVED_MODEL_NAME)
    encoding_format: str = Field(default="float")


def check_auth(
    authorization: str | None,
    x_api_key: str | None,
) -> None:
    if not API_KEY:
        return

    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    if bearer == API_KEY or x_api_key == API_KEY:
        return

    raise HTTPException(status_code=401, detail="Unauthorized")


def normalize_inputs(value: Union[str, List[str]]) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise HTTPException(status_code=400, detail="input must be string or list[string]")


def estimate_tokens(texts: List[str]) -> int:
    # Approximation simple et stable si tokenizer interne non exposé.
    return sum(max(1, len(t) // 4) for t in texts)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "BGE-M3 embeddings API",
        "model": SERVED_MODEL_NAME,
        "endpoint": "/v1/embeddings",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "flagembedding",
        "model": SERVED_MODEL_NAME,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
    }


@app.get("/v1/models")
def list_models(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    check_auth(authorization, x_api_key)
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
def embeddings(
    req: EmbeddingRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    check_auth(authorization, x_api_key)

    texts = normalize_inputs(req.input)
    started = time.time()

    try:
        out = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            max_length=MAX_MODEL_LEN,
            return_dense=True,
            return_sparse=RETURN_SPARSE,
            return_colbert_vecs=RETURN_COLBERT,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}") from e

    dense = out["dense_vecs"]
    dense = np.asarray(dense, dtype=np.float32)

    data = []
    for i, vec in enumerate(dense):
        item: dict[str, Any] = {
            "object": "embedding",
            "index": i,
            "embedding": vec.tolist(),
        }

        # Extensions non OpenAI, désactivées par défaut.
        if RETURN_SPARSE and "lexical_weights" in out:
            item["sparse_embedding"] = out["lexical_weights"][i]

        if RETURN_COLBERT and "colbert_vecs" in out:
            item["colbert_vecs"] = np.asarray(out["colbert_vecs"][i], dtype=np.float32).tolist()

        data.append(item)

    prompt_tokens = estimate_tokens(texts)

    return {
        "object": "list",
        "data": data,
        "model": SERVED_MODEL_NAME,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
        "backend": "flagembedding",
        "elapsed": round(time.time() - started, 4),
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info"),
        workers=1,
    )
