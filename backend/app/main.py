from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.config import settings
from app.errors import StageError
from app.train import build_models
from app.vocab import VocabIndex


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Thread count is set on the onnxruntime SessionOptions in app.embed, not
    # here -- there is no global "set threads" call for onnxruntime the way
    # torch.set_num_threads() was a one-time process-wide setting.

    app.state.vocab_index = VocabIndex()  # embeds the 600-word vocabulary once (~2s)

    # LOADS the live classifier plus the scored supervised/unsupervised figures
    # from the artifact scripts/fit_models.py fit once at container build time --
    # no LogisticRegression/KMeans/PCA fit happens here. See app.train.build_models.
    app.state.trained_models = build_models(settings.seed)

    yield


app = FastAPI(title="embed-lab", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StageError)
async def stage_error_handler(request: Request, exc: StageError) -> JSONResponse:
    """A user-fixable input error is a 409 with a hint, never a stack trace."""
    return JSONResponse(
        status_code=409, content={"detail": exc.detail, "hint": exc.hint}
    )


app.include_router(router)
