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
    # Any origin. This is a public, read-only, unauthenticated API: it holds
    # no user data, sets no cookies, and every endpoint is a GET that returns
    # the same thing to everybody. An allowlist would only have to be edited
    # each time the app is hosted somewhere new.
    #
    # allow_credentials MUST be False alongside "*": the CORS spec forbids
    # the wildcard with credentials, and browsers reject the pair outright --
    # which is subtle, because curl ignores CORS completely, so the API can
    # look perfectly healthy from a terminal while every browser is blocked.
    # That is exactly how this was missed: curl said 200, the deployed page
    # showed "can't reach the API".
    allow_origins=["*"],
    allow_credentials=False,
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
