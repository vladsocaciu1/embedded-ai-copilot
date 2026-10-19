import logging
import threading
from contextlib import asynccontextmanager, contextmanager

from app.logging_config import setup_logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ai_service runs load_settings() at import; configure logging first
setup_logging()
from app import ai_service  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(
        target=ai_service.warm_up, name="index-warm-up", daemon=True
    ).start()
    yield


app = FastAPI(lifespan=lifespan)

class AskRequest(BaseModel):
    question: str = Field(min_length=1, pattern=r"\S")

logger.info("FastAPI application initialized.")

CONTEXT_RANKED_LIMIT = 50


@contextmanager
def _service_errors():
    try:
        yield
    except ai_service.ProjectLoadError:
        logger.exception("Project files could not be loaded.")
        raise HTTPException(status_code=500, detail="Project files could not be loaded.")
    except ai_service.ProjectTooLargeError as error:
        logger.warning(f"Prompt is {error.token_count} tokens, limit {error.limit}.")
        if error.estimated:
            size = f"at least {error.token_count} tokens (estimated)"
        else:
            size = f"{error.token_count} tokens"
        raise HTTPException(
            status_code=413,
            detail=(
                "Project is too large for the prompt budget: "
                f"{size}, limit {error.limit}. "
                "Point PROJECT_PATH at a smaller directory or raise MAX_PROMPT_TOKENS."
            ),
        )
    except ai_service.RateLimitedError as error:
        logger.warning(
            "AI provider rate limit reached. "
            f"Retry after {error.retry_after} seconds."
        )
        raise HTTPException(
            status_code=429,
            detail=(
                "AI provider rate limit reached. "
                f"Retry after {error.retry_after} seconds."
            ),
            headers={"Retry-After": str(error.retry_after)},
        )
    except ai_service.UpstreamRejectedError:
        logger.exception("AI provider rejected the request.")
        raise HTTPException(status_code=502, detail="AI provider rejected the request.")
    except ai_service.UpstreamUnavailableError:
        logger.exception("AI provider is unavailable.")
        raise HTTPException(status_code=503, detail="AI provider is unavailable.")


@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Embedded AI Copilot is running"}

@app.post("/ask")
async def process_prompt(request: AskRequest):
    logger.info("Ask endpoint accessed.")
    with _service_errors():
        result = ai_service.generate_answer(request.question)

    return {
        "answer": result.text,
        "context": {
            "mode": result.context.mode,
            "files_total": result.context.files_total,
            "files_included": list(result.context.files_included),
            "prompt_tokens": result.context.prompt_tokens,
            "map_tier": result.context.map_tier,
        },
    }


@app.post("/context")
async def context(request: AskRequest, exact: bool = False):
    logger.info("Context endpoint accessed.")
    with _service_errors():
        prepared = ai_service.dry_run(request.question, exact=exact)
    included = set(prepared.included)
    return {
        "mode": prepared.mode,
        "limit": prepared.limit,
        "prompt_tokens": prepared.token_count,
        "token_source": prepared.token_source,
        "prompt_chars": len(prepared.prompt),
        "files_total": prepared.files_total,
        "files_included": [path.as_posix() for path in prepared.included],
        "map_tier": prepared.map_tier,
        "map_files": prepared.map_files,
        "ranked_total": len(prepared.ranked),
        "ranked": [
            {
                "path": item.path.as_posix(),
                "reason": item.reason,
                "included": item.path in included,
                "focus": item.focus,
                "tier": item.tier,
                "hits": item.hits,
                "lexical": item.lexical,
            }
            for item in prepared.ranked[:CONTEXT_RANKED_LIMIT]
        ],
    }
