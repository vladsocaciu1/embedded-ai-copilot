import logging
import threading
from contextlib import asynccontextmanager

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

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Embedded AI Copilot is running"}

@app.post("/ask")
async def process_prompt(request: AskRequest):
    logger.info("Ask endpoint accessed.")
    try:
        result = ai_service.generate_answer(request.question)
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
    except ai_service.UpstreamRejectedError:
        logger.exception("AI provider rejected the request.")
        raise HTTPException(status_code=502, detail="AI provider rejected the request.")
    except ai_service.UpstreamUnavailableError:
        logger.exception("AI provider is unavailable.")
        raise HTTPException(status_code=503, detail="AI provider is unavailable.")

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
