import logging

from app.logging_config import setup_logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ai_service runs load_settings() at import; configure logging first
setup_logging()
from app import ai_service  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI()

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
        answer = ai_service.generate_answer(request.question)
    except ai_service.ProjectLoadError:
        logger.exception("Project files could not be loaded.")
        raise HTTPException(status_code=500, detail="Project files could not be loaded.")
    except ai_service.UpstreamRejectedError:
        logger.exception("AI provider rejected the request.")
        raise HTTPException(status_code=502, detail="AI provider rejected the request.")
    except ai_service.UpstreamUnavailableError:
        logger.exception("AI provider is unavailable.")
        raise HTTPException(status_code=503, detail="AI provider is unavailable.")

    return {"answer": answer}
