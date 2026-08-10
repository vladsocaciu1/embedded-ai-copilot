import logging

from app.logging_config import setup_logging
from fastapi import FastAPI
from pydantic import BaseModel
from app import ai_service
from pydantic import ValidationError

# Set up and initialize logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

class AskRequest(BaseModel):
    question: str

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
    except ValidationError:
        logger.exception(f"Validation error.")
        return {"error": "Invalid request format."}
    except Exception:
        logger.exception(f"Error generating answer.")
        return {"error": "Gemini could not generate an answer."}

    return {"answer": answer}