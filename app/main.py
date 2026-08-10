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


@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Embedded AI Copilot is running"}

@app.post("/ask")
async def process_prompt(request: AskRequest):
    logger.info("Ask endpoint accessed.")
    try:
        answer = ai_service.generate_answer(request.question)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return {"error": "Invalid request format."}
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {"error": "Gemini could not generate an answer."}

    return {"answer": answer}