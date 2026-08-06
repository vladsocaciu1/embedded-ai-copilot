from fastapi import FastAPI
from pydantic import BaseModel
from app import ai_service
from pydantic import ValidationError

app = FastAPI()

class AskRequest(BaseModel):
    question: str


@app.get("/")
async def read_root():
    return {"message": "Embedded AI Copilot is running"}

@app.post("/ask")
async def process_prompt(request: AskRequest):
    try:
        answer = ai_service.generate_answer(request.question)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return {"error": "Invalid request format."}
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {"error": "Gemini could not generate an answer."}

    return {"answer": answer}