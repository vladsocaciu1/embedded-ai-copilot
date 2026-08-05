from fastapi import FastAPI
from pydantic import BaseModel
from app import ai_service

app = FastAPI()

class AskRequest(BaseModel):
    question: str


@app.get("/")
async def read_root():
    return {"message": "Embedded AI Copilot is running"}

@app.post("/ask")
async def process_prompt(request: AskRequest):
    answer = ai_service.generate_answer(request.question)
    return {"answer": answer}