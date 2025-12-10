from fastapi import FastAPI
from app.services.summarizer import summarize_text
from app.services.tts import generate_tts
from pydantic import BaseModel

app = FastAPI()

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 50

@app.post("/summarize")
def summarize(req: SummaryRequest):
    return {"summary": summarize_text(req.text, req.max_length, req.min_length)}
