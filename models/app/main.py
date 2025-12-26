from fastapi import FastAPI
from pydantic import BaseModel
from app.services.summarizer import summarize_text
from app.services.tts import generate_tts

app = FastAPI(
    title="NLP API",
    description="API for text summarization and text-to-speech",
    version="1.0.0"
)

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 50

@app.post("/summarize")
def summarize(req: SummaryRequest):
    summary = summarize_text(req.text, req.max_length, req.min_length)
    return {"summary": summary}

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def tts(req: TTSRequest):
    file_path = generate_tts(req.text)
    return {"audio_file": file_path}
