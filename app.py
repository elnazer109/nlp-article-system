from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
import uuid
import os

app = FastAPI()

# Load model once on startup (faster)
summarizer = pipeline("summarization", model="t5-small")

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 50

@app.post("/summarize")
def summarize_text(req: SummaryRequest):
    summary = summarizer(
        req.text,
        max_length=req.max_length,
        min_length=req.min_length,
        do_sample=False
    )
    return {"summary": summary[0]["summary_text"]}





# Load model once at startup
tts_pipeline = pipeline("text-to-speech", model="suno/bark-small")

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
def generate_tts(req: TTSRequest):
    # Generate audio
    output = tts_pipeline(req.text)

    # Create a unique filename
    file_name = f"tts_{uuid.uuid4().hex}.wav"

    # Save the audio
    with open(file_name, "wb") as f:
        f.write(output["audio"].tobytes())

    return FileResponse(
        path=file_name,
        media_type="audio/wav",
        filename="output.wav"
    )
