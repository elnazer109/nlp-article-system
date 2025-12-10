from transformers import pipeline

print("Loading summarizer model...")
summarizer = pipeline("summarization", model="t5-small")

print("Loading TTS model...")
tts_model = pipeline("text-to-speech", model="suno/bark-small")
