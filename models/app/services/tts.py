import uuid

from app.models.load_models import tts_model

def generate_tts(text):
    output = tts_model(text)

    file_name = f"tts_{uuid.uuid4().hex}.wav"

    with open(file_name, "wb") as f:
        f.write(output["audio"].tobytes())

    return file_name
