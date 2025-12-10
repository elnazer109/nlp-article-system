from app.models.load_models import summarizer

def summarize_text(text, max_length=150, min_length=50):
    output = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return output[0]["summary_text"]
