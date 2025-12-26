 NLP API 🚀

A production-ready NLP API built with FastAPI.  
Includes:

- Text Summarization (T5-small)
- Text to Speech (Bark-small)

---

## 📌 Installation

```bash
pip install -r requirements.txt
````

---

## ▶ Run the API

```bash
uvicorn app.main:app --reload
```

Visit docs:

```
http://127.0.0.1:8000/docs
```

---

## 📂 API Routes

### **POST /summarize**

Request:

```json
{
  "text": "Long text here...",
  "max_length": 150,
  "min_length": 50
}
```

Response:

```json
{
  "summary": "Shorter version of the text..."
}
```

---

### **POST /tts**

Request:

```json
{
  "text": "Hello, this is a test."
}
```

Response:

```json
{
  "audio_file": "tts_xxxxx.wav"
}
```

---

## 🛠 Project Structure

```
nlp-api/
├── app/
│   ├── main.py
│   ├── models/load_models.py
│   └── services/
│       ├── summarizer.py
│       └── tts.py
├── requirements.txt
├── README.md
└── .gitignore
```

---



Just copy these files into a folder, push to GitHub, and your project is ready! If you want, I can also generate a **Dockerfile**, **client examples**, or **CI/CD pipeline**.
```
