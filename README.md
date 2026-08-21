# Adaptive Voice Interview Engine 🎙️🤖

An intelligent, full-stack AI-driven technical mock interview platform that dynamically adapts its questioning based on candidate responses, provides real-time voice interaction, and generates an automated technical evaluation scorecard with PDF export capabilities.

---

## 🌟 Key Features

* **Adaptive Questioning Engine:** Dynamically generates contextual, non-repetitive follow-up technical questions based on the candidate's actual answer depth.
* **Speech-to-Text (STT):** Transcribes spoken answers using high-speed Groq Whisper (`whisper-large-v3`).
* **Text-to-Speech (TTS):** Converts AI questions into realistic voice audio streams (`gTTS`) for an authentic interview experience.
* **Intelligent Evaluation & Scoring:** Analyzes the full interview conversation transcript to score Technical Accuracy, Communication Clarity, and Depth of Knowledge (0-100).
* **Detailed Feedback & Export:** Provides actionable strengths, areas for improvement, and instant PDF report download.

---

## 🛠️ Tech Stack & Architecture

* **Backend:** FastAPI (Python 3.10+)
* **AI & LLM Orchestration:** Groq Cloud API (`openai/gpt-oss-20b` & `whisper-large-v3`)
* **Audio Processing:** gTTS (Google Text-to-Speech), `python-multipart`
* **Data Validation:** Pydantic V2
* **Frontend:** React / Modern Web UI (Fetch API, Web Audio API, HTML5 Canvas PDF generation)

---

## 📂 Project Structure

```text
├── main.py              # FastAPI server, endpoints, Groq LLM & voice pipelines
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (GROQ_API_KEY)
└── README.md            # Project documentation