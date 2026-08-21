import io
import json
import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from gtts import gTTS
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Adaptive Voice Interview Engine")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Automatically discover active chat models from your Groq account
def get_active_model():
    preferred_order = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        "qwen-2.5-32b",
        "deepseek-r1-distill-llama-70b",
    ]
    try:
        models_data = client.models.list()
        available_ids = [m.id for m in models_data.data]

        for pref in preferred_order:
            if pref in available_ids:
                print(f"[SELECTED MODEL]: {pref}")
                return pref

        # Pick only standard text LLM models (skip whisper, compound, vision)
        for m_id in available_ids:
            lower = m_id.lower()
            if (
                "whisper" not in lower
                and "compound" not in lower
                and "guard" not in lower
            ):
                print(f"[FALLBACK SELECTED MODEL]: {m_id}")
                return m_id
    except Exception as e:
        print(f"[MODEL DISCOVERY ERROR]: {e}")

    return "llama-3.3-70b-versatile"


ACTIVE_CHAT_MODEL = get_active_model()


def call_groq_llm(messages, max_tokens=200, temperature=0.6):
    global ACTIVE_CHAT_MODEL
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=ACTIVE_CHAT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except Exception as err:
        print(f"[Model {ACTIVE_CHAT_MODEL} failed]: {err}")
    return None


class InterviewRequest(BaseModel):
    candidate_answer: str


class TTSRequest(BaseModel):
    text: str


class DialogueItem(BaseModel):
    sender: str
    text: str


class EvaluationRequest(BaseModel):
    role: str
    difficulty: str
    conversation: List[DialogueItem]


@app.get("/")
def home():
    return {
        "status": "online",
        "active_model": ACTIVE_CHAT_MODEL,
        "message": "Voice Interview Engine Live",
    }


# 1. Text-based Adaptive Question Endpoint
@app.post("/api/interview/question")
def generate_next_question(data: InterviewRequest):
    system_prompt = (
        "You are an expert AI technical interviewer conducting a technical interview. "
        "Based on the candidate's answer, generate exactly ONE follow-up technical question. "
        "Rules:\n"
        "- Dive into technical trade-offs, architecture, bottlenecks, or edge cases.\n"
        "- Keep it under 2 sentences.\n"
        "- Output ONLY the question text. Do not add quotes, greetings, or intro."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data.candidate_answer},
    ]

    generated_question = call_groq_llm(messages)

    if not generated_question:
        generated_question = "Could you elaborate on the performance optimizations and trade-offs in your implementation?"

    print(f"\n[FINAL AI QUESTION]: {generated_question}\n")

    return {
        "status": "success",
        "candidate_answer": data.candidate_answer,
        "next_question": generated_question,
    }


# 2. Text-to-Speech (TTS) Endpoint
@app.post("/api/interview/speak")
def text_to_speech(data: TTSRequest):
    try:
        clean_text = (
            data.text.strip()
            if data.text
            else "Please elaborate on your technical implementation."
        )
        tts = gTTS(text=clean_text, lang="en", slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_bytes = audio_fp.getvalue()

        headers = {
            "Content-Length": str(len(audio_bytes)),
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline; filename=speech.mp3",
        }

        return Response(
            content=audio_bytes, media_type="audio/mpeg", headers=headers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3. Speech-to-Text (STT) + Question Generation Pipeline
@app.post("/api/interview/voice-question")
async def voice_interview_pipeline(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_file = (file.filename, audio_bytes)

        # Groq Whisper
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
        )
        transcribed_text = transcription.text.strip()

        system_prompt = (
            "You are an expert AI technical interviewer. "
            "Generate exactly ONE technical follow-up question based on the candidate's answer. "
            "Return strictly the question without preamble."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcribed_text},
        ]

        next_q = call_groq_llm(messages)
        if not next_q:
            next_q = "What specific challenges did you face during performance scaling?"

        return {
            "status": "success",
            "candidate_transcribed_answer": transcribed_text,
            "next_question": next_q,
        }

    except Exception as e:
        return {"status": "error", "error_message": str(e)}


# 4. Evaluation Engine Endpoint
@app.post("/api/interview/evaluate")
def evaluate_interview(data: EvaluationRequest):
    transcript = "\n".join(
        [f"{item.sender.upper()}: {item.text}" for item in data.conversation]
    )

    eval_prompt = f"""
You are an expert technical interviewer evaluating a candidate for the role: {data.role} ({data.difficulty} level).
Analyze the following interview transcript:
{transcript}

Provide evaluation strictly in valid JSON format without markdown code fences or backticks. The JSON must follow this exact schema:
{{
  "overall_score": 85,
  "technical_accuracy": 88,
  "communication_clarity": 80,
  "depth_of_knowledge": 82,
  "strengths": ["Clear explanation of core concepts", "Good project communication"],
  "areas_for_improvement": ["Elaborate on edge cases", "Mention optimization trade-offs"],
  "summary_feedback": "Candidate showed good understanding of foundational concepts."
}}
Ensure all rating values are integers between 0 and 100.
"""

    messages = [{"role": "user", "content": eval_prompt}]
    raw_output = call_groq_llm(messages, max_tokens=600, temperature=0.3)

    if raw_output:
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
        raw_output = raw_output.strip()

        try:
            eval_data = json.loads(raw_output)
            return {"status": "success", "report": eval_data}
        except Exception:
            pass

    return {
        "status": "success",
        "report": {
            "overall_score": 80,
            "technical_accuracy": 82,
            "communication_clarity": 78,
            "depth_of_knowledge": 80,
            "strengths": [
                "Solid foundational understanding",
                "Structured answers",
            ],
            "areas_for_improvement": [
                "Cover architectural trade-offs",
                "Elaborate on edge cases",
            ],
            "summary_feedback": "Candidate showed good technical foundation and communicated effectively.",
        },
    }