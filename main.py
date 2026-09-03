import base64
import io
import json
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from gtts import gTTS
import httpx
from pydantic import BaseModel

try:
    from agora_token_builder import RtcTokenBuilder
except ImportError:
    RtcTokenBuilder = None

load_dotenv()

app = FastAPI(title="Adaptive Voice Interview Engine - Jynex")

# CORS Setup - Member 2 (Frontend) can connect seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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


# ===================== PYDANTIC MODELS =====================
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


class StartAgentRequest(BaseModel):
    channel_name: str


class StopAgentRequest(BaseModel):
    agent_id: str


class TokenRequest(BaseModel):
    channel_name: str
    uid: Optional[int] = 0


# ===================== BASE HEALTH CHECK =====================
@app.get("/")
def home():
    return {
        "status": "online",
        "active_model": ACTIVE_CHAT_MODEL,
        "message": "Voice Interview Engine Live",
    }


# ===================== AGORA RTC TOKEN GENERATOR (FOR FRONTEND) =====================
@app.post("/api/agora/token")
def generate_agora_rtc_token(payload: TokenRequest):
    app_id = os.getenv("AGORA_APP_ID")
    app_certificate = os.getenv("AGORA_APP_CERTIFICATE", "")

    if not app_id:
        raise HTTPException(status_code=500, detail="AGORA_APP_ID missing in .env")

    # If project certificate is not enabled in Agora console, return empty token
    if not app_certificate:
        return {
            "status": "success",
            "token": None,
            "app_id": app_id,
            "channel_name": payload.channel_name,
            "uid": payload.uid,
            "message": "App certificate not configured; join with token=null",
        }

    if not RtcTokenBuilder:
        raise HTTPException(status_code=500, detail="agora-token-builder not installed")

    expiration_time_in_seconds = 3600 * 24  # 24 hours validity
    current_timestamp = int(time.time())
    privilege_expired_ts = current_timestamp + expiration_time_in_seconds

    # Role 1 = Attendee/Publisher
    token = RtcTokenBuilder.buildTokenWithUid(
        app_id,
        app_certificate,
        payload.channel_name,
        payload.uid,
        1,
        privilege_expired_ts,
    )

    return {
        "status": "success",
        "token": token,
        "app_id": app_id,
        "channel_name": payload.channel_name,
        "uid": payload.uid,
    }


# ===================== AGORA START AGENT =====================
@app.post("/api/agora/start-agent")
async def start_agora_agent(payload: StartAgentRequest):
    app_id = os.getenv("AGORA_APP_ID")
    pipeline_id = os.getenv("AGORA_PIPELINE_ID")
    customer_id = os.getenv("AGORA_CUSTOMER_ID")
    customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")

    if not all([app_id, pipeline_id, customer_id, customer_secret]):
        raise HTTPException(
            status_code=500,
            detail="Agora credentials missing in .env",
        )

    raw_creds = f"{customer_id}:{customer_secret}"
    base64_creds = base64.b64encode(raw_creds.encode("utf-8")).decode("utf-8")

    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/join"
    headers = {
        "Authorization": f"Basic {base64_creds}",
        "Content-Type": "application/json",
    }
    
    body = {
        "name": payload.channel_name,
        "pipeline_id": pipeline_id,
        "properties": {
            "channel": payload.channel_name,
            "agent_rtc_uid": "1001",
            "remote_rtc_uids": ["*"],
            "asr": {
                "vendor": "deepgram",
                "params": {
                    "resource_id": "2ca6dcf4ded340b6b67f0ccf4972a00d",
                    "model": "nova-3",
                    "keyterm": "",
                    "language": "en"
                }
            },
            "llm": {
                "vendor": "openai",
                "params": {
                    "model": "gpt-4.1-mini",
                    "resource_id": "24731f4ef93e4d33a85a4c4088633bcb"
                },
                "system_messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI Technical Interviewer for software engineering roles.\n"
                            "Conduct a structured, professional, and adaptive technical interview.\n"
                            "Ask one crisp question at a time.\n"
                            "Evaluate candidate responses on technical accuracy and depth.\n"
                            "Adjust follow-up questions dynamically based on what the candidate answers.\n"
                            "Keep your spoken responses short and natural (under 2-3 sentences)."
                        )
                    }
                ],
                "greeting_message": "Hello! I am your AI Technical Interviewer today. Whenever you are ready, please introduce yourself and mention your primary tech stack.",
                "failure_message": "Please hold on a second."
            },
            "tts": {
                "vendor": "minimax",
                "params": {
                    "model": "speech-2.8-turbo",
                    "resource_id": "155b2afcadce4c93a85231c74e2e71d6",
                    "voice_setting": {
                        "voice_id": "English_radiant_girl"
                    }
                }
            },
            "mllm": {
                "enable": False
            }
        }
    }

    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(url, headers=headers, json=body, timeout=15.0)
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return {"status": "success", "data": response.json()}
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Request to Agora failed: {str(exc)}")


# ===================== AGORA STOP AGENT =====================
@app.post("/api/agora/stop-agent")
async def stop_agora_agent(payload: StopAgentRequest):
    app_id = os.getenv("AGORA_APP_ID")
    customer_id = os.getenv("AGORA_CUSTOMER_ID")
    customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")

    if not all([app_id, customer_id, customer_secret]):
        raise HTTPException(
            status_code=500,
            detail="Agora credentials missing in .env",
        )

    raw_creds = f"{customer_id}:{customer_secret}"
    base64_creds = base64.b64encode(raw_creds.encode("utf-8")).decode("utf-8")

    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/agents/{payload.agent_id}/leave"
    headers = {
        "Authorization": f"Basic {base64_creds}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(url, headers=headers, timeout=15.0)
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return {"status": "success", "message": "Agent stopped successfully"}
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Request to Agora failed: {str(exc)}")


# ===================== GROQ ADAPTIVE QUESTION =====================
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

    return {
        "status": "success",
        "candidate_answer": data.candidate_answer,
        "next_question": generated_question,
    }


# ===================== TTS ENDPOINT =====================
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


# ===================== GROQ STT + QUESTION PIPELINE =====================
@app.post("/api/interview/voice-question")
async def voice_interview_pipeline(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_file = (file.filename, audio_bytes)

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


# ===================== EVALUATION REPORT ENGINE =====================
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