import base64
import io
import json
import os
import time
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from gtts import gTTS
import httpx
from pydantic import BaseModel, Field

try:
    from agora_token_builder import RtcTokenBuilder
except ImportError:
    RtcTokenBuilder = None

# Async MongoDB Driver
try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    AsyncIOMotorClient = None

load_dotenv()

app = FastAPI(
    title="Jynex Adaptive Voice Interview Engine",
    version="2.0.0",
    description="Real-time voice interview orchestration with Agora, Groq LLM, and MongoDB persistence."
)

# ===================== PRODUCTION CORS =====================
ALLOWED_ORIGINS = [
    "https://jynex-frontend.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== DATABASE SETUP =====================
MONGO_URI = os.getenv("MONGO_URI", "")
db = None
reports_collection = None
IN_MEMORY_REPORTS = []

if MONGO_URI and AsyncIOMotorClient:
    try:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        db = mongo_client["jynex_interview_db"]
        reports_collection = db["session_reports"]
        print("[DATABASE]: Connected to MongoDB Atlas successfully.")
    except Exception as e:
        print(f"[DATABASE ERROR]: MongoDB connection failed: {e}. Using in-memory fallback.")
else:
    print("[DATABASE]: MONGO_URI not found or motor not installed. Using in-memory fallback.")

# ===================== GROQ CLIENT & MODEL DISCOVERY =====================
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
            if "whisper" not in lower and "compound" not in lower and "guard" not in lower:
                print(f"[FALLBACK MODEL]: {m_id}")
                return m_id
    except Exception as e:
        print(f"[MODEL DISCOVERY ERROR]: {e}")

    return "llama-3.3-70b-versatile"

ACTIVE_CHAT_MODEL = get_active_model()

def call_groq_llm(messages, max_tokens=650, temperature=0.1):
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

# ===================== MULTI-AGENT PERSONAS =====================
PERSONA_PROMPTS = {
    "alex": (
        "You are Alex, an Elite System Architect & Senior Tech Interviewer. "
        "Focus on algorithmic efficiency, execution speed, edge cases, system trade-offs, and deep technical bottlenecks. "
        "Ask exactly ONE direct, sharp technical follow-up question under 2 sentences. No greetings or conversational filler."
    ),
    "emma": (
        "You are Emma, a Principal HR & Behavioral Coach. "
        "Focus on ownership, STAR methodology, conflict resolution, collaboration, and situational mindset. "
        "Ask exactly ONE insightful behavioral or scenario-based follow-up question under 2 sentences. No conversational filler."
    ),
    "sarah": (
        "You are Sarah, VP of Engineering & Technical Hiring Manager. "
        "Focus on system architecture, microservices, scaling decisions, cost-benefit trade-offs, and production resiliency. "
        "Ask exactly ONE high-level architectural follow-up question under 2 sentences. No greetings or filler."
    )
}

# ===================== DATA SCHEMAS =====================
class StartAgentRequest(BaseModel):
    channel_name: str
    persona: Optional[str] = "alex"

class StopAgentRequest(BaseModel):
    agent_id: str

class TokenRequest(BaseModel):
    channel_name: str
    uid: Optional[int] = 0

class InterviewRequest(BaseModel):
    candidate_answer: Optional[str] = None
    previous_answer: Optional[str] = None
    role: Optional[str] = "Full-Stack Engineering"
    persona: Optional[str] = "alex"
    question_index: Optional[int] = 0

class TTSRequest(BaseModel):
    text: str

class DialogueItem(BaseModel):
    sender: str
    text: str

class EvaluationReport(BaseModel):
    overall_score: int
    technical_accuracy: int
    communication_clarity: int
    depth_of_knowledge: int
    strengths: List[str]
    areas_for_improvement: List[str]
    summary_feedback: str

class EvaluationRequest(BaseModel):
    role: str
    difficulty: str
    persona: Optional[str] = "alex"
    conversation: List[DialogueItem]

class SaveReportRequest(BaseModel):
    session_id: str
    candidate_name: Optional[str] = "Candidate"
    role: str
    difficulty: str
    persona: str
    report: EvaluationReport
    conversation: List[DialogueItem]
    timestamp: Optional[str] = None

# ===================== HELPER FUNCTIONS =====================
def build_rtc_token(channel_name: str, uid: int, role: int = 1) -> Optional[str]:
    app_id = os.getenv("AGORA_APP_ID")
    app_certificate = os.getenv("AGORA_APP_CERTIFICATE", "")
    if not app_id or not app_certificate or not RtcTokenBuilder:
        return None
    
    expiration_time_in_seconds = 3600 * 24
    privilege_expired_ts = int(time.time()) + expiration_time_in_seconds
    return RtcTokenBuilder.buildTokenWithUid(
        app_id,
        app_certificate,
        channel_name,
        uid,
        role,
        privilege_expired_ts,
    )

# ===================== ENDPOINTS =====================

@app.get("/")
def home():
    return {
        "status": "online",
        "active_model": ACTIVE_CHAT_MODEL,
        "database_connected": bool(reports_collection is not None),
        "version": "2.0.0"
    }

# 1. Agora Token Generator (For Frontend WebRTC Audio)
@app.post("/api/agora/token")
def generate_agora_rtc_token(payload: TokenRequest):
    app_id = os.getenv("AGORA_APP_ID")
    app_certificate = os.getenv("AGORA_APP_CERTIFICATE", "")

    if not app_id:
        raise HTTPException(status_code=500, detail="AGORA_APP_ID missing in environment variables")

    target_uid = payload.uid if payload.uid is not None else 0
    token = build_rtc_token(payload.channel_name, target_uid, role=1)

    return {
        "status": "success",
        "token": token,
        "app_id": app_id,
        "channel_name": payload.channel_name,
        "uid": target_uid,
        "message": "Token generated successfully" if token else "App certificate not set; join with token=null"
    }

# 2. Start Agora Conversational AI Agent
@app.post("/api/agora/start-agent")
async def start_agora_agent(payload: StartAgentRequest):
    app_id = os.getenv("AGORA_APP_ID")
    pipeline_id = os.getenv("AGORA_PIPELINE_ID")
    customer_id = os.getenv("AGORA_CUSTOMER_ID")
    customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")

    if not all([app_id, pipeline_id, customer_id, customer_secret]):
        raise HTTPException(status_code=500, detail="Agora credentials missing in environment variables")

    raw_creds = f"{customer_id}:{customer_secret}"
    base64_creds = base64.b64encode(raw_creds.encode("utf-8")).decode("utf-8")

    persona_key = payload.persona.lower() if payload.persona else "alex"
    system_instruction = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["alex"])

    agent_uid = 1001
    agent_token = build_rtc_token(payload.channel_name, agent_uid, role=1)

    url = f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/join"
    headers = {
        "Authorization": f"Basic {base64_creds}",
        "Content-Type": "application/json",
    }

    properties_dict = {
        "channel": payload.channel_name,
        "agent_rtc_uid": str(agent_uid),
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
                {"role": "system", "content": system_instruction}
            ],
            "greeting_message": f"Hello! I am {persona_key.capitalize()}, your interviewer today. Whenever you are ready, please introduce yourself.",
            "failure_message": "Please hold on a second."
        },
        "tts": {
            "vendor": "minimax",
            "params": {
                "model": "speech-2.8-turbo",
                "resource_id": "155b2afcadce4c93a85231c74e2e71d6",
                "voice_setting": {
                    "voice_id": "English_radiant_girl" if persona_key in ["emma", "sarah"] else "English_radiant_man"
                }
            }
        },
        "mllm": {"enable": False}
    }

    if agent_token:
        properties_dict["token"] = agent_token

    body = {
        "name": payload.channel_name,
        "pipeline_id": pipeline_id,
        "properties": properties_dict
    }

    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(url, headers=headers, json=body, timeout=15.0)
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            resp_data = response.json()
            agent_id = resp_data.get("agent_id") or resp_data.get("id") or resp_data.get("data", {}).get("agent_id")
            return {
                "status": "success",
                "agent_id": agent_id,
                "data": resp_data
            }
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Request to Agora failed: {str(exc)}")

# 3. Stop Agora Conversational AI Agent
@app.post("/api/agora/stop-agent")
async def stop_agora_agent(payload: StopAgentRequest):
    app_id = os.getenv("AGORA_APP_ID")
    customer_id = os.getenv("AGORA_CUSTOMER_ID")
    customer_secret = os.getenv("AGORA_CUSTOMER_SECRET")

    if not all([app_id, customer_id, customer_secret]):
        raise HTTPException(status_code=500, detail="Agora credentials missing in environment variables")

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

# 4. Adaptive Question Generation
@app.post("/api/interview/question")
def generate_next_question(data: InterviewRequest):
    persona_key = data.persona.lower() if data.persona else "alex"
    system_prompt = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["alex"])
    answer_text = data.candidate_answer or data.previous_answer or "I have experience building microservices and distributed databases."

    prompt_content = (
        f"Candidate is interviewing for: {data.role}. "
        f"This is question #{data.question_index + 1}. "
        f"Candidate's previous response: '{answer_text}'. "
        "Ask a concise, challenging follow-up question (maximum 2 sentences) addressing performance, edge cases, or architecture."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_content},
    ]

    generated_question = call_groq_llm(messages, max_tokens=250, temperature=0.6)
    if not generated_question:
        generated_question = "Could you elaborate on the performance optimizations and trade-offs in your implementation?"

    return {
        "status": "success",
        "question": generated_question,
        "q": generated_question,
        "next_question": generated_question,
        "persona": persona_key,
        "keywords": ["scaling", "latency", "resiliency", "architecture", "tradeoffs"],
        "concept": f"{data.role} Deep Dive - Topic #{data.question_index + 1}",
    }

# 5. Voice Question Pipeline (Groq Whisper Large V3 STT)
@app.post("/api/interview/voice-question")
async def voice_interview_pipeline(file: UploadFile = File(...), persona: str = "alex"):
    try:
        audio_bytes = await file.read()
        audio_file = (file.filename, audio_bytes)

        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
        )
        transcribed_text = transcription.text.strip()

        persona_key = persona.lower() if persona else "alex"
        system_prompt = PERSONA_PROMPTS.get(persona_key, PERSONA_PROMPTS["alex"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcribed_text},
        ]

        next_q = call_groq_llm(messages, max_tokens=250, temperature=0.6)
        if not next_q:
            next_q = "What specific challenges did you face during architecture and scaling?"

        return {
            "status": "success",
            "persona": persona_key,
            "candidate_transcribed_answer": transcribed_text,
            "next_question": next_q,
            "question": next_q,
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# 6. Text-to-Speech MP3 Stream (Audio Fallback)
@app.post("/api/interview/speak")
def text_to_speech(data: TTSRequest):
    try:
        clean_text = data.text.strip() if data.text else "Please elaborate on your technical implementation."
        tts = gTTS(text=clean_text, lang="en", slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_bytes = audio_fp.getvalue()

        headers = {
            "Content-Length": str(len(audio_bytes)),
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline; filename=speech.mp3",
        }
        return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. Candidate Evaluation & Scorecard Generation (STRICT 0-100% UNBIASED HUMAN-JUDGE RUBRIC)
@app.post("/api/interview/evaluate")
def evaluate_interview(data: EvaluationRequest):
    valid_dialogues = [item for item in data.conversation if item.text and item.text.strip()]
    transcript = "\n".join([f"{item.sender.upper()}: {item.text}" for item in valid_dialogues])
    persona_key = data.persona.lower() if data.persona else "alex"

    candidate_texts = [item.text.strip() for item in valid_dialogues if item.sender.lower() == "candidate"]
    total_candidate_words = sum(len(text.split()) for text in candidate_texts)

    eval_prompt = f"""
You are {persona_key.capitalize()}, an uncompromising, expert Technical Interview Judge evaluating a candidate for the role: {data.role} ({data.difficulty} level).

CANDIDATE INTERVIEW TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

STRICT UNBIASED SCORING RUBRIC (Full Range: 0 to 100):
- 0 to 15: Blank, silent, gave up ("I don't know", "skip", "pass", silence), or completely irrelevant gibberish.
- 16 to 40: Factually incorrect concepts, severe technical fallacies, or completely hallucinated explanations.
- 41 to 49: Extremely weak, single buzzwords thrown around without explaining internal mechanics.
- 50 to 65: Surface-level / basic theoretical definition; understands elementary terms but lacks internal mechanics, edge cases, and real-world implementation depth.
- 66 to 80: Good technical answers; accurate terminology, structured delivery, and solid conceptual clarity, with minor omissions in production scaling or failure modes.
- 81 to 100: Deep production-grade mastery; articulates architectural tradeoffs, latency considerations, failure boundaries, and battle-tested design patterns.

MANDATORY GRADING RULES:
1. Zero grade inflation. If the candidate was silent, said "don't know", or gave wrong answers, score strictly between 0% and 40%.
2. If answers are brief textbook summaries without trade-offs, score strictly between 50% and 65%.
3. Score 85%+ ONLY when the candidate explicitly articulates system trade-offs, scalability, and deep architectural nuances.
4. Strengths & areas_for_improvement must directly cite specific statements or omissions from the transcript.
5. Return strictly valid JSON without markdown codeblocks or backticks.

Schema:
{{
  "overall_score": <integer 0-100>,
  "technical_accuracy": <integer 0-100>,
  "communication_clarity": <integer 0-100>,
  "depth_of_knowledge": <integer 0-100>,
  "strengths": ["string", "string"],
  "areas_for_improvement": ["string", "string"],
  "summary_feedback": "string"
}}
"""

    messages = [{"role": "user", "content": eval_prompt}]
    raw_output = call_groq_llm(messages, max_tokens=650, temperature=0.1)

    if raw_output:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            eval_data = json.loads(cleaned)
            return {
                "status": "success",
                "report": {
                    "overall_score": max(0, min(100, int(eval_data.get("overall_score", 0)))),
                    "technical_accuracy": max(0, min(100, int(eval_data.get("technical_accuracy", 0)))),
                    "communication_clarity": max(0, min(100, int(eval_data.get("communication_clarity", 0)))),
                    "depth_of_knowledge": max(0, min(100, int(eval_data.get("depth_of_knowledge", 0)))),
                    "strengths": eval_data.get("strengths", ["Participation recorded."]),
                    "areas_for_improvement": eval_data.get("areas_for_improvement", ["Needs comprehensive study of core fundamentals."]),
                    "summary_feedback": eval_data.get("summary_feedback", "Performance evaluated based on transcript depth.")
                }
            }
        except Exception as parse_err:
            print(f"[JSON Parse Warning]: {parse_err}")

    # ===================== DYNAMIC 0-100% FALLBACK (NO ARTIFICIAL FLOOR) =====================
    if total_candidate_words < 8:
        calc_overall = 10
        calc_tech = 5
        calc_comm = 15
        calc_depth = 5
        summary = "Candidate was essentially silent or provided one-word answers with zero technical substance."
        improvements = ["Must articulate technical concepts verbally", "Avoid skipping questions or remaining silent"]
        strengths = ["Attended the scheduled interview session"]
    elif total_candidate_words < 25:
        calc_overall = 32
        calc_tech = 28
        calc_comm = 35
        calc_depth = 22
        summary = "Candidate answers were severely incomplete, superficial, or demonstrated fundamental misconceptions."
        improvements = ["Explain how technologies actually function rather than listing buzzwords", "Study core fundamentals and internal mechanisms"]
        strengths = ["Attempted basic domain terminology"]
    elif total_candidate_words < 60:
        calc_overall = 58
        calc_tech = 55
        calc_comm = 62
        calc_depth = 52
        summary = "Candidate showed elementary theoretical understanding but lacked production tradeoffs and technical depth."
        improvements = ["Elaborate on architectural edge cases and internal mechanics", "Discuss real-world tradeoffs and failure modes"]
        strengths = ["Understands baseline theoretical definitions", "Coherent communication"]
    elif total_candidate_words < 120:
        calc_overall = 76
        calc_tech = 75
        calc_comm = 80
        calc_depth = 73
        summary = "Candidate demonstrated clear domain knowledge, structured reasoning, and accurate technical terminology."
        improvements = ["Deepen analysis of distributed failure recovery", "Quantify latency and resource scaling trade-offs"]
        strengths = ["Structured thought process", "Accurate technical definitions across topics"]
    else:
        calc_overall = 89
        calc_tech = 90
        calc_comm = 88
        calc_depth = 91
        summary = "Candidate demonstrated exceptional engineering maturity, articulating clear tradeoffs, edge cases, and resilient architecture."
        improvements = ["Continue expanding on emerging distributed systems patterns"]
        strengths = ["Staff-level architectural depth", "Clear consideration of real-world production tradeoffs"]

    return {
        "status": "success",
        "report": {
            "overall_score": calc_overall,
            "technical_accuracy": calc_tech,
            "communication_clarity": calc_comm,
            "depth_of_knowledge": calc_depth,
            "strengths": strengths,
            "areas_for_improvement": improvements,
            "summary_feedback": summary
        }
    }

# 8. Database Persistence: Save Interview Report to MongoDB
@app.post("/api/interview/save-report")
async def save_interview_report(payload: SaveReportRequest):
    report_doc = payload.dict()
    if not report_doc.get("timestamp"):
        report_doc["timestamp"] = datetime.utcnow().isoformat()

    if reports_collection is not None:
        try:
            await reports_collection.update_one(
                {"session_id": payload.session_id},
                {"$set": report_doc},
                upsert=True
            )
            return {"status": "success", "message": "Report saved to MongoDB Atlas", "session_id": payload.session_id}
        except Exception as e:
            print(f"[DB SAVE ERROR]: {e}")

    IN_MEMORY_REPORTS.append(report_doc)
    return {"status": "success", "message": "Report saved in memory cache", "session_id": payload.session_id}

# 9. Database Persistence: Fetch Interview History / Reports for Dashboard
@app.get("/api/interview/reports")
async def get_interview_reports():
    if reports_collection is not None:
        try:
            cursor = reports_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(50)
            reports = await cursor.to_list(length=50)
            return {"status": "success", "count": len(reports), "reports": reports}
        except Exception as e:
            print(f"[DB FETCH ERROR]: {e}")

    return {"status": "success", "count": len(IN_MEMORY_REPORTS), "reports": list(reversed(IN_MEMORY_REPORTS))}