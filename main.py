import os
import uuid
import shutil
import time
from datetime import datetime
from collections import deque
import re
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, Query, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from core.config import settings
from services.stt import process_incoming_audio
from services.router import process_user_input, contextualize_query
from services.translator import translate_en_to_regional
from services.tts import get_audio
from services.whatsapp import download_meta_media, send_whatsapp_text, send_whatsapp_audio, mark_whatsapp_read, send_whatsapp_interactive_reply
from services.text_processor import process_incoming_text
from services.memory import add_message, get_recent_history, clear_session_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INBOUND_DIR = os.path.join(BASE_DIR, "storage", "inbound")
OUTBOUND_DIR = os.path.join(BASE_DIR, "storage", "outbound")

os.makedirs(INBOUND_DIR, exist_ok=True)
os.makedirs(OUTBOUND_DIR, exist_ok=True)

CLEAR_MEMORY_PATTERN = re.compile(
    r"^(reset|clear|delete|forget|wipe)$|"
    r"(forget|clear|delete|erase|wipe|reset)\b.*\b(history|chat|conversation|memory|data|messages)",
    re.IGNORECASE
)


app = FastAPI(title="Narad AI Core API")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# --- CONCURRENCY & DEDUPLICATION MANAGERS ---
# 1. Self-expiring locks: {session_id: timestamp_locked}
ACTIVE_SESSION_LOCKS: dict[str, float] = {}
SESSION_LOCK_TIMEOUT_SECONDS = 60.0

def acquire_session_lock(session_id: str) -> bool:
    """Attempts to lock a session. Automatically purges locks older than 60s."""
    now = time.time()
    last_locked = ACTIVE_SESSION_LOCKS.get(session_id)
    
    if last_locked and (now - last_locked < SESSION_LOCK_TIMEOUT_SECONDS):
        return False  # Still actively locked
        
    ACTIVE_SESSION_LOCKS[session_id] = now
    return True

def release_session_lock(session_id: str):
    """Safely clears the lock for a session."""
    ACTIVE_SESSION_LOCKS.pop(session_id, None)

# 2. Meta webhook deduplication cache (stores last 500 message IDs)
PROCESSED_MESSAGE_IDS = deque(maxlen=500)


def verify_api_key(api_key: str = Security(api_key_header)):
    ...
#    if api_key != settings.NARAD_API_KEY:
#        raise HTTPException(status_code=403, detail="Invalid API Key")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Detected-Language", "X-English-Query", "X-Regional-Answer"] 
)

@app.get("/")
def read_root():
    return {"status": "Narad AI Backend is Online", "storage": "Active"}

# --- 1. CORE PIPELINE ---
def execute_ai_pipeline(audio_path: str, out_filename: str, session_id: str = None) -> dict:
    stt_result = process_incoming_audio(audio_path)
    english_query = stt_result["english_text"]
    user_lang = stt_result["detected_lang"]
    
    search_query = contextualize_query(session_id, english_query) if session_id else english_query
    
    router_result = process_user_input(search_query)
    english_answer = router_result["answer"]
    
    regional_answer = translate_en_to_regional(english_answer, user_lang)
    audio_buffer = get_audio(regional_answer, user_lang)
    
    out_path = os.path.join(OUTBOUND_DIR, out_filename)
    with open(out_path, "wb") as f:
        f.write(audio_buffer.read())
        
    return {
        "out_path": out_path,
        "english_query": english_query, 
        "english_answer": english_answer,
        "regional_answer": regional_answer,
        "user_lang": user_lang
    }

# --- 2. API ENDPOINTS FOR FRONTEND ---
class TextChatRequest(BaseModel):
    session_id: str
    query: str
    target_language: str | None = None

@app.post("/api/text-chat", dependencies=[Depends(verify_api_key)])
async def text_chat(req: TextChatRequest):
    if not acquire_session_lock(req.session_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Your previous query is still processing. Please wait a moment."}
        )
    
    try:
        norm_result = await run_in_threadpool(process_incoming_text, req.query)
        raw_english_query = norm_result["english_query"]
        user_lang = req.target_language or norm_result["detected_lang"]

        search_query = await run_in_threadpool(contextualize_query, req.session_id, raw_english_query)
        router_result = await run_in_threadpool(process_user_input, search_query)
        regional_answer = await run_in_threadpool(translate_en_to_regional, router_result["answer"], user_lang)

        add_message(req.session_id, "user", raw_english_query)
        add_message(req.session_id, "assistant", router_result["answer"])

        return {
            "session_id": req.session_id,
            "intent": router_result["intent"],
            "source": router_result["source"],
            "detected_language": user_lang,
            "english_answer": router_result["answer"],
            "regional_answer": regional_answer
        }
    finally:
        release_session_lock(req.session_id)

@app.post("/api/voice-chat", dependencies=[Depends(verify_api_key)])
async def voice_chat(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    if not acquire_session_lock(session_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Your previous audio query is still processing. Please wait a moment."}
        )
        
    try:
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        in_filename = f"{timestamp}_{request_id}_inbound_{audio_file.filename}"
        out_filename = f"{timestamp}_{request_id}_outbound.ogg"
        in_path = os.path.join(INBOUND_DIR, in_filename)

        with open(in_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        result = await run_in_threadpool(execute_ai_pipeline, in_path, out_filename, session_id)

        add_message(session_id, "user", result["english_query"])
        add_message(session_id, "assistant", result["english_answer"])

        return FileResponse(
            path=result["out_path"],
            media_type="audio/mpeg",
            headers={
                "X-Request-ID": request_id,
                "X-Detected-Language": result["user_lang"],
                "X-English-Query": result["english_query"].replace('\n', ' ').encode('utf-8').decode('latin-1', 'ignore'),
                "X-Regional-Answer": result["regional_answer"].replace('\n', ' ').encode('utf-8').decode('latin-1', 'ignore')
            }
        )
    finally:
        release_session_lock(session_id)

@app.get("/api/chat-history/{session_id}", dependencies=[Depends(verify_api_key)])
def fetch_history(session_id: str):
    return {"session_id": session_id, "history": get_recent_history(session_id, limit=20)}

@app.delete("/api/chat-history/{session_id}", dependencies=[Depends(verify_api_key)])
def forget_history(session_id: str):
    clear_session_history(session_id)
    return {"status": "success", "message": f"History for session {session_id} wiped."}

# --- 3. WHATSAPP WEBHOOK ---
@app.get("/audio/{filename}")
def serve_audio_file(filename: str):
    file_path = os.path.join(OUTBOUND_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/ogg")
    raise HTTPException(status_code=404, detail="Audio file not found")

def process_whatsapp_event(payload: dict, request_headers: dict):
    """Background worker for WhatsApp."""
    try:
        entries = payload.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])
                
                for message in messages:
                    message_id = message.get("id")
                    
                    # 1. Deduplication: Silently drop repeated Meta webhooks
                    if message_id in PROCESSED_MESSAGE_IDS:
                        print(f"[*] Duplicate webhook ignored: {message_id}")
                        continue
                    PROCESSED_MESSAGE_IDS.append(message_id)

                    sender = message.get("from")
                    msg_type = message.get("type")

                    # Dynamically determine public server URL
                    if settings.PUBLIC_SERVER_URL:
                        base_url = settings.PUBLIC_SERVER_URL.rstrip("/")
                    else:
                        proto = request_headers.get("x-forwarded-proto", "https")
                        host = request_headers.get("x-forwarded-host", request_headers.get("host", "localhost"))
                        base_url = f"{proto}://{host}"

                    # 2. Acquire user-specific processing lock
                    if not acquire_session_lock(sender):
                        send_whatsapp_text(sender, "⏳ Please wait, I am still processing your previous request...")
                        continue

                    # Process the message inside a guaranteed release block
                    try:
                        mark_whatsapp_read(message_id)
                        send_whatsapp_interactive_reply(sender, "⏳ _Narad AI is analyzing your legal query..._")

                        if msg_type == "audio":
                            media_id = message["audio"]["id"]
                            req_id = str(uuid.uuid4())[:8]
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            in_filename = f"{timestamp}_{req_id}_inbound_{media_id}.ogg"
                            out_filename = f"{timestamp}_{req_id}_outbound.ogg"
                            in_path = os.path.join(INBOUND_DIR, in_filename)

                            print(f"[*] Voice note from {sender}. Downloading...")
                            if download_meta_media(media_id, in_path):
                                result = execute_ai_pipeline(in_path, out_filename, session_id=sender)
                                add_message(sender, "user", result["english_query"])
                                add_message(sender, "assistant", result["english_answer"])

                                public_audio_url = f"{base_url}/audio/{out_filename}"
                                send_whatsapp_audio(sender, public_audio_url)
                                send_whatsapp_text(sender, f"📝 *Summary:*\n{result['regional_answer']}")
                            else:
                                send_whatsapp_text(sender, "⚠️ Unable to process your voice note.")

                        elif msg_type == "text":
                            user_query = message["text"]["body"].strip()
                            query_lower = user_query.lower()

                            # 1. SMART MEMORY WIPE CHECK (Run BEFORE sending "Analyzing...")
                            if CLEAR_MEMORY_PATTERN.search(query_lower):
                                clear_session_history(sender)
                                send_whatsapp_text(sender, "🧹 Memory wiped! Narad AI is ready for a new session.")
                                continue # Instantly skip the rest of the loop!

                            norm_result = process_incoming_text(user_query)
                            english_query = norm_result["english_query"]
                            user_lang = norm_result["detected_lang"]

                            search_query = contextualize_query(sender, english_query)
                            router_result = process_user_input(search_query)
                            english_answer = router_result["answer"]

                            add_message(sender, "user", english_query)
                            add_message(sender, "assistant", english_answer)

                            final_answer = translate_en_to_regional(english_answer, user_lang)
                            send_whatsapp_text(sender, final_answer)

                    except Exception as e:
                        print(f"[-] Error processing message for {sender}: {e}")
                        send_whatsapp_text(sender, "⚠️ *System Error:* An issue occurred. Please try again in a few moments.")
                    finally:
                        release_session_lock(sender)

    except Exception as e:
        print(f"[-] Global WhatsApp handler error: {e}")

@app.get("/webhook")
def verify_meta_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == settings.WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Verification token mismatch")

@app.post("/webhook")
async def receive_meta_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    headers_dict = dict(request.headers)
    background_tasks.add_task(process_whatsapp_event, payload, headers_dict)
    return {"status": "accepted"}