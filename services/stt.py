import whisper
from sarvamai import SarvamAI
from core.config import CONFIG, settings

# 1. Sarvam STT (Primary)
def transcribe_to_english_sarvam(file_path: str) -> dict:
    if not settings.SARVAM_API_KEY:
        raise ValueError("Sarvam API key missing.")
        
    client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
    
    # FIX: Open the file as a binary stream before passing to the SDK
    with open(file_path, "rb") as audio_file:
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model=CONFIG["stt"]["sarvam"]["model"],
            mode="translate",
            language_code="unknown"
        )
    
    return {
        "english_text": getattr(response, "transcript", ""),
        "detected_lang": getattr(response, "language_code", "hi-IN")
    }

# 2. Whisper STT (Fallback)
def transcribe_to_english_whisper(file_path: str) -> dict:
    model = whisper.load_model(CONFIG["stt"]["whisper_local"]["model_size"])
    # task="translate" forces Whisper to output English
    result = model.transcribe(file_path, task="translate") 
    
    lang_map = {"hi": "hi-IN", "gu": "gu-IN", "mr": "mr-IN", "ta": "ta-IN", "bn": "bn-IN", "te": "te-IN", "en": "en-IN"}
    detected_lang = result.get("language", "hi")
    
    return {
        "english_text": result.get("text", ""),
        "detected_lang": lang_map.get(detected_lang, "hi-IN")
    }

# 3. STT Provider Manager
def process_incoming_audio(file_path: str) -> dict:
    primary = CONFIG["stt"]["primary_provider"]
    print(f"[*] STT via: {primary.upper()}")
    
    try:
        if primary == "sarvam":
            return transcribe_to_english_sarvam(file_path)
        return transcribe_to_english_whisper(file_path)
    except Exception as e:
        print(f"[-] Primary STT failed: {e}. Switching to Fallback...")
        return transcribe_to_english_whisper(file_path)