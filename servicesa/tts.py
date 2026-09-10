import os
import io
import asyncio
import tempfile
import base64
import edge_tts
from pydub import AudioSegment
from sarvamai import SarvamAI
from core.config import CONFIG, settings

# 1. Sarvam TTS (Primary)
def generate_tts_sarvam(regional_text: str, lang_code: str) -> io.BytesIO:
    # 8-Second timeout prevents the 3-minute hang!
    client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY, timeout=8.0) 
    
    response = client.text_to_speech.convert(
        text=regional_text,
        language_code=lang_code,
        speaker=CONFIG["tts"]["sarvam"]["speaker"],
        model=CONFIG["tts"]["sarvam"]["model"]
    )
    
    if isinstance(response, bytes):
        wav_buffer = io.BytesIO(response)
    else:
        audio_b64 = getattr(response, "audios", [""])[0]
        wav_buffer = io.BytesIO(base64.b64decode(audio_b64))
        
    wav_buffer.seek(0)
    audio_segment = AudioSegment.from_wav(wav_buffer)
    
    # Export as native WhatsApp OGG (Opus codec)
    ogg_buffer = io.BytesIO()
    audio_segment.export(ogg_buffer, format="ogg", codec="libopus")
    ogg_buffer.seek(0)
    
    return ogg_buffer

# 2. Edge-TTS (Fallback)
async def _generate_tts_edge_async(text: str, lang_code: str) -> io.BytesIO:
    voice = CONFIG["tts"]["edge_tts"]["voice_map"].get(lang_code, CONFIG["tts"]["edge_tts"]["fallback_voice"])
    
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        temp_path = tmp_file.name
        
    await communicate.save(temp_path)
    
    # Read Edge's MP3 and convert it to OGG Opus for WhatsApp
    audio_segment = AudioSegment.from_mp3(temp_path)
    ogg_buffer = io.BytesIO()
    audio_segment.export(ogg_buffer, format="ogg", codec="libopus")
    ogg_buffer.seek(0)
        
    os.unlink(temp_path)
    return ogg_buffer

def generate_tts_edge(text: str, lang_code: str) -> io.BytesIO:
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(_generate_tts_edge_async(text, lang_code))

# 3. TTS Provider Manager
def get_audio(regional_text: str, lang_code: str) -> io.BytesIO:
    primary = CONFIG["tts"]["primary_provider"]
    print(f"[*] TTS via: {primary.upper()}")
    
    try:
        if primary == "sarvam":
            return generate_tts_sarvam(regional_text, lang_code)
        return generate_tts_edge(regional_text, lang_code)
    except Exception as e:
        print(f"[-] Primary TTS failed: {e}. Switching to Fallback...")
        return generate_tts_edge(regional_text, lang_code)