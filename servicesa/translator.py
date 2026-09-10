from sarvamai import SarvamAI
from langchain_groq import ChatGroq
from core.config import settings
import textwrap
def translate_en_to_regional(english_text: str, target_lang: str) -> str:
    # If the user asked in English or the detector identified English, skip translation
    if "en" in target_lang.lower():
        return english_text

    try:
        client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
        # Sarvam has a strict 2000 character limit. If it's too long, chunk it safely.
        if len(english_text) > 1950:
            print("[*] Text exceeds Sarvam limit. Chunking into smaller pieces...")
            chunks = textwrap.wrap(english_text, width=1950, replace_whitespace=False, drop_whitespace=False)
            translated_chunks = []
            
            for chunk in chunks:
                response = client.text.translate(
                    input=chunk,
                    source_language_code="en-IN",
                    target_language_code=target_lang,
                    speaker_gender="Female",
                    model="sarvam-translate:v1"
                )
                translated_chunks.append(response.get("translated_text", ""))
                
            return "".join(translated_chunks)
        else:
            # Normal < 2000 char execution
            response = client.text.translate(
                input=english_text,
                source_language_code="en-IN",
                target_language_code=target_lang,
                speaker_gender="Female",
                model="sarvam-translate:v1"
            )
            return response.translated_text
        
    except Exception as e:
        print(f"[-] Sarvam Translation failed: {e}. Switching to Fallback (GROQ)...")
        llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY, 
            model_name="openai/gpt-oss-120b", 
            temperature=0.1
        )
        prompt = (
            f"You are an expert, empathetic translator. Translate the following English text to the language represented by BCP-47 code '{target_lang}'.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. The speaker is FEMALE. You MUST use female grammatical inflections, verb endings, and pronouns in the target language (e.g., in Hindi, use 'karungi' instead of 'karunga').\n"
            "2. The tone must be warm, reassuring, and professional.\n"
            "3. Return ONLY the final translated text without any quotes, conversational filler, or explanations.\n\n"
            f"Text: {english_text}"
        )
        
        fallback_response = llm.invoke(prompt)
        return fallback_response.content.strip()