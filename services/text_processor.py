import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Initialize Gemini 3.5 Flash Lite
gemini_normalizer = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    response_mime_type="application/json"
)

NORMALIZER_PROMPT = PromptTemplate.from_template("""
You are an expert multilingual pre-processor for an Indian legal and cooperative system.
Analyze the user's input, which may be:
1. Native Indian scripts (Gujarati, Hindi, Marathi, Bengali, Tamil, Telugu, etc.)
2. Romanized/Phonetic Indian languages (e.g., "gujarat ni mukhya policy vishe janavo", "khedut sahay yojana shu che", "kisan credit card apply kaise kare")
3. Standard English

Tasks:
1. Detect the target language and return its BCP-47 tag:
   - Gujarati: "gu-IN"
   - Hindi: "hi-IN"
   - Marathi: "mr-IN"
   - Bengali: "bn-IN"
   - Tamil: "ta-IN"
   - Telugu: "te-IN"
   - Kannada: "kn-IN"
   - Malayalam: "ml-IN"
   - Punjabi: "pa-IN"
   - Odia: "or-IN"
   - English: "en-IN"
2. Translate or normalize the input into formal, clear English optimized for database legal retrieval. If the query is already in English, return it unchanged.

Return ONLY a raw JSON object:
{{"detected_lang": "gu-IN", "english_query": "Tell me about the main policies of Gujarat."}}

User Input: {user_input}
""")

normalizer_chain = NORMALIZER_PROMPT | gemini_normalizer

def process_incoming_text(raw_text: str) -> dict:
    """Detects native or Romanized language and translates to English."""
    try:
        response = normalizer_chain.invoke({"user_input": raw_text})
        
        # Robust handling for LangChain returning either a list or string
        content = response.content
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            
        clean_json = re.sub(r"^```json|```$", "", content.strip(), flags=re.MULTILINE).strip()
        data = json.loads(clean_json)
        
        return {
            "detected_lang": data.get("detected_lang", "en-IN"),
            "english_query": data.get("english_query", raw_text)
        }
        
    except Exception as e:
        print(f"[-] Text normalization fallback triggered: {e}")
        # Default fallback to en-IN so English queries are not forced to Hindi
        return {
            "detected_lang": "en-IN",
            "english_query": raw_text
        }