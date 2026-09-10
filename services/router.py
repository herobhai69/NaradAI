import json
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from core.config import CONFIG
from .rag import generate_legal_response
from services.memory import get_recent_history

PRIMARY_ROUTER = CONFIG.get("ai_router", {}).get("primary", "gemini").lower()
FALLBACK_ROUTER = CONFIG.get("ai_router", {}).get("fallback", "ollama").lower()

print(f"Loading AI Engines... (Primary: {PRIMARY_ROUTER.upper()} | Fallback: {FALLBACK_ROUTER.upper()})")

ollama_llm = ChatOllama(model="llama3.1", temperature=0, format="json")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", convert_system_message_to_human=True)

router_prompt = PromptTemplate.from_template("""
You are Narad AI, an elite, highly restricted cooperative governance and legal assistance chatbot for the Ministry of Cooperation (India).
Your defined domain includes: Cooperative laws, PACS services, agriculture, farmer welfare, rural grievances, Indian legal rights, and government citizen schemes (including education, housing, and financial benefits).

Analyze the user's input and classify its intent into exactly ONE of these categories:
1. "RAG" - The query is related to cooperative governance, Indian agriculture, legal rights, government schemes, farmer welfare, or citizen benefits (e.g., education scholarships, subsidies, crop insurance).
2. "CHITCHAT" - Polite greetings (e.g., "Hello", "Good morning").
3. "CREATOR" - Questions about your developers (Team LexFlow).
4. "REJECT" - EVERYTHING ELSE. If the user asks about celebrities, pop culture, non-agricultural global news, general programming/coding, unrelated political gossip, or illegal activities, you MUST classify it as REJECT.

Respond with ONLY a raw JSON object, like this: {{"intent": "RAG"}}

User Input: {user_input}
""")

# --- NEW: DYNAMIC CHITCHAT REWRITER ---
cHITCHAT_PROMPT = PromptTemplate.from_template("""
You are Narad AI, a warm, empathetic, and professional female legal assistant for the Ministry of Cooperation (India).
The user has just engaged in casual conversation, a greeting, or expressed gratitude.

Tasks:
1. Respond with a polite, reassuring, and distinctly human-like female persona. Do not sound robotic.
2. Keep your response brief (1 to 2 sentences max) and naturally conversational.
3. Always gently steer the conversation back to your domain (assisting with cooperative laws, agriculture, PMFBY, government schemes, or rural grievances).

User Message: {user_input}
""")

chitchat_chain = cHITCHAT_PROMPT | gemini_llm
ollama_chain = router_prompt | ollama_llm
gemini_chain = router_prompt | gemini_llm

def ask_engine(user_message: str, engine_name: str) -> str:
    if engine_name == "gemini":
        content = gemini_chain.invoke({"user_input": user_message}).content
    else:
        content = ollama_chain.invoke({"user_input": user_message}).content
        
    if isinstance(content, list):
        raw_response = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    else:
        raw_response = str(content)
        
    clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)["intent"].upper()

def process_user_input(user_message: str) -> dict:
    """Returns a unified dictionary for the downstream pipeline."""
    print(f"\nAnalyzing intent for: '{user_message}'...")
    intent = "RAG" 
    
    try:
        intent = ask_engine(user_message, PRIMARY_ROUTER)
    except Exception as e:
        print(f"-> [Primary ({PRIMARY_ROUTER.upper()}) FAILED: {e}]")
        print(f"-> [Switching to Fallback ({FALLBACK_ROUTER.upper()})...]")
        try:
            intent = ask_engine(user_message, FALLBACK_ROUTER)
        except Exception:
            intent = "RAG"

    if intent == "REJECT":
        return {"intent": intent, "source": "router", "answer": "I am a legal assistant. I cannot help you with that."}
    elif intent == "CREATOR":
        return {"intent": intent, "source": "router", "answer": "I was developed by Team LexFlow!\n- Member 1: Kartik Sharma (CodeName: Hero)\n- Member 2: Bedabrata Tarapdar (CodeName: Nomad)\n- Member 3: Mridul Patil (CodeName: Oman)\n- Member 4: Harsh Joshi (CodeName: CR)\n- Member 5: Nitya Raghuvanshi\n- Member 6: Aaryan Agrawal"}
    elif intent == "CHITCHAT":
        print("[*] Generating dynamic chitchat response...")
        try:
            # Generate a unique response based on what the user said
            chat_response = chitchat_chain.invoke({"user_input": user_message})
            
            # Safely parse Gemini's output (handling the list vs string quirk)
            content = chat_response.content
            if isinstance(content, list):
                dynamic_answer = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            else:
                dynamic_answer = str(content)
                
            answer = dynamic_answer.strip()
        except Exception as e:
            print(f"[-] Dynamic chitchat failed: {e}")
            # The bulletproof fallback
            answer = "Hello! I am Narad AI. How can I assist you with your legal and cooperative queries today?"
            
        return {"intent": intent, "source": "router", "answer": answer}
    
    print("[Routing to V2 ChromaDB & Web Search...]")
    rag_result = generate_legal_response(user_message)
    return {"intent": "RAG", "source": rag_result["source"], "answer": rag_result["answer"]}


# --- NEW: CONTEXTUAL QUERY REWRITER ---
CONDENSE_PROMPT = PromptTemplate.from_template("""
Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone, concise English search query for legal retrieval.
If the question is already standalone or not related to past history, return it exactly as is.

Chat History:
{history}

Follow-Up Query: {query}

Standalone Query:
""")

condense_chain = CONDENSE_PROMPT | gemini_llm

def contextualize_query(session_id: str, current_query: str) -> str:
    history_msgs = get_recent_history(session_id, limit=4)
    # If this is a new session with no history, skip condensation
    if not history_msgs:
        return current_query

    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_msgs])
    try:
        response = condense_chain.invoke({
            "history": formatted_history,
            "query": current_query
        })
        rewritten = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        print(f"[*] Follow-up detected. Rewritten query: {rewritten.strip()}")
        return rewritten.strip()
    except Exception as e:
        print(f"[-] Query contextualization fallback: {e}")
        return current_query