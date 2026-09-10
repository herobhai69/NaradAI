import os
import re
import torch
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from core.config import CONFIG, settings

# Robust import for DuckDuckGo Search
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = str(BASE_DIR / "chroma_db_final")

def get_vectorstore():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Embeddings on device: {device.upper()}")
    
    # 1. Place the token inside model_kwargs, NOT encode_kwargs
    model_kwargs = {"device": device}
    if getattr(settings, "HF_TOKEN", None):
        model_kwargs["token"] = settings.HF_TOKEN
        
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs
    )
    
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

VECTOR_STORE = get_vectorstore()

NARAD_LOCAL_PROMPT = """You are Narad AI, an empathetic and highly knowledgeable female legal assistant for the Ministry of Cooperation (India).
Your goal is to provide warm, reassuring, yet strictly accurate legal and scheme guidance based on the provided official context.

STRICT INSTRUCTIONS:
1. Empathy First: Adopt a supportive, human-like, and professional female persona. If a user is facing a crisis (like crop failure or drought), briefly offer a supportive word before providing facts.
2. Format: Answer in 3 to 4 clear, spoken-word friendly bullet points.
3. Fact Grounding: Base your answer EXCLUSIVELY on the facts stated in the context. Do not invent rules.
4. SOURCE CITATION: End your response by citing the exact source file from the context (e.g., "Source: <file_name>.pdf").

Context:
{context}"""

def is_context_relevant(query: str, context: str, llm) -> bool:
    """
    Prevents hallucinations: strictly tests if the retrieved documents 
    actually contain information that directly answers the query.
    """
    grading_prompt = f"""You are a strict relevance filter for an Indian governance and legal chatbot.
Determine if the provided context contains direct, specific factual facts to answer the query.
If the context is about a different law, act, or irrelevant topic (e.g., FCRA when asked about scholarships), answer 'NO'.

User Query: {query}

Context:
{context}

Answer ONLY with 'YES' or 'NO':"""
    try:
        response = llm.invoke(grading_prompt)
        text = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        return "YES" in text.strip().upper()
    except Exception as e:
        print(f"[-] Relevance grading error: {e}")
        return False

def generate_legal_response(english_query: str) -> dict:
    print(f"[*] Querying RAG Database for: '{english_query}'...")
    
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(english_query)
    
    ollama_llm = ChatOllama(
        model=CONFIG["rag_engine"]["ollama"]["model"],
        temperature=CONFIG["rag_engine"]["ollama"]["temperature"]
    )
    
    groq_llm = ChatGroq(
        groq_api_key=settings.GROQ_API_KEY, 
        model_name=CONFIG["rag_engine"]["groq"]["model"], 
        temperature=CONFIG["rag_engine"]["groq"]["temperature"]
    )

    primary_name = CONFIG.get("rag_engine", {}).get("primary", "ollama").lower()
    resilient_llm = ollama_llm.with_fallbacks([groq_llm]) if primary_name == "ollama" else groq_llm.with_fallbacks([ollama_llm])

    # 1. Local RAG Verification
    if retrieved_docs:
        context_parts = [
            f"--- Source: {doc.metadata.get('source', 'Official_Document.pdf')} ---\n{doc.page_content}" 
            for doc in retrieved_docs
        ]
        local_context = "\n\n".join(context_parts)
        
        # Use Groq for lightning-fast (<200ms) relevance verification
        print("[*] Validating local document relevance...")
        if is_context_relevant(english_query, local_context, groq_llm):
            print("[+] Verified relevant local data. Generating response...")
            prompt = ChatPromptTemplate.from_messages([
                ("system", NARAD_LOCAL_PROMPT),
                ("human", "User Query: {input}")
            ])
            chain = prompt | resilient_llm
            response = chain.invoke({"context": local_context, "input": english_query})
            return {"source": "local_db", "answer": response.content.strip()}
        else:
            print("[-] Retrieved local documents are irrelevant to this query. Dodging hallucination...")

    # 2. Live DuckDuckGo Web Search Fallback
    print("[*] Falling back to live web search via DuckDuckGo...")
    try:
        ddgs = DDGS()
        web_results = list(ddgs.text(english_query, max_results=4))
        
        if not web_results:
            raise ValueError("No search results returned.")

        # Extract authentic URLs and clean snippet bodies
        sources = []
        snippets = []
        for res in web_results:
            url = res.get("href")
            body = res.get("body", "")
            if url:
                sources.append(url)
            if body:
                snippets.append(body)

        web_context = "\n\n".join(snippets)
        
        web_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Narad AI, an empathetic and elite female legal assistant for the Ministry of Cooperation (India).
Adopt a warm, reassuring, and professional human persona. 
Provide a clear, accurate, and actionable answer in 3 to 4 bullet points based strictly on the provided web context.
Do NOT invent citations or cite PDFs. Output ONLY the answer points."""),
            ("human", "User Query: {input}\n\nContext:\n{context}")
        ])
        
        web_chain = web_prompt | resilient_llm
        web_response = web_chain.invoke({"context": web_context, "input": english_query})
        
        # Deduplicate and cleanly format real URLs
        unique_urls = list(dict.fromkeys(sources))
        url_citations = "\n".join([f"• {url}" for url in unique_urls])
        
        final_answer = f"{web_response.content.strip()}\n\n🌐 **Sources:**\n{url_citations}"
        return {"source": "live_web", "answer": final_answer}
        
    except Exception as e:
        print(f"[-] Web search failed: {e}")
        return {
            "source": "fallback", 
            "answer": "I apologize, but I currently do not have access to verified guidelines for that specific query. Please consult the nearest cooperative department or official portal."
        }