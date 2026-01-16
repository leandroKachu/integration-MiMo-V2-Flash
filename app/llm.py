import os
import re, json 
import unicodedata
from tavily import TavilyClient

from google import genai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CONFIDENCE_THRESHOLD = 0.65

key = os.getenv("GEMINI_API_KEY")
client_genai = genai.Client(api_key=key)
tavily = TavilyClient(os.getenv("TAVILY_API_KEY"))

client_mimo = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

exa_client = OpenAI(
    base_url="https://api.exa.ai",
    api_key=os.getenv("EXA_API_KEY"),
)


def ask_mimo(question: str) -> dict:
    system_prompt = "Responda SOMENTE com JSON válido."

    user_prompt = f"""
    Formato obrigatório:
    {{ "response": "" }}

    Pergunta:
    {question}

    Regras:
    - Não invente informações
    - Se não souber, diga claramente
    """

    resp = client_mimo.chat.completions.create(
        model="xiaomi/mimo-v2-flash:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return safe_json(resp.choices[0].message.content)

def ask_gemini(question: str) -> dict:
    prompt = f"""
    Responda SOMENTE com JSON válido.

    Formato:
    {{ "response": "" }}

    Pergunta:
    {question}

    Regras:
    - Seja técnico
    - Não invente
    """

    
    response = client_genai.models.generate_content (
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    
    return safe_json(response.text)

def web_search(question: str) -> str:
    response = tavily.search(
        query=question,
        search_depth="advanced",
        max_results=5
    )

    contents = []
    for r in response.get("results", []):
        if r.get("content"):
            contents.append(r["content"])

    return "\n".join(contents)


def exa_from_web(question: str, web_data: str) -> dict:
    prompt = f"""
    Use APENAS as informações abaixo para responder.

    Informações:
    {web_data}

    Pergunta:
    {question}

    Formato obrigatório:
    {{ "response": "" }}

    Regras:
    - Não invente informações
    - Se não houver resposta clara, diga isso
    - Responda no idioma da pergunta
    """

    response = exa_client.chat.completions.create(
        model="exa-pro",
        messages=[
            {"role": "system", "content": "Responda apenas com JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        stream=False,
    )

    text = response.choices[0].message.content
    return safe_json(text)


def safe_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {"response": ""}

    try:
        data = json.loads(match.group())
        return {"response": data.get("response", "").strip()}
    except json.JSONDecodeError:
        return {"response": match.group().strip()}

def is_good_answer(answer: dict) -> bool:
    text = answer.get("response", "").lower()
    if len(text) < 40:
        return False

    bad_signals = [
        "não tenho certeza",
        "não encontrei",
        "não é possível",
        "informação insuficiente",
        "não tenho"
    ]

    return not any(b in text for b in bad_signals)

def ask_llm(question: str, name: str) -> dict:
    # 1️⃣ Mimo
    if is_today_question(question):
        web_data = web_search(question)
        if web_data.strip():
            return exa_from_web(question, web_data)
        
    mimo_answer = ask_mimo(question)
    if is_good_answer(mimo_answer):
        return mimo_answer

    # 2️⃣ Gemini
    gemini_answer = ask_gemini(question)
    if is_good_answer(gemini_answer):
        return gemini_answer

    # 3️⃣ Web
    web_data = web_search(question)
    if web_data.strip():
        return exa_from_web(question, web_data)

    return {
        "response": "Não encontrei informação confiável para responder."
    }

def normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text

def is_today_question(question: str) -> bool:
    q = normalize(question)

    patterns = [
        r"\bhoje\b",
        r"\bno dia de hoje\b",
        r"\bo que aconteceu hoje\b",
        r"\baconteceu hoje\b",
        r"\bagora\b",
        r"\bneste momento\b",
        r"\bultimas noticias\b",
        r"\brecente(mente)?\b",
    ]

    return any(re.search(p, q) for p in patterns)

def is_time_sensitive(question: str) -> bool:
    q = normalize(question)

    time_words = [
        "hoje", "agora", "recente", "ultimas", "hoje em dia"
    ]

    news_words = [
        "aconteceu", "noticia", "evento", "caso", "situacao"
    ]

    return (
        any(t in q for t in time_words)
        and any(n in q for n in news_words)
    )