import os
import re, json 

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def ask_llm(question: str) -> dict:
    # context = retrieve_context(question)

    # if not context.strip():
    #     context = question

    system_prompt = """
        Você é um assistente que SEMPRE responde apenas com JSON válido.
        Nunca explique fora do JSON.
        """

    user_prompt = f"""
        Retorne apenas o JSON entre <json></json>

        <json>
        {{
        "response": ""
        }}
        </json>

        Pergunta:
        {question}

        Regras:
        - Responda com seu conhecimento geral.
        - Nunca invente informações, somente o que contiver no seu treinamento.
        - Responda no idioma da pergunta.
        - Se não souber a resposta, responda que não sabe e vai aprender.
        """

    response = client.chat.completions.create(
        model="xiaomi/mimo-v2-flash:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    text = response.choices[0].message.content

    # Extração segura do JSON
    match = re.search(r"<json>(.*?)</json>", text, re.S)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {"response": match.group(1).strip()}
    else:
        return {"response": text.strip()}