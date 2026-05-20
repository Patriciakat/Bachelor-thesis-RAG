import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0

PROMPT_TEMPLATE = """You are a legal contract analyst. Answer the question using only the exact text from the contract excerpts below. Quote the relevant parts directly. Do not add any analysis, recommendations, or commentary.

Contract excerpts:
{context}

Question: {question}

Answer:"""

def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, context: str) -> str:
    client = get_client()

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()