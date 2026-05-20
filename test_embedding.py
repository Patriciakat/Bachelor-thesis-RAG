import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.embeddings.create(
    input="liability shall not exceed five hundred dollars",
    model="text-embedding-3-small"
)

vector = response.data[0].embedding
print(f"Vector length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")