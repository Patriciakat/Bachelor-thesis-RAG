import json
from json import encoder
import tiktoken

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
LLM_MODEL = "gpt-4o-mini"

def load_cuad(path: str = "data/CUADv1.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw["data"]

def get_encoder():
    """o200k_base tokenizer"""
    return tiktoken.encoding_for_model(LLM_MODEL)

def chunk_contract(contract: dict, encoder) -> list[dict]:
    contract_id = contract["title"]
    full_text = contract["paragraphs"][0]["context"]
    
    tokens = encoder.encode(full_text)
    chunks = []
    chunk_index = 0
    start_token = 0

    while start_token < len(tokens):
        end_token = min(start_token + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start_token:end_token]
        chunk_text = encoder.decode(chunk_tokens)

        char_start = len(encoder.decode(tokens[:start_token]))
        char_end = len(encoder.decode(tokens[:end_token]))

        chunks.append({
            "text": chunk_text,
            "contract_id": contract_id,
            "chunk_index": chunk_index,
            "token_count": len(chunk_tokens),
            "char_start": char_start,
            "char_end": char_end
        })
        chunk_index += 1
        start_token += CHUNK_SIZE - CHUNK_OVERLAP

        if end_token == len(tokens):
            break
            
    return chunks

def chunk_all_contracts(path: str = "data/CUADv1.json") -> list[dict]:
    contracts = load_cuad(path)
    encoder = get_encoder()
    all_chunks = []

    for contract in contracts:
        chunks = chunk_contract(contract, encoder)
        all_chunks.extend(chunks)
    return all_chunks