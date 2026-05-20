import random
from src.store import search

RELEVANCE_THRESHOLD = 0.7

def extract_clause_type(qa_id: str) -> str:
    return qa_id.split("__")[1]

def get_answer_spans(qa: dict) -> list[dict]:
    spans = []
    for answer in qa["answers"]:
        spans.append({
            "text": answer["text"],
            "start": answer["answer_start"],
            "end": answer["answer_start"] + len(answer["text"])
        })
    return spans

def compute_overlap_fraction(chunk: dict, span: dict) -> float:
    chunk_start = chunk["char_start"]
    chunk_end = chunk["char_end"]
    answer_start = span["start"]
    answer_end = span["end"]
    answer_len = answer_end - answer_start

    if answer_len == 0:
        return 0.0

    overlap_start = max(chunk_start, answer_start)
    overlap_end = min(chunk_end, answer_end)
    overlap = max(0, overlap_end - overlap_start)

    return overlap / answer_len

def is_relevant(chunk: dict, answer_spans: list[dict]) -> bool:
    for span in answer_spans:
        if compute_overlap_fraction(chunk, span) >= RELEVANCE_THRESHOLD:
            return True
    return False

def get_relevant_chunks(all_chunks: list[dict],
                        contract_id: str,
                        answer_spans: list[dict]) -> list[dict]:
    contract_chunks = [
        c for c in all_chunks
        if c["contract_id"] == contract_id
    ]
    return [c for c in contract_chunks if is_relevant(c, answer_spans)]


def build_random_pool(all_chunks: list[dict],
                      contract_id: str,
                      answer_spans: list[dict]) -> list[dict]:
    """
    Random noise pool: chunks from completely different contracts.
    """
    return [
        c for c in all_chunks
        if c["contract_id"] != contract_id
        and not is_relevant(c, answer_spans)
    ]

def build_topical_pool(all_chunks: list[dict],
                       contract_id: str,
                       clause_type: str,
                       answer_spans: list[dict]) -> list[dict]:
    """
    Topical noise pool: same clause type, different contract.
    """
    return [
        c for c in all_chunks
        if c["contract_id"] != contract_id
        and c.get("clause_type") == clause_type
        and not is_relevant(c, answer_spans)
    ]

def build_adversarial_pool(query: str,
                           contract_id: str,
                           answer_spans: list[dict],
                           n: int) -> list[dict]:
    """
    Adversarial noise pool: semantically most similar chunks from different contracts, retrieved via vector search.
    """
    candidates = search(
        query=query,
        n_results=n * 3,
        exclude_contract_id=contract_id
    )
    pool = []
    for r in candidates:
        chunk = {
            "text": r["text"],
            "contract_id": r["metadata"]["contract_id"],
            "char_start": r["metadata"]["char_start"],
            "char_end": r["metadata"]["char_end"],
            "chunk_index": r["metadata"]["chunk_index"],
            "token_count": r["metadata"]["token_count"]
        }
        if not is_relevant(chunk, answer_spans):
            pool.append(chunk)

    return pool[:n]

def sample_noise(pool: list[dict], n: int) -> list[dict]:
    if len(pool) < n:
        raise ValueError(
            f"Noise pool too small: need {n}, have {len(pool)}"
        )
    return random.sample(pool, n)

def build_context(relevant: list[dict], noise: list[dict]) -> str:
    """
    Combine relevant and noise chunks into one context string.
    """
    all_chunks = relevant + noise
    random.shuffle(all_chunks)
    return "\n\n---\n\n".join(c["text"] for c in all_chunks)