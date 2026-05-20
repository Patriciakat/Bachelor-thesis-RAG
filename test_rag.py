from src.chunker import load_cuad, chunk_contract, get_encoder
from src.noise_builder import get_answer_spans, get_relevant_chunks, build_context
from src.rag import generate_answer

contracts = load_cuad()
contract = contracts[0]
contract_id = contract["title"]

qa = None
for q in contract["paragraphs"][0]["qas"]:
    if (not q["is_impossible"]
            and len(q["answers"]) > 0
            and len(q["answers"][0]["text"]) > 100):
        qa = q
        break

print("Question:", qa["question"][:100])
print("Expected answer:", qa["answers"][0]["text"][:150])
print()

encoder = get_encoder()
chunks = chunk_contract(contract, encoder)
spans = get_answer_spans(qa)
relevant = get_relevant_chunks(chunks, contract_id, spans)

context = build_context(relevant, [])

print(f"Context built from {len(relevant)} relevant chunk(s)")
print("Generating answer...")

answer = generate_answer(qa["question"], context)
print("Model answer:", answer)