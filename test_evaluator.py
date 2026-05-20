from src.chunker import load_cuad, chunk_contract, get_encoder
from src.noise_builder import get_answer_spans, get_relevant_chunks, build_context
from src.rag import generate_answer
from src.evaluator import evaluate_response

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

encoder = get_encoder()
chunks = chunk_contract(contract, encoder)
spans = get_answer_spans(qa)
relevant = get_relevant_chunks(chunks, contract_id, spans)

context = build_context(relevant, [])
context_texts = [c["text"] for c in relevant]

answer = generate_answer(qa["question"], context)
ground_truth = qa["answers"][0]["text"]

scores = evaluate_response(
    question=qa["question"],
    answer=answer,
    contexts=context_texts,
    ground_truth=ground_truth
)

print("RAGAS scores:")
for metric, score in scores.items():
    value = score[0] if isinstance(score, list) else score
    print(f"  {metric}: {value:.3f}")