from src.chunker import load_cuad, chunk_contract, get_encoder
from src.noise_builder import (
    extract_clause_type, get_answer_spans,
    is_relevant, get_relevant_chunks
)

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

print("Question:", qa["question"][:80])
print("Clause type:", extract_clause_type(qa["id"]))
print("Answers:")
for a in qa["answers"]:
    print(f'  start={a["answer_start"]} text="{a["text"][:60]}"')

spans = get_answer_spans(qa)
print("Spans:", spans)

encoder = get_encoder()
chunks = chunk_contract(contract, encoder)
print(f"\nTotal chunks in contract: {len(chunks)}")

relevant = get_relevant_chunks(chunks, contract_id, spans)
print(f"Relevant chunks found: {len(relevant)}")

for r in relevant:
    print(f"\n  chunk {r['chunk_index']}: chars {r['char_start']}-{r['char_end']}")
    
    for span in spans:
        if r["char_start"] <= span["start"] <= r["char_end"]:
            offset = span["start"] - r["char_start"]
            extracted = r["text"][offset:offset + len(span["text"])]
            print(f"  answer found in chunk: '{extracted[:100]}'")
            print(f"  matches original:       '{span['text'][:100]}'")
            print(f"  exact match: {extracted == span['text']}")