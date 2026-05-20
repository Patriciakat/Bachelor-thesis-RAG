from src.chunker import chunk_all_contracts
from src.qa_selector import select_questions

all_chunks = chunk_all_contracts()

main_q, cal_q = select_questions(all_chunks)

print("\nMain questions by category:")
from collections import Counter
cats = Counter(q["clause_type"] for q in main_q)
for cat, count in cats.items():
    print(f"  {cat}: {count}")

print("\nCalibration questions by category:")
cats = Counter(q["clause_type"] for q in cal_q)
for cat, count in cats.items():
    print(f"  {cat}: {count}")

print("\nSample main question:")
q = main_q[0]
print(f"  clause: {q['clause_type']}")
print(f"  contract: {q['contract_id'][:60]}")
print(f"  relevant chunks: {len(q['relevant_chunks'])}")
print(f"  answer: {q['answers'][0]['text'][:100]}")