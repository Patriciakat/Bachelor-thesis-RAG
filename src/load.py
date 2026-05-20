import json
from collections import Counter

file_path = "data/CUADv1.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

total_questions = 0
impossible_questions = 0
answerable_questions = 0
clause_types = Counter()

for document in data["data"]:
    for paragraph in document["paragraphs"]:
        for qa in paragraph["qas"]:
            total_questions += 1

            if qa["is_impossible"]:
                impossible_questions += 1
            else:
                answerable_questions += 1
            clause_type = qa["id"].split("__")[-1]
            clause_types[clause_type] += 1

print(f"Total questions: {total_questions}")
print(f"Answerable questions: {answerable_questions}")
print(f"Impossible questions: {impossible_questions}")
print(f"Unique clause types: {len(clause_types)}")