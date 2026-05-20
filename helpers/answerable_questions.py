import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunker import load_cuad

contracts = load_cuad()

categories = [
    "Governing Law",
    "Cap On Liability",
    "Termination For Convenience",
    "Ip Ownership Assignment",
    "Non-Compete"
]

counts = {cat: 0 for cat in categories}

for contract in contracts:
    for qa in contract["paragraphs"][0]["qas"]:
        clause_type = qa["id"].split("__")[1]
        if (clause_type in categories 
            and not qa["is_impossible"] 
            and len(qa["answers"]) > 0):
            counts[clause_type] += 1

for cat, count in counts.items():
    print(f"{cat}: {count} answerable questions")