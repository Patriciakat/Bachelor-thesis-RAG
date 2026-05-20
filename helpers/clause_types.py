import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunker import load_cuad

contracts = load_cuad()

clause_types = set()
for contract in contracts:
    for qa in contract["paragraphs"][0]["qas"]:
        clause_type = qa["id"].split("__")[1]
        clause_types.add(clause_type)

for ct in sorted(clause_types):
    print(ct)