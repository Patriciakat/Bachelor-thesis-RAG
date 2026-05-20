import json
import random
import pandas as pd
from src.chunker import chunk_all_contracts
from src.qa_selector import select_questions
from src.noise_builder import build_context
from src.rag import generate_answer
from src.evaluator import evaluate_response

RANDOM_SEED = 42
K_VALUES = [1, 2, 3, 4, 5]
RESULTS_PATH = "results/calibration_results.csv"

def run_calibration():
    random.seed(RANDOM_SEED)

    all_chunks = chunk_all_contracts()

    _, calibration_questions = select_questions(all_chunks)

    rows = []

    for k in K_VALUES:
        print(f"*** Testing k={k} ***")
        k_scores = []

        for i, question in enumerate(calibration_questions):
            q_text = question["question"]
            contract_id = question["contract_id"]
            clause_type = question["clause_type"]
            relevant_chunks = question["relevant_chunks"]
            ground_truth = question["answers"][0]["text"]

            available = relevant_chunks[:k]

            context = build_context(available, [])
            context_texts = [c["text"] for c in available]

            # Answer generation
            answer = generate_answer(q_text, context)

            # Evaluation
            scores = evaluate_response(
                question=q_text,
                answer=answer,
                contexts=context_texts,
                ground_truth=ground_truth
            )

            row = {
                "k": k,
                "question_index": i,
                "clause_type": clause_type,
                "contract_id": contract_id,
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"],
                "context_precision": scores["context_precision"],
                "context_recall": scores.get("context_recall", None)
            }
            rows.append(row)

            print(
                f"  q{i+1}/{len(calibration_questions)} "
                f"[{clause_type[:20]}] "
                f"faith={scores['faithfulness']:.3f} "
                f"rel={scores['answer_relevancy']:.3f}"
            )

        k_rows = [r for r in rows if r["k"] == k]
        avg_faith = sum(r["faithfulness"] for r in k_rows) / len(k_rows)
        avg_rel = sum(r["answer_relevancy"] for r in k_rows) / len(k_rows)
        print(f"  k={k} average: faithfulness={avg_faith:.3f} "
              f"answer_relevancy={avg_rel:.3f}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)

    print("\n***CALIBRATION SUMMARY***")
    summary = df.groupby("k")[
        ["faithfulness", "answer_relevancy",
         "context_precision", "context_recall"]
    ].mean().round(3)
    print(summary)

    best_k = df.groupby("k")["faithfulness"].mean().idxmax()
    print(f"\nOptimal k = {best_k}")

    return best_k


if __name__ == "__main__":
    run_calibration()