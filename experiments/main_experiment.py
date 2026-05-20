import random
import pandas as pd
from src.chunker import chunk_all_contracts
from src.qa_selector import select_questions
from src.noise_builder import (
    build_random_pool,
    build_topical_pool,
    build_adversarial_pool,
    sample_noise,
    build_context
)
from src.rag import generate_answer
from src.evaluator import evaluate_response

RANDOM_SEED = 42
OPTIMAL_K = 4
RESULTS_PATH = "results/main_experiment_results.csv"

SNR_LEVELS = {
    1.00: 0,   # baseline
    0.50: 4,   # n = k
    0.33: 8,   # n = 2k
    0.25: 12,  # n = 3k
    0.17: 20,  # n = 5k
}

NOISE_TYPES = ["random", "topical", "adversarial"]


def run_experiment():
    random.seed(RANDOM_SEED)

    all_chunks = chunk_all_contracts()

    main_questions, _ = select_questions(all_chunks)

    print(f"\nRunning main experiment:")
    print(f"  Questions: {len(main_questions)}")
    print(f"  k (relevant chunks): {OPTIMAL_K}")
    print(f"  SNR levels: {list(SNR_LEVELS.keys())}")
    print(f"  Noise types: {NOISE_TYPES}")
    total = len(main_questions) * (1 + (len(SNR_LEVELS) - 1) * len(NOISE_TYPES))
    print(f"  Total conditions: {total}")
    print()

    rows = []
    condition_count = 0

    for q_idx, question in enumerate(main_questions):
        q_text = question["question"]
        contract_id = question["contract_id"]
        clause_type = question["clause_type"]
        relevant_chunks = question["relevant_chunks"][:OPTIMAL_K]
        ground_truth = question["answers"][0]["text"]

        print(f"Question {q_idx + 1}/{len(main_questions)} "
              f"[{clause_type}] {contract_id[:40]}")

        answer_spans = [
            {
                "text": a["text"],
                "start": a["answer_start"],
                "end": a["answer_start"] + len(a["text"])
            }
            for a in question["answers"]
        ]

        random_pool = build_random_pool(
            all_chunks, contract_id, answer_spans
        )
        topical_pool = build_topical_pool(
            all_chunks, contract_id, clause_type, answer_spans
        )
        adversarial_pool = build_adversarial_pool(
            q_text, contract_id, answer_spans,
            n=max(SNR_LEVELS.values())
        )

        for snr, n_noise in SNR_LEVELS.items():

            # Baseline condition (no noise)
            if n_noise == 0:
                context = build_context(relevant_chunks, [])
                context_texts = [c["text"] for c in relevant_chunks]
                answer = generate_answer(q_text, context)
                scores = evaluate_response(
                    question=q_text,
                    answer=answer,
                    contexts=context_texts,
                    ground_truth=ground_truth
                )
                rows.append({
                    "question_index": q_idx,
                    "clause_type": clause_type,
                    "contract_id": contract_id,
                    "noise_type": "baseline",
                    "snr": snr,
                    "n_noise": n_noise,
                    **scores
                })
                condition_count += 1
                print(f"  baseline SNR=1.0 "
                      f"faith={scores['faithfulness']:.3f} "
                      f"rel={scores['answer_relevancy']:.3f}")
                continue

            # Noise conditions
            for noise_type in NOISE_TYPES:

                # Sample noise chunks
                try:
                    if noise_type == "random":
                        noise = sample_noise(random_pool, n_noise)
                    elif noise_type == "topical":
                        if len(topical_pool) < n_noise:
                            print(f"  Topical pool too small "
                                  f"({len(topical_pool)} < {n_noise}), "
                                  f"skipping")
                            continue
                        noise = sample_noise(topical_pool, n_noise)
                    elif noise_type == "adversarial":
                        if len(adversarial_pool) < n_noise:
                            print(f"  Adversarial pool too small "
                                  f"({len(adversarial_pool)} < {n_noise}), "
                                  f"skipping")
                            continue
                        noise = sample_noise(adversarial_pool, n_noise)

                except ValueError as e:
                    print(f" {e}, skipping")
                    continue

                context = build_context(relevant_chunks, noise)
                context_texts = [c["text"] for c in relevant_chunks + noise]

                answer = generate_answer(q_text, context)
                scores = evaluate_response(
                    question=q_text,
                    answer=answer,
                    contexts=context_texts,
                    ground_truth=ground_truth
                )

                rows.append({
                    "question_index": q_idx,
                    "clause_type": clause_type,
                    "contract_id": contract_id,
                    "noise_type": noise_type,
                    "snr": snr,
                    "n_noise": n_noise,
                    **scores
                })
                condition_count += 1

                print(f"  {noise_type:12} SNR={snr:.2f} "
                      f"faith={scores['faithfulness']:.3f} "
                      f"rel={scores['answer_relevancy']:.3f}")

        pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    df = pd.DataFrame(rows)

    print("\nResults by noise type and SNR:")
    summary = df.groupby(["noise_type", "snr"])[
        ["faithfulness", "answer_relevancy",
         "context_precision", "context_recall"]
    ].mean().round(3)
    print(summary)

    return df

if __name__ == "__main__":
    run_experiment()