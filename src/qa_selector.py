import random
from src.chunker import load_cuad, chunk_contract, get_encoder
from src.noise_builder import get_answer_spans, get_relevant_chunks

CLAUSE_CATEGORIES = [
    "Governing Law",
    "Cap On Liability",
    "Termination For Convenience",
    "Ip Ownership Assignment",
    "Non-Compete"
]

EXCLUDED_CONTRACTS = {
    "MANAKOASERVICESCORP_11_21_2007-EX-7.5-STRATEGIC ALLIANCE AGREEMENT",
    "CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement"
}

QUESTIONS_PER_CATEGORY = 10
CALIBRATION_QUESTIONS_PER_CATEGORY = 4
RANDOM_SEED = 42

# Skips questions with redacted answers (*** or [***])
def has_redacted_answer(qa: dict) -> bool:
    for answer in qa["answers"]:
        if "***" in answer["text"]:
            return True
    return False

# Select 50 main questions and 20 calibration questions, balanced across the 5 clause categories, with no contract overlap between the two sets
def select_questions(all_chunks: list[dict]) -> tuple[list[dict], list[dict]]:

    random.seed(RANDOM_SEED)

    contracts = load_cuad()
    encoder = get_encoder()

    by_category = {cat: [] for cat in CLAUSE_CATEGORIES}

    for contract in contracts:
        contract_id = contract["title"]
        for qa in contract["paragraphs"][0]["qas"]:
            clause_type = qa["id"].split("__")[1]

            if clause_type not in CLAUSE_CATEGORIES:
                continue
            if qa["is_impossible"] or len(qa["answers"]) == 0:
                continue
            if has_redacted_answer(qa):
                continue
            if contract_id in EXCLUDED_CONTRACTS:
                continue
            by_category[clause_type].append((contract, qa))

    main_questions = []
    calibration_questions = []

    used_contracts = set()
    
    for clause_type, candidates in by_category.items():

        random.shuffle(candidates)

        selected = []

        for contract, qa in candidates:
            contract_id = contract["title"]

            if contract_id in used_contracts:
                continue

            answer_spans = get_answer_spans(qa)
            contract_chunks = chunk_contract(contract, encoder)
            relevant = get_relevant_chunks(
                contract_chunks, contract_id, answer_spans
            )

            if len(relevant) == 0:
                continue

            for chunk in relevant:
                chunk["clause_type"] = clause_type

            for chunk in all_chunks:
                if chunk["contract_id"] == contract_id:
                    chunk["clause_type"] = clause_type

            selected.append({
                "question": qa["question"],
                "answers": qa["answers"],
                "id": qa["id"],
                "contract_id": contract_id,
                "clause_type": clause_type,
                "relevant_chunks": relevant
            })

            used_contracts.add(contract_id)

            if len(selected) == QUESTIONS_PER_CATEGORY + CALIBRATION_QUESTIONS_PER_CATEGORY:
                break

        if len(selected) < QUESTIONS_PER_CATEGORY + CALIBRATION_QUESTIONS_PER_CATEGORY:
            raise ValueError(
                f"Not enough questions for {clause_type}: "
                f"found {len(selected)}, need "
                f"{QUESTIONS_PER_CATEGORY + CALIBRATION_QUESTIONS_PER_CATEGORY}"
            )
        
        calibration_questions.extend(selected[:CALIBRATION_QUESTIONS_PER_CATEGORY])
        main_questions.extend(selected[CALIBRATION_QUESTIONS_PER_CATEGORY:])

    print(f"Selected {len(main_questions)} main questions")
    print(f"Selected {len(calibration_questions)} calibration questions")

    main_contracts = {q["contract_id"] for q in main_questions}
    cal_contracts = {q["contract_id"] for q in calibration_questions}
    overlap = main_contracts & cal_contracts
    if overlap:
        raise ValueError(f"Contract overlap between main and calibration: {overlap}")

    return main_questions, calibration_questions