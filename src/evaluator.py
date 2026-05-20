import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

def get_ragas_llm():
    return LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    )

def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    )

def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = None
) -> dict:
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    metrics = [faithfulness, answer_relevancy, context_precision]

    if ground_truth:
        data["ground_truth"] = [ground_truth]
        metrics.append(context_recall)

    dataset = Dataset.from_dict(data)

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    def _extract(value):
        return float(value[0]) if isinstance(value, list) else float(value)

    scores = {
        "faithfulness": _extract(result["faithfulness"]),
        "answer_relevancy": _extract(result["answer_relevancy"]),
        "context_precision": _extract(result["context_precision"]),
    }

    if ground_truth:
        scores["context_recall"] = _extract(result["context_recall"])

    return scores