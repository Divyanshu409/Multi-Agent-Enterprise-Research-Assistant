from __future__ import annotations
import time
import argparse
import json
import sys
from pathlib import Path
from ragas.run_config import RunConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  
from src.graph.graph import run_query  

EVAL_DIR = Path(__file__).resolve().parent


def load_testset() -> list[dict]:
    path = EVAL_DIR / "qa_testset.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_samples(testset: list[dict]) -> list[dict]:
    samples = []
    for row in testset:
        print(f"  running: {row['question'][:80]}...")
        result = run_query(row["question"])
        time.sleep(15)
        contexts = [c["text"] for c in result["retrieved_chunks"]]
        samples.append({
            "user_input": row["question"],
            "response": result["final_answer"],
            "retrieved_contexts": contexts,
            "reference": row["ground_truth"],
        })
    return samples


def build_ragas_llm_and_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    embeddings = None

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        chat = ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key)
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    elif settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat = ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key)
    else:
        raise SystemExit(
            "RAGAS evaluation needs a real LLM judge. Set LLM_PROVIDER=anthropic, "
            "openai, or gemini with a valid API key (LLM_PROVIDER=mock is for "
            "tests/ only, not for eval/)."
        )

    if settings.llm_provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", google_api_key=settings.gemini_api_key
            )
        except Exception:
            embeddings = None
    else:
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
        except Exception:
            embeddings = None

    return LangchainLLMWrapper(chat), (LangchainEmbeddingsWrapper(embeddings) if embeddings else None)


def run_evaluation(samples: list[dict]):
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    ragas_llm, ragas_embeddings = build_ragas_llm_and_embeddings()
    dataset = EvaluationDataset.from_list(samples)

    metrics = [faithfulness, context_precision, context_recall]
    if ragas_embeddings is not None:
        metrics.append(answer_relevancy) 

    result = evaluate(
        dataset=dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings,
        raise_exceptions=True,
        run_config=RunConfig(max_workers=1, timeout=120),
    )
    return result


def write_reports(result, samples: list[dict]) -> dict:
    df = result.to_pandas()
    scores = {col: float(df[col].mean()) for col in df.columns if df[col].dtype.kind in "fc"}

    (EVAL_DIR / "report.json").write_text(json.dumps({
        "scores": scores,
        "n_samples": len(samples),
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }, indent=2), encoding="utf-8")

    lines = [
        "# RAGAS Evaluation Report",
        "",
        f"- Samples: {len(samples)}",
        f"- LLM provider (drafting/critique): `{settings.llm_provider}`",
        f"- Embedding provider (retrieval): `{settings.embedding_provider}`",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]
    for metric, score in scores.items():
        lines.append(f"| {metric} | {score:.3f} |")
    (EVAL_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-under", type=float, default=None,
                         help="Exit 1 if mean faithfulness score is below this threshold (CI gate mode).")
    args = parser.parse_args()

    if settings.llm_provider == "mock":
        print("ERROR: LLM_PROVIDER=mock cannot be used for RAGAS evaluation (needs a real judge).")
        sys.exit(1)

    print(f"Loading test set...")
    testset = load_testset()[:3]
    print(f"Running {len(testset)} questions through the graph...")
    samples = collect_samples(testset)

    print("Scoring with RAGAS...")
    result = run_evaluation(samples)
    scores = write_reports(result, samples)

    print("\n=== RAGAS scores ===")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.3f}")
    print(f"\nReports written to {EVAL_DIR / 'report.json'} and {EVAL_DIR / 'report.md'}")

    if args.fail_under is not None:
        faithfulness_score = scores.get("faithfulness", 0.0)
        if faithfulness_score < args.fail_under:
            print(f"\nFAIL: faithfulness {faithfulness_score:.3f} < threshold {args.fail_under}")
            sys.exit(1)
        print(f"\nPASS: faithfulness {faithfulness_score:.3f} >= threshold {args.fail_under}")


if __name__ == "__main__":
    main()
