import json
import os
import time
from pathlib import Path
from src.baseline import run_baseline

def evaluate_configurations():
    questions_path = Path("eval/questions.json")
    if not questions_path.exists():
        print("eval/questions.json not found. Run generate_questions.py first.")
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Let's test with just the first 2 questions for now to avoid rate limits
    questions = questions[:2]

    modes = ["dense", "hybrid", "hybrid_rerank"]
    results = {mode: {"hits": 0, "total": len(questions)} for mode in modes}

    print(f"Running evaluation over {len(questions)} questions across baselines (with rate-limit throttling)...")

    for q_item in questions:
        q_text = q_item["question"]
        expected_src = q_item["expected_source"]
        q_type = q_item["type"]

        for mode in modes:
            print(f"Evaluating mode '{mode}' for question: '{q_text[:30]}...'")
            try:
                answer, docs = run_baseline(q_text, mode=mode)
                
                # Check hit rate@5
                if q_type == "in_docs" and expected_src:
                    hit = any(doc.metadata.get("source") == expected_src for doc in docs)
                    if hit:
                        results[mode]["hits"] += 1
            except Exception as e:
                print(f"Error during evaluation of {mode}: {e}")

            # Sleep 12 seconds between calls to respect free tier (5 RPM limit)
            time.sleep(12)

    # Write results summary
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Summary: Baseline Configurations\n\n")
        f.write("| Configuration | Retrieval Hit Rate@5 |\n")
        f.write("|---|---|\n")
        for mode in modes:
            hits = results[mode]["hits"]
            total = results[mode]["total"]
            rate = (hits / total) * 100 if total > 0 else 0
            f.write(f"| `{mode}` | {rate:.1f}% ({hits}/{total}) |\n")

    print(f"Evaluation complete! Summary written to {summary_path}")

if __name__ == "__main__":
    evaluate_configurations()