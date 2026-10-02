import json
import os
from pathlib import Path
from src.retrievers import get_retriever

def generate_questions():
    retriever = get_retriever()
    if not retriever.chunks:
        print("No chunks found in index. Please run ingestion first.")
        return

    questions = []
    # Pick a few sample chunks to draft basic questions from metadata
    for i, chunk in enumerate(retriever.chunks[:5]):
        source = chunk.metadata.get("source", "doc.pdf")
        page = chunk.metadata.get("page", 1)
        
        questions.append({
            "id": f"q00{i+1}",
            "question": f"What is discussed on page {page} of {source}?",
            "ground_truth": chunk.page_content[:150] + "...",
            "expected_source": source,
            "type": "in_docs"
        })
    
    # Add an out-of-docs sample
    questions.append({
        "id": f"q00{len(questions)+1}",
        "question": "What is the current weather forecast for Tokyo today?",
        "ground_truth": "NOT_IN_DOCS",
        "expected_source": None,
        "type": "out_of_docs"
    })

    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = eval_dir / "questions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Generated draft evaluation questions at {output_path}. Please review and customize them!")

if __name__ == "__main__":
    generate_questions()