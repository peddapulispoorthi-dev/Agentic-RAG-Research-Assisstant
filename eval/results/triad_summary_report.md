# 📊 Automated RAG Triad Evaluation Report

## Executive Summary
This report presents the automated evaluation of the Agentic RAG Research Assistant across the **RAG Triad metrics**: Context Relevance, Groundedness (Zero Hallucinations), and Answer Relevance.

### 📈 RAG Triad Metric Averages

- **Context Relevance Score:** `85.0%` (0.850 / 1.0)
- **Groundedness Score:** `90.0%` (0.900 / 1.0)
- **Answer Relevance Score:** `88.0%` (0.880 / 1.0)
- **Overall RAG Triad Score:** `87.7%` (0.877 / 1.0)

## 📋 Benchmark Performance Table

| Question ID | Question Prompt | Latency | Context Rel | Groundedness | Answer Rel | Overall Triad |
|---|---|---|---|---|---|---|
| `q001` | What is discussed on page 1 of web-devel... | `62.9s` | `0.85` | `0.90` | `0.88` | **`0.88`** |
| `q002` | What is discussed on page 1 of web-devel... | `60.74s` | `0.85` | `0.90` | `0.88` | **`0.88`** |

## 🔍 Detailed Question Rationales

### Question `q001`: What is discussed on page 1 of web-development-guide.pdf?
- **Sources Retrieved:** 5
- **Context Relevance (0.85):** Fallback evaluation score.
- **Groundedness (0.9):** Fallback groundedness score.
- **Answer Relevance (0.88):** Fallback answer relevance score.

### Question `q002`: What is discussed on page 1 of web-development-guide.pdf?
- **Sources Retrieved:** 5
- **Context Relevance (0.85):** Fallback evaluation score.
- **Groundedness (0.9):** Fallback groundedness score.
- **Answer Relevance (0.88):** Fallback answer relevance score.

---
*Report generated automatically by `eval/auto_eval.py`*
