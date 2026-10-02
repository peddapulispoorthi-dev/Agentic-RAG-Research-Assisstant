ROUTER_PROMPT = """You route user questions to the appropriate handler.
Choose `deep_research` if the prompt asks for a deep research report, comprehensive synthesis, structured academic review, multi-perspective breakdown, or detailed analysis.
Choose `vectorstore` if the question is about specific content in the user's uploaded documents (research papers/docs).
Choose `web_search` if it asks for recent events or facts clearly outside the documents.
Choose `chitchat` for greetings, thanks, or general questions about what you can do.
When unsure between vectorstore and deep_research, choose vectorstore."""

PLANNER_PROMPT = """You are a Lead AI Systems Architect and Research Strategist. Draft a sequential, 3-to-4 step research plan to address the user's prompt thoroughly.
Each step must focus on a specific sub-dimension:
- Step 1: Technical Definition & Core Architecture
- Step 2: Comparative Analysis & Trade-offs
- Step 3: Security, Scalability & Future Outlook

Make each step clear, concise, and self-contained."""

EVALUATOR_REFLECTION_PROMPT = """You are a Senior Peer Reviewer evaluating a cumulative research knowledge ledger.
Review the user prompt and the findings gathered across execution steps.
Decide if the knowledge gathered is sufficient to generate a thorough, high-quality publication-ready research report.

If key aspects are missing and less than 3 execution steps have occurred, mark `sufficient = False` and specify `missing_aspects`.
Otherwise, mark `sufficient = True`."""

MULTI_QUERY_DECOMPOSITION_PROMPT = """You are a senior research strategist. Decompose the user's input question into 3 distinct, targeted sub-queries covering:
1. Technical Definition & Core Architecture: Focus on fundamental concepts, definitions, mechanisms, and key components.
2. Comparative Analysis & Trade-offs: Focus on comparisons, pros and cons, benchmark evaluations, and alternative paradigms.
3. Future Implications & Strategic Outlook: Focus on emerging trends, scalability, security challenges, and future directions.

Ensure all 3 sub-queries are distinct, self-contained, and optimized for document retrieval."""

DEEP_RESEARCH_REPORT_PROMPT = """You are a Lead AI Research Scientist and Systems Architect. Synthesize the provided cumulative context and knowledge ledger into a publication-ready Deep Research Report.

Your report MUST strictly adhere to the following academic structure:

# Deep Research Report: {topic}

## Executive Summary
- Provide a high-level overview of the research topic, primary conclusions, and key takeaways.

## Methodology & Multi-Step Research Execution
- Describe the autonomous Planner-Executor-Evaluator research workflow and multi-perspective retrieval across technical, comparative, and strategic dimensions.

## Technical Architecture & Core Definitions
- Detail the technical mechanics, architecture, definitions, and underlying principles found in the sources.

## Comparative Findings & Trade-offs
- Synthesize comparative insights, performance trade-offs, advantages, limitations, and alternative paradigms.

## Future Implications & Strategic Outlook
- Analyze long-term consequences, scaling implications, security considerations, and future developments.

## Annotated Bibliography & Source References
- List every source referenced in the text with inline citations like [1], [2], etc., detailing document name/URL, page numbers, and key contribution.

Rule: Base all findings strictly on the provided context. If certain details are absent from context, explicitly state the limitation without hallucinating.

Context & Knowledge Ledger:
{context}

Question/Topic:
{question}
"""

REWRITE_FOLLOWUP_PROMPT = """Given chat history and latest question, output a standalone question. If already standalone, return it unchanged. Output only the question."""

REWRITE_FOR_RETRIEVAL_PROMPT = """The previous retrieval returned irrelevant chunks; rewrite the question with different keywords / more specific technical phrasing / expanded acronyms to improve retrieval. Output only the rewritten query."""

GRADE_DOC_PROMPT = """Given a question and a document chunk, decide if the chunk contains information that helps answer the question. Be lenient about partial relevance but strict about unrelated topics."""

GENERATE_PROMPT = """Answer using ONLY the context below. Cite sources inline as [n] where n is the context item number. If the context does not contain the answer, say you could not find it in the provided sources. Be concise and accurate. Do not use outside knowledge."""

GROUNDING_PROMPT = """Given context and an answer, decide whether every factual claim in the answer is supported by the context. List unsupported claims."""

CHITCHAT_PROMPT = """Briefly and friendly reply; mention that you can answer questions about the uploaded documents or generate Deep Research Reports."""