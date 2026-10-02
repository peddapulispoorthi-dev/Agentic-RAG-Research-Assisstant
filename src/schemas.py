from typing import Literal, List, Optional, Dict, Any
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    datasource: Literal["vectorstore", "web_search", "chitchat", "deep_research"]
    reasoning: str = Field(description="One short sentence explaining routing choice")

class MultiQueryDecomposition(BaseModel):
    technical_definition: str = Field(
        description="Sub-query focusing on technical definitions, core mechanics, and architecture."
    )
    comparative_analysis: str = Field(
        description="Sub-query focusing on comparative analysis, pros/cons, and alternatives."
    )
    future_implications: str = Field(
        description="Sub-query focusing on future implications, scaling, security, and emerging trends."
    )

    def get_queries(self) -> List[str]:
        return [self.technical_definition, self.comparative_analysis, self.future_implications]

class ResearchPlan(BaseModel):
    topic: str = Field(description="Primary research topic")
    steps: List[str] = Field(description="Sequential 3 to 4 targeted research steps to execute")

class ReflectionDecision(BaseModel):
    sufficient: bool = Field(description="True if cumulative knowledge is sufficient to answer prompt completely")
    missing_aspects: List[str] = Field(default_factory=list, description="Aspects still needing research if insufficient")
    reasoning: str = Field(description="Brief reasoning for reflection decision")

class RelevanceGrade(BaseModel):
    relevant: bool
    reasoning: str

class GroundingGrade(BaseModel):
    grounded: bool
    unsupported_claims: List[str] = Field(default_factory=list)

class Citation(BaseModel):
    source: str              # file name or URL
    page: Optional[int] = None
    snippet: str

class AnswerResult(BaseModel):
    answer: str
    report: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    path: List[str] = Field(default_factory=list)
    used_web_search: bool = False
    rewritten_query: Optional[str] = None
    sub_queries: List[str] = Field(default_factory=list)
    plan_steps: List[str] = Field(default_factory=list)
    knowledge_ledger: List[Dict[str, Any]] = Field(default_factory=list)
    reflection: Optional[Dict[str, Any]] = None
    latency_seconds: float = 0.0

class AskRequest(BaseModel):
    question: str
    history: List[dict] = Field(default_factory=list)
    is_deep_research: bool = False

# --- RAG TRIAD EVALUATION SCHEMAS ---
class ContextRelevanceEval(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0 indicating how relevant retrieved context is to question")
    reasoning: str

class GroundednessEval(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0 indicating how grounded response is in context")
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: str

class AnswerRelevanceEval(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0 indicating how relevant response is to original prompt")
    reasoning: str

class RAGTriadScore(BaseModel):
    question: str
    context_relevance: float
    groundedness: float
    answer_relevance: float
    overall_score: float
    details: Dict[str, Any] = Field(default_factory=dict)