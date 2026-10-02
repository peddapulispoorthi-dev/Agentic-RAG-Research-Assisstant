from typing import Literal, List, Optional
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
    latency_seconds: float = 0.0

class AskRequest(BaseModel):
    question: str
    history: List[dict] = Field(default_factory=list)   # [{"role": "user"|"assistant", "content": "..."}]
    is_deep_research: bool = False