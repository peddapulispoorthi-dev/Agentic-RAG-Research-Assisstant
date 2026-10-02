import logging
from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.config import settings
from src.nodes import (
    route_question,
    rewrite_followup,
    retrieve,
    decompose_query,
    parallel_retrieve,
    draft_research_plan,
    execute_plan_step,
    reflect_on_knowledge,
    grade_documents,
    transform_query,
    web_search,
    generate,
    synthesize_deep_research_report,
    check_grounding,
    regenerate_strict,
    chitchat
)

logger = logging.getLogger(__name__)

def decide_after_routing(state: GraphState) -> str:
    route = state.get("route", "vectorstore")
    is_deep = state.get("is_deep_research", False)
    
    if route == "chitchat":
        return "chitchat"
    elif route == "deep_research" or is_deep:
        return "draft_research_plan"
    return "rewrite_followup"

def decide_after_reflection(state: GraphState) -> str:
    reflection = state.get("reflection", {})
    step_idx = state.get("current_step_idx", 0)
    plan_steps = state.get("plan_steps", [])
    
    sufficient = reflection.get("sufficient", True)
    
    if sufficient or step_idx >= len(plan_steps):
        logger.info("Knowledge ledger evaluation complete. Proceeding to synthesis...")
        return "synthesize_deep_research_report"
    else:
        logger.info(f"Reflection indicates gap. Executing step {step_idx + 1}...")
        return "execute_plan_step"

def decide_after_grading(state: GraphState) -> str:
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    is_deep = state.get("is_deep_research", False)
    
    if len(docs) > 0:
        if is_deep:
            return "synthesize_deep_research_report"
        return "generate"
    elif retry_count < settings.MAX_QUERY_RETRIES:
        return "transform_query"
    else:
        return "web_search"

def decide_after_web_search(state: GraphState) -> str:
    if state.get("is_deep_research", False):
        return "synthesize_deep_research_report"
    return "generate"

def decide_after_grounding(state: GraphState) -> str:
    grounded = state.get("grounded", True)
    regen_count = state.get("regen_count", 0)
    
    if grounded or regen_count >= settings.MAX_REGENERATIONS:
        return "end"
    else:
        return "regenerate_strict"

def build_graph():
    workflow = StateGraph(GraphState)

    # Add all graph nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("chitchat", chitchat)
    workflow.add_node("rewrite_followup", rewrite_followup)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("decompose_query", decompose_query)
    workflow.add_node("parallel_retrieve", parallel_retrieve)
    
    # Autonomous Planner-Executor-Evaluator Nodes
    workflow.add_node("draft_research_plan", draft_research_plan)
    workflow.add_node("execute_plan_step", execute_plan_step)
    workflow.add_node("reflect_on_knowledge", reflect_on_knowledge)

    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("synthesize_deep_research_report", synthesize_deep_research_report)
    workflow.add_node("check_grounding", check_grounding)
    workflow.add_node("regenerate_strict", regenerate_strict)

    # Set entry point
    workflow.set_entry_point("route_question")

    # Conditional routing after initial question classification
    workflow.add_conditional_edges(
        "route_question",
        decide_after_routing,
        {
            "chitchat": "chitchat",
            "draft_research_plan": "draft_research_plan",
            "rewrite_followup": "rewrite_followup"
        }
    )

    workflow.add_edge("chitchat", END)
    
    # Standard single-query branch
    workflow.add_edge("rewrite_followup", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Autonomous Planner-Executor-Evaluator Research Loop branch
    workflow.add_edge("draft_research_plan", "execute_plan_step")
    workflow.add_edge("execute_plan_step", "reflect_on_knowledge")
    
    workflow.add_conditional_edges(
        "reflect_on_knowledge",
        decide_after_reflection,
        {
            "execute_plan_step": "execute_plan_step",
            "synthesize_deep_research_report": "synthesize_deep_research_report"
        }
    )

    # Grading decision logic for standard retrieval
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "synthesize_deep_research_report": "synthesize_deep_research_report",
            "transform_query": "transform_query",
            "web_search": "web_search"
        }
    )

    workflow.add_edge("transform_query", "retrieve")
    
    # Decision after web search
    workflow.add_conditional_edges(
        "web_search",
        decide_after_web_search,
        {
            "generate": "generate",
            "synthesize_deep_research_report": "synthesize_deep_research_report"
        }
    )

    # Synthesis -> Grounding verification -> Output
    workflow.add_edge("generate", "check_grounding")
    workflow.add_edge("synthesize_deep_research_report", "check_grounding")

    workflow.add_conditional_edges(
        "check_grounding",
        decide_after_grounding,
        {
            "end": END,
            "regenerate_strict": "regenerate_strict"
        }
    )

    workflow.add_edge("regenerate_strict", END)

    return workflow.compile()

# Export compiled graph instance
app_graph = build_graph()