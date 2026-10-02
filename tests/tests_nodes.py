import pytest
from src.state import GraphState
from src.nodes import route_question, transform_query

def test_transform_query():
    state = {"rewritten_question": "old query", "retry_count": 0, "path": []}
    # Test that retry count increments properly even if mocked/fallback occurs
    res = transform_query(state)
    assert res["retry_count"] == 1
    assert "transform_query" in res["path"]