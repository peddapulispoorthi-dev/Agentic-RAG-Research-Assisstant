from src.service import run_agent_query

if __name__ == "__main__":
    print("Testing Agentic RAG Research Assistant workflow...")
    response = run_agent_query("What is discussed on page 1 of the documents?")
    print("\n--- AGENT RESPONSE ---")
    print(response["answer"])
    print("\n--- EXECUTION PATH ---")
    print(response["path"])