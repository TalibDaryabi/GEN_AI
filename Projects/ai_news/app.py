import os

from Projects.ai_news.graph import create_graph
from Projects.ai_news.load_env import load_env_class

if __name__ == "__main__":

    # Initialize and run workflow
    load_key = load_env_class("H:\My_LangGraph_toturial\.env")
    os.environ["OPENAI_API_KEY"] = load_key.get_open_ai_key()
    os.environ["TAVILY_API_KEY"] = load_key.load_travily_api_key()

    graph = create_graph()
    workflow = graph.compile()


    final_state = workflow.invoke({
        "articles": None,
        "summaries": None,
        "report": None
    })

    # Display results
    print("\n=== AI/ML Weekly News Report ===\n")
    print(final_state['report'])