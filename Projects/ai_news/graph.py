import os
from typing import Dict, Any

from langgraph.graph import StateGraph
from tavily import TavilyClient

from Projects.ai_news.LLMs.open_ai import OPEN_AI_LLM
from Projects.ai_news.agents.news_searcher import NewsSearcher
from Projects.ai_news.agents.publisher import Publisher
from Projects.ai_news.agents.summarize import Summarizer
from Projects.ai_news.load_env import load_env_class
from Projects.ai_news.states.states import GraphState


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for article search

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        Dict[str, Any]: Updated state with found articles
    """

    tavily_search = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    searcher = NewsSearcher()
    # introduce interrupt to get the query
    # question = input("What do you want to search for? ")
    question = "What is the latest news on AI?"
    state['articles'] = searcher.search(inquiry=question , tavily=tavily_search )
    return state


def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for article summarization

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        Dict[str, Any]: Updated state with summaries
    """
    summarizer = Summarizer()
    state['summaries'] = []
    llm = OPEN_AI_LLM(model_name="gpt-4o-mini")
    llm = llm.get_llm()
    for article in state['articles']:  # Uses articles from previous node
        summary = summarizer.summarize(article , llm)
        state['summaries'] = state['summaries'] + [{
            'title': article.title,
            'summary': summary,
            'url': article.url
        }]
    return state


def publish_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node for report generation

    Args:
        state (Dict[str, Any]): Current workflow state

    Returns:
        Dict[str, Any]: Updated state with final report
    """
    publisher = Publisher()
    llm = OPEN_AI_LLM(model_name="gpt-4o-mini")
    llm = llm.get_llm()
    report_content = publisher.create_report(state['summaries'], llm)
    state['report'] = report_content
    return state


def create_graph() -> StateGraph:
    """
    Constructs and configures the workflow graph
    search -> summarize -> publish

    Returns:
        StateGraph: Compiled workflow ready for execution
    """

    # Create a workflow (graph) initialized with our state schema
    graph = StateGraph(state_schema=GraphState)

    # Add processing nodes that we will flow between
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("publish", publish_node)

    # Define the flow with edges
    graph.add_edge("search", "summarize")  # search results flow to summarizer
    graph.add_edge("summarize", "publish")  # summaries flow to publisher

    # Set where to start
    graph.set_entry_point("search")

    return graph

def draw_graph(graph):
    """
    Draw the graph
    """
    pass

