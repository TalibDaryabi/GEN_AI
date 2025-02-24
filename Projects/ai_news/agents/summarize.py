from langchain_core.messages import SystemMessage, HumanMessage

from Projects.ai_news.LLMs.open_ai import OPEN_AI_LLM
from Projects.ai_news.states.states import Article


class Summarizer:
    """
    Agent that processes articles and generates accessible summaries
    using gpt-4o-mini
    """

    def __init__(self):
        self.system_prompt = """
        You are an AI expert who makes complex topics accessible 
        to general audiences. Summarize this article in 2-3 sentences, focusing on the key points 
        and explaining any technical terms simply.
        """

    def summarize(self, article: Article , llm) -> str:
        """
        Generates an accessible summary of a single article

        Args:
            article (Article): Article to summarize

        Returns:
            str: Generated summary
        """

        response = llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Title: {article.title}\n\nContent: {article.content}")
        ])
        return response.content