import os
from typing import List

from Projects.ai_news.load_env import load_env_class
from Projects.ai_news.states.states import Article
from tavily import TavilyClient


class NewsSearcher:
    """
    Agent responsible for finding relevant AI/ML news articles
    using the Tavily search API
    """

    def search(self , tavily, inquiry) -> List[Article]:
        """
        Performs news search with configured parameters

        Returns:
            List[Article]: Collection of found articles
        """
        response = tavily.search(
            query=inquiry,
            topic="news",
            time_period="1w",
            search_depth="advanced",
            max_results=5
        )

        articles = []
        for result in response['results']:
            articles.append(Article(
                title=result['title'],
                url=result['url'],
                content=result['content']
            ))

        return articles


