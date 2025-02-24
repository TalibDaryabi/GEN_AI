from datetime import datetime
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage


class Publisher:
    """
    Agent that compiles summaries into a formatted report
    and saves it to disk
    """

    def create_report(self, summaries: List[Dict] , llm) -> str:
        """
        Creates and saves a formatted markdown report

        Args:
            summaries (List[Dict]): Collection of article summaries

        Returns:
            str: Generated report content
        """
        prompt = """
        Create a weekly AI/ML news report for the general public. 
        Format it with:
        1. A brief introduction
        2. The main news items with their summaries
        3. Links for further reading

        Make it engaging and accessible to non-technical readers.
        """

        # Format summaries for the LLM
        summaries_text = "\n\n".join([
            f"Title: {item['title']}\nSummary: {item['summary']}\nSource: {item['url']}"
            for item in summaries
        ])

        # Generate report
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=summaries_text)
        ])

        # Add metadata and save
        current_date = datetime.now().strftime("%Y-%m-%d")
        markdown_content = f"""
        Generated on: {current_date}

        {response.content}
        """

        filename = f"ai_news_report_{current_date}.md"
        with open(filename, 'w') as f:
            f.write(markdown_content)

        return response.content