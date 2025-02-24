import os
from dotenv import load_dotenv

class load_env_class:
    def __init__(self, path: str):
        load_dotenv(path)

    def load_all_env_key(self):
        open_ai_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = open_ai_key

    def load_travily_api_key(self):
        return os.getenv("TAVILY_API_KEY")

    def get_open_ai_key(self):
        return os.getenv("OPENAI_API_KEY")

