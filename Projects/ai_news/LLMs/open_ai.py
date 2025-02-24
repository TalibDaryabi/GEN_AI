from langchain_openai import ChatOpenAI

class OPEN_AI_LLM:
    def __init__(self, model_name , temperature = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = 1000
        self.llm = None

    def get_llm(self):
        if self.llm is None:
            self.llm = ChatOpenAI(model_name=self.model_name,
                              temperature=self.temperature,
                              max_tokens=self.max_tokens)
        return self.llm

