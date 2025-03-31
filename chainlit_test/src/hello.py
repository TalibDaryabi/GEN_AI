import os
from dotenv import load_dotenv # type: ignore
from typing import cast
import chainlit as cl # type: ignore
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel # type: ignore
from agents.run import RunConfig # type: ignore

# Load the environment variables from the .env file
load_dotenv("H:\My_LangGraph_toturial\chainlit_test\.env")

gemini_api_key = os.getenv("GOOGLE_API_KEY")
print(gemini_api_key)

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")
