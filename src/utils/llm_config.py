import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def get_together_llm():
    return LLM(
        model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key=os.getenv("TOGETHER_API_KEY"),
        temperature=0.0,
    )