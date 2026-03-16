from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os

load_dotenv(override=True)

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model names
PLANNER_MODEL   = "llama-3.3-70b-versatile"   # Groq — deep thinking, planning, evaluation
CODER_MODEL     = "openai/gpt-4o-mini"        # OpenRouter — smarter code generation
EXPLAINER_MODEL = "llama-3.1-8b-instant"      # Groq — fast, lightweight for README

# Pipeline settings
MAX_EVAL_ITERATIONS = 3
EVAL_PASS_SCORE     = 88


def get_groq_llm(model: str, temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(api_key=GROQ_API_KEY, model=model, temperature=temperature)


def get_openrouter_llm(model: str, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=model,
        temperature=temperature,
    )

