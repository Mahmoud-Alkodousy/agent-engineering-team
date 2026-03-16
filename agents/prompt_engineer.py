from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import PipelineState
from config import GROQ_API_KEY, PLANNER_MODEL

llm = ChatGroq(api_key=GROQ_API_KEY, model=PLANNER_MODEL, temperature=0.3)

SYSTEM = """You are a senior software architect specializing in requirements analysis.
Transform the user's rough idea into precise, detailed technical specifications.

Your output must follow this exact structure:

## Project Name
One clear name.

## Description
2-3 sentences describing what the app does and who it's for.

## Core Features
List every feature the app must have. Be specific.

## API Endpoints
For each endpoint:
- METHOD /path — description
Include: auth endpoints if needed, CRUD for each resource, any special actions.

## Database Schema
For each table: table name, columns (name, type, constraints), relationships.

## Pages & UI Structure
List every page/view the frontend needs, what it shows, what actions are available.

## Technical Notes
Any edge cases, validations, business rules to implement.

Be thorough. This will be consumed by specialized coding agents."""


def run_prompt_engineer(state: PipelineState) -> dict:
    print("[PromptEngineer] Analyzing idea...")
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"User idea: {state['user_idea']}")
    ]
    response = llm.invoke(messages)
    print("[PromptEngineer] Done.")
    return {"specs": response.content}
