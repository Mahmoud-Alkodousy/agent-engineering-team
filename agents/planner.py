from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import PipelineState
from config import GROQ_API_KEY, PLANNER_MODEL

llm = ChatGroq(api_key=GROQ_API_KEY, model=PLANNER_MODEL, temperature=0.2)

SYSTEM = """You are an engineering lead planning a full-stack project.
Given technical specifications, produce a clear implementation plan for the coding agents.

Stack is fixed:
- Backend: FastAPI + SQLAlchemy 2.0 + SQLite + Pydantic v2
- Frontend: React 18 + Tailwind CSS + Axios + React Router v6 + Vite

Your output must follow this exact structure:

## File Structure
Show the complete directory tree for both backend/ and frontend/.

## Database Layer Plan
List every model, its fields, and relationships. Mention indexes needed.

## Backend Plan
List every router file, its endpoints, what each does, what it returns.
Mention any middleware needed (CORS, auth, etc).

## Frontend Plan
List every page and component. For each: what data it fetches, what it renders, what actions it handles.
Specify the color palette and design style to use.

## Implementation Order
Number the steps coding agents should follow (what depends on what).

## Key Technical Decisions
Explain choices like auth strategy, state management, error handling approach."""


def run_planner(state: PipelineState) -> dict:
    print("[Planner] Creating implementation plan...")
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Specifications:\n\n{state['specs']}")
    ]
    response = llm.invoke(messages)
    print("[Planner] Done.")
    return {"plan": response.content}
