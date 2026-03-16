from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import PipelineState
from config import GROQ_API_KEY, EXPLAINER_MODEL

llm = ChatGroq(api_key=GROQ_API_KEY, model=EXPLAINER_MODEL, temperature=0.5)

SYSTEM = """You are a friendly technical writer explaining a generated app to a non-technical user.
Write a clear, encouraging README in Markdown.

Structure:
# [App Name]
> One sentence tagline

## What this app does
2-3 sentences in plain English.

## How to run it

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Then open http://localhost:5173

## What was built
Table: File | What it does (one line each)

## How to customize
3-4 common things they might want to change, with simple instructions.

Tone: warm, simple, encouraging. No jargon."""


def run_explainer(state: PipelineState) -> dict:
    print("[Explainer] Writing README...")

    all_files = list(state["db_code"].keys()) + \
                list(state["backend_code"].keys()) + \
                list(state["frontend_code"].keys())

    # Pull key files for context
    key_code = ""
    for name in ["main.py", "models.py", "App.jsx", "src/App.jsx"]:
        code = state["backend_code"].get(name) or state["db_code"].get(name) or state["frontend_code"].get(name)
        if code:
            key_code += f"\n### {name}\n{code[:600]}\n"

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"User asked for: {state['user_idea']}\n\n"
            f"Generated files:\n" + "\n".join(f"- {f}" for f in all_files) +
            f"\n\nKey files (for context):\n{key_code}"
        ))
    ]
    response = llm.invoke(messages)
    print("[Explainer] Done.")
    return {"explanation": response.content}
