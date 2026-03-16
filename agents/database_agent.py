from langchain_core.messages import SystemMessage, HumanMessage
from state import PipelineState
from config import CODER_MODEL, get_openrouter_llm

llm = get_openrouter_llm(CODER_MODEL, temperature=0.1)


def _call_llm(system: str, user: str) -> str:
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ])
    code = response.content.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.rstrip().endswith("```"):
        code = code.rstrip()[:-3].rstrip()
    return code.strip()


DATABASE_SYSTEM = """You are a database engineer. Generate ONLY the database.py file for a FastAPI + SQLAlchemy project.

Requirements:
- SQLAlchemy 2.0 engine with SQLite: SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"
- SessionLocal with autocommit=False, autoflush=False
- Base = DeclarativeBase()
- get_db() dependency function
- Export: engine, SessionLocal, Base, get_db

Output ONLY raw Python code. No markdown fences, no backticks, no explanation."""

MODELS_SYSTEM = """You are a database engineer. Generate ONLY the models.py file using SQLAlchemy 2.0.

Requirements:
- Import Base from database
- One class per table, inheriting from Base
- All columns with proper types and constraints
- Add created_at = Column(DateTime, default=datetime.utcnow) to every model
- Add relationships between models where needed
- Add __repr__ to every model

Output ONLY raw Python code. No markdown fences, no backticks, no explanation."""

SCHEMAS_SYSTEM = """You are a backend engineer. Generate ONLY the schemas.py file using Pydantic v2.

Requirements:
- Separate Create, Update, Response schema per model
- Response schemas: model_config = ConfigDict(from_attributes=True)
- Use Optional fields for Update schemas
- Include all fields from the models

Output ONLY raw Python code. No markdown fences, no backticks, no explanation."""


def run_database_agent(state: PipelineState) -> dict:
    feedback = state.get("eval_feedback", {}).get("database", "")
    iteration = state.get("iteration", 0)

    if feedback and iteration > 0:
        feedback_section = (
            f"\n\n[IMPORTANT — PREVIOUS ATTEMPT WAS REJECTED]\n"
            f"You must fix these specific issues:\n{feedback}\n"
            f"Do not repeat the same mistakes."
        )
    else:
        feedback_section = ""

    context = f"Specifications:\n{state['specs']}\n\nPlan:\n{state['plan']}{feedback_section}"

    print("[DatabaseAgent] Generating database.py...")
    database_py = _call_llm(DATABASE_SYSTEM, context)

    print("[DatabaseAgent] Generating models.py...")
    models_py = _call_llm(MODELS_SYSTEM, context)

    print("[DatabaseAgent] Generating schemas.py...")
    schemas_py = _call_llm(SCHEMAS_SYSTEM, f"{context}\n\nModels already written:\n{models_py}")

    files = {
        "database.py": database_py,
        "models.py":   models_py,
        "schemas.py":  schemas_py,
    }
    print(f"[DatabaseAgent] Generated: {list(files.keys())}")
    return {"db_code": files}

