from langchain_core.messages import SystemMessage, HumanMessage
from state import PipelineState
from config import CODER_MODEL, get_openrouter_llm

llm = get_openrouter_llm(CODER_MODEL, temperature=0.1)

MAIN_SYSTEM = """You are a senior FastAPI backend engineer.
Generate ONLY the main.py file for a FastAPI application.

Requirements for main.py:
- Create FastAPI app with title, description, version
- Add CORSMiddleware: allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
- Include all routers with prefix and tags
- Add startup event to create all tables: Base.metadata.create_all(bind=engine)
- Add GET /health endpoint returning {"status": "ok"}
- Import everything correctly
- At the end of the file, add this block exactly:

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

Output ONLY raw Python code. No markdown fences, no explanation, no backticks."""

ROUTER_SYSTEM = """You are a senior FastAPI backend engineer.
Generate ONLY a single FastAPI router file.

Requirements:
- Use APIRouter with prefix and tags
- Async endpoints with db: Session = Depends(get_db)
- Full CRUD: GET all, GET by id, POST, PUT, DELETE
- Proper HTTP status codes (201 for create, 404 for not found)
- HTTPException with descriptive detail messages
- Import models and schemas from parent directory (e.g. from ..models import X)

Output ONLY raw Python code. No markdown fences, no explanation, no backticks."""

REQUIREMENTS_CONTENT = """fastapi==0.111.0
uvicorn[standard]==0.29.0
sqlalchemy==2.0.30
pydantic==2.7.1
python-dotenv==1.0.1
"""


def _call_llm(system: str, user: str) -> str:
    """Single LLM call, returns clean code string."""
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user)
    ])
    code = response.content.strip()
    # Strip markdown fences if model added them
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:])
    if code.rstrip().endswith("```"):
        code = code.rstrip()[:-3].rstrip()
    return code.strip()


def _extract_resources(specs: str, plan: str) -> list[str]:
    """Ask the model to list the resource names needed (one word each)."""
    response = llm.invoke([
        SystemMessage(content="You are a backend architect. List ONLY the resource names (lowercase, singular) needed for this app, one per line. Nothing else — no explanation, no numbering, just the names."),
        HumanMessage(content=f"Specifications:\n{specs}\n\nPlan:\n{plan}")
    ])
    resources = []
    for line in response.content.strip().split("\n"):
        name = line.strip().lower().strip("-•* ")
        if name and len(name) < 30 and " " not in name:
            resources.append(name)
    return resources[:6]  # cap at 6 routers


def run_backend_agent(state: PipelineState) -> dict:
    feedback = state.get("eval_feedback", {}).get("backend", "")
    iteration = state.get("iteration", 0)

    if feedback and iteration > 0:
        feedback_section = (
            f"\n\n[IMPORTANT — PREVIOUS ATTEMPT WAS REJECTED]\n"
            f"You must fix these specific issues in this attempt:\n{feedback}\n"
            f"Do not repeat the same mistakes."
        )
    else:
        feedback_section = ""

    db_context = "\n\n".join(
        f"# {name}\n{code}" for name, code in state["db_code"].items()
    )

    print("[BackendAgent] Detecting resources...")
    resources = _extract_resources(state["specs"], state["plan"])
    print(f"[BackendAgent] Resources: {resources}")

    files: dict[str, str] = {}

    # Generate one router per resource
    router_imports = []
    for resource in resources:
        print(f"[BackendAgent] Generating routers/{resource}.py ...")
        code = _call_llm(
            ROUTER_SYSTEM,
            f"Resource name: {resource}\n\n"
            f"Specifications:\n{state['specs']}\n\n"
            f"Database layer:\n{db_context}"
            f"{feedback_section}"
        )
        files[f"routers/{resource}.py"] = code
        router_imports.append(resource)

    # Generate main.py
    print("[BackendAgent] Generating main.py ...")
    router_list = "\n".join(f"- {r}" for r in router_imports)
    main_code = _call_llm(
        MAIN_SYSTEM,
        f"Routers to include: {router_list}\n\n"
        f"Database layer:\n{db_context}\n\n"
        f"Specifications:\n{state['specs']}"
        f"{feedback_section}"
    )
    files["main.py"] = main_code
    files["requirements.txt"] = REQUIREMENTS_CONTENT

    print(f"[BackendAgent] Generated: {list(files.keys())}")
    return {"backend_code": files}

