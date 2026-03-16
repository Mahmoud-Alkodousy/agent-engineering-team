import asyncio
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from state import PipelineState
from config import MAX_EVAL_ITERATIONS
from agents.prompt_engineer import run_prompt_engineer
from agents.planner import run_planner
from agents.database_agent import run_database_agent
from agents.backend_agent import run_backend_agent
from agents.frontend_agent import run_frontend_agent
from agents.evaluator import run_evaluator
from agents.explainer import run_explainer


# ─── Node wrappers ────────────────────────────────────────────────────────────

def node_prompt_engineer(state: PipelineState) -> dict:
    return run_prompt_engineer(state)


def node_planner(state: PipelineState) -> dict:
    return run_planner(state)


def node_code_generation(state: PipelineState) -> dict:
    """Run DB → Backend → Frontend agents with per-agent error handling."""
    # --- Database ---
    try:
        db_result = run_database_agent(state)
    except Exception as e:
        print(f"[ERROR] DatabaseAgent failed: {e}")
        db_result = {"db_code": {}}

    state_with_db = {**state, **db_result}

    # --- Backend ---
    try:
        backend_result = run_backend_agent(state_with_db)
        if not backend_result.get("backend_code"):
            print("[WARNING] BackendAgent returned no files — retrying once...")
            backend_result = run_backend_agent(state_with_db)
    except Exception as e:
        print(f"[ERROR] BackendAgent failed: {e}")
        backend_result = {"backend_code": {}}

    state_with_backend = {**state_with_db, **backend_result}

    # --- Frontend ---
    try:
        frontend_result = run_frontend_agent(state_with_backend)
        if not frontend_result.get("frontend_code"):
            print("[WARNING] FrontendAgent returned no files — retrying once...")
            frontend_result = run_frontend_agent(state_with_backend)
    except Exception as e:
        print(f"[ERROR] FrontendAgent failed: {e}")
        frontend_result = {"frontend_code": {}}

    return {
        **db_result,
        **backend_result,
        **frontend_result,
    }


def node_evaluator(state: PipelineState) -> dict:
    return run_evaluator(state)


def node_explainer(state: PipelineState) -> dict:
    return run_explainer(state)


def node_write_files(state: PipelineState) -> dict:
    """Write all generated files to disk."""
    output_dir = Path(state.get("output_dir", "output"))

    backend_dir = output_dir / "backend"
    frontend_dir = output_dir / "frontend"

    _write(backend_dir, state["db_code"])
    _write(backend_dir, state["backend_code"])
    _write(frontend_dir, state["frontend_code"])

    readme_path = output_dir / "README.md"
    readme_path.write_text(state["explanation"], encoding="utf-8")

    (output_dir / "START.md").write_text(_start_guide(), encoding="utf-8")

    run_backend = output_dir / "run_backend.py"
    run_backend.write_text(
        "import subprocess, sys, os\n"
        "os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))\n"
        "subprocess.run([sys.executable, 'main.py'])\n",
        encoding="utf-8"
    )

    run_frontend = output_dir / "run_frontend.py"
    run_frontend.write_text(
        "import subprocess, os\n"
        "frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')\n"
        "subprocess.run(['npm', 'install'], cwd=frontend_dir, shell=True)\n"
        "subprocess.run(['npm', 'run', 'dev'], cwd=frontend_dir, shell=True)\n",
        encoding="utf-8"
    )

    print(f"[Writer] All files written to {output_dir}/")
    return {}


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_eval(state: PipelineState) -> str:
    iteration = state.get("iteration", 0)
    if state["eval_passed"] or iteration >= MAX_EVAL_ITERATIONS:
        if not state["eval_passed"]:
            print(f"[Router] Max iterations reached. Proceeding with score {state['eval_score']}/100.")
        return "explainer"
    else:
        print(f"[Router] Score {state['eval_score']}/100 — retrying code generation.")
        return "code_generation"


# ─── Graph build ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("prompt_engineer", node_prompt_engineer)
    g.add_node("planner",         node_planner)
    g.add_node("code_generation", node_code_generation)
    g.add_node("evaluator",       node_evaluator)
    g.add_node("explainer",       node_explainer)
    g.add_node("write_files",     node_write_files)

    g.add_edge(START,             "prompt_engineer")
    g.add_edge("prompt_engineer", "planner")
    g.add_edge("planner",         "code_generation")
    g.add_edge("code_generation", "evaluator")
    g.add_conditional_edges(
        "evaluator",
        route_after_eval,
        {"code_generation": "code_generation", "explainer": "explainer"}
    )
    g.add_edge("explainer",       "write_files")
    g.add_edge("write_files",     END)

    return g.compile()


# ─── Streaming runner ─────────────────────────────────────────────────────────

STEP_LABELS = {
    "prompt_engineer": "Step 1/5 — Analyzing your idea...",
    "planner":         "Step 2/5 — Creating implementation plan...",
    "code_generation": "Step 3/5 — Generating code...",
    "evaluator":       "Step 4/5 — Reviewing code quality...",
    "explainer":       "Step 5/5 — Writing documentation...",
    "write_files":     "Writing files to disk...",
}


async def run_pipeline(user_idea: str, output_dir: str = "output"):
    """Stream status updates as the pipeline runs."""
    graph = build_graph()

    initial_state: PipelineState = {
        "user_idea":     user_idea,
        "specs":         "",
        "plan":          "",
        "db_code":       {},
        "backend_code":  {},
        "frontend_code": {},
        "eval_score":    0,
        "eval_passed":   False,
        "eval_issues":   [],
        "eval_feedback": {},
        "iteration":     0,
        "explanation":   "",
        "output_dir":    output_dir,
    }

    log = ""

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")

            if kind == "on_chain_start" and name in STEP_LABELS:
                msg = STEP_LABELS[name]
                log += f"\n\n**{msg}**"
                yield log

            elif kind == "on_chain_end" and name == "code_generation":
                data   = event.get("data", {})
                output = data.get("output", {})
                files_generated = (
                    list(output.get("db_code", {}).keys()) +
                    list(output.get("backend_code", {}).keys()) +
                    list(output.get("frontend_code", {}).keys())
                )
                if files_generated:
                    log += "\n- Files generated: " + ", ".join(f"`{f}`" for f in files_generated[:8])
                    if len(files_generated) > 8:
                        log += f" +{len(files_generated) - 8} more"
                    yield log

            elif kind == "on_chain_end" and name == "evaluator":
                data   = event.get("data", {})
                output = data.get("output", {})
                score  = output.get("eval_score", "?")
                passed = output.get("eval_passed", False)
                issues = output.get("eval_issues", [])
                status = "✅ Passed" if passed else "🔄 Retrying"
                log += f"\n- Score: **{score}/100** — {status}"
                if issues:
                    log += "\n- Issues found:\n" + "\n".join(f"  - {i}" for i in issues)
                yield log

            elif kind == "on_chain_end" and name == "explainer":
                data        = event.get("data", {})
                output      = data.get("output", {})
                explanation = output.get("explanation", "")
                if explanation:
                    log += f"\n\n---\n\n{explanation}"
                    yield log

            elif kind == "on_chain_end" and name == "write_files":
                log += f"\n\n---\n\n✅ **Done! Your app is in `{output_dir}/`**"
                yield log

    except Exception as e:
        log += f"\n\n❌ **Pipeline error:** {e}\n\nPartial output may have been saved to `{output_dir}/`."
        yield log


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write(base_dir: Path, files: dict) -> None:
    for filepath, code in files.items():
        full = base_dir / filepath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(code, encoding="utf-8")
        print(f"  [Writer] {full}")


def _start_guide() -> str:
    return """# How to Start Your App

## Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
API will be at: http://localhost:8000
API docs at:    http://localhost:8000/docs

## Frontend
```bash
cd frontend
npm install
npm run dev
```
App will be at: http://localhost:5173
"""
