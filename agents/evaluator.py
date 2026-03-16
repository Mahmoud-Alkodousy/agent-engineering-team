import ast
import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from state import PipelineState
from config import GROQ_API_KEY, PLANNER_MODEL, EVAL_PASS_SCORE

llm = ChatGroq(api_key=GROQ_API_KEY, model=PLANNER_MODEL, temperature=0.1)

SYSTEM = """You are a strict senior code reviewer for full-stack applications.
Evaluate the generated code and return ONLY a JSON object — no other text.

Scoring guide:
- Start at 100
- Deduct 20 if CSS variables (--bg, --surface, --border, --text, --green, --red) are missing from index.css
- Deduct 15 if Syne or DM Mono fonts are not imported
- Deduct 15 if sidebar layout with grid(220px 1fr) is missing
- Deduct 10 if Chart.js is not used for data visualization (when data exists)
- Deduct 10 if loading skeletons are missing from any page
- Deduct 10 if any API call has no try/catch error handling
- Deduct 10 if backend endpoints don't match what frontend calls
- Deduct 5 if Chart.destroy() is missing in useEffect cleanup
- Deduct 5 if any CRUD operation is missing from backend
- Deduct 5 if no empty states in list pages

feedback_for_agents must be SPECIFIC — say exactly which file and what to fix.

Return ONLY this JSON (no markdown, no explanation):
{
  "score": <0-100>,
  "passed": <true if score >= 88>,
  "issues": ["specific issue 1", "specific issue 2"],
  "feedback_for_agents": {
    "database": "<exact fix needed or null>",
    "backend": "<exact fix needed or null>",
    "frontend": "<exact fix needed or null>"
  }
}"""


# --- Static Analysis ----------------------------------------------------------

def _check_python_syntax(files: dict) -> list:
    errors = []
    for filename, code in files.items():
        if not filename.endswith(".py"):
            continue
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"SyntaxError in {filename} line {e.lineno}: {e.msg}")
    return errors


def _check_js_basics(files: dict) -> list:
    errors = []
    for filename, code in files.items():
        if not filename.endswith((".js", ".jsx")):
            continue
        open_b = code.count("{")
        close_b = code.count("}")
        if abs(open_b - close_b) > 5:
            errors.append(f"Possible unmatched braces in {filename} ({open_b} open vs {close_b} close)")
        if filename.endswith(".jsx") and "from 'react'" not in code and 'from "react"' not in code:
            errors.append(f"Missing React import in {filename}")
    return errors


def _check_endpoint_alignment(backend_files: dict, frontend_files: dict) -> list:
    issues = []
    backend_paths = set()
    path_re = re.compile(r'@(?:router|app)\.\w+\(["\'](/[^"\']*)["\']')
    for code in backend_files.values():
        for m in path_re.finditer(code):
            backend_paths.add(m.group(1).rstrip("/"))

    if not backend_paths:
        return issues

    axios_re = re.compile(r'(?:axios|client)\.\w+\([`"\']([^`"\']+)[`"\']')
    missing = []
    for filename, code in frontend_files.items():
        for m in axios_re.finditer(code):
            path = m.group(1).split("$")[0].rstrip("/")
            if path and not any(path.startswith(bp) or bp.startswith(path) for bp in backend_paths):
                missing.append(f"{filename}: '{path}'")

    if missing:
        issues.append("Frontend calls endpoints not in backend: " + "; ".join(missing[:5]))
    return issues


def _static_penalty(db_files: dict, backend_files: dict, frontend_files: dict) -> tuple:
    all_issues = []
    penalty = 0

    py_errors = _check_python_syntax({**db_files, **backend_files})
    if py_errors:
        all_issues.extend(py_errors)
        penalty += min(len(py_errors) * 10, 30)

    js_errors = _check_js_basics(frontend_files)
    if js_errors:
        all_issues.extend(js_errors)
        penalty += min(len(js_errors) * 5, 20)

    align_issues = _check_endpoint_alignment(backend_files, frontend_files)
    if align_issues:
        all_issues.extend(align_issues)
        penalty += 10

    return penalty, all_issues


# --- File formatting (smart budget, no silent truncation) --------------------

def _summarise_for_llm(name: str, files: dict, char_budget: int = 12_000) -> str:
    out = f"\n\n=== {name} Layer ===\n"
    used = 0
    skipped = []
    for fname, code in files.items():
        if used + len(code) <= char_budget:
            out += f"\n--- {fname} ---\n{code}\n"
            used += len(code)
        else:
            skipped.append(fname)
    if skipped:
        out += f"\n[Not shown (budget): {', '.join(skipped)}]\n"
    return out


# --- Main runner -------------------------------------------------------------

def run_evaluator(state: PipelineState) -> dict:
    iteration = state.get("iteration", 0) + 1
    print(f"[Evaluator] Reviewing code (iteration {iteration})...")

    db_code       = state["db_code"]
    backend_code  = state["backend_code"]
    frontend_code = state["frontend_code"]

    # 1. Static analysis
    static_penalty, static_issues = _static_penalty(db_code, backend_code, frontend_code)
    if static_issues:
        print(f"[Evaluator] Static issues: {static_issues}")

    # 2. LLM review
    all_code  = _summarise_for_llm("Database", db_code)
    all_code += _summarise_for_llm("Backend",  backend_code)
    all_code += _summarise_for_llm("Frontend", frontend_code)

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Original specifications:\n{state['specs']}\n\n"
            f"Generated code:\n{all_code}"
        ))
    ]
    response = llm.invoke(messages)

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "score": 60,
            "passed": False,
            "issues": ["Could not parse evaluator response"],
            "feedback_for_agents": {"database": None, "backend": None, "frontend": None}
        }

    llm_score  = data.get("score", 0)
    llm_issues = data.get("issues", [])
    feedback   = data.get("feedback_for_agents", {})

    # 3. Apply static penalty on top of LLM score
    final_score = max(0, llm_score - static_penalty)
    all_issues  = static_issues + llm_issues
    passed      = final_score >= EVAL_PASS_SCORE

    print(f"[Evaluator] LLM: {llm_score} | Static penalty: -{static_penalty} | Final: {final_score}/100 | Passed: {passed}")
    for issue in all_issues:
        print(f"  - {issue}")

    if static_issues:
        static_summary = "Fix these static analysis issues: " + "; ".join(static_issues)
        feedback["backend"]  = ((feedback.get("backend")  or "") + "\n" + static_summary).strip()
        feedback["frontend"] = ((feedback.get("frontend") or "") + "\n" + static_summary).strip()

    return {
        "eval_score":    final_score,
        "eval_passed":   passed,
        "eval_issues":   all_issues,
        "eval_feedback": feedback,
        "iteration":     iteration,
    }
