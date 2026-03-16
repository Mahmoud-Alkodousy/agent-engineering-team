from typing import TypedDict, Optional


class PipelineState(TypedDict):
    # Input
    user_idea: str

    # Stage outputs
    specs: str
    plan: str
    db_code: dict[str, str]       # filename -> code
    backend_code: dict[str, str]  # filename -> code
    frontend_code: dict[str, str] # filepath -> code

    # Evaluator
    eval_score: int
    eval_passed: bool
    eval_issues: list[str]
    eval_feedback: dict[str, str]  # per-agent feedback
    iteration: int

    # Final
    explanation: str
    output_dir: str
