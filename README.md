<h1 align="center">🤖 Agent Engineering Team</h1>
<h3 align="center">A multi-agent AI pipeline that transforms a plain text idea into a production-ready React + FastAPI application.</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2+-FF6B35?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Generated-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-Generated-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white"/>
</p>

---

### 🌟 What is this?

**Agent Engineering Team** is an agentic AI system where a team of specialized LLM agents collaborate like a real engineering team — each with a distinct role — to generate a complete, working full-stack application from a single plain-text idea.

You describe what you want. The agents handle everything else.

---

### ⚙️ How It Works

The pipeline is orchestrated by **LangGraph** and runs through 6 sequential stages:

```
Your Idea
    │
    ▼
┌─────────────────────┐
│  1. Prompt Engineer │  → Turns rough idea into precise technical specs
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  2. Planner         │  → Designs file structure, DB schema, API contract, UI layout
└─────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  3. Code Generation (3 parallel agents)  │
│     ├── Database Agent                   │ → models.py, schemas.py, database.py (SQLAlchemy 2.0)
│     ├── Backend Agent                    │ → FastAPI routers + main.py (one router per resource)
│     └── Frontend Agent                   │ → React 18 + Vite (pages, components, Chart.js, dark UI)
└──────────────────────────────────────────┘
    │
    ▼
┌─────────────────────┐
│  4. Evaluator       │  → Static analysis (AST, brace check, endpoint alignment)
│                     │    + LLM code review → score/100
│                     │    If score < 88 → loops back to Code Generation (max 3x)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  5. Explainer       │  → Writes a clear README for the generated app
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  6. File Writer     │  → Saves backend/ + frontend/ + README.md to disk
└─────────────────────┘
    │
    ▼
Your Production-Ready App 🚀
```

---

### 🧠 The Agent Team

| Agent | Model | Role |
|-------|-------|------|
| **Prompt Engineer** | Llama 3.3 70B (Groq) | Transforms vague ideas into detailed technical specs |
| **Planner** | Llama 3.3 70B (Groq) | Architects the full system — DB, API, UI, file structure |
| **Database Agent** | GPT-4o Mini (OpenRouter) | Generates SQLAlchemy models, Pydantic schemas, DB config |
| **Backend Agent** | GPT-4o Mini (OpenRouter) | Generates FastAPI routers (one per resource) + main.py |
| **Frontend Agent** | GPT-4o Mini (OpenRouter) | Generates React pages, components, Chart.js dashboards |
| **Evaluator** | Llama 3.3 70B (Groq) | Reviews code quality with static analysis + LLM scoring |
| **Explainer** | Llama 3.1 8B (Groq) | Writes the README for the generated app |

---

### 📦 What Gets Generated

For every run, the pipeline outputs a fully structured project:

```
generated_apps/<run_id>/
├── backend/
│   ├── database.py        # SQLAlchemy engine + session + Base
│   ├── models.py          # ORM models with relationships
│   ├── schemas.py         # Pydantic v2 request/response schemas
│   ├── routers/
│   │   └── <resource>.py  # One router per resource (full CRUD)
│   ├── main.py            # FastAPI app with CORS, startup events
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css      # Full dark design system (CSS variables)
│       ├── api/client.js  # Axios instance
│       ├── pages/         # One page per route
│       └── components/    # Layout, charts, reusable components
├── README.md              # Auto-generated run instructions
├── START.md               # Quick start guide
├── run_backend.py         # One-click backend launcher
└── run_frontend.py        # One-click frontend launcher
```

---

### 🛠️ Tech Stack

**Pipeline**
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration & state machine
- [LangChain](https://github.com/langchain-ai/langchain) — LLM abstraction layer
- [Gradio](https://gradio.app) — streaming web UI
- [Groq](https://groq.com) — fast inference for planning & evaluation
- [OpenRouter](https://openrouter.ai) — GPT-4o Mini for code generation

**Generated Apps**
- **Backend**: FastAPI · SQLAlchemy 2.0 · Pydantic v2 · SQLite · Uvicorn
- **Frontend**: React 18 · Vite · Chart.js · Axios · React Router v6

---

### 🚀 Getting Started

#### 1. Clone the repo
```bash
git clone https://github.com/Mahmoud-Alkodousy/agent-engineering-team.git
cd agent-engineering-team
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Set up environment variables
```bash
cp .env.example .env
# Then open .env and add your API keys
```

```env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> Get your keys: [Groq Console](https://console.groq.com) · [OpenRouter](https://openrouter.ai/keys)

#### 4. Run the app
```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

---

### 💡 Example Prompts

> *"A project management tool where teams create projects, add tasks with deadlines and priorities, assign them to members, and track progress on a kanban board."*

> *"An expense tracker where users log income and expenses by category, set monthly budgets, and see spending trends on charts."*

> *"A simple CRM where salespeople manage leads, log calls and meetings, and track deals through a pipeline."*

---

### 📁 Project Structure

```
agent-engineering-team/
├── app.py                     # Gradio UI + entry point
├── graph.py                   # LangGraph pipeline definition
├── state.py                   # PipelineState TypedDict
├── config.py                  # Models, API keys, pipeline settings
├── requirements.txt
├── .env.example               # Environment variable template
├── .gitignore
└── agents/
    ├── prompt_engineer.py
    ├── planner.py
    ├── database_agent.py
    ├── backend_agent.py
    ├── frontend_agent.py
    ├── evaluator.py           # Static analysis + LLM review loop
    └── explainer.py
```

---

### 🔧 Configuration

Edit `config.py` to customize the pipeline:

```python
MAX_EVAL_ITERATIONS = 3   # Max retries if code quality score < threshold
EVAL_PASS_SCORE     = 88  # Minimum score (out of 100) to accept generated code
PLANNER_MODEL       = "llama-3.3-70b-versatile"   # Groq
CODER_MODEL         = "openai/gpt-4o-mini"         # OpenRouter
EXPLAINER_MODEL     = "llama-3.1-8b-instant"       # Groq
```

---

### 🌐 Connect With Me

<p align="center">
  <a href="https://www.linkedin.com/in/mahmoud-khalid-8b4ab2309/" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
  <a href="https://kaggle.com/mahmoudalkodousy" target="_blank"><img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/></a>
  <a href="mailto:mahmoudkhaledalkoudosy@gmail.com"><img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
</p>

---

### ✨ Quote

> *"Give it an idea. The team handles the rest."*

---

<p align="center">Built with ❤️ by <a href="https://github.com/Mahmoud-Alkodousy">Mahmoud Alkodousy</a></p>
