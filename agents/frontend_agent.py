from langchain_core.messages import SystemMessage, HumanMessage
from state import PipelineState
from config import CODER_MODEL, get_openrouter_llm

llm = get_openrouter_llm(CODER_MODEL, temperature=0.2)

SYSTEM = """You are a world-class frontend engineer and UI/UX designer.
Generate a stunning, professional React frontend — NO Tailwind CSS, use pure CSS-in-JS via inline styles and a single index.css file.

Output files separated by: ### FILE: filepath

Files to generate:
1. index.html
2. package.json
3. vite.config.js
4. src/main.jsx
5. src/App.jsx
6. src/index.css
7. src/api/client.js
8. src/pages/[PageName].jsx — one per page
9. src/components/Layout.jsx — sidebar + topbar shell
10. src/components/[ComponentName].jsx — reusable components

════════════════════════════════════════
DESIGN SYSTEM — MANDATORY
════════════════════════════════════════

Fonts (import in index.html via Google Fonts):
  - Display/Headings: Syne (weights 400,700,800)
  - Monospace numbers: DM Mono (weights 300,400,500)
  - Body: Syne

index.css must define ONLY these CSS variables and base styles:

:root {
  --bg: #080c14;
  --surface: #0d1320;
  --surface2: #131b2e;
  --surface3: #1a2440;
  --border: #1e2d4a;
  --border2: #263552;
  --text: #e2eaf8;
  --muted: #5a7299;
  --faint: #2d4066;
  --green: #00d97e;
  --red: #ff4d6a;
  --blue: #4d9fff;
  --amber: #ffb830;
  --purple: #a78bfa;
  --green-dim: #003d22;
  --red-dim: #3d0012;
  --blue-dim: #0d2a4d;
  --amber-dim: #3d2800;
  --purple-dim: #1a1040;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'Syne', sans-serif; font-size: 14px; line-height: 1.5; }
::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: var(--surface); } ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
input, select, textarea { font-family: 'Syne', sans-serif; }

════════════════════════════════════════
LAYOUT — USE THIS EXACT SHELL
════════════════════════════════════════

Layout.jsx must render:
  <div style={{ display:'grid', gridTemplateColumns:'220px 1fr', minHeight:'100vh' }}>
    <aside style={{ background:'var(--surface)', borderRight:'1px solid var(--border)', padding:'28px 16px', display:'flex', flexDirection:'column', gap:'4px' }}>
      Logo + nav items + profile at bottom
    </aside>
    <main style={{ padding:'28px 32px', overflowY:'auto', background:'var(--bg)' }}>
      {children}
    </main>
  </div>

Logo style: font-size:18px, fontWeight:800, letterSpacing:'-0.5px', accent color on last char
Nav item style: display:flex, alignItems:center, gap:10px, padding:'9px 10px', borderRadius:10px, cursor:pointer, fontSize:13px, fontWeight:500
  - Default: color: var(--muted)
  - Active: background:var(--surface3), color:var(--text), border:'1px solid var(--border2)'
  - Hover: background:var(--surface2), color:var(--text)

Use SVG icons inline (not emoji) for nav items. Keep them 18x18.

════════════════════════════════════════
COMPONENT STYLES — COPY EXACTLY
════════════════════════════════════════

STAT CARDS (for summary numbers):
  style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:14, padding:'18px 20px' }}
  Label: fontSize:11, fontWeight:600, letterSpacing:'0.08em', textTransform:'uppercase', color:'var(--muted)', marginBottom:10
  Value: fontFamily:'DM Mono,monospace', fontSize:24, fontWeight:500, letterSpacing:'-1px'
  Change: fontSize:11, fontFamily:'DM Mono,monospace', marginTop:6
    - positive: color:'var(--green)'  prefix: ↑
    - negative: color:'var(--red)'    prefix: ↓

PANELS (chart containers, lists):
  style={{ background:'var(--surface)', border:'1px solid var(--border)', borderRadius:14, padding:20 }}
  Panel header: display:flex, justifyContent:space-between, alignItems:center, marginBottom:16
  Panel title: fontSize:11, fontWeight:700, letterSpacing:'0.08em', textTransform:'uppercase', color:'var(--muted)'

TRANSACTION ROWS:
  Container: display:flex, alignItems:center, gap:12, padding:'11px 14px', background:'var(--surface2)', borderRadius:10, border:'1px solid transparent'
  Hover: border:'1px solid var(--border2)'
  Icon box: width:36, height:36, borderRadius:10, display:flex, alignItems:center, justifyContent:center
  Name: fontSize:13, fontWeight:600
  Category: fontSize:11, color:'var(--muted)', marginTop:1
  Amount positive: color:'var(--green)', fontFamily:'DM Mono'
  Amount negative: color:'var(--text)', fontFamily:'DM Mono'
  Date: fontSize:11, color:'var(--muted)', fontFamily:'DM Mono'

PROGRESS BARS:
  Track: height:6, background:'var(--surface3)', borderRadius:3, overflow:hidden
  Fill: height:'100%', borderRadius:3, transition:'width 0.8s cubic-bezier(0.4,0,0.2,1)'
  Color: use fill color from category, switch to var(--red) if > 90%

BADGES:
  Base: display:inline-flex, alignItems:center, padding:'2px 8px', borderRadius:20, fontSize:10, fontWeight:700, letterSpacing:'0.05em', textTransform:'uppercase'
  Green: background:'var(--green-dim)', color:'var(--green)'
  Red:   background:'var(--red-dim)', color:'var(--red)'
  Amber: background:'var(--amber-dim)', color:'var(--amber)'
  Blue:  background:'var(--blue-dim)', color:'var(--blue)'

BUTTONS:
  Primary: background:'var(--green)', color:'#000', border:'none', padding:'9px 18px', borderRadius:10, fontSize:13, fontWeight:700, cursor:pointer
  Ghost:   background:'transparent', color:'var(--muted)', border:'1px solid var(--border)', padding:'9px 18px', borderRadius:10, fontSize:13, cursor:pointer
  Danger:  background:'var(--red-dim)', color:'var(--red)', border:'1px solid rgba(255,77,106,0.3)', padding:'9px 18px', borderRadius:10, fontSize:13, cursor:pointer

FORM INPUTS:
  background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:10,
  padding:'10px 14px', color:'var(--text)', fontSize:13, outline:'none', width:'100%'
  onFocus: border:'1px solid var(--blue)'

MODALS:
  Backdrop: position:fixed, inset:0, background:'rgba(0,0,0,0.7)', display:flex, alignItems:center, justifyContent:center, zIndex:50
  Panel: background:'var(--surface)', border:'1px solid var(--border2)', borderRadius:16, padding:28, width:'100%', maxWidth:480

LOADING SKELETON:
  background:'var(--surface2)', borderRadius:8, animation:'pulse 1.5s ease-in-out infinite'
  Add @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} } to index.css

EMPTY STATE:
  padding:'64px 0', display:flex, flexDirection:column, alignItems:center, gap:16
  Icon: fontSize:48, opacity:0.3
  Heading: fontSize:16, fontWeight:700, color:'var(--muted)'
  Description: fontSize:13, color:'var(--faint)', textAlign:center

════════════════════════════════════════
CHARTS — USE CHART.JS
════════════════════════════════════════

Install chart.js in package.json. Import in components directly.
Chart colors: NEVER use CSS variables inside Chart.js — hardcode hex values from the design system.
  green: '#00d97e', red: '#ff4d6a', blue: '#4d9fff', amber: '#ffb830', purple: '#a78bfa', muted: '#5a7299'

Chart.js global defaults for dark theme (set once in App.jsx):
  Chart.defaults.color = '#5a7299';
  Chart.defaults.borderColor = '#1e2d4a';
  Chart.defaults.font.family = "'DM Mono', monospace";
  Chart.defaults.font.size = 11;

Tooltip style:
  plugins: { tooltip: { backgroundColor:'#0d1320', borderColor:'#1e2d4a', borderWidth:1, titleColor:'#e2eaf8', bodyColor:'#5a7299', padding:10 }}

Use useRef + useEffect to create/destroy Chart instances. Always call chart.destroy() in the useEffect cleanup to prevent "canvas already in use" errors.

════════════════════════════════════════
CODE QUALITY RULES
════════════════════════════════════════

1. Every page shows a loading skeleton (3-4 pulsing bars) while fetching
2. Every form validates inputs and shows inline errors
3. Every DELETE has a confirm dialog before calling the API
4. All axios calls wrapped in try/catch, errors shown in a toast or inline banner
5. axios instance in src/api/client.js with baseURL: 'http://localhost:8000'
6. All API calls use the axios instance — never fetch()
7. Use useState + useEffect for data fetching, useCallback for handlers
8. react-router-dom v6: useNavigate, Link, NavLink for active state
9. Numbers always formatted: use .toLocaleString() for currency, toFixed(1) for percentages
10. Hover effects on all interactive elements via onMouseEnter/onMouseLeave state

════════════════════════════════════════
FILE CONFIGS
════════════════════════════════════════

package.json:
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.23.0",
    "axios": "^1.7.0",
    "chart.js": "^4.4.1"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "@vitejs/plugin-react": "^4.2.1"
  },
  "scripts": { "dev": "vite", "build": "vite build" }
}

vite.config.js:
  import { defineConfig } from 'vite'
  import react from '@vitejs/plugin-react'
  export default defineConfig({ plugins: [react()] })

index.html: import Syne + DM Mono from Google Fonts, root div, main.jsx script

Output ONLY raw code — no markdown fences, no backticks, no explanation."""


def run_frontend_agent(state: PipelineState) -> dict:
    feedback = state.get("eval_feedback", {}).get("frontend", "")
    iteration = state.get("iteration", 0)

    if feedback and iteration > 0:
        feedback_section = (
            f"\n\n[IMPORTANT — PREVIOUS ATTEMPT WAS REJECTED]\n"
            f"You must fix these specific issues in this attempt:\n{feedback}\n"
            f"Do not repeat the same mistakes."
        )
    else:
        feedback_section = ""

    # Give frontend only the API contract (routes), not full backend code
    api_context = "\n\n".join(
        f"### {name}\n{code}"
        for name, code in state["backend_code"].items()
        if any(k in name for k in ["main", "router", "route"])
    )

    print("[FrontendAgent] Generating React frontend (pure CSS, no Tailwind)...")
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Specifications:\n{state['specs']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Backend API contract (do not rewrite — just consume these endpoints):\n{api_context}"
            f"{feedback_section}"
        ))
    ]
    response = llm.invoke(messages)
    files = _parse_files(response.content)
    print(f"[FrontendAgent] Generated: {list(files.keys())}")
    return {"frontend_code": files}


def _parse_files(raw: str) -> dict[str, str]:
    import re
    files = {}
    parts = re.split(r'#{2,3}\s*FILE:\s*', raw)
    for part in parts[1:]:
        lines = part.strip().split("\n", 1)
        if len(lines) < 2:
            continue
        filepath = lines[0].strip().strip("`").strip()
        code = lines[1].strip()
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3].rstrip()
        if filepath and code:
            files[filepath] = code.strip()
    return files
