# hacknation

Generic hackathon skeleton: a three-stage pipeline (parse → retrieve → analyze)
fronted by a Next.js UI, backed by a FastAPI service. Nothing challenge-specific
is baked in yet — fill in `context/brief.md` and extend the agents once the
challenge is revealed.

## Structure

- `frontend/` — Next.js 15 (App Router, TypeScript, Tailwind). One page, three
  result panels. Browser only talks to same-origin `app/api/*` routes, which
  proxy to the backend — no CORS config needed.
- `backend/` — FastAPI app.
  - `main.py` — routes: `GET /health`, `POST /api/parse`, `POST /api/retrieve`,
    `POST /api/analyze`.
  - `agents/` — `parse.py`, `retrieve.py`, `analyze.py`.
  - `core/llm.py` — the one LLM wrapper every agent goes through.
- `context/` — the only channel agents read/write through: `brief.md`,
  `requirements.md`, `retrieved.md`, `analysis.md`, `decisions.md`,
  `token_budget.md`.

## Run commands

Backend:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # LLM_PROVIDER defaults to mock — no key needed
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
cp .env.local.example .env.local   # BACKEND_URL defaults to http://127.0.0.1:8000
npm run dev
```

## Agent contract

Every agent in `backend/agents/` follows the same rules:

- One job, one exported function, no loops, no retries.
- Reads its inputs from `context/*.md` files (or accepts them as function args
  for direct chaining) — never from chat history.
- Writes its output back to its own `context/*.md` file.
- Any call to an LLM goes through `core.llm.call_llm(prompt, schema_hint, agent_name)`:
  temperature 0, forced JSON object output, defensive parsing. It never raises —
  on any failure it returns `{"error": ..., "fallback": True}`.
- Every `call_llm` invocation is logged to `context/token_budget.md` (agent
  name, input/output tokens, running total).

API endpoints always return HTTP 200 with a complete payload
(`{"data": ..., "metadata": {"fallbackReasons": [...]}}`), even when an agent
fails — failures are reported in `metadata.fallbackReasons`, never as a 5xx.

## Decisions log

Any non-obvious decision made while building this project gets a line in
`context/decisions.md`. Check it before assuming why something is the way it
is; add to it before changing something that isn't obviously wrong.
