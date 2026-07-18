from fastapi import FastAPI
from pydantic import BaseModel

from agents.analyze import analyze
from agents.parse import parse
from agents.retrieve import retrieve

app = FastAPI(title="hacknation backend")


class ParseRequest(BaseModel):
    brief_text: str | None = None


class RetrieveRequest(BaseModel):
    requirements: dict | None = None


class AnalyzeRequest(BaseModel):
    requirements: dict | None = None
    retrieved: dict | None = None


def _run_agent(agent_fn, *args):
    """Runs an agent and never lets a 5xx reach the caller. Returns
    (data, fallback_reasons)."""
    try:
        result = agent_fn(*args)
        if isinstance(result, dict) and result.get("fallback"):
            return result, [result.get("error", "agent returned a fallback response")]
        return result, []
    except Exception as exc:
        return {"error": str(exc), "fallback": True}, [str(exc)]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/parse")
def api_parse(req: ParseRequest):
    data, fallback_reasons = _run_agent(parse, req.brief_text)
    return {"data": data, "metadata": {"fallbackReasons": fallback_reasons}}


@app.post("/api/retrieve")
def api_retrieve(req: RetrieveRequest):
    data, fallback_reasons = _run_agent(retrieve, req.requirements)
    return {"data": data, "metadata": {"fallbackReasons": fallback_reasons}}


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest):
    data, fallback_reasons = _run_agent(analyze, req.requirements, req.retrieved)
    return {"data": data, "metadata": {"fallbackReasons": fallback_reasons}}
