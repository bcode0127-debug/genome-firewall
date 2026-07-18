import json
from pathlib import Path

from core.llm import call_llm

CONTEXT_DIR = Path(__file__).resolve().parent.parent.parent / "context"
REQUIREMENTS_PATH = CONTEXT_DIR / "requirements.md"
RETRIEVED_PATH = CONTEXT_DIR / "retrieved.md"
ANALYSIS_PATH = CONTEXT_DIR / "analysis.md"

SCHEMA_HINT = {
    "summary": "string",
    "findings": ["string"],
    "recommendations": ["string"],
}


def analyze(requirements=None, retrieved=None):
    """Reads context/requirements.md and context/retrieved.md (or uses the
    given args), returns result JSON, and writes it to context/analysis.md."""
    if requirements is None:
        requirements = REQUIREMENTS_PATH.read_text() if REQUIREMENTS_PATH.exists() else ""
    if retrieved is None:
        retrieved = RETRIEVED_PATH.read_text() if RETRIEVED_PATH.exists() else ""

    prompt = (
        "Analyze the following requirements and retrieved context, and return JSON "
        f"matching this shape: {json.dumps(SCHEMA_HINT)}\n\n"
        f"Requirements:\n{requirements}\n\nRetrieved:\n{retrieved}"
    )
    result = call_llm(prompt, SCHEMA_HINT, agent_name="analyze")

    ANALYSIS_PATH.write_text(
        "# Analysis\n\n"
        "Output of the `analyze` agent, derived from `requirements.md` and `retrieved.md`.\n\n"
        "## Latest Output\n\n"
        f"```json\n{json.dumps(result, indent=2)}\n```\n"
    )

    return result
