import json
from pathlib import Path

from core.llm import call_llm

CONTEXT_DIR = Path(__file__).resolve().parent.parent.parent / "context"
BRIEF_PATH = CONTEXT_DIR / "brief.md"
REQUIREMENTS_PATH = CONTEXT_DIR / "requirements.md"

SCHEMA_HINT = {
    "goals": ["string"],
    "constraints": ["string"],
    "deliverables": ["string"],
}


def parse(brief_text=None):
    """Reads context/brief.md (or uses brief_text if given), returns structured
    requirements JSON, and writes the result to context/requirements.md."""
    if brief_text is None:
        brief_text = BRIEF_PATH.read_text() if BRIEF_PATH.exists() else ""

    prompt = (
        "Extract structured requirements from this hackathon brief. "
        f"Return JSON matching this shape: {json.dumps(SCHEMA_HINT)}\n\n"
        f"Brief:\n{brief_text}"
    )
    result = call_llm(prompt, SCHEMA_HINT, agent_name="parse")

    REQUIREMENTS_PATH.write_text(
        "# Requirements\n\n"
        "Structured output of the `parse` agent, derived from `brief.md`.\n\n"
        "## Latest Output\n\n"
        f"```json\n{json.dumps(result, indent=2)}\n```\n"
    )

    return result
