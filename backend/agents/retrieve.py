import json
from pathlib import Path

CONTEXT_DIR = Path(__file__).resolve().parent.parent.parent / "context"
REQUIREMENTS_PATH = CONTEXT_DIR / "requirements.md"
RETRIEVED_PATH = CONTEXT_DIR / "retrieved.md"


def retrieve(requirements=None):
    """Reads context/requirements.md (or uses requirements if given). Stub:
    returns an empty results list and writes it to context/retrieved.md."""
    if requirements is None:
        REQUIREMENTS_PATH.read_text() if REQUIREMENTS_PATH.exists() else ""

    result = {"results": []}

    RETRIEVED_PATH.write_text(
        "# Retrieved\n\n"
        "Output of the `retrieve` agent, derived from `requirements.md`.\n\n"
        "## Latest Output\n\n"
        f"```json\n{json.dumps(result, indent=2)}\n```\n"
    )

    return result
