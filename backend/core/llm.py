import json
import os
from datetime import datetime, timezone
from pathlib import Path

CONTEXT_DIR = Path(__file__).resolve().parent.parent.parent / "context"
TOKEN_BUDGET_PATH = CONTEXT_DIR / "token_budget.md"

_running_total = {"input": 0, "output": 0}


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def _log_call(agent_name, input_tokens, output_tokens):
    _running_total["input"] += input_tokens
    _running_total["output"] += output_tokens
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    if not TOKEN_BUDGET_PATH.exists():
        TOKEN_BUDGET_PATH.write_text(
            "# Token Budget\n\n"
            "Running log of every LLM call made by an agent.\n\n"
            "| Timestamp | Agent | Input Tokens | Output Tokens | Running Total |\n"
            "|---|---|---|---|---|\n"
        )
    timestamp = datetime.now(timezone.utc).isoformat()
    running_total = _running_total["input"] + _running_total["output"]
    with TOKEN_BUDGET_PATH.open("a") as f:
        f.write(
            f"| {timestamp} | {agent_name} | {input_tokens} | {output_tokens} | {running_total} |\n"
        )


def _mock_response(prompt, schema_hint):
    return {
        "mock": True,
        "note": "LLM_PROVIDER=mock — this is a stubbed response, no API call was made.",
        "schema_hint": schema_hint,
        "prompt_echo": prompt[:200],
    }


def _call_openai(prompt, schema_hint):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    system = "You output only a single valid JSON object matching this shape hint: " + json.dumps(schema_hint)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return content, input_tokens, output_tokens


def _call_anthropic(prompt, schema_hint):
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    system = (
        "You output only a single valid JSON object and nothing else, matching this "
        "shape hint: " + json.dumps(schema_hint)
    )
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    content = "".join(block.text for block in response.content if block.type == "text")
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return content, input_tokens, output_tokens


def call_llm(prompt, schema_hint, agent_name):
    """Single LLM entrypoint. Never raises — returns a dict, always."""
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()

    try:
        if provider == "mock":
            result = _mock_response(prompt, schema_hint)
            content = json.dumps(result)
            input_tokens = _estimate_tokens(prompt)
            output_tokens = _estimate_tokens(content)
        elif provider == "openai":
            content, input_tokens, output_tokens = _call_openai(prompt, schema_hint)
        elif provider == "anthropic":
            content, input_tokens, output_tokens = _call_anthropic(prompt, schema_hint)
        else:
            _log_call(agent_name, 0, 0)
            return {"error": f"unknown LLM_PROVIDER: {provider}", "fallback": True}
    except Exception as exc:
        _log_call(agent_name, 0, 0)
        return {"error": str(exc), "fallback": True}

    _log_call(agent_name, input_tokens, output_tokens)

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM output was not a JSON object")
        return parsed
    except Exception as exc:
        return {"error": f"failed to parse LLM output as JSON: {exc}", "fallback": True}
