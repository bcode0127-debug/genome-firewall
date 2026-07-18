# Requirements

Structured output of the `parse` agent, derived from `brief.md`.

## Latest Output

```json
{
  "mock": true,
  "note": "LLM_PROVIDER=mock \u2014 this is a stubbed response, no API call was made.",
  "schema_hint": {
    "goals": [
      "string"
    ],
    "constraints": [
      "string"
    ],
    "deliverables": [
      "string"
    ]
  },
  "prompt_echo": "Extract structured requirements from this hackathon brief. Return JSON matching this shape: {\"goals\": [\"string\"], \"constraints\": [\"string\"], \"deliverables\": [\"string\"]}\n\nBrief:\nBuild a tool that summa"
}
```
