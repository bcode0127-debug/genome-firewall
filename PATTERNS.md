# Shared Patterns

Five repos: `foldforward` (FF), `ai-scientist` (AIS), `N-3-AI-Scientist` (N3),
`agentic-labmate` (AL), `lab-mindai` (LM). Only conventions common to 3+ are kept.

---

## 1. LLM Wrapper

**One thin `(system, user) -> text` function per repo. Provider varies (OpenAI /
Anthropic / Groq / Gemini), the shape doesn't.** JSON is forced two ways, always both:
`response_format:{type:"json_object"}` (FF, N3, AL, LM) **and** "Return ONLY valid JSON"
in the system prompt (all 5). Temperature is pinned low: 0.1–0.3.

```python
# N3 ai_scientist/llm_clients.py — canonical form
request = {"model": model, "response_format": {"type": "json_object"},
           "temperature": 0.1,
           "messages": [{"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(payload)}]}
```
```javascript
// LM backend/src/server.js:470 — JSON mode, retry once without it on 400
let resp = await groqFetch(true);
if (!resp.ok && resp.status === 400) resp = await groqFetch(false);
```

**Bad-JSON handling is the same helper in every repo: strip ```json fences → try
parse → regex-grab the outer `{...}` → parse again → give up.**

```python
# FF _parse_json_object_loose / N3 _parse_json_object / AL _extract_json
def parse_loose(text):
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    m = re.search(r"\{[\s\S]*\}", text)          # or \{.*\} DOTALL
    return json.loads(m.group(0)) if m else None
```
```javascript
// LM json-extract.js / AIS inline / AL — same three moves
const fence = text.match(/```json\s*([\s\S]*?)```/i);
const c = fence ? fence[1] : text;
const s = c.indexOf("{"), e = c.lastIndexOf("}");
try { return JSON.parse(c.slice(s, e + 1)); } catch { return null; }
```

**On failure: never propagate — return a deterministic stub with the valid empty
shape (FF, N3, AL, LM; AIS is the exception, returns HTTP 502).** The stub is a
full plan skeleton so downstream stages/gates still run.

```python
# AL plan_gen.py — parse fails → minimal valid plan, not an exception
plan = {"protocol": {"steps": []}, "materials": {"items": []},
        "budget": {"total_usd": 0.0, "breakdown": []},
        "timeline": {"total_weeks": 0, "phases": []},
        "validation": {"approach": raw[:500], "success_criteria": ""}}
```
```python
# FF pattern, repeated per artifact: try LLM, record reason, fall to determinism
try: plan = _masterplan_from_inputs(req)
except Exception as exc: fallback_reasons.append(f"Agent2 plan: {exc}")
if plan is None: plan = _masterplan_fallback(req)
```
Multi-provider repos (AL, LM) extend this: on truncation/429, **retry with a bigger
token budget, then fail over to the next provider in an ordered list**, stub only if
all fail.

---

## 2. Streaming (SSE)

Present in FF, AIS, N3, AL (LM + N3-chat are plain POST→JSON). **Server = async
generator yielding `event:/data:` frames; the formatter is byte-identical across
the Python repos.**

```python
# FF main.py / N3 app.py / AL streaming.py — same one-liner
def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.get("/api/papers/stream")                       # FF; N3 & AL: POST + queue
def stream(query: str):
    def gen():
        yield sse("agent1_start", {"message": "reasoning…"})
        for p in papers: yield sse("paper", p)
        yield sse("done", {})
    return StreamingResponse(gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```
AL/N3 run the LLM work in a thread pool and bridge events over an
`asyncio.Queue` (`loop.call_soon_threadsafe(queue.put_nowait, ...)`), the generator
drains the queue with a timeout so it can emit keep-alive `progress` frames.

**Client = one of two shapes.** Native `EventSource` with per-event listeners (FF),
or manual `reader.read()` + buffer split on `\n`, keep the `data:` lines (AIS, AL):

```typescript
// AIS useAgentStream.ts / AL test util — manual SSE reader, the common JS form
const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
while (true) {
  const { done, value } = await reader.read(); if (done) break;
  buf += dec.decode(value, { stream: true });
  const lines = buf.split("\n"); buf = lines.pop() ?? "";
  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    const p = line.slice(6).trim(); if (p === "[DONE]") continue;
    try { onDelta(JSON.parse(p)); } catch { /* ignore partial frame */ }
  }
}
```
Conventions: header trio `text/event-stream` + `no-cache` + `keep-alive`; terminal
sentinel (`event:done` or `data: [DONE]`); every JSON.parse wrapped in an empty
`catch` so a split frame never kills the loop.

---

## 3. Agent Definition Shape

**Every repo models the same team: one specialist agent per plan section —
`protocol · materials · budget · timeline · validation`** (AIS `PLAN_AGENTS`, AL
`SPECIALISTS`, LM stages, FF agent1/2/4, N3 characters). System prompts live
**inline in a static definition object/dataclass**, keyed by id, with a persona
string + a `getPrompt(hypothesis)` template.

```typescript
// AIS agents.ts — array of definitions, system prompt is a field
interface AgentDefinition { id: AgentId; label: string; system: string;
                            getPrompt: (h: string) => string; }
{ id: "protocol", label: "Protocol Architect",
  system: "You are a senior research scientist... write precise protocols.",
  getPrompt: h => `Design a step-by-step protocol for:\n"${h}"` }
```
```python
# AL council.py — same team as a frozen dataclass list
@dataclass(frozen=True)
class Specialist: agent: str; section: str; persona: str
SPECIALISTS = [Specialist("ProtocolArchitect", "protocol", "Senior protocol scientist."),
               Specialist("BudgetAnalyst",     "budget",   "Lab ops manager, 2025 pricing."),
               ...]  # returns ONLY JSON {section, content, notes}
```

**What differs between agents: persona + section + prompt only. Model and
temperature are uniform across the team** — set globally, not per agent. Extras seen
in 2+: a `DevilsAdvocate`/critique agent over the drafts (AL, LM gates), and prior
scientist corrections appended to the system prompt (AIS, LM feedback).

```typescript
// AIS — corrections grafted onto the shared system prompt at call time
const systemPrompt = corrections.length
  ? `${agentDef.system}\n\nSCIENTIST CORRECTIONS — apply without being asked:\n${corrections.join("\n")}`
  : agentDef.system;
```

---

## 4. Conventions (3+ repos)

- **Env-configured provider + model.** `<PROVIDER>_API_KEY` (`OPENAI_/ANTHROPIC_/
  GROQ_/GOOGLE_API_KEY`) plus a per-role model override read from env, with a
  hardcoded default. Model IDs seen: `gpt-4o-mini`, `claude-sonnet-4-6` /
  `claude-3-5-haiku`, `gemini-2.5-flash`, `llama-3.3-70b-versatile`.
- **The 5-section plan domain** (protocol / materials / budget / timeline /
  validation) is the shared data model everywhere.
- **JSON-extract helper** (fence-strip + outer-`{}` regex) — its own function in
  FF, N3, AL, LM.
- **Fallback stub** with the full valid empty shape so gates/UI never see null.
- **Tag or flag provenance**: N3 stamps `_llm_provider`/`_llm_model` on every
  result; LM/FF carry a `releaseDegraded` / `fallback_reasons` field; results are a
  discriminated `{ok:true,data} | {ok:false,message}` union (FF, LM).
- **Frontend → backend URL** from `BACKEND_URL` / `NEXT_PUBLIC_BACKEND_URL` with a
  `127.0.0.1` default; Next.js route handlers proxy and pass the stream body through
  untouched.
