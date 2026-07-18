"use client";

import { useState } from "react";

type StageStatus = "idle" | "loading" | "done" | "error";

type StageResult = {
  status: StageStatus;
  data: unknown;
  fallbackReasons: string[];
};

const EMPTY_STAGE: StageResult = { status: "idle", data: null, fallbackReasons: [] };

async function callStage(path: string, body: unknown): Promise<StageResult> {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const json = await res.json();
    return {
      status: "done",
      data: json.data,
      fallbackReasons: json.metadata?.fallbackReasons ?? [],
    };
  } catch (err) {
    return {
      status: "error",
      data: null,
      fallbackReasons: [err instanceof Error ? err.message : "request failed"],
    };
  }
}

function Panel({ title, result }: { title: string; result: StageResult }) {
  return (
    <div className="flex-1 min-w-0 rounded-lg border border-zinc-200 dark:border-zinc-800 p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-lg">{title}</h2>
        <span
          className={
            "text-xs px-2 py-0.5 rounded-full " +
            (result.status === "loading"
              ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
              : result.status === "done"
                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                : result.status === "error"
                  ? "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                  : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400")
          }
        >
          {result.status}
        </span>
      </div>
      {result.status === "loading" && (
        <p className="text-sm text-zinc-500 animate-pulse">Running…</p>
      )}
      {result.fallbackReasons.length > 0 && (
        <ul className="text-xs text-red-600 dark:text-red-400 list-disc list-inside">
          {result.fallbackReasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      )}
      {result.data != null && (
        <pre className="text-xs bg-zinc-50 dark:bg-zinc-900 rounded p-3 overflow-auto max-h-80 whitespace-pre-wrap">
          {JSON.stringify(result.data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default function Home() {
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [parseResult, setParseResult] = useState<StageResult>(EMPTY_STAGE);
  const [retrieveResult, setRetrieveResult] = useState<StageResult>(EMPTY_STAGE);
  const [analyzeResult, setAnalyzeResult] = useState<StageResult>(EMPTY_STAGE);

  async function runPipeline() {
    setRunning(true);
    setParseResult({ ...EMPTY_STAGE, status: "loading" });
    setRetrieveResult(EMPTY_STAGE);
    setAnalyzeResult(EMPTY_STAGE);

    const parsed = await callStage("/api/parse", { brief_text: input || null });
    setParseResult(parsed);

    setRetrieveResult({ ...EMPTY_STAGE, status: "loading" });
    const retrieved = await callStage("/api/retrieve", { requirements: parsed.data });
    setRetrieveResult(retrieved);

    setAnalyzeResult({ ...EMPTY_STAGE, status: "loading" });
    const analyzed = await callStage("/api/analyze", {
      requirements: parsed.data,
      retrieved: retrieved.data,
    });
    setAnalyzeResult(analyzed);

    setRunning(false);
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black p-8">
      <main className="max-w-5xl mx-auto flex flex-col gap-6">
        <h1 className="text-2xl font-semibold">hacknation pipeline</h1>

        <div className="flex flex-col gap-3">
          <textarea
            className="w-full rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 p-3 text-sm min-h-32"
            placeholder="Paste the brief here (optional — falls back to context/brief.md)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            className="self-start rounded-full bg-foreground text-background px-5 py-2 text-sm font-medium disabled:opacity-50"
            onClick={runPipeline}
            disabled={running}
          >
            {running ? "Running…" : "Run pipeline"}
          </button>
        </div>

        <div className="flex flex-col md:flex-row gap-4">
          <Panel title="Parse" result={parseResult} />
          <Panel title="Retrieve" result={retrieveResult} />
          <Panel title="Analyze" result={analyzeResult} />
        </div>
      </main>
    </div>
  );
}
