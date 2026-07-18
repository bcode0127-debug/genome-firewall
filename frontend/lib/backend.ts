export const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function proxyToBackend(path: string, body: unknown) {
  try {
    const res = await fetch(`${BACKEND_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    const json = await res.json();
    return Response.json(json, { status: res.status });
  } catch (err) {
    return Response.json(
      {
        data: null,
        metadata: {
          fallbackReasons: [
            err instanceof Error ? err.message : "failed to reach backend",
          ],
        },
      },
      { status: 200 },
    );
  }
}
