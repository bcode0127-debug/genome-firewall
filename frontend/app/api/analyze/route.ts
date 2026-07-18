import { proxyToBackend } from "@/lib/backend";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  return proxyToBackend("/api/analyze", body);
}
