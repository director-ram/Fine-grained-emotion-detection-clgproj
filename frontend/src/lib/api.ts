export type PredictResponse = {
  sarcastic: boolean
  label: string
  score: number
  message: string
}

const DEFAULT_API_BASE_URL = 'http://localhost:8000'

function getApiBaseUrl(): string {
  // Vite only exposes env vars prefixed with `VITE_`.
  return import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL
}

export async function predictSarcasm(text: string): Promise<PredictResponse> {
  const trimmed = text.trim()
  const apiBaseUrl = getApiBaseUrl()

  const res = await fetch(`${apiBaseUrl}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: trimmed }),
  })

  if (!res.ok) {
    // FastAPI typically returns JSON like: { "detail": "..." }
    let detail: string | undefined
    try {
      const data = (await res.json()) as { detail?: string }
      detail = data.detail
    } catch {
      // Ignore JSON parse errors; we’ll fall back to status text below.
    }

    const msg = detail || `Request failed with status ${res.status}.`
    throw new Error(msg)
  }

  return (await res.json()) as PredictResponse
}

