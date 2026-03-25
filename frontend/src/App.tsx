import { useEffect, useRef, useState } from 'react'
import { predictSarcasm } from './lib/api'

export default function App() {
  const [text, setText] = useState('')
  const [message, setMessage] = useState<string | null>(null)
  const [sarcastic, setSarcastic] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const copyTimerRef = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
    }
  }, [])

  async function onAnalyze() {
    setError(null)
    setMessage(null)
    setSarcastic(null)

    const trimmed = text.trim()
    if (!trimmed) {
      setError('Please enter a sentence.')
      return
    }

    setLoading(true)
    try {
      const resp = await predictSarcasm(trimmed)
      setSarcastic(resp.sarcastic)
      setMessage(resp.message)
    } catch (e) {
      setMessage(null)
      setSarcastic(null)
      setError(e instanceof Error ? e.message : 'Request failed.')
    } finally {
      setLoading(false)
    }
  }

  function onClear() {
    setText('')
    setMessage(null)
    setSarcastic(null)
    setError(null)
    setCopied(false)
  }

  async function onCopy() {
    if (!message) return
    try {
      await navigator.clipboard.writeText(message)
      setCopied(true)
      if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
      copyTimerRef.current = window.setTimeout(() => setCopied(false), 1200)
    } catch {
      setError('Could not copy to clipboard.')
    }
  }

  const resultAccent =
    sarcastic === true
      ? 'ring-cyan-300/30'
      : sarcastic === false
        ? 'ring-purple-300/30'
        : 'ring-white/10'

  return (
    <div className="min-h-screen px-4">
      <div className="mx-auto flex min-h-screen max-w-3xl flex-col justify-center">
        <header className="mb-7 text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-(--stroke) bg-[rgba(255,255,255,0.05)] px-4 py-2 shadow-sm">
            <span className="text-xs font-semibold tracking-wide text-(--muted)">
              Sarcasm detection
            </span>
            <span className="text-xs font-medium text-[rgba(255,255,255,0.78)]">
              FastAPI + Transformer
            </span>
          </div>
          <h1 className="mt-4 text-4xl font-semibold leading-tight md:text-5xl">
            Sarcasm Oracle
          </h1>
          <p className="mt-3 text-[15px] leading-relaxed text-(--muted)">
            Type a sentence and get a single yes/no response.
          </p>
        </header>

        <main className="relative rounded-3xl border border-(--stroke) bg-[rgba(255,255,255,0.04)] p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)] backdrop-blur md:p-7">
          <div className="relative overflow-hidden rounded-2xl border border-(--stroke) bg-[rgba(0,0,0,0.10)] p-4 md:p-6">
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 bg-[radial-gradient(500px_circle_at_20%_10%,rgba(192,132,252,0.28),transparent_50%),radial-gradient(500px_circle_at_90%_0%,rgba(34,211,238,0.22),transparent_55%)]"
            />

            <div className="relative">
              <label
                htmlFor="sentence"
                className="block text-sm font-semibold text-(--muted)"
              >
                Sentence
              </label>
              <textarea
                id="sentence"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="e.g., Oh great, another bug in production."
                className="mt-2 min-h-[110px] w-full resize-none rounded-xl border border-(--stroke) bg-[rgba(255,255,255,0.03)] p-4 text-[15px] leading-relaxed text-[rgba(255,255,255,0.92)] outline-none transition focus:border-[rgba(192,132,252,0.65)] focus:ring-2 focus:ring-[rgba(192,132,252,0.25)]"
              />

              <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
                <button
                  onClick={onAnalyze}
                  disabled={loading}
                  className="inline-flex h-[44px] items-center justify-center rounded-xl bg-[rgba(192,132,252,0.18)] px-4 font-semibold text-white transition hover:bg-[rgba(192,132,252,0.26)] disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {loading ? (
                    <span className="inline-flex items-center gap-2">
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/60 border-t-white" />
                      Analyzing…
                    </span>
                  ) : (
                    'Analyze'
                  )}
                </button>
                <button
                  onClick={onClear}
                  disabled={loading}
                  className="inline-flex h-[44px] items-center justify-center rounded-xl border border-(--stroke) bg-[rgba(255,255,255,0.03)] px-4 font-semibold text-(--muted) transition hover:bg-[rgba(255,255,255,0.06)] disabled:cursor-not-allowed disabled:opacity-70"
                >
                  Clear
                </button>
              </div>

              <div className="mt-5 min-h-[92px]">
                {error ? (
                  <div className="rounded-xl border border-red-400/30 bg-red-500/10 p-4 text-sm text-red-200">
                    {error}
                  </div>
                ) : null}

                {!error && message ? (
                  <div
                    className={`result-reveal mt-0 rounded-2xl border border-(--stroke) bg-[rgba(255,255,255,0.05)] p-5 ${resultAccent}`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="mt-0 text-2xl font-semibold">
                          {message}
                        </div>
                      </div>
                      <button
                        onClick={onCopy}
                        className="rounded-xl border border-(--stroke) bg-[rgba(0,0,0,0.10)] px-3 py-2 text-sm font-semibold text-[rgba(255,255,255,0.88)] transition hover:bg-[rgba(255,255,255,0.06)]"
                      >
                        {copied ? 'Copied' : 'Copy result'}
                      </button>
                    </div>
                  </div>
                ) : null}

                {!error && !message && !loading ? (
                  <div className="pt-2 text-sm text-(--muted)">
                    Enter text and click <span className="font-semibold">Analyze</span>.
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <div className="mt-6 text-center text-xs text-(--muted)">
            Privacy note: your text is sent to the local backend at{' '}
            <code className="rounded border border-(--stroke) bg-[rgba(255,255,255,0.03)] px-2 py-1 text-[11px]">
              {import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}
            </code>
            .
          </div>
        </main>
      </div>
    </div>
  )
}
