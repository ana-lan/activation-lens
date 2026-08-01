'use client';

import { useState } from 'react';

interface TokenLogEntry {
  step: number;
  token: string;
  score: number;
  eligible_for_alert: boolean;
  above_threshold: boolean;
  alert_active: boolean;
}

interface SafetyResult {
  generated_text: string;
  per_token_log: TokenLogEntry[];
  flagged: boolean;
  flagged_at_token: number | null;
  threshold: number;
  min_prefix_tokens: number;
  disclaimer: string;
}

const getErrorMessage = (detail: unknown, status: number): string => {
  if (typeof detail === 'string') {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item: unknown) => {
        if (typeof item === 'object' && item !== null && 'msg' in item && typeof item.msg === 'string') {
          return item.msg;
        }
        return 'Invalid request';
      })
      .join('; ');
  }

  return `Request failed (${status})`;
};

export default function SafetyMonitor() {
  const [prompt, setPrompt] = useState('The best way to deal with a difficult situation is');
  const [data, setData] = useState<SafetyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runMonitor = async () => {
    if (!prompt.trim()) {
      setError('Enter a prompt first.');
      return;
    }

    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch('http://localhost:8000/safety/monitor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), max_new_tokens: 30 }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(getErrorMessage(body?.detail, res.status));
      }

      const json: SafetyResult = await res.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The monitor request failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 p-8">
      <h1 className="text-2xl font-bold mb-2">ActivationLens — Safety Monitor Trace (Experimental)</h1>

      <div className="bg-yellow-950 border border-yellow-700 rounded p-4 mb-6 text-sm text-yellow-200">
        <strong>Experimental prototype, not a production safety system.</strong> This computes the
        monitor token-by-token internally, but shows the full trace after generation finishes —
        a per-token trace, not a live-streaming UI. Expect frequent false alarms on benign text.
      </div>

      {error && (
        <div className="mb-6 rounded border border-red-800 bg-red-950 p-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="flex gap-2 mb-8">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="bg-neutral-800 border border-neutral-700 rounded px-3 py-2 flex-1"
          placeholder="Enter a prompt..."
        />
        <button
          onClick={runMonitor}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50 rounded px-4 py-2 font-medium whitespace-nowrap"
        >
          {loading ? 'Generating...' : 'Run Monitor'}
        </button>
      </div>

      {data && (
        <div>
          <div className="mb-4 rounded border border-yellow-700 bg-yellow-950 p-4 text-sm text-yellow-200">
            <strong>Experimental prototype.</strong> {data.disclaimer}
          </div>

          <div className="mb-4 p-3 rounded bg-neutral-900 border border-neutral-800">
            <span className={data.flagged ? 'text-red-400 font-semibold' : 'text-green-400 font-semibold'}>
              {data.flagged ? `⚠ Alert triggered at token ${data.flagged_at_token}` : '✓ No alert triggered'}
            </span>
            <span className="text-neutral-400 ml-4 text-sm">
              threshold={data.threshold.toFixed(4)} | min_prefix={data.min_prefix_tokens}
            </span>
          </div>

          <p className="text-neutral-300 mb-6 whitespace-pre-wrap">{data.generated_text}</p>

          <h2 className="text-lg font-semibold mb-3">Per-token trace</h2>
          <div className="space-y-1">
            {data.per_token_log.map((entry) => (
              <div
                key={entry.step}
                className={`flex gap-4 p-2 rounded border text-sm ${
                  entry.alert_active ? 'bg-red-950 border-red-800' : 'bg-neutral-900 border-neutral-800'
                }`}
              >
                <span className="w-8 text-neutral-500">{entry.step}</span>
                <span className="w-24 font-mono whitespace-pre">{JSON.stringify(entry.token)}</span>
                <span className="w-20 text-blue-400 font-mono">{entry.score.toFixed(4)}</span>
                <span className="text-neutral-500">
                  {!entry.eligible_for_alert ? 'not yet eligible' : entry.above_threshold ? 'above threshold' : 'below threshold'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}