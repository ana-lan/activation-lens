'use client';

import { useState } from 'react';

interface FeatureExample {
  activation: number;
  context: string;
}

interface FeatureResponse {
  feature_idx: number;
  examples: FeatureExample[];
}

export default function Home() {
  const [featureIdx, setFeatureIdx] = useState('500');
  const [data, setData] = useState<FeatureResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchFeature = async () => {
    setLoading(true);
    const res = await fetch(`http://localhost:8000/features/${featureIdx}`);
    const json: FeatureResponse = await res.json();
    setData(json);
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 p-8">
      <h1 className="text-2xl font-bold mb-6">ActivationLens — Feature Browser</h1>

      <div className="flex gap-2 mb-8">
        <input
          type="number"
          value={featureIdx}
          onChange={(e) => setFeatureIdx(e.target.value)}
          className="bg-neutral-800 border border-neutral-700 rounded px-3 py-2 w-40"
          placeholder="Feature index"
        />
        <button
          onClick={fetchFeature}
          className="bg-blue-600 hover:bg-blue-500 rounded px-4 py-2 font-medium"
        >
          {loading ? 'Loading...' : 'Load Feature'}
        </button>
      </div>

      {data && (
        <div>
          <h2 className="text-lg font-semibold mb-4">
            Feature #{data.feature_idx} — Top Activating Examples
          </h2>
          <div className="space-y-2">
            {data.examples.map((ex, i) => (
              <div
                key={i}
                className="bg-neutral-900 border border-neutral-800 rounded p-3 flex gap-4"
              >
                <span className="text-blue-400 font-mono w-20 shrink-0">
                  {ex.activation.toFixed(2)}
                </span>
                <span className="text-neutral-300">{ex.context}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}