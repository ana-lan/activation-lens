'use client';

import { useState } from 'react';
import Nav from '@/components/Nav';
import FeatureBrowser from '@/components/FeatureBrowser';
import SafetyMonitor from '@/components/SafetyMonitor';

export default function DemoPage() {
  const [tab, setTab] = useState<'features' | 'safety'>('features');

  return (
    <main className="min-h-screen bg-[#0a0810] text-neutral-300">
      <Nav />

      <div className="max-w-5xl mx-auto px-6 pt-6">
        <div className="flex items-center gap-1 border border-neutral-800 rounded-lg p-1 bg-[#100c1a] w-fit mb-6">
          <button
            onClick={() => setTab('features')}
            className={`px-4 py-2 rounded-md text-sm transition-colors ${
              tab === 'features' ? 'bg-violet-950 text-violet-300 font-medium' : 'text-neutral-400 hover:text-white'
            }`}
          >
            Feature Browser
          </button>
          <button
            onClick={() => setTab('safety')}
            className={`px-4 py-2 rounded-md text-sm transition-colors ${
              tab === 'safety' ? 'bg-violet-950 text-violet-300 font-medium' : 'text-neutral-400 hover:text-white'
            }`}
          >
            Safety Monitor
          </button>
        </div>

        {tab === 'features' ? <FeatureBrowser /> : <SafetyMonitor />}
      </div>
    </main>
  );
}