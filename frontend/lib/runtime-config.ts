export type DemoMode = 'precomputed' | 'live';

export const DEMO_MODE: DemoMode =
  (process.env.NEXT_PUBLIC_DEMO_MODE as DemoMode) ?? 'precomputed';

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

export const isPrecomputed = DEMO_MODE === 'precomputed';
export const isLive = DEMO_MODE === 'live';