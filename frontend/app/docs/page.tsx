import Nav from "@/components/Nav";

export default function DocsPage() {
  return (
    <main className="min-h-screen bg-[#0a0810] text-neutral-300">
      <Nav />
      <div className="max-w-5xl mx-auto px-6 pt-6 pb-20">
        <div className="rounded-2xl border border-neutral-800 bg-[#100c1a] px-10 py-12 mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">Docs</h1>
          <p className="text-neutral-400">Setup, reproduction, architecture, and known limitations.</p>
        </div>

        <DocSection title="Setup">
          <Code>{`git clone https://github.com/ana-lan/activation-lens
cd activation-lens
conda create -n activation-lens python=3.11 -y
conda activate activation-lens
pip install -r requirements.txt`}</Code>
        </DocSection>

        <DocSection title="Build the SAE">
          <Code>{`python3 -m model.build_activation_cache
python3 -m model.train_sae`}</Code>
          <p className="text-neutral-500 text-sm mt-4 leading-relaxed">
            Dead-feature diagnosis: early runs showed 52–82% of features going permanently dead.
            Decoder weight normalization + periodic resampling reduced this to 0.77%.
          </p>
        </DocSection>

        <DocSection title="Reproduce the experiments">
          <Code>{`# Phase 5 — quantization benchmark
python3 -m benchmarks.feature_survival

# Phase 6 — safety monitor
python3 -m safety.find_harm_feature
python3 -m safety.multi_feature_classifier
python3 -m safety.calibrate_live_multifeature
python3 -m safety.benchmark_overhead

# Phase 7 — steering
python3 -m steering.screen_candidates
python3 -m steering.stage2_test_candidates
python3 -m steering.stage2_persistent_test`}</Code>
        </DocSection>

        <DocSection title="Run the dashboard locally">
          <Code>{`uvicorn backend.api.main:app --reload

# in a second terminal:
cd frontend
npm install
npm run dev`}</Code>
          <p className="text-neutral-500 text-sm mt-4 leading-relaxed">
            The public demo runs in precomputed mode. Set <span className="text-violet-300 font-mono">NEXT_PUBLIC_DEMO_MODE=live</span> locally for real inference on arbitrary prompts and features.
          </p>
        </DocSection>

        <DocSection title="Environment & hardware">
          <ul className="text-neutral-400 text-sm space-y-2 list-disc list-inside leading-relaxed">
            <li>Apple Silicon Mac; all final experiments run CPU-only</li>
            <li>TransformerLens warned the installed PyTorch/MPS combination could produce incorrect results</li>
            <li>Python 3.11, conda environment (see requirements.txt)</li>
            <li>Overhead benchmark: 4 PyTorch CPU threads, no KV-cache</li>
          </ul>
        </DocSection>

        <DocSection title="Limitations">
          <ul className="text-neutral-400 text-sm space-y-2 list-disc list-inside leading-relaxed">
            <li>Quantization is simulated via uniform min-max weight rounding, not a production backend (GPTQ/AWQ)</li>
            <li>Single model (GPT-2-small), single layer (6), single SAE training seed</li>
            <li>Steering tested at one layer only; negative result may not generalize</li>
            <li>Safety monitor has no baseline comparison against simpler methods</li>
          </ul>
        </DocSection>

        <DocSection title="Future work" last>
          <ul className="text-neutral-400 text-sm space-y-2 list-disc list-inside leading-relaxed">
            <li>Real quantization backends (GPTQ/AWQ)</li>
            <li>Steering on a larger open model (Llama 3 8B / Gemma 2) with a correspondingly larger SAE</li>
            <li>Layer sweep for both interpretability and steering</li>
            <li>Raw-residual / plain-text-classifier baselines for the safety monitor</li>
            <li>KV-caching, batching, and speculative-decoding + steering interaction</li>
          </ul>
        </DocSection>

        <div className="rounded-2xl border border-neutral-800 bg-[#100c1a] px-10 py-8 mt-6 text-center">
          <p className="text-neutral-500 text-sm">
            Full methodology, exact metrics, and committed JSON result artifacts: see the{" "}
            <a href="https://github.com/ana-lan/activation-lens" className="text-violet-300 hover:text-violet-200 underline underline-offset-2">
              repository README
            </a>.
          </p>
        </div>
      </div>
    </main>
  );
}

function DocSection({ title, children, last }: { title: string; children: React.ReactNode; last?: boolean }) {
  return (
    <div className={`rounded-2xl border border-neutral-800 bg-[#100c1a] px-10 py-8 ${last ? "" : "mb-6"}`}>
      <h2 className="text-lg font-semibold text-white mb-5">{title}</h2>
      {children}
    </div>
  );
}

function Code({ children }: { children: string }) {
  return (
    <pre className="bg-[#0c0916] border border-neutral-800 rounded-xl p-5 text-xs text-neutral-400 font-mono overflow-x-auto leading-relaxed">
      {children}
    </pre>
  );
}