import Nav from "@/components/Nav";

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-[#0a0810] text-neutral-300">
      <Nav />

      <div className="max-w-5xl mx-auto px-6 pt-6">
        {/* Hero + stats — ONE bordered panel */}
        <div className="rounded-2xl border border-neutral-800 bg-[#100c1a] px-10 py-12">
          <span className="inline-flex items-center gap-2 text-xs text-neutral-400 border border-neutral-700 rounded-full px-3 py-1 mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400" />
            Open source · Benchmarked on GPT-2-small
          </span>

          <h1 className="text-5xl font-bold leading-tight mb-6 max-w-3xl">
            <span className="text-white">Does </span>
            <span className="text-indigo-400">quantization</span>
            <span className="text-white"> quietly break </span>
            <span className="text-violet-400">interpretability</span>
            <span className="text-white">?</span>
          </h1>

          <p className="text-neutral-400 max-w-2xl mb-8 leading-relaxed">
            A sparse autoencoder built from scratch on GPT-2-small, benchmarked against
            simulated quantization and converted into a live, low-overhead safety monitor.
          </p>

          <div className="flex items-center gap-3 mb-12">
            <a href="/docs" className="px-4 py-2.5 rounded-lg bg-violet-950 border border-violet-800 text-violet-300 text-sm font-medium hover:bg-violet-900/50 transition-colors">
              Read the docs
            </a>
            <a href="https://github.com/ana-lan/activation-lens" className="px-4 py-2.5 rounded-lg border border-neutral-700 text-neutral-300 text-sm font-medium hover:border-neutral-600 transition-colors">
              View on GitHub ↗
            </a>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard value="0.796" label="Safety-probe AUROC" sub="validated on untouched holdout" />
            <StatCard value="+0.66%" label="Live monitor overhead" sub="not distinguishable from zero" />
            <StatCard value="8,192" label="SAE features" sub="trained from scratch on GPT-2-small" />
            <StatCard value="500" label="Held-out documents" sub="per benchmark, zero overlap with training" />
          </div>
        </div>

        {/* Experiment 1 */}
        <ExperimentSection eyebrow="EXPERIMENT 1 · QUANTIZATION" title="Does quantization break interpretability?">
          <p className="text-neutral-400 mb-6 leading-relaxed">
            Trained a sparse autoencoder on GPT-2-small&apos;s layer-6 residual stream. Simulated 16/8/4-bit
            quantization via uniform weight rounding, evaluated on 500 held-out documents using three
            independent metrics: activating-position recall, per-feature correlation, and behavioral checks.
          </p>

          <div className="grid grid-cols-3 gap-4 mb-8">
            <MiniStat value="1.99×" label="Perplexity ratio at 8-bit" />
            <MiniStat value="0.641" label="Feature correlation at 8-bit" />
            <MiniStat value="32,328×" label="Perplexity ratio at 4-bit" color="text-amber-400" />
          </div>

          <ResultTable
            headers={["Precision", "PPL ratio", "Top-1 agree", "Correlation", "Recall@5"]}
            rows={[
              ["16-bit", "1.00×", "99.92%", "1.000", "99.83%"],
              ["8-bit", "1.99×", "59.33%", "0.641", "29.44%"],
              ["4-bit", "32,328×", "0.38%", "0.004", "0.01%"],
            ]}
          />

          <p className="text-neutral-500 text-sm mt-6 leading-relaxed">
            Behavior and interpretability degrade together, not independently. 16-bit is a clean negative
            control validating the pipeline; 4-bit destroys the model outright rather than revealing a
            subtle interpretability-specific danger.
          </p>
        </ExperimentSection>

        {/* Experiment 2 */}
        <ExperimentSection eyebrow="EXPERIMENT 2 · SAFETY MONITOR" title="A live, low-overhead safety monitor">
          <p className="text-neutral-400 mb-6 leading-relaxed">
            Searched all 8,192 SAE features against BeaverTails using a strict discovery/validation/test
            split. A single feature was weak and confounded; a 100-feature regularized probe was real,
            validated on a genuinely untouched holdout, and converted into a live per-token monitor.
          </p>

          <div className="grid grid-cols-3 gap-4 mb-8">
            <MiniStat value="0.653 → 0.796" label="AUROC, single → 100-feature" />
            <MiniStat value="−23.6pp" label="False-positive rate reduction" />
            <MiniStat value="+0.66%" label="Optimized monitor overhead" />
          </div>

          <ResultTable
            headers={["Model", "AUROC", "AUPRC", "Recall", "FPR"]}
            rows={[
              ["Single feature", "0.653", "0.652", "92.0%", "84.4%"],
              ["100-feature probe", "0.796", "0.801", "91.2%", "60.8%"],
            ]}
          />

          <p className="text-neutral-500 text-sm mt-6 leading-relaxed">
            Computing only the 100 needed encoder features (skipping the full decoder and 8,092 unused
            features) cut monitoring overhead from a measurable +4.12% to +0.66% — not statistically
            distinguishable from zero.
          </p>
        </ExperimentSection>

        {/* Experiment 3 */}
        <ExperimentSection eyebrow="EXPERIMENT 3 · FEATURE STEERING" title="Golden Gate Claude-inspired steering">
          <p className="text-neutral-400 mb-6 leading-relaxed">
            Additive decoder-direction steering reliably raised targeted SAE activations — confirming
            causal control of the encoded coordinate. Tested across 8 candidates including clean lexical
            controls and a matched random-direction control, under both last-position and persistent
            intervention policies.
          </p>
          <p className="text-neutral-500 text-sm leading-relaxed">
            Target and random directions produced comparable output degeneration at matched strength.
            Reported as an honest, controlled negative result — not a hidden limitation.
          </p>
        </ExperimentSection>

        {/* Architecture */}
        <div className="rounded-2xl border border-neutral-800 bg-[#100c1a] px-10 py-10 mt-6">
          <h2 className="text-xs font-mono text-neutral-500 uppercase tracking-wider mb-6">Architecture &amp; stack</h2>
          <pre className="text-xs text-neutral-400 overflow-x-auto font-mono leading-relaxed mb-6">
{`Base LLM (GPT-2-small, frozen)
         │
   Hook/Capture layer ──────→ Quantized variants (16/8/4-bit)
         │                               │
    SAE (trained once) ←────────────────┘
         │
  ┌──────┼──────────┬──────────────┐
Feature  Safety     Steering    Benchmark
Browser  Monitor    (Phase 7)   Layer`}
          </pre>
          <div className="flex flex-wrap gap-2">
            {["Python", "PyTorch", "TransformerLens", "scikit-learn", "FastAPI", "Next.js 15", "TypeScript", "Tailwind"].map((t) => (
              <span key={t} className="text-xs px-2.5 py-1 rounded-md border border-neutral-700 text-neutral-400 font-mono">{t}</span>
            ))}
          </div>
        </div>

        {/* Closing CTA */}
        <div className="rounded-2xl border border-neutral-800 bg-[#100c1a] px-10 py-12 my-6 text-center">
          <h2 className="text-2xl font-semibold text-white mb-3">Let&apos;s talk interpretability &amp; inference</h2>
          <p className="text-neutral-400 mb-6 max-w-xl mx-auto">
            ActivationLens is open source, built by Anagha — MS Computer Science, UT Dallas.
            Open to ML and SDE new-grad roles.
          </p>
          <div className="flex items-center justify-center gap-3">
            <a href="https://github.com/ana-lan/activation-lens" className="px-4 py-2.5 rounded-lg border border-neutral-700 text-neutral-300 text-sm font-medium hover:border-neutral-600 transition-colors">
              View on GitHub ↗
            </a>
            <a href="https://www.linkedin.com/in/anagha-langhe/" className="px-4 py-2.5 rounded-lg bg-violet-950 border border-violet-800 text-violet-300 text-sm font-medium hover:bg-violet-900/50 transition-colors">
              LinkedIn
            </a>
          </div>
          <p className="text-neutral-600 text-xs mt-8">
            Open source · MIT · Numbers are measured. See <code className="text-neutral-500">results/*.json</code>.
          </p>
        </div>
      </div>
    </main>
  );
}

function StatCard({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-[#0c0916] px-5 py-5">
      <div className="text-3xl font-bold text-violet-400 mb-2">{value}</div>
      <div className="text-sm text-white font-medium mb-1">{label}</div>
      <div className="text-xs text-neutral-500 leading-snug">{sub}</div>
    </div>
  );
}

function MiniStat({ value, label, color }: { value: string; label: string; color?: string }) {
  return (
    <div className="rounded-xl border border-neutral-800 bg-[#0c0916] px-5 py-4">
      <div className={`text-xl font-bold mb-1 ${color ?? "text-violet-400"}`}>{value}</div>
      <div className="text-xs text-neutral-500 leading-snug">{label}</div>
    </div>
  );
}

function ExperimentSection({ eyebrow, title, children }: { eyebrow: string; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-neutral-800 bg-[#100c1a] px-10 py-10 mt-6">
      <div className="text-xs font-mono text-indigo-400 tracking-wider mb-3">{eyebrow}</div>
      <h2 className="text-2xl font-semibold text-white mb-6">{title}</h2>
      {children}
    </div>
  );
}

function ResultTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="rounded-xl border border-neutral-800 overflow-hidden">
      <table className="w-full text-sm font-mono">
        <thead>
          <tr className="bg-[#0c0916]">
            {headers.map((h) => (
              <th key={h} className="text-left px-4 py-2.5 text-neutral-500 font-normal text-xs border-b border-neutral-800">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j} className={`px-4 py-2.5 border-b border-neutral-900 last:border-0 ${j === 0 ? "text-neutral-200" : "text-neutral-400"}`}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}