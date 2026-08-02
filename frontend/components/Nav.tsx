'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Nav() {
  const pathname = usePathname();

  const linkClass = (active: boolean) =>
    `px-3 py-1.5 rounded-md text-sm transition-colors ${
      active ? 'bg-violet-950 text-violet-300 font-medium' : 'text-neutral-400 hover:text-white'
    }`;

  return (
    <nav className="max-w-5xl mx-auto px-6 pt-8 pb-2 flex items-center justify-between">
      <div>
        <div className="text-lg font-bold text-white">ActivationLens</div>
        <div className="text-xs text-neutral-500">Interpretability × inference research</div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1 border border-neutral-800 rounded-lg p-1 bg-[#100c1a]">
          <Link href="/#results" className={linkClass(pathname === '/')}>Results</Link>
          <Link href="/docs" className={linkClass(pathname === '/docs')}>Docs</Link>
          <Link href="/demo" className={linkClass(pathname === '/demo')}>Live Demo</Link>
        </div>
        <a href="https://github.com/ana-lan/activation-lens" className="px-3 py-1.5 rounded-lg border border-neutral-800 text-neutral-300 text-sm hover:border-neutral-700 transition-colors">
          GitHub
        </a>
      </div>
    </nav>
  );
}