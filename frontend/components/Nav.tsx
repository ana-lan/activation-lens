'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Nav() {
  const pathname = usePathname();

  const linkClass = (active: boolean) =>
    `px-2.5 py-1.5 rounded-md text-xs sm:text-sm whitespace-nowrap transition-colors ${
      active ? 'bg-violet-950 text-violet-300 font-medium' : 'text-neutral-400 hover:text-white'
    }`;

  return (
    <nav className="max-w-5xl mx-auto px-4 sm:px-6 pt-6 sm:pt-8 pb-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <div>
        <div className="text-lg font-bold text-white">ActivationLens</div>
        <div className="text-xs text-neutral-500">Interpretability × inference research</div>
      </div>

      <div className="flex items-center gap-2 flex-wrap w-full sm:w-auto">
        <div className="flex items-center gap-1 border border-neutral-800 rounded-lg p-1 bg-[#100c1a]">
          <Link href="/#results" className={linkClass(pathname === '/')}>Results</Link>
          <Link href="/docs" className={linkClass(pathname === '/docs')}>Docs</Link>
          <Link href="/demo" className={linkClass(pathname === '/demo')}>Live Demo</Link>
        </div>
        <a href="https://github.com/ana-lan/activation-lens" className="px-2.5 py-1.5 rounded-lg border border-neutral-800 text-neutral-300 text-xs sm:text-sm hover:border-neutral-700 transition-colors whitespace-nowrap">
          GitHub
        </a>
      </div>
    </nav>
  );
}