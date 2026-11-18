import React from 'react';

export function getSeverityStyle(score) {
  if (score == null) {
    return { color: '#64748b', bg: 'bg-slate-100 dark:bg-slate-800', text: 'text-slate-600 dark:text-slate-300', label: 'Awaiting Data' };
  }
  if (score < 30) {
    return { color: '#10b981', bg: 'bg-emerald-500/15', text: 'text-emerald-600 dark:text-emerald-400', label: 'No / Mild Pain' };
  }
  if (score < 60) {
    return { color: '#f59e0b', bg: 'bg-amber-500/15', text: 'text-amber-600 dark:text-amber-400', label: 'Moderate Pain' };
  }
  if (score < 80) {
    return { color: '#f97316', bg: 'bg-orange-500/15', text: 'text-orange-600 dark:text-orange-400', label: 'High Pain' };
  }
  return { color: '#ef4444', bg: 'bg-red-500/15', text: 'text-red-600 dark:text-red-400', label: 'Severe Pain' };
}

export default function PainGauge({ score, ci, uncertaintyStd, uncertaintyLevel, groundTruth }) {
  const safeScore = score != null ? Math.max(0, Math.min(100, Number(score))) : 0;
  const sev = getSeverityStyle(score);
  const radius = 84;
  const circumference = Math.PI * radius; // semicircle
  const dashOffset = circumference * (1 - safeScore / 100);

  return (
    <div className="flex flex-col items-center justify-center py-2">
      <div className="relative w-56 h-36 flex items-end justify-center">
        <svg viewBox="0 0 210 125" className="w-full h-full overflow-visible">
          {/* Background track */}
          <path
            d="M 21 108 A 84 84 0 0 1 189 108"
            fill="none"
            stroke="currentColor"
            className="text-slate-200 dark:text-slate-800"
            strokeWidth="16"
            strokeLinecap="round"
          />
          {/* Value arc */}
          <path
            d="M 21 108 A 84 84 0 0 1 189 108"
            fill="none"
            stroke={sev.color}
            strokeWidth="16"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{ transition: 'stroke-dashoffset 0.45s ease, stroke 0.35s ease' }}
          />
          <text x="16" y="124" className="fill-slate-400 text-[10px]" fontWeight="600">0</text>
          <text x="98" y="18" className="fill-slate-400 text-[10px]" fontWeight="600">50</text>
          <text x="176" y="124" className="fill-slate-400 text-[10px]" fontWeight="600">100</text>
        </svg>
        <div className="absolute bottom-2 flex flex-col items-center">
          <div className="text-4xl font-bold tracking-tight">
            {score != null ? Math.round(score) : '--'}
            <span className="text-lg font-medium text-slate-400">/100</span>
          </div>
          <span className={`mt-1 px-2.5 py-0.5 rounded-full text-xs font-semibold ${sev.bg} ${sev.text}`}>
            {sev.label}
          </span>
        </div>
      </div>

      <div className="mt-3 w-full grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-lg bg-slate-50 dark:bg-slate-800/70 p-2.5 text-center">
          <div className="text-slate-500 dark:text-slate-400">95% Prediction Interval</div>
          <div className="font-semibold mt-0.5 text-sm">
            {ci ? `[${ci[0].toFixed(1)}, ${ci[1].toFixed(1)}]` : '--'}
          </div>
        </div>
        <div className="rounded-lg bg-slate-50 dark:bg-slate-800/70 p-2.5 text-center">
          <div className="text-slate-500 dark:text-slate-400">Uncertainty (σ)</div>
          <div className="font-semibold mt-0.5 text-sm">
            {uncertaintyStd != null ? `±${uncertaintyStd.toFixed(1)} (${uncertaintyLevel})` : '--'}
          </div>
        </div>
      </div>

      {groundTruth && (
        <div className="mt-2 w-full rounded-lg border border-sky-200 dark:border-sky-800 bg-sky-50/70 dark:bg-sky-950/40 px-3 py-1.5 text-xs flex items-center justify-between">
          <span className="text-sky-700 dark:text-sky-300 font-medium">
            Observed Self-Report (@ {groundTruth.timestamp_sec}s)
          </span>
          <span className="font-bold text-sky-900 dark:text-sky-100">
            {groundTruth.pain_score_0_100}/100 (NRS {groundTruth.pain_rate_nrs}/10)
          </span>
        </div>
      )}
    </div>
  );
}

