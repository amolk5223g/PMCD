import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

const ALL_MODS = ['BVP', 'EDA', 'Resp', 'Tmp', 'EMG', 'Grip'];

export default function SensorStatusBar({ disabledModalities, onToggleModality, interactive = true }) {
  return (
    <div className="card p-3.5">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            PMPDB Sensor Modalities (6 Available · 8 Channels @ 250 Hz)
          </span>
          {disabledModalities.length > 0 && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/15 text-amber-600 dark:text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5" />
              {disabledModalities.length} Missing (Masked, Zero Fabrication)
            </span>
          )}
        </div>
        <span className="text-[11px] text-slate-500 dark:text-slate-400">
          {interactive ? 'Click a modality to simulate sensor dropout' : 'Real-time availability status'}
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        {ALL_MODS.map((mod) => {
          const isOff = disabledModalities.includes(mod);
          return (
            <button
              key={mod}
              type="button"
              disabled={!interactive}
              onClick={() => interactive && onToggleModality(mod)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                isOff
                  ? 'border-amber-400/60 bg-amber-500/10 text-amber-700 dark:text-amber-300'
                  : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/20'
              } ${!interactive ? 'cursor-default' : 'cursor-pointer'}`}
            >
              {isOff ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
              {mod}
              <span className="text-[10px] opacity-75">{isOff ? 'DROPPED' : 'ONLINE'}</span>
            </button>
          );
        })}

        {/* Explicitly disclose absent modalities */}
        {['ECG', 'EEG'].map((absent) => (
          <div
            key={absent}
            title={`${absent} is NOT recorded in PMPDB`}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border border-slate-200 dark:border-slate-800 bg-slate-100/70 dark:bg-slate-800/50 text-slate-400"
          >
            <XCircle className="w-3.5 h-3.5" />
            {absent}
            <span className="text-[10px]">NOT IN PMPDB</span>
          </div>
        ))}
      </div>
    </div>
  );
}

