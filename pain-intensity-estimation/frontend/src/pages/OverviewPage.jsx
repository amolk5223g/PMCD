import React from 'react';
import {
  Users,
  FileSpreadsheet,
  Activity,
  Cpu,
  Gauge,
  AlertCircle,
  CheckCircle2,
  PlayCircle,
  ShieldAlert,
} from 'lucide-react';
import PainGauge from '../components/PainGauge';
import SensorStatusBar from '../components/SensorStatusBar';

export default function OverviewPage({
  datasetSummary,
  modelStatus,
  latestPrediction,
  evalSummary,
  disabledModalities,
  onToggleModality,
  onNavigate,
}) {
  const bestRow = evalSummary?.model_comparison_table?.[0];

  const kpis = [
    {
      label: 'Participants in PMPDB',
      value: datasetSummary ? datasetSummary.num_participants : '--',
      sub: datasetSummary
        ? `${datasetSummary.participants_with_2_sessions} with 2 sessions · ${datasetSummary.participants_with_1_session} with 1`
        : 'Loading...',
      icon: Users,
      accent: 'text-sky-600 dark:text-sky-400 bg-sky-500/10',
    },
    {
      label: 'Total Recordings',
      value: datasetSummary ? datasetSummary.num_main_recordings : '--',
      sub: datasetSummary
        ? `+ ${datasetSummary.num_runup_recordings} run-up baselines (${datasetSummary.duration_stats_sec?.total_hours} hrs)`
        : 'Loading...',
      icon: FileSpreadsheet,
      accent: 'text-indigo-600 dark:text-indigo-400 bg-indigo-500/10',
    },
    {
      label: 'Active Sensor Modalities',
      value: `${6 - disabledModalities.length} / 6`,
      sub: 'BVP, EDA, Resp, Tmp, EMG, Grip (No ECG/EEG)',
      icon: Activity,
      accent: 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10',
    },
    {
      label: 'Trained Checkpoints',
      value: modelStatus?.models_ready ? 'READY' : 'LOADING',
      sub: modelStatus
        ? `${modelStatus.checkpoint_files?.length || 0} saved checkpoints loaded`
        : 'Checking backend...',
      icon: Cpu,
      accent: modelStatus?.models_ready
        ? 'text-teal-600 dark:text-teal-400 bg-teal-500/10'
        : 'text-amber-600 dark:text-amber-400 bg-amber-500/10',
    },
  ];

  return (
    <div className="space-y-5">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {kpis.map((k) => {
          const Icon = k.icon;
          return (
            <div key={k.label} className="card p-4 flex items-start justify-between">
              <div>
                <div className="text-xs font-medium text-slate-500 dark:text-slate-400">{k.label}</div>
                <div className="text-2xl font-bold mt-1">{k.value}</div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">{k.sub}</div>
              </div>
              <div className={`p-2.5 rounded-xl ${k.accent}`}>
                <Icon className="w-5 h-5" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Sensor Status Interactive Bar */}
      <SensorStatusBar
        disabledModalities={disabledModalities}
        onToggleModality={onToggleModality}
        interactive={true}
      />

      {/* Main Overview Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Latest Pain Prediction Card */}
        <div className="card p-5 lg:col-span-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-bold flex items-center gap-2">
                <Gauge className="w-4 h-4 text-sky-500" />
                Latest Continuous Pain Estimate
              </h2>
              {latestPrediction && (
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800">
                  {latestPrediction.recording_id} @ {latestPrediction.window_end_sec}s
                </span>
              )}
            </div>
            <PainGauge
              score={latestPrediction?.predicted_pain_0_100}
              ci={latestPrediction?.confidence_interval_95}
              uncertaintyStd={latestPrediction?.uncertainty_std}
              uncertaintyLevel={latestPrediction?.uncertainty_level}
              groundTruth={latestPrediction?.nearest_ground_truth}
            />
          </div>

          <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs">
            <span className="text-slate-500 dark:text-slate-400">
              Model: <strong className="text-slate-700 dark:text-slate-200">{latestPrediction?.model_used || 'Not run yet'}</strong>
            </span>
            <button
              onClick={() => onNavigate('prediction')}
              className="inline-flex items-center gap-1 font-semibold text-sky-600 dark:text-sky-400 hover:underline"
            >
              <PlayCircle className="w-3.5 h-3.5" /> Open Live Playback
            </button>
          </div>
        </div>

        {/* System Status & Best Validation Benchmark */}
        <div className="card p-5 lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold">System Integrity & Validation Snapshot</h2>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 font-semibold">
              Participant-Independent CV
            </span>
          </div>

          {bestRow ? (
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800/70 p-3">
                <div className="text-[11px] text-slate-500">Top Model</div>
                <div className="font-bold text-xs mt-1 truncate" title={bestRow.model}>{bestRow.model}</div>
              </div>
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800/70 p-3">
                <div className="text-[11px] text-slate-500">MAE (0–100)</div>
                <div className="font-bold text-lg mt-0.5 text-sky-600 dark:text-sky-400">{bestRow.mae.toFixed(2)}</div>
              </div>
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800/70 p-3">
                <div className="text-[11px] text-slate-500">RMSE</div>
                <div className="font-bold text-lg mt-0.5">{bestRow.rmse.toFixed(2)}</div>
              </div>
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800/70 p-3">
                <div className="text-[11px] text-slate-500">Pearson r</div>
                <div className="font-bold text-lg mt-0.5 text-emerald-600 dark:text-emerald-400">{bestRow.pearson_r.toFixed(3)}</div>
              </div>
              <div className="rounded-lg bg-slate-50 dark:bg-slate-800/70 p-3">
                <div className="text-[11px] text-slate-500">Lin's CCC</div>
                <div className="font-bold text-lg mt-0.5 text-indigo-600 dark:text-indigo-400">{bestRow.ccc.toFixed(3)}</div>
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-500">Evaluation metrics loading...</div>
          )}

          {/* Checkpoints Loaded */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            {modelStatus?.loaded_checkpoints &&
              Object.entries(modelStatus.loaded_checkpoints).map(([k, ok]) => (
                <div
                  key={k}
                  className="flex items-center justify-between px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-800"
                >
                  <span className="font-mono text-[11px]">{k}</span>
                  {ok ? (
                    <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Loaded
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-amber-500 font-semibold">
                      <AlertCircle className="w-3.5 h-3.5" /> Pending
                    </span>
                  )}
                </div>
              ))}
          </div>

          {/* Ground Truth Transparency Box */}
          <div className="rounded-lg border border-amber-300/60 dark:border-amber-700/50 bg-amber-50/70 dark:bg-amber-950/30 p-3 text-xs text-amber-900 dark:text-amber-200 flex gap-2.5">
            <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5 text-amber-600 dark:text-amber-400" />
            <div>
              <strong>Dataset & Research Disclosure:</strong> All signals and predictions come strictly from your local{' '}
              <strong>PMPDB</strong> dataset (49 participants, 90 sessions, 250 Hz synchronized BVP, EDA, Resp, Tmp, EMG, Grip) and
              actual trained model checkpoints. <strong>No ECG or EEG</strong> exists in PMPDB, and pain ratings are event-stamped
              1–10 NRS self-reports mapped to 0–100 (not 250 Hz CoVAS). Playback mode uses prerecorded sessions — not live clinical hardware.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

