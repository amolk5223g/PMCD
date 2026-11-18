import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Legend,
} from 'recharts';
import { Play, Pause, RefreshCw, AlertTriangle, Trash2, Clock } from 'lucide-react';
import PainGauge from '../components/PainGauge';
import SensorStatusBar from '../components/SensorStatusBar';

export default function PainPredictionPage({
  selectedRecording,
  currentTime,
  setCurrentTime,
  isPlaying,
  setIsPlaying,
  modelChoice,
  setModelChoice,
  disabledModalities,
  onToggleModality,
  latestPrediction,
  timelineData,
  loadingTimeline,
  predictionHistory,
  onClearHistory,
  onRunWindowNow,
}) {
  const chartData = (timelineData?.timeline || []).map((pt) => ({
    ...pt,
    ciBand: [pt.ci_low, pt.ci_high],
  }));

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="card p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => setIsPlaying((p) => !p)}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? 'Pause' : 'Play Playback'}
          </button>

          <select
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value)}
            className="px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-semibold"
          >
            <option value="hybrid">Hybrid Ensemble (MM_Gated + XGBoost)</option>
            <option value="mm_gated">MM_Gated (Deep Gated Multimodal + MC Dropout)</option>
            <option value="xgboost">XGBoost Regressor (Biomedical Features)</option>
            <option value="random_forest">Random Forest Regressor</option>
          </select>

          <button
            onClick={onRunWindowNow}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-semibold"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Re-Evaluate Window
          </button>
        </div>

        <div className="flex items-center gap-3 min-w-[240px]">
          <input
            type="range"
            min={6}
            max={timelineData?.duration_sec || 120}
            step={1}
            value={currentTime}
            onChange={(e) => setCurrentTime(parseFloat(e.target.value))}
            className="flex-1 accent-sky-600"
          />
          <span className="font-mono text-xs font-bold">{currentTime.toFixed(1)}s</span>
        </div>
      </div>

      <SensorStatusBar
        disabledModalities={disabledModalities}
        onToggleModality={onToggleModality}
        interactive={true}
      />

      {latestPrediction?.missing_modality_warning && (
        <div className="rounded-xl border border-amber-400/70 bg-amber-50 dark:bg-amber-950/40 px-4 py-3 text-xs text-amber-800 dark:text-amber-200 flex items-center gap-2.5">
          <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" />
          <span>{latestPrediction.missing_modality_warning}</span>
        </div>
      )}

      {/* Main Gauge + Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="card p-5 lg:col-span-4 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold">Continuous Pain Estimate</h2>
            <span className="text-[11px] font-mono text-slate-500">{selectedRecording}</span>
          </div>

          <PainGauge
            score={latestPrediction?.predicted_pain_0_100}
            ci={latestPrediction?.confidence_interval_95}
            uncertaintyStd={latestPrediction?.uncertainty_std}
            uncertaintyLevel={latestPrediction?.uncertainty_level}
            groundTruth={latestPrediction?.nearest_ground_truth}
          />

          {/* Component Model Breakdown */}
          {latestPrediction?.component_predictions && (
            <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-1.5 text-xs">
              <div className="font-semibold text-slate-500 dark:text-slate-400 text-[11px] uppercase">
                Checkpoint Breakdown
              </div>
              <div className="flex justify-between">
                <span>MM_Gated (Deep Waveform):</span>
                <span className="font-mono font-bold">{latestPrediction.component_predictions.mm_gated ?? '--'}/100</span>
              </div>
              <div className="flex justify-between">
                <span>XGBoost (Engineered Features):</span>
                <span className="font-mono font-bold">{latestPrediction.component_predictions.xgboost ?? '--'}/100</span>
              </div>
              <div className="flex justify-between">
                <span>Random Forest (Tree Ensemble):</span>
                <span className="font-mono font-bold">{latestPrediction.component_predictions.random_forest ?? '--'}/100</span>
              </div>
            </div>
          )}

          {/* Auxiliary Ordinal Probabilities */}
          {latestPrediction?.ordinal_class_probabilities && (
            <div className="pt-3 border-t border-slate-200 dark:border-slate-800 space-y-1.5 text-xs">
              <div className="font-semibold text-slate-500 dark:text-slate-400 text-[11px] uppercase">
                Auxiliary 3-Class Ordinal Head
              </div>
              {[
                ['No / Low Pain (0)', latestPrediction.ordinal_class_probabilities.no_low_pain, 'bg-emerald-500'],
                ['Moderate Pain (1)', latestPrediction.ordinal_class_probabilities.moderate_pain, 'bg-amber-500'],
                ['Severe Pain (2)', latestPrediction.ordinal_class_probabilities.severe_pain, 'bg-red-500'],
              ].map(([lbl, prob, barColor]) => (
                <div key={lbl}>
                  <div className="flex justify-between text-[11px] mb-0.5">
                    <span>{lbl}</span>
                    <span className="font-mono">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div className={`h-full ${barColor}`} style={{ width: `${Math.min(100, prob * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Continuous Trajectory Chart */}
        <div className="card p-5 lg:col-span-8 flex flex-col">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
            <div>
              <h2 className="text-sm font-bold">Continuous Pain-Intensity Trajectory & 95% Uncertainty Band</h2>
              <p className="text-[11px] text-slate-500">
                Sliding 6-second causal inference over {selectedRecording} overlaid with actual observed NRS self-reports (red dots)
              </p>
            </div>
            <span className="text-[11px] px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 font-medium">
              Prerecorded Dataset Playback
            </span>
          </div>

          <div className="h-80 w-full">
            {loadingTimeline ? (
              <div className="h-full flex items-center justify-center text-sm text-slate-500">
                Running continuous checkpoint inference across {selectedRecording}...
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 18, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.18} />
                  <XAxis dataKey="t" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}s`} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ fontSize: '12px', borderRadius: '8px' }}
                    labelFormatter={(l) => `Recording Time: ${l}s`}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  {timelineData?.no_pain_threshold_0_100 != null && (
                    <ReferenceLine
                      y={timelineData.no_pain_threshold_0_100}
                      stroke="#10b981"
                      strokeDasharray="4 4"
                      label={{ value: 'No-Pain Thr', fontSize: 10, fill: '#10b981', position: 'insideTopRight' }}
                    />
                  )}
                  {timelineData?.severe_pain_threshold_0_100 != null && (
                    <ReferenceLine
                      y={timelineData.severe_pain_threshold_0_100}
                      stroke="#f97316"
                      strokeDasharray="4 4"
                      label={{ value: 'Severe Thr', fontSize: 10, fill: '#f97316', position: 'insideTopRight' }}
                    />
                  )}
                  <ReferenceLine x={currentTime} stroke="#0284c7" strokeWidth={2} />
                  <Area
                    type="monotone"
                    dataKey="ciBand"
                    name="95% Prediction Interval"
                    fill="#0ea5e9"
                    fillOpacity={0.16}
                    stroke="none"
                  />
                  <Line
                    type="monotone"
                    dataKey="pred"
                    name="Predicted Pain (0–100)"
                    stroke="#0284c7"
                    strokeWidth={2.4}
                    dot={false}
                  />
                  <Scatter
                    dataKey="gt"
                    name="Observed Self-Report (PMPDB)"
                    fill="#dc2626"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Prediction History Log */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold flex items-center gap-2">
            <Clock className="w-4 h-4 text-sky-500" />
            Recent Prediction History ({predictionHistory.length})
          </h3>
          {predictionHistory.length > 0 && (
            <button
              onClick={onClearHistory}
              className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-red-500"
            >
              <Trash2 className="w-3.5 h-3.5" /> Clear
            </button>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
              <tr>
                <th className="py-2 pr-3">Timestamp (UTC)</th>
                <th className="py-2 pr-3">Recording @ Sec</th>
                <th className="py-2 pr-3">Model</th>
                <th className="py-2 pr-3">Predicted (0–100)</th>
                <th className="py-2 pr-3">95% CI</th>
                <th className="py-2 pr-3">Observed GT</th>
                <th className="py-2">Missing Modalities</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {predictionHistory.slice(0, 12).map((h) => (
                <tr key={h.prediction_id}>
                  <td className="py-2 pr-3 font-mono text-[11px]">{h.created_at?.slice(11, 19)}</td>
                  <td className="py-2 pr-3 font-mono">{h.recording_id} @ {h.timestamp_sec}s</td>
                  <td className="py-2 pr-3">{h.model_used}</td>
                  <td className="py-2 pr-3 font-bold text-sky-600 dark:text-sky-400">{h.predicted_pain_0_100}/100</td>
                  <td className="py-2 pr-3 font-mono">[{h.confidence_interval_95?.[0]}, {h.confidence_interval_95?.[1]}]</td>
                  <td className="py-2 pr-3 font-mono">{h.ground_truth_0_100 != null ? `${h.ground_truth_0_100}/100` : '—'}</td>
                  <td className="py-2">
                    {h.missing_modalities?.length ? (
                      <span className="text-amber-600 dark:text-amber-400 font-semibold">{h.missing_modalities.join(', ')}</span>
                    ) : (
                      <span className="text-emerald-600 dark:text-emerald-400">All 6 Active</span>
                    )}
                  </td>
                </tr>
              ))}
              {predictionHistory.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-6 text-center text-slate-400">
                    No logged predictions yet. Scrub the timeline or press Play to run real checkpoint inference.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

