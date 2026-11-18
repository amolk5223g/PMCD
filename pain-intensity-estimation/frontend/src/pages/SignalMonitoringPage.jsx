import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { Activity, HeartPulse, Thermometer, Wind, Zap, Hand, Play, Pause, AlertTriangle } from 'lucide-react';
import SensorStatusBar from '../components/SensorStatusBar';

export default function SignalMonitoringPage({
  recordingData,
  selectedRecording,
  currentTime,
  setCurrentTime,
  isPlaying,
  setIsPlaying,
  disabledModalities,
  onToggleModality,
  latestPrediction,
  loadingRecording,
}) {
  const windowStart = Math.max(0, currentTime - 6.0);

  // Show a 35-second sliding viewport centered near currentTime for crisp waveform readability
  const visibleSignals = useMemo(() => {
    if (!recordingData?.signals) return [];
    const viewStart = Math.max(0, currentTime - 25);
    const viewEnd = Math.min(recordingData.duration_sec, viewStart + 35);
    return recordingData.signals.filter((s) => s.t >= viewStart && s.t <= viewEnd);
  }, [recordingData, currentTime]);

  const painEventsInView = useMemo(() => {
    if (!recordingData?.observed_pain_events || visibleSignals.length === 0) return [];
    const t0 = visibleSignals[0].t;
    const t1 = visibleSignals[visibleSignals.length - 1].t;
    return recordingData.observed_pain_events.filter((e) => e.timestamp_sec >= t0 && e.timestamp_sec <= t1);
  }, [recordingData, visibleSignals]);

  const phys = latestPrediction?.key_physiological_metrics || {};

  const renderWaveform = (title, modKey, icon, lines, metricText, unit) => {
    const isDropped = disabledModalities.includes(modKey);
    const Icon = icon;
    return (
      <div className={`card p-4 ${isDropped ? 'opacity-65 border-amber-400/60' : ''}`}>
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <Icon className="w-4 h-4 text-sky-500" />
            <h3 className="text-xs font-bold uppercase tracking-wide">{title}</h3>
            {isDropped && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/15 text-amber-600 dark:text-amber-400 text-[10px] font-semibold">
                <AlertTriangle className="w-3 h-3" /> DROPPED — MASKED OUT (NO FABRICATION)
              </span>
            )}
          </div>
          <div className="text-xs font-mono text-slate-600 dark:text-slate-300">
            {isDropped ? 'Sensor Offline' : `${metricText} ${unit}`}
          </div>
        </div>

        <div className="h-36 w-full">
          {isDropped ? (
            <div className="h-full rounded-lg border border-dashed border-amber-400/50 bg-amber-50/40 dark:bg-amber-950/20 flex items-center justify-center text-xs text-amber-700 dark:text-amber-300">
              {modKey} modality simulated offline — zero synthetic waveform substituted.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={visibleSignals} margin={{ top: 5, right: 12, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis dataKey="t" tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}s`} />
                <YAxis tick={{ fontSize: 10 }} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ fontSize: '11px', borderRadius: '8px' }}
                  labelFormatter={(l) => `Time: ${l}s`}
                />
                <ReferenceArea x1={windowStart} x2={currentTime} fill="#0ea5e9" fillOpacity={0.12} />
                <ReferenceLine x={currentTime} stroke="#0ea5e9" strokeWidth={2} />
                {painEventsInView.map((ev) => (
                  <ReferenceLine
                    key={ev.sample_idx}
                    x={ev.timestamp_sec}
                    stroke="#ef4444"
                    strokeDasharray="3 3"
                    strokeOpacity={0.65}
                  />
                ))}
                {lines.map((ln) => (
                  <Line
                    key={ln.key}
                    type="monotone"
                    dataKey={ln.key}
                    name={ln.name}
                    stroke={ln.color}
                    dot={false}
                    strokeWidth={1.6}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Playback & Time Scrubber */}
      <div className="card p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying((p) => !p)}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold transition"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? 'Pause Playback' : 'Play Recording'}
          </button>
          <div>
            <div className="text-xs font-bold">
              Recording: <span className="font-mono text-sky-600 dark:text-sky-400">{selectedRecording}</span>
            </div>
            <div className="text-[11px] text-slate-500">
              Causal 6s Inference Window: [{windowStart.toFixed(1)}s – {currentTime.toFixed(1)}s] /{' '}
              {recordingData?.duration_sec || '--'}s · Red dashed lines = observed pain ratings
            </div>
          </div>
        </div>

        <div className="flex-1 max-w-md flex items-center gap-3">
          <input
            type="range"
            min={6}
            max={recordingData?.duration_sec || 100}
            step={0.5}
            value={currentTime}
            onChange={(e) => setCurrentTime(parseFloat(e.target.value))}
            className="w-full accent-sky-600"
          />
          <span className="font-mono text-xs font-bold w-14 text-right">{currentTime.toFixed(1)}s</span>
        </div>
      </div>

      <SensorStatusBar
        disabledModalities={disabledModalities}
        onToggleModality={onToggleModality}
        interactive={true}
      />

      {loadingRecording ? (
        <div className="card p-10 text-center text-sm text-slate-500">Loading 250 Hz PMPDB recording {selectedRecording}...</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {renderWaveform(
            'BVP / PPG Waveform (Empatica E4 — No ECG in PMPDB)',
            'BVP',
            HeartPulse,
            [{ key: 'Bvp', name: 'Bvp (E4)', color: '#ef4444' }],
            phys.pulse_rate_bpm != null ? `PR: ${phys.pulse_rate_bpm} BPM · RMSSD: ${phys.prv_rmssd_ms} ms` : '--',
            ''
          )}
          {renderWaveform(
            'Electrodermal Activity (EDA Tonic SCL + Phasic SCR)',
            'EDA',
            Activity,
            [
              { key: 'Eda_RB', name: 'Eda_RB Clean', color: '#0284c7' },
              { key: 'Eda_RB_Tonic', name: 'Tonic (SCL)', color: '#64748b' },
            ],
            phys.eda_scl_rb != null ? `SCL: ${phys.eda_scl_rb} µS · SCR Peaks: ${phys.eda_scr_peaks}` : '--',
            ''
          )}
          {renderWaveform(
            'Respiration (RespiBAN Chest Belt, 0.05–1 Hz)',
            'Resp',
            Wind,
            [{ key: 'Resp', name: 'Respiration', color: '#10b981' }],
            phys.resp_rate_bpm != null ? `Rate: ${phys.resp_rate_bpm}` : '--',
            'br/min'
          )}
          {renderWaveform(
            'Skin Temperature (Empatica E4)',
            'Tmp',
            Thermometer,
            [{ key: 'Tmp', name: 'Skin Temp (°C)', color: '#8b5cf6' }],
            phys.skin_temp_c != null ? `${phys.skin_temp_c}` : '--',
            '°C'
          )}
          {renderWaveform(
            'Surface EMG (RespiBAN, 50 Hz Notch + 20–120 Hz)',
            'EMG',
            Zap,
            [{ key: 'Emg', name: 'EMG (mV)', color: '#eab308' }],
            phys.emg_rms_ratio != null ? `RMS / RunUp: ${phys.emg_rms_ratio}x` : '--',
            ''
          )}
          {renderWaveform(
            'Grip Force (RespiBAN Hand Dynamometer)',
            'Grip',
            Hand,
            [{ key: 'Grip', name: 'Grip Force', color: '#6366f1' }],
            phys.grip_rel_z != null ? `RunUp Δz: ${phys.grip_rel_z}` : '--',
            ''
          )}
        </div>
      )}
    </div>
  );
}

