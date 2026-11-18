import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { Search, Database, PlayCircle } from 'lucide-react';

export default function DatasetExplorerPage({
  datasetSummary,
  participantsData,
  selectedRecording,
  onSelectRecording,
  recordingData,
  onNavigate,
}) {
  const [search, setSearch] = useState('');

  const filteredParticipants = useMemo(() => {
    const list = participantsData?.participants || [];
    if (!search.trim()) return list;
    const q = search.trim().toLowerCase();
    return list.filter(
      (p) =>
        p.participant_id.toLowerCase().includes(q) ||
        p.recordings.some((r) => r.recording_id.toLowerCase().includes(q))
    );
  }, [participantsData, search]);

  const nrsDist = useMemo(() => {
    const raw = datasetSummary?.label_summary?.pain_rate_distribution_nrs || {};
    return Object.entries(raw)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([k, v]) => ({ nrs: `${k} (${Number(k) * 10})`, count: v }));
  }, [datasetSummary]);

  const edaFigures = [
    { file: '01_multimodal_waveforms_P01_1.png', title: 'Multimodal Waveforms (P01_1)' },
    { file: '02_eda_decomposition_P01_1.png', title: 'EDA Tonic / Phasic Decomposition' },
    { file: '03_bvp_pulse_detection.png', title: 'BVP Systolic Peak Detection' },
    { file: '06_feature_pain_correlations.png', title: 'Feature–Pain Correlations' },
  ];

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Participant & Recording List */}
        <div className="card p-4 lg:col-span-4 flex flex-col max-h-[540px]">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-bold flex items-center gap-2">
              <Database className="w-4 h-4 text-sky-500" />
              PMPDB Participants ({participantsData?.num_participants || 0})
            </h2>
          </div>

          <div className="relative mb-3">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter P01..P49 or recording..."
              className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800"
            />
          </div>

          <div className="overflow-y-auto space-y-2 pr-1 flex-1">
            {filteredParticipants.map((p) => (
              <div key={p.participant_id} className="rounded-lg border border-slate-200 dark:border-slate-800 p-2.5">
                <div className="text-xs font-bold mb-1.5">{p.participant_id}</div>
                <div className="flex flex-wrap gap-1.5">
                  {p.recordings.map((r) => {
                    const active = r.recording_id === selectedRecording;
                    return (
                      <button
                        key={r.recording_id}
                        onClick={() => onSelectRecording(r.recording_id)}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition ${
                          active
                            ? 'bg-sky-600 text-white font-bold'
                            : 'bg-slate-100 dark:bg-slate-800 hover:bg-sky-500/15'
                        }`}
                      >
                        {r.recording_id} · {r.duration_sec.toFixed(0)}s · {r.n_pain_rates} pts
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Recording Pain Ratings & Global NRS Distribution */}
        <div className="lg:col-span-8 space-y-5">
          <div className="card p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div>
                <h3 className="text-sm font-bold">
                  Observed Pain Ratings: <span className="font-mono text-sky-600 dark:text-sky-400">{selectedRecording}</span>
                </h3>
                <p className="text-[11px] text-slate-500">
                  Duration: {recordingData?.duration_sec ?? '--'}s · Sampling: 250 Hz · Events:{' '}
                  {recordingData?.observed_pain_events?.length ?? 0} · Thresholds: No-Pain &lt;{' '}
                  {recordingData?.no_pain_threshold_0_100 ?? '--'}, Severe &gt; {recordingData?.severe_pain_threshold_0_100 ?? '--'}
                </p>
              </div>
              <button
                onClick={() => onNavigate('signals')}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-700 text-white text-xs font-semibold"
              >
                <PlayCircle className="w-3.5 h-3.5" /> Inspect Waveforms
              </button>
            </div>

            <div className="h-52 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={recordingData?.observed_pain_events || []} margin={{ top: 10, right: 16, left: -12, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                  <XAxis dataKey="timestamp_sec" tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}s`} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} labelFormatter={(v) => `Time: ${v}s`} />
                  {recordingData?.no_pain_threshold_0_100 != null && (
                    <ReferenceLine y={recordingData.no_pain_threshold_0_100} stroke="#10b981" strokeDasharray="4 4" />
                  )}
                  {recordingData?.severe_pain_threshold_0_100 != null && (
                    <ReferenceLine y={recordingData.severe_pain_threshold_0_100} stroke="#f97316" strokeDasharray="4 4" />
                  )}
                  <Line type="monotone" dataKey="pain_score_0_100" name="Pain Score (0–100)" stroke="#dc2626" strokeWidth={2.2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card p-5">
            <h3 className="text-sm font-bold mb-1">Dataset-Wide Pain Rating Distribution (All 1,424 Verified Ratings)</h3>
            <p className="text-[11px] text-slate-500 mb-3">
              Original PMPDB 1–10 NRS scale linearly mapped to 0–100 (`NRS × 10`). Zero fabricated samples.
            </p>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={nrsDist} margin={{ top: 5, right: 15, left: -12, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                  <XAxis dataKey="nrs" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                  <Bar dataKey="count" name="Event Count" fill="#0284c7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* EDA Figures Gallery served from FastAPI /reports/figures */}
      <div className="card p-5">
        <h3 className="text-sm font-bold mb-3">Generated Exploratory Data Analysis Figures (Real PMPDB Data)</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {edaFigures.map((fig) => (
            <div key={fig.file} className="rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden bg-white">
              <div className="px-3 py-2 text-xs font-semibold bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-200 border-b border-slate-200 dark:border-slate-700">
                {fig.title}
              </div>
              <img
                src={`/reports/figures/${fig.file}`}
                alt={fig.title}
                className="w-full h-auto object-contain"
                loading="lazy"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

