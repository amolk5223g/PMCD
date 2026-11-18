import React, { useEffect, useState, useCallback } from 'react';
import {
  LayoutDashboard,
  Activity,
  Gauge,
  Sparkles,
  Database,
  BarChart3,
  Sun,
  Moon,
  Radio,
  AlertTriangle,
} from 'lucide-react';
import { api } from './services/api';
import OverviewPage from './pages/OverviewPage';
import SignalMonitoringPage from './pages/SignalMonitoringPage';
import PainPredictionPage from './pages/PainPredictionPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import DatasetExplorerPage from './pages/DatasetExplorerPage';
import ModelPerformancePage from './pages/ModelPerformancePage';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'signals', label: 'Signal Monitoring', icon: Activity },
  { id: 'prediction', label: 'Pain Prediction', icon: Gauge },
  { id: 'explainability', label: 'Explainable AI', icon: Sparkles },
  { id: 'explorer', label: 'Dataset Explorer', icon: Database },
  { id: 'performance', label: 'Model Performance', icon: BarChart3 },
];

export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [errorBanner, setErrorBanner] = useState(null);

  const [datasetSummary, setDatasetSummary] = useState(null);
  const [participantsData, setParticipantsData] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [evalSummary, setEvalSummary] = useState(null);
  const [globalXai, setGlobalXai] = useState(null);

  const [selectedRecording, setSelectedRecording] = useState('P01_1');
  const [recordingData, setRecordingData] = useState(null);
  const [loadingRecording, setLoadingRecording] = useState(false);

  const [modelChoice, setModelChoice] = useState('hybrid');
  const [disabledModalities, setDisabledModalities] = useState([]);
  const [currentTime, setCurrentTime] = useState(56.5);
  const [isPlaying, setIsPlaying] = useState(false);

  const [latestPrediction, setLatestPrediction] = useState(null);
  const [timelineData, setTimelineData] = useState(null);
  const [loadingTimeline, setLoadingTimeline] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState([]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  // Initial metadata & evaluation fetch
  useEffect(() => {
    Promise.all([
      api.getDatasetSummary(),
      api.getParticipants(),
      api.getModelStatus(),
      api.getEvaluationSummary(),
      api.getGlobalXai(),
      api.getPredictionHistory(50),
    ])
      .then(([ds, parts, ms, ev, gx, hist]) => {
        setDatasetSummary(ds);
        setParticipantsData(parts);
        setModelStatus(ms);
        setEvalSummary(ev);
        setGlobalXai(gx);
        setPredictionHistory(hist.items || []);
        setErrorBanner(null);
      })
      .catch((err) => {
        setErrorBanner(`Backend connection issue: ${err.message}. Ensure FastAPI is running on port 8000.`);
      });
  }, []);

  // Load selected real PMPDB recording waveforms
  useEffect(() => {
    let cancelled = false;
    setLoadingRecording(true);
    setIsPlaying(false);
    api
      .getRecording(selectedRecording, 8)
      .then((data) => {
        if (cancelled) return;
        setRecordingData(data);
        const firstEv = data.observed_pain_events?.[2]?.timestamp_sec || data.observed_pain_events?.[0]?.timestamp_sec || 20.0;
        setCurrentTime(Math.max(6.0, Math.round(firstEv * 2) / 2));
      })
      .catch((err) => !cancelled && setErrorBanner(err.message))
      .finally(() => !cancelled && setLoadingRecording(false));
    return () => {
      cancelled = true;
    };
  }, [selectedRecording]);

  // Compute continuous timeline when recording, model, or disabled modalities change
  useEffect(() => {
    let cancelled = false;
    setLoadingTimeline(true);
    api
      .predictTimeline({
        recording_id: selectedRecording,
        step_sec: 4.0,
        model_choice: modelChoice,
        disabled_modalities: disabledModalities,
      })
      .then((tl) => {
        if (!cancelled) setTimelineData(tl);
      })
      .catch((err) => !cancelled && setErrorBanner(err.message))
      .finally(() => !cancelled && setLoadingTimeline(false));
    return () => {
      cancelled = true;
    };
  }, [selectedRecording, modelChoice, disabledModalities]);

  // Run single-window prediction + SHAP + IG explanation
  const runWindowInference = useCallback(async () => {
    try {
      const res = await api.predictRecording({
        recording_id: selectedRecording,
        timestamp_sec: currentTime,
        model_choice: modelChoice,
        disabled_modalities: disabledModalities,
        include_explainability: true,
      });
      setLatestPrediction(res);
      const hist = await api.getPredictionHistory(50);
      setPredictionHistory(hist.items || []);
    } catch (err) {
      setErrorBanner(err.message);
    }
  }, [selectedRecording, currentTime, modelChoice, disabledModalities]);

  useEffect(() => {
    const timer = setTimeout(() => {
      runWindowInference();
    }, 220);
    return () => clearTimeout(timer);
  }, [runWindowInference]);

  // Playback loop (advances 2.0s every 1.4s through prerecorded PMPDB session)
  useEffect(() => {
    if (!isPlaying || !recordingData) return;
    const id = setInterval(() => {
      setCurrentTime((prev) => {
        const next = prev + 2.0;
        if (next >= recordingData.duration_sec) {
          setIsPlaying(false);
          return recordingData.duration_sec;
        }
        return Math.round(next * 10) / 10;
      });
    }, 1400);
    return () => clearInterval(id);
  }, [isPlaying, recordingData]);

  const handleToggleModality = (mod) => {
    setDisabledModalities((prev) => {
      if (prev.includes(mod)) return prev.filter((m) => m !== mod);
      if (prev.length >= 5) return prev; // Keep at least 1 modality active
      return [...prev, mod];
    });
  };

  const handleClearHistory = async () => {
    await api.clearPredictionHistory();
    setPredictionHistory([]);
  };

  const allRecordingIds = (participantsData?.participants || []).flatMap((p) =>
    p.recordings.map((r) => r.recording_id)
  );

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Sidebar */}
      <aside className="w-full lg:w-64 shrink-0 border-b lg:border-b-0 lg:border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2.5 pb-4 mb-4 border-b border-slate-200 dark:border-slate-800">
            <div className="w-9 h-9 rounded-xl bg-sky-600 text-white flex items-center justify-center font-bold shadow">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className="font-bold text-sm leading-tight">PainMonit AI</div>
              <div className="text-[11px] text-slate-500">Multimodal Pain Estimation</div>
            </div>
          </div>

          <nav className="space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-xs font-semibold transition ${
                    active
                      ? 'bg-sky-600 text-white shadow-sm'
                      : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-800 space-y-3 text-[11px] text-slate-500">
          <div>
            <label className="block font-semibold mb-1 text-slate-600 dark:text-slate-300">Active PMPDB Session</label>
            <select
              value={selectedRecording}
              onChange={(e) => setSelectedRecording(e.target.value)}
              className="w-full px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 font-mono text-xs text-slate-900 dark:text-slate-100"
            >
              {(allRecordingIds.length ? allRecordingIds : ['P01_1']).map((rid) => (
                <option key={rid} value={rid}>
                  {rid}
                </option>
              ))}
            </select>
          </div>

          <div className="rounded-lg bg-slate-100 dark:bg-slate-800/70 p-2.5 leading-relaxed">
            <strong className="text-slate-700 dark:text-slate-200">Research Prototype Only:</strong> Not intended for clinical diagnosis, triage, or medical treatment decisions.
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header Bar */}
        <header className="sticky top-0 z-20 border-b border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 backdrop-blur px-5 py-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-base font-bold">
              {NAV_ITEMS.find((n) => n.id === activeTab)?.label}
            </h1>
            <p className="text-[11px] text-slate-500">
              Explainable Multimodal Continuous Pain-Intensity Estimation · Local PMPDB Dataset
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20">
              <Radio className="w-3.5 h-3.5" />
              Prerecorded Dataset Playback (Not Live Hardware)
            </span>

            <button
              onClick={() => setDarkMode((d) => !d)}
              className="p-2 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
              title="Toggle Dark / Light Theme"
            >
              {darkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-700" />}
            </button>
          </div>
        </header>

        {errorBanner && (
          <div className="mx-5 mt-4 rounded-xl border border-red-400/60 bg-red-50 dark:bg-red-950/40 px-4 py-2.5 text-xs text-red-700 dark:text-red-300 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorBanner}</span>
          </div>
        )}

        <main className="p-5 flex-1">
          {activeTab === 'overview' && (
            <OverviewPage
              datasetSummary={datasetSummary}
              modelStatus={modelStatus}
              latestPrediction={latestPrediction}
              evalSummary={evalSummary}
              disabledModalities={disabledModalities}
              onToggleModality={handleToggleModality}
              onNavigate={setActiveTab}
            />
          )}
          {activeTab === 'signals' && (
            <SignalMonitoringPage
              recordingData={recordingData}
              selectedRecording={selectedRecording}
              currentTime={currentTime}
              setCurrentTime={setCurrentTime}
              isPlaying={isPlaying}
              setIsPlaying={setIsPlaying}
              disabledModalities={disabledModalities}
              onToggleModality={handleToggleModality}
              latestPrediction={latestPrediction}
              loadingRecording={loadingRecording}
            />
          )}
          {activeTab === 'prediction' && (
            <PainPredictionPage
              selectedRecording={selectedRecording}
              currentTime={currentTime}
              setCurrentTime={setCurrentTime}
              isPlaying={isPlaying}
              setIsPlaying={setIsPlaying}
              modelChoice={modelChoice}
              setModelChoice={setModelChoice}
              disabledModalities={disabledModalities}
              onToggleModality={handleToggleModality}
              latestPrediction={latestPrediction}
              timelineData={timelineData}
              loadingTimeline={loadingTimeline}
              predictionHistory={predictionHistory}
              onClearHistory={handleClearHistory}
              onRunWindowNow={runWindowInference}
            />
          )}
          {activeTab === 'explainability' && (
            <ExplainabilityPage latestPrediction={latestPrediction} globalXai={globalXai} />
          )}
          {activeTab === 'explorer' && (
            <DatasetExplorerPage
              datasetSummary={datasetSummary}
              participantsData={participantsData}
              selectedRecording={selectedRecording}
              onSelectRecording={setSelectedRecording}
              recordingData={recordingData}
              onNavigate={setActiveTab}
            />
          )}
          {activeTab === 'performance' && <ModelPerformancePage evalSummary={evalSummary} />}
        </main>
      </div>
    </div>
  );
}

