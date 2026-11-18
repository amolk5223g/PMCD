import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import { Sparkles, BrainCircuit, Info } from 'lucide-react';

const MOD_COLORS = {
  EMG: '#eab308',
  Grip: '#6366f1',
  EDA: '#0ea5e9',
  BVP: '#ef4444',
  Resp: '#10b981',
  Tmp: '#8b5cf6',
  Temporal: '#64748b',
};

export default function ExplainabilityPage({ latestPrediction, globalXai }) {
  const localExp = latestPrediction?.explainability;

  const localShapData = (localExp?.shap?.top_features || []).map((f) => ({
    feature: f.feature,
    shap: f.shap_value,
    modality: f.modality,
    value: f.feature_value,
  }));

  const localModData = ['BVP', 'EDA', 'Resp', 'Tmp', 'EMG', 'Grip'].map((m) => ({
    modality: m,
    SHAP: localExp?.shap?.modality_contributions?.[m]?.pct_contribution || 0,
    IntegratedGradients: localExp?.integrated_gradients?.modality_attributions?.[m]?.pct_contribution || 0,
    NeuralGate: localExp?.neural_gate_weights_pct?.[m] || 0,
  }));

  const temporalSalience = localExp?.integrated_gradients?.temporal_salience || [];

  const globalTopFeatures = (globalXai?.shap_top_25_features || []).slice(0, 15).map((f) => ({
    feature: f.feature,
    importance: f.mean_abs_shap,
    modality: f.modality,
  }));

  const globalModComp = ['BVP', 'EDA', 'Resp', 'Tmp', 'EMG', 'Grip'].map((m) => ({
    modality: m,
    SHAP: globalXai?.shap_modality_contribution_pct?.[m] || 0,
    IntegratedGradients: globalXai?.integrated_gradients_modality_pct?.[m] || 0,
    NeuralGate: globalXai?.neural_gate_modality_weights_pct?.[m] || 0,
  }));

  return (
    <div className="space-y-5">
      {/* Local Explanation Summary */}
      <div className="card p-5 border-l-4 border-l-sky-500">
        <div className="flex items-start gap-3">
          <Sparkles className="w-5 h-5 text-sky-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-sm font-bold">Local Window Explanation</h2>
              {latestPrediction && (
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800">
                  {latestPrediction.recording_id} [{latestPrediction.window_start_sec}s – {latestPrediction.window_end_sec}s]
                </span>
              )}
            </div>
            <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
              {localExp?.summary || 'Run a prediction on any PMPDB recording window to inspect local SHAP & Integrated Gradients attributions.'}
            </p>
            <p className="text-[11px] text-slate-500 flex items-center gap-1 pt-1">
              <Info className="w-3.5 h-3.5" />
              {localExp?.limitations_note || 'Explanations cover strictly BVP, EDA, Resp, Tmp, EMG, and Grip. No EEG or ECG channels exist in PMPDB.'}
            </p>
          </div>
        </div>
      </div>

      {/* Local SHAP + Local Modality Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <h3 className="text-sm font-bold mb-1">Local SHAP Feature Attributions (XGBoost)</h3>
          <p className="text-[11px] text-slate-500 mb-3">
            Red bars push predicted pain higher; blue bars push predicted pain lower (Base value: {localExp?.shap?.base_value ?? '--'})
          </p>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={localShapData} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ fontSize: '11px', borderRadius: '8px' }}
                  formatter={(v, _n, item) => [`${v > 0 ? '+' : ''}${v} pts (val=${item.payload.value})`, `SHAP (${item.payload.modality})`]}
                />
                <Bar dataKey="shap" radius={[0, 4, 4, 0]}>
                  {localShapData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.shap >= 0 ? '#ef4444' : '#0284c7'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-bold mb-1">Local Modality Contribution (SHAP vs. Integrated Gradients vs. Neural Gates)</h3>
          <p className="text-[11px] text-slate-500 mb-3">
            Cross-method agreement across the 6 PMPDB physiological/biomechanical modalities (%)
          </p>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={localModData} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis dataKey="modality" tick={{ fontSize: 11, fontWeight: 600 }} />
                <YAxis tick={{ fontSize: 10 }} unit="%" />
                <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="SHAP" name="SHAP (XGBoost)" fill="#0284c7" radius={[4, 4, 0, 0]} />
                <Bar dataKey="IntegratedGradients" name="Integrated Gradients (MM_Gated)" fill="#f97316" radius={[4, 4, 0, 0]} />
                <Bar dataKey="NeuralGate" name="Gated Fusion Weights" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Temporal Salience */}
      <div className="card p-5">
        <h3 className="text-sm font-bold mb-1 flex items-center gap-2">
          <BrainCircuit className="w-4 h-4 text-orange-500" />
          Integrated Gradients Temporal Salience Across 6-Second Causal Window
        </h3>
        <p className="text-[11px] text-slate-500 mb-3">
          Shows which half-second time segments preceding the prediction (`-6.0s` to `0.0s`) drove the MM_Gated neural network most strongly
        </p>
        <div className="h-44 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={temporalSalience} margin={{ top: 5, right: 15, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
              <XAxis dataKey="window_offset_sec" tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}s`} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} labelFormatter={(v) => `Offset: ${v}s`} />
              <Area type="monotone" dataKey="salience" name="Normalized IG Salience" stroke="#f97316" fill="#f97316" fillOpacity={0.22} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Global XAI across all 1,424 PMPDB Windows */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <h3 className="text-sm font-bold mb-1">Global Top-15 Feature Importance (Mean |SHAP| Across PMPDB)</h3>
          <p className="text-[11px] text-slate-500 mb-3">Color-coded by physiological modality across all 1,424 labeled windows</p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={globalTopFeatures} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="feature" width={145} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                <Bar dataKey="importance" name="Mean |SHAP|" radius={[0, 4, 4, 0]}>
                  {globalTopFeatures.map((entry, idx) => (
                    <Cell key={idx} fill={MOD_COLORS[entry.modality] || '#64748b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-bold mb-1">Global Modality Importance Across PMPDB Dataset</h3>
          <p className="text-[11px] text-slate-500 mb-3">Aggregated over 49 participants & 90 sessions (strictly BVP, EDA, Resp, Tmp, EMG, Grip)</p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={globalModComp} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis dataKey="modality" tick={{ fontSize: 11, fontWeight: 600 }} />
                <YAxis tick={{ fontSize: 10 }} unit="%" />
                <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="SHAP" name="Global SHAP" fill="#0284c7" radius={[4, 4, 0, 0]} />
                <Bar dataKey="IntegratedGradients" name="Global Integrated Gradients" fill="#f97316" radius={[4, 4, 0, 0]} />
                <Bar dataKey="NeuralGate" name="Mean Neural Gate Weights" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

