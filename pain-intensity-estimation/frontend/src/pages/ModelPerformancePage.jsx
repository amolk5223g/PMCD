import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { Award, ShieldCheck, Layers } from 'lucide-react';

export default function ModelPerformancePage({ evalSummary }) {
  const [selectedParticipantModel, setSelectedParticipantModel] = useState('XGBoost');
  const table = evalSummary?.model_comparison_table || [];

  const ablationRaw = evalSummary?.baseline?.modality_ablation || {};
  const ablationChart = Object.entries(ablationRaw).map(([name, m]) => ({
    modality: name.replace('All_Phys_No_Temporal', 'All Phys').replace('All_Multimodal', 'Multimodal'),
    MAE: m.mae,
    Pearson_r: Number((m.pearson_r * 100).toFixed(1)),
    CCC: Number((m.ccc * 100).toFixed(1)),
  }));

  const stressRaw = evalSummary?.deep_learning?.missing_modality_stress_test || {};
  const stressChart = Object.entries(stressRaw).map(([name, m]) => ({
    scenario: name.replace('Wearable_E4_Only (BVP+EDA+Tmp)', 'E4 Only'),
    MAE: m.mae,
    RMSE: m.rmse,
    MeanUncStd: m.uncertainty?.mean_std || 0,
  }));

  const participantOptions = {
    XGBoost: evalSummary?.baseline?.participant_metrics?.XGBoost || [],
    RandomForest: evalSummary?.baseline?.participant_metrics?.RandomForest || [],
    MM_Gated: evalSummary?.deep_learning?.participant_metrics?.MM_Gated || [],
    MM_Attention: evalSummary?.deep_learning?.participant_metrics?.MM_Attention || [],
  };
  const participantMetrics = participantOptions[selectedParticipantModel] || [];

  return (
    <div className="space-y-5">
      {/* Full Model Comparison Table */}
      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h2 className="text-sm font-bold flex items-center gap-2">
              <Award className="w-4 h-4 text-sky-500" />
              Participant-Independent Validation Results (All Models)
            </h2>
            <p className="text-[11px] text-slate-500">
              Strict zero-leakage 5-Fold GroupKFold (all models) & 49-Fold Leave-One-Subject-Out (ML baselines) across 1,424 verified PMPDB ratings
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
              <tr>
                <th className="py-2 pr-3">Model</th>
                <th className="py-2 pr-3">Family</th>
                <th className="py-2 pr-3">MAE ↓</th>
                <th className="py-2 pr-3">RMSE ↓</th>
                <th className="py-2 pr-3">Pearson r ↑</th>
                <th className="py-2 pr-3">Lin's CCC ↑</th>
                <th className="py-2 pr-3">R² ↑</th>
                <th className="py-2 pr-3">LOSO MAE / r</th>
                <th className="py-2">PICP 95% / Ord F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {table.map((row, idx) => (
                <tr key={row.model} className={idx === 0 ? 'bg-sky-500/5 font-semibold' : ''}>
                  <td className="py-2.5 pr-3 font-mono">{row.model}</td>
                  <td className="py-2.5 pr-3">{row.family}</td>
                  <td className="py-2.5 pr-3 text-sky-600 dark:text-sky-400 font-bold">{row.mae.toFixed(2)}</td>
                  <td className="py-2.5 pr-3">{row.rmse.toFixed(2)}</td>
                  <td className="py-2.5 pr-3 text-emerald-600 dark:text-emerald-400">{row.pearson_r.toFixed(3)}</td>
                  <td className="py-2.5 pr-3 text-indigo-600 dark:text-indigo-400">{row.ccc.toFixed(3)}</td>
                  <td className="py-2.5 pr-3">{row.r2.toFixed(3)}</td>
                  <td className="py-2.5 pr-3 font-mono text-[11px]">
                    {row.loso_mae != null ? `${row.loso_mae.toFixed(2)} / ${row.loso_pearson_r.toFixed(3)}` : '—'}
                  </td>
                  <td className="py-2.5 font-mono text-[11px]">
                    {row.picp_95 != null ? `${(row.picp_95 * 100).toFixed(1)}%` : '—'}
                    {row.ordinal_f1 != null ? ` · F1=${row.ordinal_f1.toFixed(2)}` : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Single Modality vs Multimodal + Missing Modality Stress Test */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card p-5">
          <h3 className="text-sm font-bold mb-1 flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-500" />
            Single-Modality vs. Multimodal Ablation (5-Fold GroupKFold)
          </h3>
          <p className="text-[11px] text-slate-500 mb-3">Lower MAE and higher Pearson r indicate stronger pain prediction</p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ablationChart} margin={{ top: 5, right: 15, left: -12, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis dataKey="modality" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="MAE" name="MAE (0–100, lower better)" fill="#0284c7" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Pearson_r" name="Pearson r × 100" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-5">
          <h3 className="text-sm font-bold mb-1 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-500" />
            Missing-Modality Resilience Stress Test (OOF MM_Gated)
          </h3>
          <p className="text-[11px] text-slate-500 mb-3">
            Zero-fabrication masking evaluation when individual sensors or entire RespiBAN unit drop out
          </p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stressChart} margin={{ top: 5, right: 15, left: -12, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
                <XAxis dataKey="scenario" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="MAE" name="MAE (0–100)" fill="#f97316" radius={[4, 4, 0, 0]} />
                <Bar dataKey="MeanUncStd" name="Mean Uncertainty σ" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Participant-Wise Evaluation Across All 49 Participants */}
      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-bold">Participant-Wise Out-of-Fold MAE Across All 49 Participants</h3>
            <p className="text-[11px] text-slate-500">Illustrates inter-subject variability in physiological pain generalization</p>
          </div>
          <select
            value={selectedParticipantModel}
            onChange={(e) => setSelectedParticipantModel(e.target.value)}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-semibold"
          >
            <option value="XGBoost">XGBoost (49-Fold LOSO)</option>
            <option value="RandomForest">Random Forest (49-Fold LOSO)</option>
            <option value="MM_Gated">MM_Gated (5-Fold GroupKFold)</option>
            <option value="MM_Attention">MM_Attention (5-Fold GroupKFold)</option>
          </select>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={participantMetrics} margin={{ top: 5, right: 10, left: -12, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
              <XAxis dataKey="participant_id" tick={{ fontSize: 9 }} angle={-60} textAnchor="end" interval={0} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ fontSize: '11px', borderRadius: '8px' }}
                formatter={(v, n) => [v, n]}
              />
              <Bar dataKey="mae" name="Participant MAE (0–100)" fill="#0ea5e9" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

