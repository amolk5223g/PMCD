const API_BASE = import.meta.env.VITE_API_BASE || '';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      msg = err.detail || msg;
    } catch {
      // ignore json parse failure
    }
    throw new Error(msg);
  }
  return res.json();
}

export const api = {
  getHealth: () => request('/api/health'),
  getModelStatus: () => request('/api/models/status'),
  getDatasetSummary: () => request('/api/dataset/summary'),
  getParticipants: () => request('/api/dataset/participants'),
  getRecording: (recordingId, downsampleHz = 8) =>
    request(`/api/dataset/recordings/${encodeURIComponent(recordingId)}?downsample_hz=${downsampleHz}`),
  getEvaluationSummary: () => request('/api/evaluation/summary'),
  getGlobalXai: () => request('/api/explain/global'),
  getPredictionHistory: (limit = 50) => request(`/api/predictions/history?limit=${limit}`),
  clearPredictionHistory: () => request('/api/predictions/history', { method: 'DELETE' }),
  predictRecording: (payload) =>
    request('/api/predict/recording', { method: 'POST', body: JSON.stringify(payload) }),
  predictTimeline: (payload) =>
    request('/api/predict/timeline', { method: 'POST', body: JSON.stringify(payload) }),
  explainWindow: (payload) =>
    request('/api/explain/window', { method: 'POST', body: JSON.stringify(payload) }),
};

