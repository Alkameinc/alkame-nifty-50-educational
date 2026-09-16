const API_BASE = "/api/v1";
const API_KEY = import.meta.env.VITE_API_KEY ?? "dev-alkame-readonly-key";

function headers() {
  return { "X-API-Key": API_KEY, "Content-Type": "application/json" };
}

async function handleResponse(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const fetchSignal   = (symbol) => fetch(`${API_BASE}/signal/${encodeURIComponent(symbol)}`, { headers: headers() }).then(handleResponse);
export const fetchSymbols  = ()       => fetch(`${API_BASE}/symbols`, { headers: headers() }).then(handleResponse);
export const fetchHealth   = ()       => fetch(`/healthz`, { headers: headers() }).then(handleResponse);
export const fetchChart    = (symbol, horizon = "INTRADAY") => fetch(`${API_BASE}/chart/${encodeURIComponent(symbol)}?horizon=${horizon}`, { headers: headers() }).then(handleResponse);
export const fetchScalping = ()       => fetch(`${API_BASE}/scalping`, { headers: headers() }).then(handleResponse);
export const fetchRiskToggle = ()     => fetch(`${API_BASE}/risk/toggle`, { headers: headers() }).then(handleResponse);
export const toggleRisk = (enabled)   => fetch(`${API_BASE}/risk/toggle?enabled=${enabled}`, { method: 'POST', headers: headers() }).then(handleResponse);

// We implement streamSignal using fetch + ReadableStream instead of EventSource 
// so that we can pass the X-API-Key authentication header.
export async function streamSignal(symbol, onData, onError) {
  const url = `${API_BASE}/signal/stream/${encodeURIComponent(symbol)}`;
  try {
    const res = await fetch(url, { headers: headers() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // keep the last incomplete chunk

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          onData(data);
        }
      }
    }
  } catch (err) {
    onError(err);
  }
}