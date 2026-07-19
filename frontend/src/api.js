const API_BASE_URL = "http://127.0.0.1:8000";

async function handleResponse(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function fetchSignal(symbol) {
  const response = await fetch(`${API_BASE_URL}/signal/${encodeURIComponent(symbol)}`);
  return handleResponse(response);
}

export async function submitOverride({
  symbol,
  original_action,
  overridden_action,
  reason,
  created_by,
}) {
  const response = await fetch(`${API_BASE_URL}/override`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol,
      original_action,
      overridden_action,
      reason,
      created_by,
    }),
  });
  return handleResponse(response);
}