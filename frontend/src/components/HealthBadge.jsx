import { useEffect, useState } from 'react';
import { fetchHealth } from '../api';

export default function HealthBadge() {
  const [health, setHealth] = useState(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const checkHealth = () => {
      fetchHealth()
        .then(data => setHealth(data))
        .catch(err => setHealth({ overall: 'DOWN', diagnostics: [{ component: 'API', status: 'DOWN', message: err.message }] }));
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  if (!health) return <div className="health-badge">Checking health...</div>;

  let color = 'var(--text)';
  if (health.overall === 'OK') color = '#1a7f37';
  if (health.overall === 'DEGRADED') color = '#d97706';
  if (health.overall === 'DOWN') color = '#cf222e';

  return (
    <div className="health-container" style={{ margin: '1rem auto', maxWidth: '600px', textAlign: 'left' }}>
      <button 
        onClick={() => setExpanded(!expanded)} 
        style={{
          border: `1px solid ${color}`,
          backgroundColor: `${color}15`,
          color,
          padding: '4px 12px',
          borderRadius: '16px',
          cursor: 'pointer',
          fontWeight: 'bold',
          fontSize: '0.85rem'
        }}
      >
        System Status: {health.overall} {expanded ? '▼' : '▶'}
      </button>

      {expanded && health.diagnostics && (
        <div style={{ marginTop: '0.5rem', padding: '1rem', border: '1px solid var(--border)', borderRadius: '4px', background: 'var(--bg)', fontSize: '0.85rem' }}>
          <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th>Component</th>
                <th>Status</th>
                <th>Message</th>
              </tr>
            </thead>
            <tbody>
              {health.diagnostics.map((diag, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '4px 0' }}>{diag.component}</td>
                  <td style={{ color: diag.status === 'OK' ? '#1a7f37' : '#cf222e' }}>{diag.status}</td>
                  <td>{diag.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
