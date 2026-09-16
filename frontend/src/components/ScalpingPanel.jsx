import { useEffect, useState } from 'react';
import { fetchScalping } from '../api';

export default function ScalpingPanel() {
  const [setups, setSetups] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchScalping()
      .then(data => setSetups(data.setups || []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div>Loading scalping setups...</div>;
  if (setups.length === 0) return <div>No scalping setups currently available.</div>;

  return (
    <div className="scalping-panel" style={{ marginTop: '2rem', padding: '1rem', border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--bg)' }}>
      <h2 style={{ marginBottom: '1rem' }}>Intraday Scalping Opportunities</h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--border)' }}>
              <th>Symbol</th>
              <th>Action</th>
              <th>Entry</th>
              <th>Target</th>
              <th>Stop</th>
              <th>RR</th>
              <th>Conf</th>
            </tr>
          </thead>
          <tbody>
            {setups.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '8px 0', fontWeight: 'bold' }}>{s.symbol}</td>
                <td style={{ color: s.action === 'BUY' ? '#1a7f37' : s.action === 'SELL' ? '#cf222e' : 'inherit', fontWeight: 'bold' }}>{s.action}</td>
                <td>₹{s.entry.toFixed(2)}</td>
                <td>₹{s.target.toFixed(2)}</td>
                <td>₹{s.stop.toFixed(2)}</td>
                <td>{s.rr.toFixed(2)}</td>
                <td>{(s.confidence * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
