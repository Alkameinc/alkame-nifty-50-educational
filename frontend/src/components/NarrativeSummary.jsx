import { useState } from 'react';

export default function NarrativeSummary({ narrative }) {
  const [expanded, setExpanded] = useState(false);

  if (!narrative) return null;

  return (
    <div style={{ margin: '1.5rem 0', padding: '1rem', background: 'var(--accent-bg)', border: '1px solid var(--accent-border)', borderRadius: '8px', textAlign: 'left' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0, color: 'var(--accent)' }}>Overall Narrative</h3>
        <button 
          onClick={() => setExpanded(!expanded)}
          style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontWeight: 'bold' }}
        >
          {expanded ? 'Collapse ▲' : 'Expand ▼'}
        </button>
      </div>
      
      {expanded && (
        <div style={{ marginTop: '1rem', fontStyle: 'italic', lineHeight: '1.5' }}>
          {narrative}
        </div>
      )}
    </div>
  );
}
