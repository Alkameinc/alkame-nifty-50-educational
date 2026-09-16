

export default function HorizonTabs({ horizons, selected, onSelect }) {
  return (
    <div className="horizon-tabs" style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
      {horizons.map(h => (
        <button
          key={h}
          onClick={() => onSelect(h)}
          style={{
            padding: '0.5rem 1rem',
            border: selected === h ? '2px solid var(--accent)' : '1px solid var(--border)',
            backgroundColor: selected === h ? 'var(--accent-bg)' : 'var(--bg)',
            color: selected === h ? 'var(--text-h)' : 'var(--text)',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: selected === h ? 'bold' : 'normal'
          }}
        >
          {h}
        </button>
      ))}
    </div>
  );
}
