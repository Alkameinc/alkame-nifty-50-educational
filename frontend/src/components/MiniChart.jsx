import { useEffect, useState } from 'react';
import { fetchChart } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function MiniChart({ symbol, horizon }) {
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    
    const loadData = async () => {
      setLoading(true);
      try {
        const data = await fetchChart(symbol, horizon);
        if (!cancelled) {
          setChartData(data.chart || []);
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadData();

    return () => { cancelled = true; };
  }, [symbol, horizon]);

  if (loading) return <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading chart...</div>;
  if (!chartData || chartData.length === 0) return <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>No chart data available</div>;

  return (
    <div style={{ height: 250, width: '100%', marginTop: '1rem' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <XAxis 
            dataKey="time" 
            tick={{ fontSize: 10, fill: 'var(--text)' }} 
            minTickGap={30}
          />
          <YAxis 
            domain={['auto', 'auto']} 
            tick={{ fontSize: 10, fill: 'var(--text)' }} 
            width={60}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: 'var(--bg)', borderColor: 'var(--border)', color: 'var(--text-h)' }}
          />
          <Line 
            type="monotone" 
            dataKey="close" 
            stroke="var(--accent)" 
            strokeWidth={2} 
            dot={false} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
