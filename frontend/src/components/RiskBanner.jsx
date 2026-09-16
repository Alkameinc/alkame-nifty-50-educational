import { useEffect, useState } from 'react';
import { fetchRiskToggle } from '../api';

export default function RiskBanner() {
  const [riskData, setRiskData] = useState(null);

  useEffect(() => {
    fetchRiskToggle()
      .then(data => setRiskData(data))
      .catch(console.error);
      
    // Poll every 60 seconds
    const interval = setInterval(() => {
      fetchRiskToggle().then(setRiskData).catch(console.error);
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  if (!riskData || !riskData.enabled) return null;

  return (
    <div className="risk-banner" style={{ 
      backgroundColor: '#cf222e', 
      color: 'white', 
      padding: '0.75rem', 
      textAlign: 'center', 
      fontWeight: 'bold',
      width: '100%',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      ⚠️ Global risk mode is active — signals are penalised. ({riskData.reason})
    </div>
  );
}
