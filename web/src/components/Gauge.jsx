// Gauge.jsx - a ring that shows "readiness" (how close the person is to what the job needs).
// Dark arc = now, light arc = what it becomes after the recommended path (an estimate).
import { motion } from "framer-motion";

export default function Gauge({ now = 0, after = null, size = 150 }) {
  const r = 58, c = 2 * Math.PI * r;
  const arc = (pct) => c * (1 - Math.max(0, Math.min(100, pct)) / 100);
  return (
    <svg width={size} height={size} viewBox="0 0 150 150" role="img" aria-label={`Readiness ${now}%`}>
      <circle cx="75" cy="75" r={r} fill="none" stroke="#e8dec3" strokeWidth="13" />
      {after != null && (
        <motion.circle cx="75" cy="75" r={r} fill="none" stroke="#a9bfa3" strokeWidth="13" strokeLinecap="round"
          strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: arc(after) }}
          transition={{ duration: 1.2, ease: "easeOut" }} transform="rotate(-90 75 75)" />
      )}
      <motion.circle cx="75" cy="75" r={r} fill="none" stroke="#1f4d3a" strokeWidth="13" strokeLinecap="round"
        strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: arc(now) }}
        transition={{ duration: 1, ease: "easeOut" }} transform="rotate(-90 75 75)" />
      <text x="75" y="74" textAnchor="middle" fontFamily="Fraunces, serif" fontSize="30" fontWeight="600" fill="#14352a">{Math.round(now)}%</text>
      <text x="75" y="94" textAnchor="middle" fontSize="9.5" fontWeight="700" letterSpacing="1.2" fill="#6b8f71">READY NOW</text>
    </svg>
  );
}
