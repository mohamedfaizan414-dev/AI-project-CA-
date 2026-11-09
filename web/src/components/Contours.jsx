// Contours.jsx - decorative "topographic map" lines drawn with SVG, used behind the hero.
// Each path is a gentle wave; stacking many of them looks like contour lines of a hill.
export default function Contours() {
  const lines = Array.from({ length: 16 }, (_, i) => {
    const y = 60 + i * 34;
    const a = 26 + (i % 5) * 6;                     // wave height changes per line
    return `M-20 ${y} C 220 ${y - a}, 380 ${y + a * 1.4}, 620 ${y} S 1000 ${y - a * 1.2}, 1260 ${y + a * .5}`;
  });
  return (
    <svg className="contours" viewBox="0 0 1240 640" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      {lines.map((d, i) => (
        <path key={i} d={d} fill="none" stroke={i % 4 === 0 ? "#6b8f71" : "#a9bfa3"} strokeWidth={i % 4 === 0 ? 1.3 : 0.8} opacity={0.75 - i * 0.025} />
      ))}
    </svg>
  );
}
