// CountUp.jsx - a number that smoothly counts from its old value to its new value.
import { useEffect, useRef, useState } from "react";

export default function CountUp({ value, decimals = 0, suffix = "", duration = 900 }) {
  const [shown, setShown] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    const start = performance.now();
    const a = from.current, b = Number(value) || 0;
    let raf;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);          // ease-out: fast at first, slow at the end
      setShown(a + (b - a) * eased);
      if (t < 1) raf = requestAnimationFrame(tick); else from.current = b;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  return <>{shown.toFixed(decimals)}{suffix}</>;
}
