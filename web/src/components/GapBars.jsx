// GapBars.jsx - one row per competency: a bar for "where you are", a black marker for "what the job needs".
import { motion } from "framer-motion";

const COLORS = { 0: "#6b8f71", 1: "#c79a3b", 2: "#a4452c" };      // none / moderate / critical
const CHIP = { 0: "green", 1: "ochre", 2: "rust" };

export default function GapBars({ gaps }) {
  // bar length = level / 5 (levels run 1..5, so we map 1..5 to 0..100% of the track after subtracting 1)
  const pct = (lvl) => ((lvl - 1) / 4) * 100;
  return (
    <div>
      {gaps.map((g, i) => (
        <div className="gap-row" key={g.competency_id}>
          <div className="gap-head">
            <span className="gap-name">{g.name} <span className="chip sand" style={{ marginLeft: 6 }}>{g.group}</span></span>
            <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span className="ev" title="Evidence available: Quiz, Self-rating, Manager rating">
                <span className={g.evidence.quiz != null ? "on" : ""}>Q</span>
                <span className={g.evidence.self != null ? "on" : ""}>S</span>
                <span className={g.evidence.supervisor != null ? "on" : ""}>M</span>
              </span>
              <span className={`chip ${CHIP[g.class]}`}>{g.label}</span>
            </span>
          </div>
          <div className="track">
            <motion.div className="fill" style={{ background: COLORS[g.class] }}
              initial={{ width: 0 }} whileInView={{ width: `${pct(g.estimated_level)}%` }} viewport={{ once: true }}
              transition={{ duration: 0.9, delay: Math.min(i * 0.04, 0.5), ease: "easeOut" }} />
            <div className="need" style={{ left: `calc(${pct(g.required)}% - 1px)` }} />
          </div>
          <div className="ticks"><span>1 Aware</span><span>2</span><span>3 Working</span><span>4</span><span>5 Expert</span></div>
          <p className="muted" style={{ fontSize: ".8rem", marginTop: 4 }}>
            Estimated level <b>{g.estimated_level.toFixed(1)}</b> · role needs <b>{g.required}</b> · gap <b>{g.gap_size.toFixed(1)}</b> levels
            {g.evidence.quiz == null && <> · <i>no quiz — inferred from related skills</i></>}
          </p>
        </div>
      ))}
    </div>
  );
}
