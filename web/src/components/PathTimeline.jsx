// PathTimeline.jsx - the recommended learning path, drawn as a vertical timeline of course cards.
import { motion } from "framer-motion";
import { ExternalLink, Clock, Star } from "lucide-react";

export default function PathTimeline({ path }) {
  if (!path.length) return <p className="muted">No courses needed — no meaningful gaps found.</p>;
  return (
    <div className="timeline">
      {path.map((c, i) => (
        <motion.div className="node" key={c.course_id}
          initial={{ opacity: 0, x: -10 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true, margin: "-30px" }}
          transition={{ duration: 0.45, delay: Math.min(i * 0.05, 0.3) }}>
          <div className="dot">{c.step}</div>
          <div className="card">
            <div className="meta">
              <span className="chip green">{c.primary_skill}</span>
              <span className="chip sand">{c.level}</span>
              <span className="chip">{c.provider}</span>
            </div>
            <h3>{c.title}</h3>
            <div className="meta muted" style={{ fontSize: ".82rem" }}>
              <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}><Clock size={13} /> {c.hours}h{c.hours_estimated ? " (est.)" : ""}</span>
              {c.rating && <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}><Star size={13} /> {c.rating}</span>}
              <span>· +{c.expected_level_gain} level{c.expected_level_gain === 1 ? "" : "s"} expected</span>
            </div>
            <p className="why">{c.why}{c.also_covers.length > 0 && <> Also touches: {c.also_covers.join(", ")}.</>}</p>
            <a className="btn ghost small" style={{ marginTop: 12 }} href={c.url} target="_blank" rel="noreferrer">Open course <ExternalLink size={13} /></a>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
