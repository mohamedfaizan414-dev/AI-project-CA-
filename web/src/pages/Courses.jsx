// Courses.jsx - browse the CLEANED catalogue. Every course here passed the cleaning + tagging pipeline.
import { useEffect, useState } from "react";
import { Search, ExternalLink } from "lucide-react";
import { get } from "../api.js";
import Reveal from "../components/Reveal.jsx";

export default function Courses() {
  const [meta, setMeta] = useState(null);
  const [q, setQ] = useState(""); const [comp, setComp] = useState(""); const [level, setLevel] = useState(""); const [prov, setProv] = useState("");
  const [data, setData] = useState({ total: 0, items: [] });
  const [limit, setLimit] = useState(24);

  useEffect(() => { get("/api/meta").then(setMeta).catch(() => {}); }, []);
  useEffect(() => {
    const t = setTimeout(() => {
      const p = new URLSearchParams({ q, competency: comp, level, provider: prov, limit });
      get("/api/courses?" + p).then(setData).catch(() => {});
    }, 200);
    return () => clearTimeout(t);
  }, [q, comp, level, prov, limit]);

  return (
    <div className="wrap section">
      <Reveal><div className="eyebrow">Course catalogue</div>
        <h2 style={{ marginTop: 8 }}>{meta ? meta.catalogue_size.toLocaleString() : "…"} cleaned, competency-tagged courses</h2>
        <p className="muted" style={{ marginTop: 6, maxWidth: 680 }}>Filter by the competency a course teaches. The tags were learned by the course-tagging model, not typed by hand.</p></Reveal>
      <div className="card" style={{ marginTop: 20 }}>
        <div className="grid g4">
          <label className="field">Search title
            <div style={{ position: "relative" }}><Search size={15} style={{ position: "absolute", left: 11, top: 12, color: "#6b8f71" }} />
              <input style={{ paddingLeft: 32 }} value={q} onChange={(e) => { setQ(e.target.value); setLimit(24); }} placeholder="e.g. regression" /></div></label>
          <label className="field">Competency
            <select value={comp} onChange={(e) => { setComp(e.target.value); setLimit(24); }}>
              <option value="">All</option>{meta?.competencies.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
          <label className="field">Level
            <select value={level} onChange={(e) => setLevel(e.target.value)}><option value="">All</option><option>Beginner</option><option>Intermediate</option><option>Advanced</option></select></label>
          <label className="field">Provider
            <select value={prov} onChange={(e) => setProv(e.target.value)}><option value="">All</option><option>Coursera</option><option>edX</option><option>Udemy</option></select></label>
        </div>
      </div>
      <p className="muted" style={{ margin: "16px 2px", fontSize: ".85rem" }}>{data.total.toLocaleString()} courses match</p>
      <div className="grid g3">
        {data.items.map((c, i) => (
          <Reveal key={c.course_id} delay={(i % 3) * 0.05}>
            <div className="card" style={{ height: "100%", display: "flex", flexDirection: "column" }}>
              <div className="meta"><span className="chip sand">{c.level}</span><span className="chip">{c.provider}</span></div>
              <h3 style={{ margin: "10px 0 6px", fontSize: "1rem" }}>{c.title}</h3>
              <p className="muted" style={{ fontSize: ".8rem" }}>{c.institution}</p>
              <div className="meta" style={{ marginTop: 10 }}>{c.skills.map((s) => <span key={s} className="chip green">{s}</span>)}</div>
              <div style={{ marginTop: "auto", paddingTop: 14, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span className="muted" style={{ fontSize: ".8rem" }}>{c.hours}h{c.hours_estimated ? " est." : ""}{c.rating ? ` · ★ ${c.rating}` : ""}</span>
                <a href={c.url} target="_blank" rel="noreferrer" className="btn ghost small">Open <ExternalLink size={12} /></a>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
      {data.items.length < data.total && <div style={{ textAlign: "center", marginTop: 26 }}><button className="btn ghost" onClick={() => setLimit(limit + 24)}>Show more</button></div>}
    </div>
  );
}
