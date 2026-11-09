// Dashboard.jsx - THE MAIN PAGE. Pick (or build) a learner -> see gaps -> see the recommended path.
import { useEffect, useMemo, useRef, useState } from "react";
import { Sparkles, Loader2 } from "lucide-react";
import { get, post } from "../api.js";
import Reveal from "../components/Reveal.jsx";
import CountUp from "../components/CountUp.jsx";
import Gauge from "../components/Gauge.jsx";
import GapBars from "../components/GapBars.jsx";
import PathTimeline from "../components/PathTimeline.jsx";

export default function Dashboard() {
  const [meta, setMeta] = useState(null);          // roles + competencies (for the form)
  const [demos, setDemos] = useState([]);          // ready-made example learners
  const [profile, setProfile] = useState(null);    // the learner currently shown
  const [maxCourses, setMaxCourses] = useState(8);
  const [budget, setBudget] = useState("");        // study-hours budget ("" = no limit)
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [active, setActive] = useState(0);
  const reqId = useRef(0);                         // so a slow old answer cannot overwrite a newer one

  // load meta + demo learners once, then show the first demo learner
  useEffect(() => {
    Promise.all([get("/api/meta"), get("/api/demo-learners")]).then(([m, d]) => {
      setMeta(m); setDemos(d); setProfile(d[0]);
    }).catch((e) => setErr("Could not reach the Python server. Start it with: uvicorn server.app:app --port 8000  (" + e.message + ")"));
  }, []);

  // whenever the learner or the controls change -> ask the server to analyse (wait 250ms so typing feels smooth)
  useEffect(() => {
    if (!profile) return;
    const id = ++reqId.current;
    const timer = setTimeout(() => {
      setBusy(true); setErr("");
      post("/api/analyze", {
        role_id: profile.role_id, exp_years: Number(profile.exp_years) || 0, education: profile.education,
        assessments: profile.assessments, completed_course_ids: profile.completed_course_ids || [],
        max_courses: maxCourses, hours_budget: budget ? Number(budget) : null,
      }).then((r) => { if (id === reqId.current) setResult(r); })
        .catch((e) => setErr(e.message))
        .finally(() => { if (id === reqId.current) setBusy(false); });
    }, 250);
    return () => clearTimeout(timer);
  }, [profile, maxCourses, budget]);

  const role = useMemo(() => meta?.roles.find((r) => r.id === profile?.role_id), [meta, profile]);
  const compName = (id) => meta?.competencies.find((c) => c.id === id)?.name || id;

  // ---- editing helpers
  const pick = (i) => { setActive(i); setProfile(demos[i]); };
  const custom = () => { setActive(-1); setProfile({ name: "You", role_id: meta.roles[2].id, exp_years: 3, education: 0, assessments: {}, completed_course_ids: [] }); };
  const setField = (k, v) => setProfile({ ...profile, [k]: v });
  const setScore = (cid, field, v) => {
    const cur = { ...(profile.assessments[cid] || {}) };
    if (v === "") delete cur[field]; else cur[field] = Math.max(1, Math.min(5, Number(v)));
    const next = { ...profile.assessments, [cid]: cur };
    if (!Object.keys(cur).length) delete next[cid];
    setProfile({ ...profile, assessments: next });
  };

  if (err && !meta) return <div className="wrap section"><div className="note">{err}</div></div>;
  if (!meta || !profile) return <div className="wrap section"><span className="spinner" /> Loading…</div>;
  const s = result?.summary;

  return (
    <div className="wrap section">
      <Reveal>
        <div className="eyebrow">Learner dashboard</div>
        <h2 style={{ marginTop: 8 }}>{profile.name !== "You" ? profile.name : "Your profile"} <span className="muted" style={{ fontWeight: 400 }}>· {role?.name}</span></h2>
        <p className="muted" style={{ marginTop: 6, maxWidth: 680 }}>Choose an example learner, or build your own. The gap model reads whatever evidence exists and infers the rest.</p>
      </Reveal>

      <div className="card" style={{ marginTop: 22 }}>
        <div className="persona-row">
          {demos.map((d, i) => (
            <button key={d.name} className={`persona ${active === i ? "active" : ""}`} onClick={() => pick(i)}>
              {d.name}<small>{meta.roles.find((r) => r.id === d.role_id)?.name}</small>
            </button>
          ))}
          <button className={`persona ${active === -1 ? "active" : ""}`} onClick={custom}>+ Build your own</button>
        </div>

        <div className="grid g4" style={{ marginTop: 18 }}>
          <label className="field">Role
            <select value={profile.role_id} onChange={(e) => setProfile({ ...profile, role_id: e.target.value, assessments: {} })}>
              {meta.roles.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
            </select></label>
          <label className="field">Years of experience
            <input type="number" min="0" max="35" value={profile.exp_years} onChange={(e) => setField("exp_years", e.target.value)} /></label>
          <label className="field">Education
            <select value={profile.education} onChange={(e) => setField("education", Number(e.target.value))}>
              {meta.education.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select></label>
          <label className="field">Path length: {maxCourses} courses
            <input type="range" min="4" max="12" value={maxCourses} onChange={(e) => setMaxCourses(Number(e.target.value))} style={{ padding: 0 }} /></label>
        </div>
        <div className="grid g4" style={{ marginTop: 14 }}>
          <label className="field">Study-hours budget
            <select value={budget} onChange={(e) => setBudget(e.target.value)}>
              <option value="">No limit</option><option value="20">20 hours</option><option value="40">40 hours</option><option value="80">80 hours</option>
            </select></label>
        </div>

        <details style={{ marginTop: 16 }}>
          <summary style={{ cursor: "pointer", fontWeight: 700, color: "#1f4d3a" }}>Assessment evidence ({Object.keys(profile.assessments).length} of {role?.core.length} skills have data) — edit</summary>
          <p className="muted" style={{ fontSize: ".82rem", margin: "10px 0" }}>Scores use a 1–5 scale. Leave a box empty if that evidence does not exist — the model handles missing data.</p>
          <div style={{ overflowX: "auto" }}>
            <table className="assess">
              <thead><tr><th>Competency</th><th>Needs</th><th>Quiz</th><th>Self</th><th>Manager</th></tr></thead>
              <tbody>
                {role?.core.map((c) => {
                  const a = profile.assessments[c.id] || {};
                  return (
                    <tr key={c.id}>
                      <td>{compName(c.id)}</td><td>{c.required}</td>
                      <td><input type="number" step="0.1" min="1" max="5" value={a.quiz ?? ""} onChange={(e) => setScore(c.id, "quiz", e.target.value)} /></td>
                      <td><input type="number" step="0.1" min="1" max="5" value={a.self_rating ?? ""} onChange={(e) => setScore(c.id, "self_rating", e.target.value)} /></td>
                      <td><input type="number" step="0.1" min="1" max="5" value={a.supervisor ?? ""} onChange={(e) => setScore(c.id, "supervisor", e.target.value)} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      </div>

      {err && <div className="note" style={{ marginTop: 16 }}>{err}</div>}

      {s && (
        <>
          <div className="grid g4" style={{ marginTop: 22, opacity: busy ? 0.6 : 1, transition: "opacity .2s" }}>
            <Reveal><div className="card stat" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}><Gauge now={s.readiness} after={s.projected?.readiness_after} /></div></Reveal>
            <Reveal delay={0.05}><div className="card stat"><div className="label">After the path</div>
              <div className="value">{s.projected ? <CountUp value={s.projected.readiness_after} decimals={0} suffix="%" /> : "–"}</div>
              <div className="sub">estimated readiness if the courses are completed</div></div></Reveal>
            <Reveal delay={0.1}><div className="card stat"><div className="label">Critical gaps</div>
              <div className="value" style={{ color: "#a4452c" }}><CountUp value={s.critical} /></div>
              <div className="sub">plus {s.moderate} moderate · {s.no_gap} skills already at target</div></div></Reveal>
            <Reveal delay={0.15}><div className="card stat"><div className="label">Study time</div>
              <div className="value"><CountUp value={s.path_hours} decimals={0} suffix="h" /></div>
              <div className="sub">across {result.path.length} courses</div></div></Reveal>
          </div>

          <div className="grid" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", marginTop: 30, alignItems: "start", gap: 28 }} id="dash-split">
            <section>
              <h3 style={{ marginBottom: 4 }}>Competency gap map</h3>
              <p className="muted" style={{ fontSize: ".85rem", marginBottom: 10 }}>Bar = estimated current level. Black marker = level the role needs. Q/S/M show which evidence the model had.</p>
              <div className="card"><GapBars gaps={result.gaps} /></div>
            </section>
            <section>
              <h3 style={{ marginBottom: 4 }}>Recommended learning path</h3>
              <p className="muted" style={{ fontSize: ".85rem", marginBottom: 10 }}>Biggest gaps first; within a skill, beginner before advanced.</p>
              <PathTimeline path={result.path} />
            </section>
          </div>
          <style>{`@media (max-width: 960px){ #dash-split{ grid-template-columns: 1fr !important; } }`}</style>
        </>
      )}
      {busy && !result && <p style={{ marginTop: 20 }}><Loader2 className="spinner" size={16} /> Analysing…</p>}
    </div>
  );
}
