// Evidence.jsx - the "show your work" page: data cleaning, tagging model, gap model, recommender, limitations.
// Every number is read from the JSON reports the Python pipeline wrote, so this page can never go out of date.
import { useEffect, useState } from "react";
import { get } from "../api.js";
import Reveal from "../components/Reveal.jsx";

const f3 = (x) => (x == null ? "–" : x.toFixed(3));

function Bars({ rows, max = 1, clay = false, fmt = f3 }) {
  return rows.map(([label, v]) => (
    <div className="hbar" key={label}><span>{label}</span>
      <div className="bar"><i className={clay ? "clay" : ""} style={{ width: `${Math.min(100, (v / max) * 100)}%` }} /></div><b>{fmt(v)}</b></div>
  ));
}

export default function Evidence() {
  const [r, setR] = useState(null);
  useEffect(() => { get("/api/reports").then(setR).catch(() => {}); }, []);
  if (!r) return <div className="wrap section"><span className="spinner" /> Loading…</div>;
  const { cleaning_report: cl, tagging_report: tg, gap_model_report: gm, recsys_report: rs } = r;
  const cm = gm.confusion_matrix_test, names = ["None", "Moderate", "Critical"];
  const systems = Object.entries(rs.systems);

  return (
    <div className="wrap section">
      <Reveal><div className="eyebrow">Evidence</div><h2 style={{ marginTop: 8 }}>How it was built and how well it works</h2></Reveal>

      <Reveal><div className="note" style={{ marginTop: 20 }}><b>Read this first.</b> The course catalogue ({cl.summary.final_rows.toLocaleString()} courses) is real public data. Learners, assessments and expert paths are <b>simulated</b> (real iGOT data is not public), so the numbers below show the pipeline works and which design choices help — they are not a forecast of accuracy on real officials.</div></Reveal>

      <section style={{ marginTop: 40 }}>
        <Reveal><h3>1 · Dataset cleaning</h3><p className="muted" style={{ margin: "6px 0 14px", maxWidth: 700 }}>Three real datasets with three different layouts were merged, then cleaned. {cl.summary.final_rows.toLocaleString()} courses survive.</p></Reveal>
        <Reveal><div className="card"><table className="tbl"><thead><tr><th>Step</th><th>Before</th><th>After</th><th>Removed</th></tr></thead><tbody>
          {cl.steps.filter((s) => s.removed > 0).map((s) => <tr key={s.step}><td>{s.step}</td><td>{s.rows_before}</td><td>{s.rows_after}</td><td>{s.removed}</td></tr>)}
        </tbody></table></div></Reveal>
      </section>

      <section style={{ marginTop: 40 }}>
        <Reveal><h3>2 · Which competency does each course teach?</h3>
          <p className="muted" style={{ margin: "6px 0 14px", maxWidth: 720 }}>No dataset had these labels. Keyword rules give noisy labels; a second classifier that <i>cannot see the keywords</i> learns the surrounding vocabulary. It re-discovers the rule labels with mean F1 <b>{tg.context_classifier_mean_f1}</b> from context alone — proof it learned real topic words. {tg.courses_relevant.toLocaleString()} of {tg.courses_in.toLocaleString()} courses were relevant to our framework.</p></Reveal>
        <Reveal><div className="card"><div className="eyebrow" style={{ marginBottom: 8 }}>Courses per competency (after tagging)</div>
          <Bars max={Math.max(...Object.values(tg.coverage).map((c) => c.courses))} fmt={(x) => x}
            rows={Object.entries(tg.coverage).sort((a, b) => b[1].courses - a[1].courses).map(([k, v]) => [k.replaceAll("_", " "), v.courses])} /></div></Reveal>
      </section>

      <section style={{ marginTop: 40 }}>
        <Reveal><h3>3 · Gap model</h3><p className="muted" style={{ margin: "6px 0 14px", maxWidth: 720 }}>Three classes: no gap / moderate / critical. Winner on validation: <b>{gm.winner}</b>. The test set was touched once.</p></Reveal>
        <div className="grid g2">
          <Reveal><div className="card"><div className="eyebrow" style={{ marginBottom: 8 }}>Macro-F1 on test learners</div>
            <Bars rows={[["Majority class", gm.baselines["Majority class"].macro_f1], ["Rule: self-rating", gm.baselines["Rule: trust self-rating"].macro_f1],
              ["Rule: quiz else self", gm.baselines["Rule: quiz else self-rating"].macro_f1], [gm.winner, gm.test.macro_f1]]} />
            <p className="muted" style={{ fontSize: ".82rem", marginTop: 10 }}>Per-class F1: none {f3(gm.test.per_class_f1[0])}, moderate {f3(gm.test.per_class_f1[1])}, critical {f3(gm.test.per_class_f1[2])}. Quadratic kappa {f3(gm.test.quadratic_kappa)}.</p>
            <p style={{ fontSize: ".88rem", marginTop: 10 }}>For skills with <b>no quiz</b> the model scores <b>{f3(gm.test_skills_WITHOUT_quiz.macro_f1)}</b> vs {f3(gm.baseline_rule_quiz_else_self_WITHOUT_quiz.macro_f1)} for the rule — it infers from related skills.</p></div></Reveal>
          <Reveal delay={0.08}><div className="card"><div className="eyebrow" style={{ marginBottom: 8 }}>What each evidence source adds (validation F1)</div>
            <Bars rows={gm.ablation.map((a) => [a.information_used, a.validation_macro_f1])} clay />
            <div className="eyebrow" style={{ margin: "16px 0 8px" }}>Confusion matrix (rows = truth)</div>
            <div className="cm">{cm.map((row, i) => row.map((v, j) => {
              const share = v / row.reduce((a, b) => a + b, 0);
              return <div key={`${i}${j}`} style={{ background: `rgba(31,77,58,${0.08 + share * 0.85})`, color: share > 0.5 ? "#faf6ec" : "#1b2a22" }} title={`true ${names[i]} → predicted ${names[j]}`}>{Math.round(share * 100)}%</div>;
            }))}</div></div></Reveal>
        </div>
      </section>

      <section style={{ marginTop: 40 }}>
        <Reveal><h3>4 · Recommender vs baselines</h3><p className="muted" style={{ margin: "6px 0 14px", maxWidth: 740 }}>{rs.test_learners} unseen learners. Relevant = chosen by at least 2 of 3 simulated experts. 95% bootstrap intervals are in the report file.</p></Reveal>
        <Reveal><div className="card" style={{ overflowX: "auto" }}><table className="tbl"><thead><tr><th>System</th><th>Recall@5</th><th>Recall@10</th><th>Recall@20</th><th>NDCG@10</th><th>Gap coverage@10</th></tr></thead><tbody>
          {systems.map(([n, t]) => <tr key={n} className={n.includes("final") ? "hl" : ""}><td>{n}</td>{["recall@5", "recall@10", "recall@20", "ndcg@10", "gapcov@10"].map((m) => <td key={m}>{f3(t[m].mean)}</td>)}</tr>)}
        </tbody></table></div></Reveal>
        <Reveal><p className="muted" style={{ fontSize: ".85rem", marginTop: 12, maxWidth: 760 }}>The final system beats popularity by {f3(rs.final_vs_popularity_paired["recall@10"].mean_difference)} recall@10 (95% CI {rs.final_vs_popularity_paired["recall@10"].ci95.map(f3).join(" – ")}). The last row gives the same ranker the <i>true</i> gaps: the distance to it is what gap-inference errors cost.</p></Reveal>
      </section>

      <section style={{ marginTop: 40 }}>
        <Reveal><h3>5 · Honest limitations</h3></Reveal>
        <Reveal><div className="card"><ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
          <li>Learners, assessments and expert paths are simulated; the expert panel is rule-based. Absolute scores are optimistic — the ordering of systems is the real finding.</li>
          <li>Course-to-competency tags come from weak supervision, not human labels. Use ml/data/processed/gold_review_sample.csv to hand-check 80 courses.</li>
          <li>Coursera/edX have no course length for most rows; hours are estimated from level (flagged “est.”). Udemy has real hours and subscriber counts.</li>
          <li>The catalogue is general-purpose, not statistics-specific, so some competencies (e.g. survey design) have few courses. Real iGOT Karmayogi content would fix this.</li>
          <li>“Finance, Budgeting & Accounting” has by far the most courses because Udemy’s finance section is large; it is corporate-finance heavy, not public-finance specific.</li>
          <li>No live iGOT Karmayogi API is connected (no public API); the catalogue schema is designed so iGOT data can replace it.</li>
        </ul></div></Reveal>
      </section>
    </div>
  );
}
