// Home.jsx - the landing page: what the product does + the headline numbers from our evaluation.
import { useEffect, useState } from "react";
import { ArrowRight, Target, Route, ClipboardCheck } from "lucide-react";
import { motion } from "framer-motion";
import Reveal from "../components/Reveal.jsx";
import Contours from "../components/Contours.jsx";
import CountUp from "../components/CountUp.jsx";
import { get } from "../api.js";

const STEPS = [
  { icon: Target, t: "Find the gap", d: "Role requirements, quiz scores, self-ratings and supervisor ratings are combined by a trained model that also infers skills nobody tested." },
  { icon: Route, t: "Rank the courses", d: "From 2,100 real courses, a learned ranker picks the ones that close your gaps at your level, then orders them as a path." },
  { icon: ClipboardCheck, t: "Check understanding", d: "Upload any learning material and get multiple-choice questions generated from it." },
];

export default function Home({ go }) {
  const [rep, setRep] = useState(null);
  useEffect(() => { get("/api/reports").then(setRep).catch(() => {}); }, []);
  const gap = rep?.gap_model_report, rs = rep?.recsys_report?.systems;
  const fin = rs?.["Ours: learned ranker (final)"], pop = rs?.["Popularity (baseline)"];

  return (
    <>
      <section className="hero">
        <Contours />
        <div className="wrap">
          <motion.div className="hero-inner" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <div className="eyebrow">Smart India Hackathon 2026 · MoSPI · SIH26101</div>
            <h1 style={{ marginTop: 14 }}>Know the gap. <em>Close it</em> with the right course.</h1>
            <p className="lead">An AI platform for India's Official Statistical System that infers each official's competency gaps and recommends a ranked learning path — instead of leaving them alone with a catalogue of thousands.</p>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <button className="btn" onClick={() => go("dashboard")}>Open learner dashboard <ArrowRight size={16} /></button>
              <button className="btn ghost" onClick={() => go("evidence")}>See the evidence</button>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 20 }}>
        <div className="wrap grid g3">
          {STEPS.map((s, i) => (
            <Reveal key={s.t} delay={i * 0.08}>
              <div className="card" style={{ height: "100%" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <s.icon size={22} color="#1f4d3a" /><span className="step-num">0{i + 1}</span>
                </div>
                <h3 style={{ marginTop: 14 }}>{s.t}</h3>
                <p className="muted" style={{ marginTop: 8, fontSize: ".93rem" }}>{s.d}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section" style={{ paddingTop: 10 }}>
        <div className="wrap">
          <Reveal><div className="eyebrow">Measured on unseen test learners</div><h2 style={{ marginTop: 8, marginBottom: 22 }}>Does it actually work?</h2></Reveal>
          <div className="grid g3">
            <Reveal><div className="card"><div className="stat-big">{gap ? <CountUp value={gap.test.macro_f1} decimals={2} /> : "–"}</div>
              <p style={{ fontWeight: 700, marginTop: 8 }}>Gap classification macro-F1</p>
              <p className="muted" style={{ fontSize: ".85rem" }}>vs {gap ? gap.baselines["Rule: quiz else self-rating"].macro_f1.toFixed(2) : "–"} for the best rule-based baseline</p></div></Reveal>
            <Reveal delay={0.08}><div className="card"><div className="stat-big">{fin ? <CountUp value={fin["recall@10"].mean} decimals={2} /> : "–"}</div>
              <p style={{ fontWeight: 700, marginTop: 8 }}>Recall@10 against expert paths</p>
              <p className="muted" style={{ fontSize: ".85rem" }}>vs {pop ? pop["recall@10"].mean.toFixed(3) : "–"} for the popularity-ranked baseline</p></div></Reveal>
            <Reveal delay={0.16}><div className="card"><div className="stat-big">{fin ? <CountUp value={fin["ndcg@10"].mean} decimals={2} /> : "–"}</div>
              <p style={{ fontWeight: 700, marginTop: 8 }}>NDCG@10</p>
              <p className="muted" style={{ fontSize: ".85rem" }}>vs {pop ? pop["ndcg@10"].mean.toFixed(3) : "–"} for popularity</p></div></Reveal>
          </div>
          <Reveal><div className="note" style={{ marginTop: 22 }}>Courses are real public data. Learners and expert paths are <b>simulated</b> because real iGOT data is not public — so read these numbers as proof that the pipeline works, not as a promise about real-world accuracy. The Evidence page explains exactly how.</div></Reveal>
        </div>
      </section>
    </>
  );
}
