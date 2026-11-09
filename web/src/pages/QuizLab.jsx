// QuizLab.jsx - upload learning material (txt / md / pdf) or paste text -> multiple-choice questions.
import { useState } from "react";
import { Upload, FileText, Check, X } from "lucide-react";
import { postForm } from "../api.js";
import Reveal from "../components/Reveal.jsx";

const SAMPLE = `Sampling is the process of selecting a subset of units from a population to estimate its characteristics. A sampling frame is a list of all units in the population from which the sample is drawn. Simple random sampling gives every unit in the population an equal chance of being selected. Stratified sampling divides the population into homogeneous groups called strata and draws a separate sample from each stratum. Cluster sampling selects whole groups of units, such as villages or census blocks, which reduces travel cost in household surveys. The sampling error measures how much an estimate from a sample is expected to differ from the true population value. A larger sample size reduces the sampling error but increases the cost of data collection. Non-response bias occurs when the people who do not answer a survey differ systematically from those who do. Weighting adjusts survey estimates so that the sample represents the population correctly. The design effect compares the variance of a complex sample design with the variance of a simple random sample of the same size. Official statistics agencies publish the margin of error alongside survey estimates so that users understand their precision. A questionnaire should use clear and neutral wording to avoid measurement error. Pilot testing a questionnaire helps to detect confusing questions before the main survey begins.`;

export default function QuizLab() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [n, setN] = useState(8);
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});      // question index -> chosen option index
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const generate = async () => {
    setBusy(true); setErr(""); setQuiz(null); setAnswers({});
    const fd = new FormData();
    if (file) fd.append("file", file); else fd.append("text", text);
    fd.append("n", n);
    try { setQuiz(await postForm("/api/quiz", fd)); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  const answered = Object.keys(answers).length;
  const score = quiz ? quiz.questions.filter((q, i) => answers[i] === q.answer_index).length : 0;

  return (
    <div className="wrap section">
      <Reveal><div className="eyebrow">Quiz lab</div><h2 style={{ marginTop: 8 }}>Turn any material into a quiz</h2>
        <p className="muted" style={{ marginTop: 6, maxWidth: 700 }}>Upload a .txt, .md or .pdf — or paste text. Questions are built with classic NLP (sentence ranking + key-term extraction), so every question points back to a sentence in your material.</p></Reveal>
      <div className="card" style={{ marginTop: 20 }}>
        <textarea placeholder="Paste learning material here (at least ~60 words)…" value={text} disabled={!!file} onChange={(e) => setText(e.target.value)} />
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", marginTop: 14 }}>
          <label className="btn ghost small" style={{ cursor: "pointer" }}><Upload size={14} /> {file ? file.name : "Upload file"}
            <input type="file" accept=".txt,.md,.pdf" hidden onChange={(e) => setFile(e.target.files[0] || null)} /></label>
          {file && <button className="btn ghost small" onClick={() => setFile(null)}>Remove file</button>}
          <button className="btn ghost small" onClick={() => { setFile(null); setText(SAMPLE); }}><FileText size={14} /> Use sample text</button>
          <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: ".85rem" }}>Questions
            <select value={n} onChange={(e) => setN(Number(e.target.value))} style={{ width: 80 }}>{[5, 8, 10, 15].map((v) => <option key={v}>{v}</option>)}</select></label>
          <button className="btn" onClick={generate} disabled={busy || (!file && text.trim().split(/\s+/).length < 60)} style={{ marginLeft: "auto" }}>
            {busy ? <><span className="spinner" /> Generating…</> : "Generate quiz"}</button>
        </div>
      </div>
      {err && <div className="note" style={{ marginTop: 16 }}>{err}</div>}
      {quiz && quiz.questions.length === 0 && <div className="note" style={{ marginTop: 16 }}>{quiz.note}</div>}
      {quiz && quiz.questions.length > 0 && (
        <div style={{ marginTop: 26 }}>
          <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8, marginBottom: 12 }}>
            <p className="muted" style={{ fontSize: ".85rem" }}>{quiz.sentences_analysed} sentences analysed · key terms: {quiz.key_terms.slice(0, 6).join(", ")}</p>
            <b>Score: {score} / {quiz.questions.length} ({answered} answered)</b>
          </div>
          <div className="grid">
            {quiz.questions.map((q, i) => {
              const chosen = answers[i], done = chosen !== undefined;
              return (
                <Reveal key={i}><div className="card">
                  <div className="meta"><span className="chip">{q.type}</span><span className="chip sand">{q.difficulty}</span></div>
                  <p style={{ fontWeight: 600, margin: "10px 0 4px" }}>{i + 1}. {q.question}</p>
                  {q.options.map((o, j) => {
                    const cls = !done ? "" : j === q.answer_index ? "right" : j === chosen ? "wrong" : "";
                    return <button key={j} className={`opt ${cls}`} disabled={done} onClick={() => setAnswers({ ...answers, [i]: j })}>
                      {done && j === q.answer_index && <Check size={14} style={{ marginRight: 6 }} />}{done && j === chosen && j !== q.answer_index && <X size={14} style={{ marginRight: 6 }} />}{o}</button>;
                  })}
                  {done && <p className="muted" style={{ fontSize: ".82rem", marginTop: 10 }}>{q.explanation}</p>}
                </div></Reveal>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
