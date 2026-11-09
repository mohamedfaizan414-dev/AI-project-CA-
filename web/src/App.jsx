// App.jsx - the "frame" of the website: top navigation + which page to show.
// We use the URL hash (#dashboard, #courses ...) as a super-simple router, so no extra library is needed.
import { useEffect, useState } from "react";
import { Leaf } from "lucide-react";
import Home from "./pages/Home.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Courses from "./pages/Courses.jsx";
import QuizLab from "./pages/QuizLab.jsx";
import Evidence from "./pages/Evidence.jsx";

const PAGES = [
  ["home", "Overview"],
  ["dashboard", "Learner dashboard"],
  ["courses", "Courses"],
  ["quiz", "Quiz lab"],
  ["evidence", "Evidence"],
];

export default function App() {
  const read = () => (window.location.hash.replace("#", "") || "home");
  const [page, setPage] = useState(read());
  useEffect(() => {
    const onHash = () => { setPage(read()); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = (p) => { window.location.hash = p; };

  return (
    <>
      <header className="nav">
        <div className="wrap nav-inner">
          <button className="brand" onClick={() => go("home")}>
            <span className="brand-mark"><Leaf size={17} /></span><span>CompetencyAI</span>
          </button>
          <nav className="nav-links">
            {PAGES.map(([id, label]) => (
              <button key={id} className={page === id ? "active" : ""} onClick={() => go(id)}>{label}</button>
            ))}
          </nav>
        </div>
      </header>
      <main>
        {page === "home" && <Home go={go} />}
        {page === "dashboard" && <Dashboard />}
        {page === "courses" && <Courses />}
        {page === "quiz" && <QuizLab />}
        {page === "evidence" && <Evidence />}
      </main>
      <footer className="footer">
        <div className="wrap">CompetencyAI · SIH 2026 · PS SIH26101 (MoSPI) · Course data: public Coursera, edX and Udemy datasets. Learners are simulated — see the Evidence page.</div>
      </footer>
    </>
  );
}
