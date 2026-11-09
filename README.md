# 🌿 CompetencyAI — Competency Gap Analysis & Training Recommendation

**Smart India Hackathon 2026 · Problem SIH26101 · MoSPI · AI Foundations course project**

> *Officials are told to "upskill" from a huge course catalogue, but nobody tells them **which** courses close **their** gaps.*
> This project finds each person's skill gaps and builds a ranked learning path, then proves with numbers that it beats the "show the most popular courses" approach.

This README is written so that **a person who has never done machine learning can understand every part**. Read it top to bottom once, and you will be able to present and defend the project.

---

## 0. The whole project in 60 seconds (explained like you're ten)

Imagine a school library with 2,000 books about everything. A teacher says to a student: *"Read more!"* That is useless. A **good** teacher would:

1. **Test the student** to see what they are bad at (the *gap*).
2. **Look at what the job needs** (a cricket coach must know fitness; a mathematician must know algebra).
3. **Pick the few books** that fix *this student's* weak spots — not too easy, not too hard — and put them **in a sensible order**.

CompetencyAI is that good teacher, made of three small "brains" (AI models):

| Brain | Question it answers | How it learned |
|---|---|---|
| 🧠 **Course tagger** | "Which skills does each course teach?" | Learned from keyword hints + context words |
| 🧠 **Gap model** | "How far is this person from what their job needs, in each skill?" | Learned from quizzes, self-ratings, manager ratings |
| 🧠 **Ranker** | "Which courses should they take first?" | Learned from what (simulated) experts would choose |

Plus a **Quiz Lab** that turns any uploaded notes into multiple-choice questions.

---

## 1. Honest note (say this to your teacher before he asks)

| Part | Real or simulated? |
|---|---|
| **6,988 courses** (Coursera, edX, Udemy) | ✅ **Real**, public datasets |
| Competency framework & role requirements | ✍️ Written by us, shaped like iGOT Karmayogi's (Behavioural / Functional / Domain) |
| **Learners, assessments, expert learning paths** | 🧪 **Simulated** with explicit rules |

**Why simulated?** Real officials' assessment data and expert-approved paths belong to the government (iGOT) and are not public. The problem statement itself says *"Student-sourced public data"*. So we built a careful simulator — **and kept it separate from the models**, so the models still have to genuinely learn.

**What the numbers mean:** they prove the *pipeline works* and show *which design choices help*. They are **not** a promise of accuracy on real officials. When MoSPI data exists, replace the tables from step 4 and everything else re-runs.

---

## 2. The big picture

```
 RAW DATA (3 real course files)
        │  02_clean_catalogue.py      wash the data (duplicates, junk, wrong labels)
        ▼
 6,988 CLEAN courses
        │  03_tag_courses.py          🧠 #1 course tagger  → "course X teaches SQL, level Beginner"
        ▼
 2,107 courses relevant to statistics work, each tagged with skills
        │
        │   framework.py  (31 competencies, 11 job roles, level needed per role)
        │        │
        │        ▼  04_simulate_learners.py   4,000 simulated people + noisy tests + 3 expert panelists
        │        │
        │        ▼  05_train_gap_model.py     🧠 #2 gap model  → "gap in Survey Design: critical"
        ▼        ▼
   06_train_recommender.py                    🧠 #3 ranker     → "take these 8 courses in this order"
        │
        ▼  07_evaluate.py          the EXAM on 800 unseen people vs. popularity baseline
        │
        ▼  server/app.py (FastAPI) + web/ (React)    the website
```

---

## 3. How to run it

### 3.1 Install (once)

You need **Python 3.10+** and **Node.js 18+**.

```bash
# from the project folder (the one containing this README)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # also downloads the spaCy English model used by the Quiz Lab
```

### 3.2 Easiest run (everything is already trained and built)

```bash
python -m uvicorn server.app:app --port 8000
```
Open **http://localhost:8000** — the website is served by the same server. ✅

### 3.3 Development run (edit the React code and see changes live)

Terminal 1:
```bash
python -m uvicorn server.app:app --port 8000
```
Terminal 2:
```bash
cd web
npm install
npm run dev          # opens http://localhost:5173 (it forwards /api calls to port 8000)
```
To rebuild the production site after edits: `cd web && npm run build`.

### 3.4 Re-train all the AI from scratch (takes ~3–4 minutes)

```bash
cd ml
python run_all.py
```
This re-creates everything in `ml/data/processed`, `ml/models` and `ml/reports`. The random seeds are fixed, so you get the same numbers every time.

---

## 4. Folder map

```
competencyai/
├── README.md               ← you are here
├── requirements.txt        ← Python libraries
├── ml/                     ← ALL THE AI (this is the important part)
│   ├── framework.py            31 competencies + 11 roles (the "answer key" of what jobs need)
│   ├── 02_clean_catalogue.py   data cleaning
│   ├── 03_tag_courses.py       AI #1 – course → skills
│   ├── 04_simulate_learners.py simulated people, assessments, expert paths
│   ├── features.py             turns a person into numbers (shared by training & website)
│   ├── 05_train_gap_model.py   AI #2 – gap classification
│   ├── recommender.py          scoring, ranking, diversity, path order, metrics (shared)
│   ├── 06_train_recommender.py AI #3 – learning-to-rank
│   ├── 07_evaluate.py          final exam + figures
│   ├── run_all.py              runs 02→07
│   ├── data/raw/               the 3 original datasets (untouched)
│   ├── data/processed/         cleaned + tagged data, simulated learners
│   ├── models/                 saved trained models (.joblib)
│   └── reports/                every metric as JSON + figures/*.png for your slides
├── server/
│   ├── app.py                  the Python web server (API)
│   └── quizgen.py              MCQ generator
└── web/                    ← the React website
    └── src/ (App.jsx, pages/, components/, styles.css, api.js)
```

---

## 5. The data (the most important part, explained step by step)

### 5.1 Where it comes from

| File | Courses | What it has | What it lacks |
|---|---|---|---|
| `coursera.csv` | 3,522 | title, university, difficulty, rating, description, skills text | course length, popularity |
| `edx.csv` | 720 | title, difficulty, description | rating, skills, length |
| `udemy.csv` | 3,678 | title, level, **hours**, **subscriber count**, subject | description, rating |

These are widely used public Kaggle datasets. We obtained the files from public GitHub repositories that host copies (`deepvpatel26/In22Labs` for Coursera/edX, `Jeevan-Sannapaneni/udemy-courses-analysis` for Udemy). **Check each dataset's licence on Kaggle before publishing the code publicly.**

**Why these three?** We looked for datasets that fit official statistics (SQL, Python, statistics, Excel, communication, leadership…). Coursera and edX cover data/statistics/management topics; Udemy adds Excel, finance and business courses plus real subscriber counts. Datasets we **rejected**: the teammate's file (6 made-up courses); O\*NET / ESCO (the real occupation-skill databases — ideal, but their websites were not reachable from our build environment; plugging them in is listed as future work).

### 5.2 What was wrong with the data (we *looked* before cleaning)

| Problem we found | Evidence | Fix in `02_clean_catalogue.py` |
|---|---|---|
| Three different column layouts | different names for the same thing | rename into **one schema** |
| Out-of-domain courses | 680 Udemy *Musical Instruments* courses | dropped |
| Duplicates | 105 same-URL rows; 22 same-title; 66 near-identical titles | 3-level dedupe (exact URL → exact title → **TF-IDF similarity ≥ 0.92**) |
| Non-English text | 57 courses with mostly non-Latin characters | dropped |
| Junk in text | HTML tags, links, coupon codes, odd unicode, broken `�` characters | `clean_text()` |
| No real description | 2 courses | dropped |
| Messy difficulty labels | `Conversant`, `Not Calibrated`, `All Levels`, `Expert Level` | mapped to Beginner / Intermediate / Advanced; if unknown, **guessed from title words** and **flagged** (`level_inferred`, 24% of rows) |
| Ratings as text | `"Not Calibrated"` | converted to numbers; unknown stays *missing* (not faked) |
| Missing course length | Coursera/edX have none | read "10 hours"/"4 weeks" from text; otherwise typical value per level, **flagged** `hours_imputed` |

Result: **7,240 raw rows → 6,988 clean courses** (`reports/cleaning_report.json` records every step).

> 💡 Rule we followed: **never silently invent data.** Every guess carries a flag, and the website shows "(est.)" for estimated hours.

### 5.3 Dataset selection by relevance

A course on screenwriting is useless for a statistical officer. After tagging (section 7) we keep only courses that clearly teach at least one of our 31 competencies: **6,988 → 2,107** (Coursera 1,503 · Udemy 382 · edX 222).

---

## 6. The competency framework (`framework.py`)

* **31 competencies** in three groups, mirroring iGOT Karmayogi's idea: **Domain** (e.g. Survey Design & Sampling), **Functional** (e.g. SQL, Python), **Behavioural** (e.g. Leadership).
* **11 roles** (Statistical Investigator … Deputy Director). Each role lists the **level (1–5)** it needs per competency.
  * Level words: 1 Aware · 2 Basic · 3 Working · 4 Advanced · 5 Expert.
  * Competencies *not* listed for a role are "nice to have": level 1, importance 0.3.
* Every competency has **keywords** (e.g. `spreadsheets` → *excel, pivot, google sheets*). They are only a starting hint for the tagger.

Key lines to understand:
```python
required = [role["needs"].get(cid, 1) for cid in COMP_IDS]     # level needed, default 1
importance = [1.0 if cid in role["needs"] else 0.3 for cid in COMP_IDS]
```
"For every competency, look up what the role needs; if the role doesn't mention it, say level 1 with low importance."

---

## 7. AI #1 — the course tagger (`03_tag_courses.py`)

**Problem:** nobody labelled "this course teaches SQL". Labelling 7,000 courses by hand is unrealistic.

**Idea: weak supervision with two independent views.**

**View 1 — rules (labelling functions).** Four small judges each vote, scores range 0–1:
1. keyword in the **title** (most trusted),
2. keywords in the **skills** text,
3. keywords in the **description**,
4. **TF-IDF similarity** between the course and the competency's description.

They are combined with a **noisy-OR**: `score = 1 − (1−a)(1−b)(1−c)(1−d)`. Meaning: high if *any* judge is confident, higher if several agree.

*Ambiguous words* (like "survey", "policy", "writing") only count 60% in titles — otherwise a "Charts in Google Sheets" course containing the word "survey" would be tagged *Survey Design*.

**View 2 — context classifier.** A **logistic regression** is trained to predict View 1's confident labels, but **all keywords are deleted from its input text**. To still get the right answer it must learn *other* words that go with a topic (e.g. "cells, formulas, worksheet" → spreadsheets). This is real learning that **generalises beyond keywords**.

Honest evaluation: with **5-fold out-of-fold prediction** (each course is predicted by a model that never saw it), the keyword-blind classifier re-discovers the rule labels with **mean F1 = 0.60**. If it only memorised keywords it would score ≈ 0.

**Blend:** `R = 0.6 × rules + 0.4 × context`, plus a +0.15 bonus when both views agree. `R[course, skill]` ∈ 0..1 says *how strongly the course teaches the skill*.

Each course also gets a **level range**: Beginner takes you from level 1→2, Intermediate 2→4, Advanced 3→5 *(an explicit assumption)* and a **quality** score from its rating (neutral if unrated).

> ⚠️ Limitation: tags are weak labels, not human labels. `ml/data/processed/gold_review_sample.csv` has 80 random courses so your team can hand-label them and measure true tagger accuracy — a great extra for the viva.

---

## 8. The simulator (`04_simulate_learners.py`)

For each of **4,000** people we create:

1. **Who:** role, years of experience, education.
2. **The hidden truth:** a true level (1–5) for every competency. People near their role's requirement minus a "still learning" shortfall that shrinks with experience; related skills move together (a hidden "digital talent" lifts SQL *and* Python).
3. **What we observe (noisy, like real life):**
   * **Quiz score** = truth + noise — but only for ~55% of competencies (nobody tests everything).
   * **Self-rating** = truth + bias — weak people *over*-rate themselves (the **Dunning–Kruger effect**), plus a personal over-confidence trait.
   * **Manager rating** = truth + small noise, for ~40% of competencies.
4. **History:** courses already completed (65% chosen because they need the skill, 35% "herding" onto popular courses). Udemy popularity uses **real subscriber counts**; for Coursera/edX a propensity is simulated.
5. **Expert panel (the answer key):** three simulated experts each pick the best **10** courses using the **true** gaps and the **strict rule tags** (not our learned tags). The value of a course = how much of the person's gap it closes (right skill *and* right level), nudged by rating and length. Experts disagree a little (12% noise). A course's **grade** = number of experts who picked it (1–3).

The **gap truth** is `max(0, required − true level)`, turned into 3 classes:
`0 = no gap (<0.5 level)`, `1 = moderate (0.5–1.5)`, `2 = critical (≥1.5)`.
In our data: 39% none / 47% moderate / 14% critical (core skills).

**Splits:** 60% train / 20% validation / 20% test **by person** — nobody appears in two splits.

> Why it's fair: the experts use *true* gaps and *rule* tags. Our system sees only *noisy* evidence and *learned* tags. Different information on both sides.

---

## 9. AI #2 — the gap model (`features.py` + `05_train_gap_model.py`)

**Task:** for each person and each job-critical skill, predict **no / moderate / critical gap** from noisy evidence. For ~45% of skills there is **no quiz**, so the model must **infer from related skills**.

### 9.1 Features (`features.py`) — turning a person into numbers

For every (person, skill) row we compute 23 numbers, in five families:

| Family | Examples | Why |
|---|---|---|
| Job & person | required level, role, experience, education | tells what is expected |
| Direct evidence | quiz, self, manager scores, and `score − required` | the strongest clues |
| Indirect evidence | average quiz/self/manager score of **other skills in the same group**, average shortfall elsewhere | **infers untested skills** |
| Bias correction | `overconf = mean(self − quiz)` on other skills; `self_corrected = self − overconf` | un-does the over-rater's inflation |
| History | number of finished courses on this skill | learning already done |

Missing values stay `NaN` (unknown) — never filled with fake numbers; gradient boosting handles them natively.
*The same file is used by the website*, so training and live predictions can never disagree.

### 9.2 Models compared (validation macro-F1)

| Model | Validation macro-F1 |
|---|---|
| Logistic Regression | 0.716 |
| Random Forest | 0.730 |
| **Gradient Boosting** ← winner | **0.743** |

We chose the winner **on validation only**, then evaluated on test **once**.

### 9.3 Test results (11,739 held-out rows)

| System | Macro-F1 |
|---|---|
| Majority class | 0.209 |
| Rule: trust self-rating | 0.535 |
| Rule: quiz, else self-rating (**baseline**) | 0.627 |
| **Gradient Boosting (ours)** | **0.745** |

* Per-class F1: none 0.810 · moderate 0.755 · critical 0.672. Quadratic kappa 0.732.
* **Gap size** (in levels) MAE: **0.24** vs 0.37 for the rule.
* For skills with **no quiz**: **0.698** vs 0.536 for the rule → the model really infers from related skills. With a quiz: 0.783.

### 9.4 Ablation — what is each evidence source worth?

| Information used (cumulative) | Validation F1 |
|---|---|
| Role only | 0.355 |
| + experience / education | 0.383 |
| + self-rating | 0.653 |
| + manager rating | 0.691 |
| + quiz scores | 0.741 |
| + course history | 0.740 (no gain — honest result) |

Reading: self-ratings are the cheapest big win; quizzes are the best single source; history adds nothing *in this simulator*.

---

## 10. AI #3 — the recommender (`recommender.py` + `06_train_recommender.py`)

### 10.1 No cheating: out-of-fold gaps
The recommender must use the gap model's **estimates**. But the gap model was trained on the training people, so its predictions *for them* are unrealistically good. Fix: **5-fold out-of-fold prediction** grouped by person, so every training person's gap is predicted by a model that never saw them.

### 10.2 The pipeline for one person
1. **Coverage score** per course: `Σ over skills ( importance × how much the course teaches it × how much of the gap it can close )`. A course that starts more than one level above the person is skipped.
2. **Shortlist** the top 80 courses.
3. **Learned ranker:** each (person, course) pair gets 15 numbers (coverage, share of total gap, level mismatch, rating, length, popularity, …). A **gradient-boosting regressor** learns to predict the **number of expert votes** (0–3) — this is *pointwise learning-to-rank*. Three hyper-parameter sets were tried; the best on **validation** NDCG@10 won (0.542).
4. **Diversify:** after choosing a course for a skill, the next course for the *same* skill loses score (`penalty = 1.2`), so you don't get 8 SQL courses. The penalty was tuned on **validation**: more diversity costs ~2 points of expert agreement but gains ~5 points of real gap coverage (table in `recommender.py`).
5. **Order as a path:** biggest gaps first; inside one skill, beginner → advanced.

### 10.3 Metrics (explained simply)

| Metric | Plain meaning |
|---|---|
| **Recall@K** | The experts chose N courses. How many of them are in our top K? (1.0 = all) |
| **NDCG@K** | Like recall but rewards putting courses *more experts agreed on* **higher** in the list (1.0 = perfect order) |
| **Gap coverage@10** | Of the person's *real* gap, how much has at least one useful course (right skill **and** level) in our top 10? — an outcome metric that doesn't rely on the experts |
| **Macro-F1** | For classification: average of F1 over the 3 classes (so rare "critical" counts equally) |

*Relevant* = chosen by **≥ 2 of 3** experts. 95% intervals come from **bootstrap** (resample the 800 test people 1,000 times).

### 10.4 Results — 800 unseen test learners

| System | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | Gap cov.@10 |
|---|---|---|---|---|---|
| Random | 0.004 | 0.006 | 0.011 | 0.007 | 0.146 |
| **Popularity (required baseline)** | 0.005 | 0.008 | 0.036 | 0.015 | 0.272 |
| Popularity within job skills | 0.007 | 0.019 | 0.051 | 0.023 | 0.435 |
| Ours: heuristic only | 0.163 | 0.256 | 0.409 | 0.276 | **0.746** |
| **Ours: learned ranker (final)** | **0.355** | **0.475** | **0.600** | **0.537** | 0.737 |
| Reference: same ranker with *true* gaps | 0.426 | 0.583 | 0.714 | 0.650 | 0.766 |

* Final vs popularity: **+0.467 recall@10** (95% CI 0.452–0.482), **+0.522 NDCG@10**, **+0.464 gap coverage**.
* The learned ranker agrees far more with experts (0.475 vs 0.256 recall@10) while covering about the same real gaps (0.737 vs 0.746) — it learned *expert taste* (quality, length, level fit), not just coverage.
* The last row shows what **imperfect gap inference costs** (0.475 → 0.583).

**How to read this honestly:** popularity is a weak baseline here because it ignores the person completely, so the huge margin is expected. The experts are simulated, so absolute numbers are optimistic. The solid findings are the **ordering** of systems, the ablation, and the ~0.1 gap between estimated and true gaps.

---

## 11. Quiz Lab (`server/quizgen.py`) — MCQs from uploaded material

Required by the problem statement: *"generating quizzes and MCQs from uploaded learning materials"*. No paid AI is used — everything is explainable:

1. **spaCy** splits sentences and tags each word's part of speech.
2. **TextRank** (PageRank for sentences): sentences similar to many others are central → important.
3. **Key terms** = noun phrases ranked by TF-IDF (common here, not an everyday phrase). Singular/plural merge via the word's lemma.
4. **Question:** "definition" style if the sentence looks like *"X is/refers to …"*, otherwise **fill-in-the-blank** (hide the best key term).
5. **Distractors:** other key terms that appear in *similar* sentences (co-occurrence) but **not** in this sentence.
6. Accepts `.txt`, `.md`, `.pdf` (via `pypdf`) or pasted text. Every question shows the source sentence as its explanation.

**Limitations:** it is classic NLP, not an LLM — questions are mostly recall-style, and distractor quality improves with longer material (short texts have few repeated terms). It could be upgraded by adding an LLM later.

---

## 12. The website (`web/`) — explained

* **React** builds the screens; **Vite** is the tool that runs/builds it; **framer-motion** gives subtle animations; **lucide-react** gives icons.
* **Design:** forest greens + sand beige; a fixed paper-grain texture made from an SVG noise filter (no image file); topographic contour lines in the hero; Fraunces (serif) headings + Manrope body.
* **Pages** (switched by the `#hash` in the URL, no router library): Overview · Learner dashboard · Courses · Quiz lab · Evidence.
* **Dashboard flow** (`pages/Dashboard.jsx`): pick an example learner or "Build your own" → every change waits 250 ms, then `POST /api/analyze` → server returns gaps + path → **GapBars** (bar = estimated level, black marker = required level, Q/S/M badges = which evidence existed) and **PathTimeline**.
* **Evidence page** reads the JSON reports, so it can never show stale numbers.

### How to read a React component (example: `CountUp.jsx`)
```jsx
const [shown, setShown] = useState(0);        // "shown" is the number on screen; starts at 0
useEffect(() => { ... }, [value]);            // run this code whenever "value" changes
const eased = 1 - Math.pow(1 - t, 3);         // fast at first, slow at the end → feels natural
```

---

## 13. API reference (`server/app.py`)

| Endpoint | Does |
|---|---|
| `GET /api/meta` | roles, competencies, education options |
| `GET /api/demo-learners` | 6 example test learners (never used for training) |
| `POST /api/analyze` | learner data → `{summary, gaps[], path[]}` |
| `GET /api/courses?q=&competency=&level=&provider=` | browse the catalogue |
| `GET /api/reports` | all evaluation JSON |
| `POST /api/quiz` | file or text → MCQs |

Interactive docs: **http://localhost:8000/docs**.

---

## 14. Glossary (kid-friendly)

* **Competency** — a skill needed for a job (e.g. "SQL").
* **Gap** — how far you are below what the job needs.
* **Feature** — one number describing something (e.g. "quiz score minus required level").
* **Training / validation / test** — study material / practice exam (to choose the best model) / final exam (used once).
* **Overfitting** — memorising the practice questions instead of understanding.
* **Out-of-fold (OOF)** — predicting each item with a model that never saw it.
* **TF-IDF** — a number saying how special a word is to one text (frequent here, rare elsewhere).
* **Logistic regression** — draws a line that separates "yes" from "no" and gives a probability.
* **Gradient boosting** — many small decision trees, each fixing the previous ones' mistakes.
* **Weak supervision** — training on rough, automatically made labels instead of perfect hand labels.
* **Learning to rank** — teaching a model to put the best items at the top of a list.
* **Ablation** — removing/adding one ingredient to see what it is worth.
* **Bootstrap CI** — re-sampling the test people many times to see how much the result could wobble.
* **Dunning–Kruger** — people who know little tend to over-rate themselves.

---

## 15. Presentation guide (5–7 minutes)

1. **Problem (30 s):** "A huge catalogue, no way to know which course closes *my* gap."
2. **Data (60 s):** show `reports/figures/cleaning_funnel.png` — three real datasets, 7,240 → 6,988 clean → 2,107 relevant. Mention what you *rejected* and why.
3. **Course tagger (45 s):** "No labels existed, so rules give noisy labels and a keyword-blind classifier learns the context — F1 0.60."
4. **Gap model (60 s):** `gap_ablation.png` + `gap_confusion.png` — F1 0.745 vs 0.627 baseline; works even without a quiz (0.698 vs 0.536).
5. **Recommender (60 s):** `recommender_comparison.png` — recall@10 0.475 vs popularity 0.008.
6. **Live demo (90 s):** Dashboard → pick Ananya → gaps → path → change the study budget to 40 hours → Quiz Lab with sample text.
7. **Honesty slide (30 s):** simulated learners; what changes with real iGOT data.

### Likely questions from sir — and answers

**Q: Why simulated learners? Isn't that fake?**
Real iGOT data isn't public; the PS says "student-sourced public data". The courses are real. The simulator is explicit, separate from the models, and the same code runs on real data when available. We state clearly that results show the pipeline works, not real-world accuracy.

**Q: Isn't it circular — you built the experts and the model?**
Partly, and we say so. We reduced it: experts use *true* hidden gaps and *strict rule tags*; the system sees *noisy* evidence and *learned tags*. The comparison "ranker with true gaps vs estimated gaps" quantifies the difference. The strongest evidence is the *relative* comparison and the ablation, not the absolute score.

**Q: Why gradient boosting?** It won on validation (0.743 vs 0.730 RF vs 0.716 LR), handles missing values natively, and captures interactions like "quiz is high but self-rating is also very high".

**Q: What stops overfitting / leakage?** Splits by person; model choice on validation only; test used once; out-of-fold predictions for the recommender's training; `features.py` excludes the current skill when averaging related skills.

**Q: Why F1 and not accuracy?** The "critical" class is rare (14%); macro-F1 averages per-class F1 so it can't be ignored.

**Q: Why NDCG?** Ranking order matters; NDCG rewards putting more-agreed courses higher.

**Q: What would you do with real data?** Replace `catalogue_tagged.csv` with the iGOT catalogue, `learners.pkl` with real assessments and mentor-approved paths, re-run `run_all.py`.

**Q: How is iGOT Karmayogi integrated?** No public API exists, so there is no live connection. The catalogue and competency framework follow iGOT's structure; an adapter that replaces `catalogue_tagged.csv` is the integration point.

---

## 16. Known limitations & future work

1. Learners and expert paths are simulated (see section 1).
2. Course tags are weak labels — hand-label `gold_review_sample.csv` to measure real accuracy.
3. Catalogue is general-purpose; coverage is thin for some skills (survey design 22 courses, official quality 16, geospatial 18, research methods 18). "Finance, Budgeting & Accounting" is large (468) and corporate-finance heavy.
4. Course hours are mostly estimated for Coursera/edX (flagged "est.").
5. Quiz generator is classic NLP; questions are recall-style.
6. Future: real O\*NET / ESCO occupation-skill data; sentence-embedding models for tagging; real iGOT integration; difficulty-aware quizzes; per-user progress tracking.

## 17. Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard says "Could not reach the Python server" | Start it: `python -m uvicorn server.app:app --port 8000` |
| `spaCy model missing` in Quiz Lab | `python -m spacy download en_core_web_sm` |
| `ModuleNotFoundError` | activate the virtual environment, then `pip install -r requirements.txt` |
| Fonts look plain | Google Fonts needs internet; the site falls back to system fonts |
| Port already in use | use `--port 8001` (and edit `web/vite.config.js` proxy if using `npm run dev`) |
