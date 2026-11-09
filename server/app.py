"""
app.py  --  THE BACKEND (a small web server written with FastAPI)
=================================================================
The React website cannot run Python models by itself. So we run this little server.
React asks it questions over HTTP ("here is a learner, what are their gaps?") and it answers
with JSON using the SAME trained models and the SAME feature code we trained with.

Endpoints (all start with /api):
   GET  /api/meta            roles + competencies (to build the forms)
   GET  /api/demo-learners   a few ready-made example learners
   POST /api/analyze         learner data  ->  gaps + recommended learning path
   GET  /api/courses         browse / search the cleaned course catalogue
   GET  /api/reports         all evaluation numbers (for the "Model report" page)
   POST /api/quiz            uploaded material (txt/md/pdf) or pasted text -> MCQs

Run it from the project root:   uvicorn server.app:app --reload --port 8000
"""
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
sys.path.insert(0, str(ML))                       # so we can import framework.py, features.py, recommender.py
sys.path.insert(0, str(ROOT / "server"))          # so we can import quizgen.py

from features import FEATURE_NAMES, build_features, history_counts   # noqa: E402
from framework import COMP_IDS, COMPETENCIES, LEVEL_NAMES, ROLES, ROLE_INDEX, role_requirements  # noqa: E402
from recommender import Catalogue, candidate_table, diversify, gain_matrix, order_as_path, primary_skill  # noqa: E402
import quizgen                                    # noqa: E402

PROC, REP, MOD = ML / "data/processed", ML / "reports", ML / "models"
K = len(COMP_IDS)
CLASS_NAMES = ["No gap", "Moderate gap", "Critical gap"]
COMP_BY_ID = {c["id"]: c for c in COMPETENCIES}

# --------------------------------------------------------------------------- load everything once, at start-up
df = pd.read_csv(PROC / "catalogue_tagged.csv").fillna({"institution": "", "tags": "", "description": ""})
R = np.load(PROC / "course_comp_matrix.npy")
pop = np.load(PROC / "popularity.npy")
cat = Catalogue(df, R, pop)
CID_TO_ROW = {cid: i for i, cid in enumerate(df["course_id"])}
clf = joblib.load(MOD / "gap_classifier.joblib")
reg = joblib.load(MOD / "gap_regressor.joblib")
ranker = joblib.load(MOD / "ranker.joblib")

app = FastAPI(title="CompetencyAI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- helpers
def course_card(c, extra=None):
    row = df.iloc[c]
    d = {
        "course_id": row["course_id"], "title": row["title"], "provider": row["provider"],
        "institution": row["institution"], "url": row["url"], "level": row["level"],
        "hours": round(float(row["hours"]), 1), "hours_estimated": bool(row["hours_imputed"]),
        "rating": None if pd.isna(row["rating"]) else float(row["rating"]),
        "skills": [COMP_BY_ID[t]["name"] for t in str(row["tags"]).split("|") if t in COMP_BY_ID],
        "level_from": int(row["level_from"]), "level_to": int(row["level_to"]),
    }
    if extra:
        d.update(extra)
    return d


class Assessment(BaseModel):
    quiz: Optional[float] = None          # quiz/test score on the 1-5 scale
    self_rating: Optional[float] = None   # what the person says about themselves, 1-5
    supervisor: Optional[float] = None    # what the manager says, 1-5


class AnalyzeRequest(BaseModel):
    role_id: str
    exp_years: float = 3
    education: int = 0                    # 0 graduate, 1 postgraduate, 2 stats/maths/CS degree
    assessments: Dict[str, Assessment] = {}
    completed_course_ids: List[str] = []
    max_courses: int = 8
    hours_budget: Optional[float] = None


def analyze(req_: AnalyzeRequest):
    if req_.role_id not in ROLE_INDEX:
        raise HTTPException(400, "Unknown role")
    role_idx = ROLE_INDEX[req_.role_id]
    req, imp = (np.array(a, float) for a in role_requirements(req_.role_id))
    core = imp >= 1.0

    quiz = np.full(K, np.nan); selfr = np.full(K, np.nan); sup = np.full(K, np.nan)
    for cid, a in req_.assessments.items():
        if cid not in COMP_IDS:
            continue
        k = COMP_IDS.index(cid)
        clip = lambda v: None if v is None else float(np.clip(v, 1, 5))
        if clip(a.quiz) is not None: quiz[k] = clip(a.quiz)
        if clip(a.self_rating) is not None: selfr[k] = clip(a.self_rating)
        if clip(a.supervisor) is not None: sup[k] = clip(a.supervisor)
    completed = [CID_TO_ROW[c] for c in req_.completed_course_ids if c in CID_TO_ROW]
    hist = history_counts([completed], R)

    # ---- 1) GAP MODEL
    X = build_features(np.array([role_idx]), np.array([req_.exp_years]), np.array([req_.education]),
                       req[None], quiz[None], selfr[None], sup[None], hist)[0]      # (K, features)
    Xc = X[core]
    proba = clf.predict_proba(Xc)
    size = np.clip(reg.predict(Xc), 0, 4)
    gap_est = np.zeros(K)
    gap_est[core] = size
    cls = proba.argmax(1)

    gaps = []
    for n, k in enumerate(np.where(core)[0]):
        gaps.append({
            "competency_id": COMP_IDS[k], "name": COMPETENCIES[k]["name"], "group": COMPETENCIES[k]["group"],
            "required": int(req[k]), "estimated_level": round(float(np.clip(req[k] - size[n], 1, 5)), 2),
            "gap_size": round(float(size[n]), 2), "label": CLASS_NAMES[int(cls[n])], "class": int(cls[n]),
            "probabilities": [round(float(p), 3) for p in proba[n]],
            "evidence": {"quiz": None if np.isnan(quiz[k]) else float(quiz[k]),
                         "self": None if np.isnan(selfr[k]) else float(selfr[k]),
                         "supervisor": None if np.isnan(sup[k]) else float(sup[k])},
        })
    gaps.sort(key=lambda g: (-g["class"], -g["gap_size"]))

    # ---- 2) RECOMMENDER
    ids, F = candidate_table(cat, gap_est, req, imp, completed)
    path, projected = [], None
    if len(ids):
        sc = ranker.predict(F)
        prim = primary_skill(cat, ids, gap_est, req, imp)
        ranked = diversify(ids, sc, cat, prim, top_n=max(req_.max_courses, 1))
        score_of = {int(c): float(s) for c, s in zip(ids, sc)}
        if req_.hours_budget:                                  # respect a study-time budget
            keep, total = [], 0.0
            for c in ranked:
                if total + cat.hours[c] <= req_.hours_budget or not keep:
                    keep.append(c); total += cat.hours[c]
            ranked = keep
        gain = gain_matrix(cat, req - gap_est, req) * 2        # in levels
        ordered = order_as_path(ranked, cat, gap_est, req, imp)
        for step, (c, k) in enumerate(ordered, 1):
            covers = [COMPETENCIES[j]["name"] for j in np.where((cat.Rt[c] >= 0.35) & core & (gap_est > 0.3))[0]]
            lvl_gain = float(min(cat.lt[c], req[k]) - max(cat.lf[c], req[k] - gap_est[k]))
            why = (f"Targets your {gaps[[g['competency_id'] for g in gaps].index(COMP_IDS[k])]['label'].lower()} in "
                   f"{COMPETENCIES[k]['name']} (about level {req[k] - gap_est[k]:.1f}, your role needs {int(req[k])}). "
                   f"This {df.iloc[c]['level'].lower()} course moves learners from level {int(cat.lf[c])} to {int(cat.lt[c])}.")
            path.append(course_card(c, {
                "step": step, "primary_skill_id": COMP_IDS[k], "primary_skill": COMPETENCIES[k]["name"],
                "also_covers": [n for n in covers if n != COMPETENCIES[k]["name"]][:3],
                "expected_level_gain": round(max(lvl_gain, 0), 1), "why": why, "rank_score": round(score_of[int(c)], 3),
            }))
        # projected gaps: conservative = only the BEST single course per skill counts
        best = np.zeros(K)
        for c, _ in ordered:
            best = np.maximum(best, (gain[c] * (cat.Rt[c] >= 0.5)))
        new_gap = np.maximum(gap_est - best, 0)
        denom = float(((req - 1) * core).sum())
        projected = {"readiness_now": round(100 * (1 - gap_est[core].sum() / denom), 1),
                     "readiness_after": round(100 * (1 - new_gap[core].sum() / denom), 1)}
    summary = {
        "readiness": round(100 * (1 - gap_est[core].sum() / float(((req - 1) * core).sum())), 1),
        "critical": int((cls == 2).sum()), "moderate": int((cls == 1).sum()), "no_gap": int((cls == 0).sum()),
        "path_hours": round(sum(p["hours"] for p in path), 1), "projected": projected,
        "assessed_skills": int(np.sum(~np.isnan(quiz[core]))),
    }
    return {"role": ROLES[role_idx]["name"], "summary": summary, "gaps": gaps, "path": path}


# --------------------------------------------------------------------------- routes
@app.get("/api/meta")
def meta():
    roles = []
    for r in ROLES:
        req, imp = role_requirements(r["id"])
        roles.append({"id": r["id"], "name": r["name"], "seniority": r["seniority"],
                      "core": [{"id": COMP_IDS[k], "required": req[k]} for k in range(K) if imp[k] >= 1.0]})
    return {"roles": roles, "competencies": [{k: v for k, v in c.items() if k != "keywords"} for c in COMPETENCIES],
            "level_names": LEVEL_NAMES,
            "catalogue_size": int(len(df)),
            "education": [{"value": 0, "label": "Graduate"}, {"value": 1, "label": "Postgraduate"},
                          {"value": 2, "label": "Statistics / Maths / CS degree"}]}


_DEMO = None


@app.get("/api/demo-learners")
def demo_learners():
    """A few simulated test learners (never used for training) so the demo works instantly."""
    global _DEMO
    if _DEMO is None:
        L = pickle.load(open(PROC / "learners.pkl", "rb"))
        names = ["Ananya Iyer", "Rohit Verma", "Meera Nair", "Imran Qureshi", "Lakshmi Reddy", "Arjun Mehta"]
        want_roles = ["stat_officer", "investigator", "data_analyst", "senior_officer", "dissemination", "gis_analyst"]
        out = []
        test = np.where(L["split"] == "test")[0]
        for nm, rid in zip(names, want_roles):
            ri = ROLE_INDEX[rid]
            cand = [i for i in test if L["role_idx"][i] == ri and L["exp"][i] < 12]
            # pick a learner with a realistic MIX of results (about 45% of skills fine, few critical gaps),
            # so the demo is neither "perfect" nor hopeless
            def mix_score(i):
                cls = L["gap_cls"][i][L["imp"][i] >= 1.0]
                return abs((cls == 0).mean() - 0.45) + 0.08 * abs((cls == 2).sum() - 2)
            i = min(cand, key=mix_score)
            ass = {}
            for k in range(K):
                if L["imp"][i][k] < 1.0:
                    continue
                a = {}
                if not np.isnan(L["quiz"][i][k]): a["quiz"] = round(float(L["quiz"][i][k]), 1)
                if not np.isnan(L["selfr"][i][k]): a["self_rating"] = round(float(L["selfr"][i][k]), 1)
                if not np.isnan(L["sup"][i][k]): a["supervisor"] = round(float(L["sup"][i][k]), 1)
                if a: ass[COMP_IDS[k]] = a
            out.append({"name": nm, "role_id": rid, "exp_years": round(float(L["exp"][i]), 1), "education": int(L["edu"][i]),
                        "assessments": ass,
                        "completed": [course_card(c) for c in L["hist"][i]],
                        "completed_course_ids": [df.iloc[c]["course_id"] for c in L["hist"][i]]})
        _DEMO = out
    return _DEMO


@app.post("/api/analyze")
def analyze_route(body: AnalyzeRequest):
    return analyze(body)


@app.get("/api/courses")
def courses(q: str = "", competency: str = "", level: str = "", provider: str = "", limit: int = 40, offset: int = 0):
    mask = np.ones(len(df), bool)
    if q:
        mask &= df["title"].str.contains(q, case=False, regex=False).to_numpy()
    if competency in COMP_IDS:
        mask &= R[:, COMP_IDS.index(competency)] >= 0.5
    if level:
        mask &= (df["level"] == level).to_numpy()
    if provider:
        mask &= (df["provider"] == provider).to_numpy()
    idx = np.where(mask)[0]
    if competency in COMP_IDS:
        idx = idx[np.argsort(-R[idx, COMP_IDS.index(competency)])]
    return {"total": int(len(idx)), "items": [course_card(int(c)) for c in idx[offset:offset + limit]]}


@app.get("/api/reports")
def reports():
    out = {}
    for name in ["cleaning_report", "tagging_report", "gap_model_report", "ranker_report", "recsys_report"]:
        p = REP / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    return out


@app.post("/api/quiz")
async def quiz(file: Optional[UploadFile] = File(None), text: str = Form(""), n: int = Form(8)):
    content = text
    if file is not None:
        content = quizgen.extract_text(file.filename, await file.read())
    if len(content.split()) < 60:
        raise HTTPException(400, "Please provide at least ~60 words of learning material.")
    return quizgen.generate_mcqs(content, n_questions=max(3, min(n, 20)))


# If the React app has been built (npm run build), serve it from the same server -> ONE command to run everything.
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
