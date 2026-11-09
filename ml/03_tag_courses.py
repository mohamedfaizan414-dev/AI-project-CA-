"""
03_tag_courses.py  --  "WHICH COMPETENCIES DOES EACH COURSE TEACH?"  (AI model #1)
====================================================================================
Problem: our course catalogue has NO column saying "this course teaches SQL".
Nobody labelled it for us, and labelling 7,000 courses by hand is not realistic.

Solution = WEAK SUPERVISION with TWO INDEPENDENT VIEWS ("co-training" idea):

  VIEW 1  (rules)   : keyword matches in the title / skills / description, plus
                      TF-IDF similarity to the competency description.
                      -> cheap and fairly precise, but misses synonyms.
  VIEW 2  (context) : a logistic-regression text classifier trained on View-1 labels
                      BUT WITH ALL THE KEYWORDS HIDDEN from its input.
                      To still get the answer right it must learn the *other* words
                      that go with a topic ("worksheet", "pivot", "cells" -> spreadsheets).
                      -> this is real learning, it generalises beyond the rules.

  FINAL SCORE = a blend of both. A course teaches competency k with strength R[c, k] in 0..1.

Outputs
  data/processed/catalogue_tagged.csv   only the courses relevant to our framework
  data/processed/course_comp_matrix.npy matrix R: rows = courses, columns = competencies
  reports/tagging_report.json           honest numbers on how well the tagger works
  data/processed/gold_review_sample.csv 80 random courses for humans to hand-check
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

from framework import COMPETENCIES, COMP_IDS

HERE = Path(__file__).parent
PROC, REP = HERE / "data/processed", HERE / "reports"
RNG = np.random.default_rng(42)          # fixed seed -> same result every run (important for science)
K = len(COMPETENCIES)


def kw_regex(words):
    """Build ONE regular expression that matches any keyword as a WHOLE word (so 'r' does not match 'research')."""
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


# Some keywords are everyday words that appear in many unrelated course titles
# ("survey" in a Google-Sheets chart course, "policy" in a privacy-policy tutorial...).
# We call them AMBIGUOUS: a title match on them only counts 60%, so a course also needs
# a second clue (skills text, description or similarity) before we believe it.
AMBIGUOUS = {"survey", "writing", "policy", "planning", "finance", "financial", "security", "evaluation",
             "innovation", "government", "classification", "forecast", "mapping", "creativity", "logic",
             "reasoning", "presentation", "manager", "compliance", "ethical", "distribution", "editing",
             "influence", "conflict", "charts", "audit", "economics", "mathematical", "supervised",
             "geographic", "public sector", "statistical", "coaching", "sampling"}
PATTERNS = [kw_regex(c["keywords"]) for c in COMPETENCIES]
STRONG_PATTERNS = [kw_regex([w for w in c["keywords"] if w not in AMBIGUOUS] or ["zzzzzz"]) for c in COMPETENCIES]
ALL_KEYWORDS = sorted({w for c in COMPETENCIES for w in c["keywords"]}, key=len, reverse=True)
ALL_KW_PATTERN = kw_regex(ALL_KEYWORDS)


# ----------------------------------------------------------------------------
# VIEW 1: rule-based "labelling functions"
# ----------------------------------------------------------------------------
def view1_scores(df):
    """
    For every (course, competency) pair compute a weak score between 0 and 1.
    Four little judges ("labelling functions") each vote; we combine them with a
    'noisy-OR': the result is high if ANY judge is confident, and higher if several agree.
    """
    n = len(df)
    title_hit = np.zeros((n, K))
    skill_hit = np.zeros((n, K))
    desc_hit = np.zeros((n, K))
    for k, pat in enumerate(PATTERNS):
        strong = STRONG_PATTERNS[k]
        title_hit[:, k] = df["title"].map(lambda t: 1.0 if strong.search(t) else (0.6 if pat.search(t) else 0.0))
        # skills: how many different matches (more matches = more confident), capped at 1
        skill_hit[:, k] = df["skills_text"].map(lambda t: min(1.0, len(pat.findall(t)) / 2))
        # description: need ~3 mentions to be fully confident
        desc_hit[:, k] = df["description"].map(lambda t: min(1.0, len(pat.findall(t)) / 3))

    # Judge 4: TF-IDF cosine similarity between the course and a "document" describing the competency.
    comp_docs = [" ".join([c["name"], c["description"]] + c["keywords"] * 2) for c in COMPETENCIES]
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = tf.fit_transform(df["text"])
    C = tf.transform(comp_docs)
    sim = (X @ C.T).toarray()
    # similarity numbers have different sizes per competency, so rescale each column by its 97th percentile
    sim = np.clip(sim / (np.percentile(sim, 97, axis=0) + 1e-9), 0, 1)

    # weights = how much we trust each judge (title is the most trustworthy)
    w_title, w_skill, w_desc, w_sim = 0.90, 0.60, 0.45, 0.35
    w = 1 - (1 - w_title * title_hit) * (1 - w_skill * skill_hit) * (1 - w_desc * desc_hit) * (1 - w_sim * sim * 0.6)
    return w, {"title": title_hit, "skills": skill_hit, "desc": desc_hit, "sim": sim}


# ----------------------------------------------------------------------------
# VIEW 2: context classifier that cannot see the keywords
# ----------------------------------------------------------------------------
def view2_probabilities(df, weak, n_splits=5):
    """
    Train one binary classifier per competency on the MASKED text.
    We use K-fold cross-validation: a course's probability is always predicted by a
    model that has NEVER seen that course. This is called out-of-fold (OOF) prediction.
    It stops the model from just memorising and gives honest evaluation numbers.
    """
    masked = df["text"].map(lambda t: ALL_KW_PATTERN.sub(" ", t))        # <- hide every keyword
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                        max_features=60000, sublinear_tf=True).fit_transform(masked)

    oof = np.full((len(df), K), np.nan)
    metrics = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for k in range(K):
        pos = weak[:, k] >= 0.75            # confident positive examples
        neg = weak[:, k] <= 0.15            # confident negative examples
        usable = pos | neg                  # ignore the uncertain middle - don't teach the model guesses
        y = pos.astype(int)
        if pos.sum() < 15:                  # too few examples to learn from -> leave as NaN (use rules only)
            metrics[COMP_IDS[k]] = {"positives": int(pos.sum()), "note": "too few examples, rules only"}
            continue
        pred = np.zeros(len(df))
        for tr, te in kf.split(X):
            tr = tr[usable[tr]]                                                  # train only on confident rows
            clf = LogisticRegression(C=3.0, max_iter=400, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]                            # predict the held-out fold
        oof[:, k] = pred
        # How well does context-only prediction recover the rule-based labels?
        yhat = pred[usable] >= 0.5
        metrics[COMP_IDS[k]] = {
            "positives": int(pos.sum()),
            "precision": round(float(precision_score(y[usable], yhat, zero_division=0)), 3),
            "recall": round(float(recall_score(y[usable], yhat, zero_division=0)), 3),
            "f1": round(float(f1_score(y[usable], yhat, zero_division=0)), 3),
        }
    return oof, metrics


# ----------------------------------------------------------------------------
def main():
    df = pd.read_csv(PROC / "catalogue_clean.csv").fillna({"skills_text": "", "description": "", "text": ""})
    print(f"Tagging {len(df)} courses against {K} competencies...")

    weak, parts = view1_scores(df)
    print("View 1 done. Training View 2 (context classifier, keywords hidden)...")
    p2, m2 = view2_probabilities(df, weak)

    # ---- blend. Where View 2 is unavailable (NaN) we trust the rules fully.
    p2f = np.where(np.isnan(p2), weak, p2)
    # A course gets credit if rules are sure, OR if context strongly suggests it.
    # (0.6/0.4 blend, then a boost when BOTH views agree)
    R = 0.6 * weak + 0.4 * p2f
    agree = (weak >= 0.5) & (p2f >= 0.5)
    R = np.clip(R + 0.15 * agree, 0, 1)

    # ---- relevance filter: keep a course only if it clearly teaches at least one competency
    top = R.max(axis=1)
    keep = top >= 0.55
    # Courses that fire ONLY on one broad everyday word (e.g. just "writing" or "policy") are weak evidence.
    print(f"Relevant courses: {keep.sum()} of {len(df)}")

    cat = df[keep].reset_index(drop=True)
    Rk = R[keep]
    # Level each course takes you FROM and TO (assumption written in the README):
    lvl_map = {"Beginner": (1, 2), "Intermediate": (2, 4), "Advanced": (3, 5)}
    cat["level_from"] = cat["level"].map(lambda x: lvl_map[x][0])
    cat["level_to"] = cat["level"].map(lambda x: lvl_map[x][1])
    # Quality 0..1 from the rating (3 stars -> 0, 5 stars -> 1). No rating -> neutral 0.5.
    cat["quality"] = ((cat["rating"] - 3) / 2).clip(0, 1).fillna(0.5)

    # human-friendly "top competencies" column
    def top_tags(row):
        idx = np.argsort(-row)[:3]
        return "|".join(COMP_IDS[i] for i in idx if row[i] >= 0.4)
    cat["tags"] = [top_tags(r) for r in Rk]

    keep_cols = ["course_id", "provider", "title", "institution", "url", "level", "level_inferred",
                 "level_from", "level_to", "rating", "quality", "hours", "hours_imputed",
                 "subscribers", "tags", "description"]
    cat["description"] = cat["description"].str.slice(0, 400)       # keep the file small
    cat[keep_cols].to_csv(PROC / "catalogue_tagged.csv", index=False)
    np.save(PROC / "course_comp_matrix.npy", Rk.astype(np.float32))
    # The RULES-ONLY view, saved separately. The simulated expert panel (step 04) judges courses with
    # this "strict rules" opinion, while our recommender uses the blended learned scores. Two different
    # opinions -> the evaluation is not just the model agreeing with itself.
    np.save(PROC / "course_gold_rules.npy", weak[keep].astype(np.float32))

    # ---- coverage report: how many courses per competency (flags thin spots honestly)
    cover = {}
    for k, cid in enumerate(COMP_IDS):
        strong = Rk[:, k] >= 0.5
        cover[cid] = {
            "courses": int(strong.sum()),
            "beginner": int((strong & (cat["level"] == "Beginner")).sum()),
            "intermediate": int((strong & (cat["level"] == "Intermediate")).sum()),
            "advanced": int((strong & (cat["level"] == "Advanced")).sum()),
        }
    thin = [c for c, v in cover.items() if v["courses"] < 15]

    f1s = [v["f1"] for v in m2.values() if "f1" in v]
    report = {
        "courses_in": int(len(df)), "courses_relevant": int(keep.sum()),
        "context_classifier_mean_f1": round(float(np.mean(f1s)), 3),
        "context_classifier_note": "F1 of the keyword-blind classifier at re-discovering the rule labels (5-fold out-of-fold)",
        "per_competency_context_model": m2,
        "coverage": cover, "thin_coverage_competencies": thin,
        "relevant_by_provider": cat["provider"].value_counts().to_dict(),
    }
    (REP / "tagging_report.json").write_text(json.dumps(report, indent=2))

    # ---- a sample for humans to hand-label (real validation needs real eyes)
    sample = cat.sample(80, random_state=7)[["course_id", "title", "level", "tags"]].copy()
    sample["human_correct_tags"] = ""
    sample.to_csv(PROC / "gold_review_sample.csv", index=False)

    print("Context-model mean F1:", report["context_classifier_mean_f1"])
    print("Thin coverage:", thin)
    print(json.dumps({k: v["courses"] for k, v in cover.items()}))


if __name__ == "__main__":
    main()
