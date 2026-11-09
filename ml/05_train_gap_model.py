"""
05_train_gap_model.py  --  "WHERE IS THIS PERSON WEAK?"  (AI model #2: the GAP MODEL)
=====================================================================================
Task (a classification problem with 3 classes), for every (person, job-critical skill):
      0 = no gap     1 = moderate gap (0.5-1.5 levels short)     2 = critical gap (>=1.5 levels short)
The TRUE gap is hidden. The model sees only noisy evidence (quizzes, self-ratings, supervisor
ratings, role requirement, experience, history) - and for ~45% of skills there is NO quiz at all,
so it must INFER from related skills. That inference is where machine learning earns its keep.

We compare:
   * 3 dumb baselines (majority class, "trust self-rating", "trust quiz else self-rating")
   * 3 learned models (Logistic Regression, Random Forest, Gradient Boosting)
We pick the winner on the VALIDATION set only, then report the TEST set once (no peeking).
We also run an ABLATION: add one information source at a time to show what each one is worth.
"""
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, SOURCES, build_features, history_counts
from framework import COMP_IDS

HERE = Path(__file__).parent
PROC, REP, MOD = HERE / "data/processed", HERE / "reports", HERE / "models"
MOD.mkdir(exist_ok=True)
SEED = 42
CLASS_NAMES = ["No gap", "Moderate gap", "Critical gap"]


def to_class(gap):
    return np.where(gap >= 1.5, 2, np.where(gap >= 0.5, 1, 0))


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro"))


def main():
    L = pickle.load(open(PROC / "learners.pkl", "rb"))
    R = np.load(PROC / "course_comp_matrix.npy")
    N, K = L["req"].shape

    hist_cnt = history_counts(L["hist"], R)
    X3 = build_features(L["role_idx"], L["exp"], L["edu"], L["req"], L["quiz"], L["selfr"], L["sup"], hist_cnt)

    # We only predict gaps for the person's JOB-CRITICAL competencies (importance == 1).
    core = L["imp"] >= 1.0
    person = np.repeat(np.arange(N)[:, None], K, 1)[core]          # which learner each row belongs to
    X = X3[core]                                                    # (rows, features)
    y = L["gap_cls"][core]
    y_gap = L["gap"][core]
    split = L["split"][person]
    has_quiz = ~np.isnan(L["quiz"][core])
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"{len(y)} rows | train {tr.sum()} val {va.sum()} test {te.sum()} | with quiz {has_quiz.mean():.0%}")

    # ------------------------------------------------------------------ baselines
    req = L["req"][core]
    selfr, quiz = L["selfr"][core], L["quiz"][core]
    rule_self = to_class(np.maximum(req - np.nan_to_num(selfr, nan=req), 0))          # trust self-rating
    rule_qs = to_class(np.maximum(req - np.where(np.isnan(quiz), np.nan_to_num(selfr, nan=req), quiz), 0))
    majority = np.full(len(y), np.bincount(y[tr]).argmax())
    baselines = {"Majority class": majority, "Rule: trust self-rating": rule_self, "Rule: quiz else self-rating": rule_qs}

    # ------------------------------------------------------------------ learned models
    cat_idx = [FEATURE_NAMES.index("role_idx"), FEATURE_NAMES.index("comp_idx")]   # these two are CATEGORIES, not quantities
    models = {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=600, C=1.0)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0,
            categorical_features=cat_idx, early_stopping=True, validation_fraction=0.1,
            random_state=SEED),
    }
    val_scores, fitted = {}, {}
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        val_scores[name] = macro_f1(y[va], m.predict(X[va]))
        fitted[name] = m
        print(f"  {name:<22} validation macro-F1 = {val_scores[name]:.3f}")
    best = max(val_scores, key=val_scores.get)
    print("Winner on validation:", best)
    clf = fitted[best]

    # ------------------------------------------------------------------ the one-time TEST report
    def report(pred, mask):
        return {
            "macro_f1": round(macro_f1(y[mask], pred[mask]), 4),
            "accuracy": round(float(accuracy_score(y[mask], pred[mask])), 4),
            "per_class_f1": [round(float(v), 4) for v in f1_score(y[mask], pred[mask], average=None, labels=[0, 1, 2])],
            "quadratic_kappa": round(float(cohen_kappa_score(y[mask], pred[mask], weights="quadratic")), 4),
        }
    test_pred = clf.predict(X)
    results = {"baselines": {n: report(p, te) for n, p in baselines.items()},
               "models_validation_macro_f1": {k: round(v, 4) for k, v in val_scores.items()},
               "winner": best,
               "test": report(test_pred, te),
               "test_skills_with_quiz": report(test_pred, te & has_quiz),
               "test_skills_WITHOUT_quiz": report(test_pred, te & ~has_quiz),
               "baseline_rule_quiz_else_self_WITHOUT_quiz": report(rule_qs, te & ~has_quiz),
               "confusion_matrix_test": confusion_matrix(y[te], test_pred[te], labels=[0, 1, 2]).tolist(),
               "class_names": CLASS_NAMES}

    # ------------------------------------------------------------------ ablation: what is each data source worth?
    ablation, used = [], []
    for step, feats in SOURCES.items():
        used += feats
        cols = [FEATURE_NAMES.index(f) for f in used]
        cat_local = [cols.index(i) for i in cat_idx if i in cols]
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=24, l2_regularization=1.0,
                                           categorical_features=cat_local or None, early_stopping=True, random_state=SEED)
        m.fit(X[tr][:, cols], y[tr])
        p = m.predict(X[va][:, cols])
        ablation.append({"information_used": step, "validation_macro_f1": round(macro_f1(y[va], p), 4)})
        print(f"  ablation {step:<26} F1 = {ablation[-1]['validation_macro_f1']:.3f}")
    results["ablation"] = ablation

    # ------------------------------------------------------------------ which features matter most?
    sub = np.where(va)[0][:6000]
    imp = permutation_importance(clf, X[sub], y[sub], n_repeats=3, random_state=SEED, scoring="f1_macro", n_jobs=-1)
    order = np.argsort(-imp.importances_mean)[:10]
    results["top_features"] = [{"feature": FEATURE_NAMES[i], "importance": round(float(imp.importances_mean[i]), 4)} for i in order]

    # ------------------------------------------------------------------ severity regressor: predicts the actual gap in levels (0-4)
    # The recommender uses this number to decide which gap is the BIGGEST problem.
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0,
                                        categorical_features=cat_idx, early_stopping=True, random_state=SEED)
    reg.fit(X[tr], y_gap[tr])
    pg = np.clip(reg.predict(X), 0, 4)
    base_gap = np.maximum(req - np.where(np.isnan(quiz), np.nan_to_num(selfr, nan=req), quiz), 0)
    results["gap_size_mae_levels"] = {"model": round(float(mean_absolute_error(y_gap[te], pg[te])), 4),
                                      "baseline_rule": round(float(mean_absolute_error(y_gap[te], base_gap[te])), 4)}
    results["rows"] = {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}

    # ------------------------------------------------------------------ save everything the website needs
    joblib.dump(clf, MOD / "gap_classifier.joblib")
    joblib.dump(reg, MOD / "gap_regressor.joblib")
    json.dump(results, open(REP / "gap_model_report.json", "w"), indent=2)
    print("\nTEST macro-F1:", results["test"]["macro_f1"], "| baseline (quiz else self):",
          results["baselines"]["Rule: quiz else self-rating"]["macro_f1"])
    print("Without any quiz for the skill -> model", results["test_skills_WITHOUT_quiz"]["macro_f1"],
          "vs rule", results["baseline_rule_quiz_else_self_WITHOUT_quiz"]["macro_f1"])


if __name__ == "__main__":
    main()
