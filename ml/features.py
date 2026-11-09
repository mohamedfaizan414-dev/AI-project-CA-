"""
features.py  --  TURNING A LEARNER'S DATA INTO NUMBERS THE MODEL CAN READ
==========================================================================
A machine-learning model only understands tables of numbers. For every
(person, competency) pair we build one ROW of numbers called "features".

IMPORTANT: this ONE file is used both when TRAINING (05_train_gap_model.py) and
when the website asks for a live prediction (server/app.py). Using the same code in
both places prevents the classic bug "the model works in the notebook but not in the app".

Missing values: in real life many assessments are missing. We keep them as NaN
("not a number" = unknown). The gradient-boosting model knows how to deal with NaN.
"""
import numpy as np

from framework import COMPETENCIES, ROLES

GROUP_OF = np.array([["Domain", "Functional", "Behavioural"].index(c["group"]) for c in COMPETENCIES])
SENIORITY = np.array([r["seniority"] for r in ROLES], float)

FEATURE_NAMES = [
    # --- who is the person and what does the job need
    "req", "role_idx", "seniority", "exp_years", "education", "comp_idx", "group_idx",
    # --- direct evidence about THIS competency (may be missing)
    "quiz", "self", "supervisor", "quiz_minus_req", "self_minus_req", "sup_minus_req",
    # --- indirect evidence: how the person did on RELATED competencies
    "grp_quiz_mean", "grp_self_mean", "grp_sup_mean", "all_quiz_mean", "n_quiz",
    "avg_quiz_shortfall",
    # --- correcting self-rating bias (are they an over-rater?)
    "overconf_vs_quiz", "overconf_vs_sup", "self_corrected",
    # --- history
    "hist_cnt",
]
# Which features belong to which "information source" (used in the ablation experiment)
SOURCES = {
    "role only": ["req", "role_idx", "seniority", "comp_idx", "group_idx"],
    "+ experience/education": ["exp_years", "education"],
    "+ self rating": ["self", "self_minus_req", "grp_self_mean", "overconf_vs_quiz", "overconf_vs_sup", "self_corrected"],
    "+ supervisor": ["supervisor", "sup_minus_req", "grp_sup_mean"],
    "+ quiz scores": ["quiz", "quiz_minus_req", "grp_quiz_mean", "all_quiz_mean", "n_quiz", "avg_quiz_shortfall"],
    "+ course history": ["hist_cnt"],
}


def _mean_excluding_self(X, mask_same):
    """
    For each person i and competency k: average of X[i, j] over the OTHER competencies j (j != k)
    that are in `mask_same[k]` and are not missing.  Returns NaN when there is nothing to average.
    X: (N, K) with NaN for missing.  mask_same: (K, K) boolean, [k, j] = True if j counts for k.
    """
    obs = ~np.isnan(X)
    Xz = np.where(obs, X, 0.0)
    num = Xz @ mask_same.T.astype(float)          # sum of observed values, per (i, k)
    den = obs.astype(float) @ mask_same.T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def build_features(role_idx, exp, edu, req, quiz, selfr, sup, hist_cnt):
    """
    All inputs are numpy arrays. N = number of people, K = number of competencies.
      role_idx (N,)  exp (N,)  edu (N,)
      req, quiz, selfr, sup, hist_cnt : (N, K)    (quiz/selfr/sup have NaN where missing)
    Returns an array of shape (N, K, number_of_features).
    """
    N, K = req.shape
    same_group = (GROUP_OF[:, None] == GROUP_OF[None, :]) & ~np.eye(K, dtype=bool)   # related = same group, not itself
    everyone_else = ~np.eye(K, dtype=bool)

    f = {}
    f["req"] = req
    f["role_idx"] = np.repeat(role_idx[:, None], K, 1).astype(float)
    f["seniority"] = np.repeat(SENIORITY[role_idx][:, None], K, 1)
    f["exp_years"] = np.repeat(exp[:, None], K, 1)
    f["education"] = np.repeat(edu[:, None], K, 1).astype(float)
    f["comp_idx"] = np.repeat(np.arange(K)[None, :], N, 0).astype(float)
    f["group_idx"] = np.repeat(GROUP_OF[None, :], N, 0).astype(float)

    f["quiz"], f["self"], f["supervisor"] = quiz, selfr, sup
    f["quiz_minus_req"] = quiz - req
    f["self_minus_req"] = selfr - req
    f["sup_minus_req"] = sup - req

    f["grp_quiz_mean"] = _mean_excluding_self(quiz, same_group)
    f["grp_self_mean"] = _mean_excluding_self(selfr, same_group)
    f["grp_sup_mean"] = _mean_excluding_self(sup, same_group)
    f["all_quiz_mean"] = _mean_excluding_self(quiz, everyone_else)
    f["n_quiz"] = (~np.isnan(quiz)).sum(1, keepdims=True).repeat(K, 1).astype(float)
    f["avg_quiz_shortfall"] = _mean_excluding_self(quiz - req, everyone_else)

    # Over-confidence: how much does this person's self-rating exceed what quizzes/supervisors say,
    # on the competencies where we have BOTH? (excluding the current competency to avoid peeking)
    f["overconf_vs_quiz"] = _mean_excluding_self(selfr - quiz, everyone_else)
    f["overconf_vs_sup"] = _mean_excluding_self(selfr - sup, everyone_else)
    oc = np.where(np.isnan(f["overconf_vs_quiz"]), f["overconf_vs_sup"], f["overconf_vs_quiz"])
    f["self_corrected"] = selfr - np.where(np.isnan(oc), 0.0, oc)

    f["hist_cnt"] = hist_cnt
    return np.stack([f[n] for n in FEATURE_NAMES], axis=-1)


def history_counts(hist, R):
    """
    hist: list (one entry per person) of lists of course row-numbers the person already finished.
    R   : course x competency matrix.
    Returns hist_cnt[i, k] = how many finished courses of person i teach competency k (R >= 0.5).
    """
    H = np.zeros((len(hist), R.shape[1]))
    for i, h in enumerate(hist):
        for c in h:
            H[i] += (R[c] >= 0.5)
    return H
