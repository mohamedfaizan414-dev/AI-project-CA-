"""
recommender.py  --  FROM "GAPS" TO "A RANKED LEARNING PATH"
===========================================================
Used by 06_train_recommender.py (training), 07_evaluate.py (grading) and the website server.

The pipeline for ONE learner:
  1. INPUT   : estimated gap size for every skill (from the gap model) + what the job requires.
  2. SCORE   : for every course, "how much of this person's gap would it close?"  -> coverage score.
  3. SHORTLIST: keep the 80 best-covering courses (we don't need to judge all 2,110).
  4. RANK    : a trained model (gradient boosting) re-scores the shortlist using richer clues
               (quality, level fit, popularity, length...). It learned from expert paths.
  5. DIVERSIFY: don't show 10 SQL courses - penalise repeats of the same main skill.
  6. ORDER   : turn the ranked list into a PATH: biggest gaps first, beginner -> advanced.
"""
import numpy as np

N_CANDIDATES = 80
PROVIDER_CODE = {"Coursera": 0, "edX": 1, "Udemy": 2}
CAND_FEATURES = [
    "coverage", "coverage_share", "coverage_vs_best", "max_rel_gap", "n_targets", "primary_gap",
    "start_mismatch", "end_mismatch", "quality", "log_hours", "level_from", "level_to",
    "provider", "log_pop", "primary_rel",
]


class Catalogue:
    """Holds the course table and the numbers we need again and again."""

    def __init__(self, cat_df, R, pop_counts):
        self.df = cat_df.reset_index(drop=True)
        self.R = R.astype(np.float32)                      # (courses, competencies) learned teaching strength
        self.lf = self.df["level_from"].to_numpy(float)    # level the course starts at
        self.lt = self.df["level_to"].to_numpy(float)      # level it brings you to
        self.hours = self.df["hours"].to_numpy(float)
        self.quality = self.df["quality"].to_numpy(float)
        self.provider = self.df["provider"].map(PROVIDER_CODE).to_numpy(float)
        self.pop = np.asarray(pop_counts, float)           # how many learners (in training data) completed each course
        # soft tags: a course "teaches" skill k only if its score >= 0.35; below that we treat as 0
        self.Rt = np.where(self.R >= 0.35, self.R, 0.0)


def gain_matrix(cat, theta_hat, req):
    """
    gain[c, k] in 0..1 = how much of the (gap) from the learner's level up to the job requirement
    course c could close for skill k. Skipped when the course starts >1 level above the learner.
    """
    th, rq = theta_hat[None, :], req[None, :]
    start = np.maximum(cat.lf[:, None], th)
    end = np.minimum(cat.lt[:, None], rq)
    g = np.clip(end - start, 0, 2) / 2.0
    return g * (cat.lf[:, None] <= th + 1.0)


def candidate_table(cat, gap_est, req, imp, completed):
    """
    Build the shortlist and its feature matrix for one learner.
    gap_est: estimated gap size per skill (levels, >= 0).  Returns (candidate_ids, feature_matrix).
    """
    theta_hat = req - gap_est                                # estimated current level
    gain = gain_matrix(cat, theta_hat, req)
    contrib = cat.Rt * imp[None, :] * gain                   # (courses, skills): value of course c for skill k
    cov = contrib.sum(1)
    if len(completed):
        cov[list(completed)] = -1                            # never re-recommend finished courses
    ids = np.argsort(-cov)[:N_CANDIDATES]
    ids = ids[cov[ids] > 0]
    if len(ids) == 0:
        return ids, np.zeros((0, len(CAND_FEATURES)))

    total_gap_mass = float((imp * gap_est).sum()) + 1e-6
    c_ = contrib[ids]
    primary = c_.argmax(1)                                   # the main skill each candidate helps with
    rows = np.arange(len(ids))
    F = np.column_stack([
        cov[ids],
        cov[ids] / total_gap_mass,
        cov[ids] / cov[ids].max(),
        (cat.Rt[ids] * gap_est[None, :]).max(1),
        (c_ > 0.05).sum(1),
        gap_est[primary],
        cat.lf[ids] - theta_hat[primary],
        cat.lt[ids] - req[primary],
        cat.quality[ids],
        np.log1p(cat.hours[ids]),
        cat.lf[ids], cat.lt[ids], cat.provider[ids],
        np.log1p(cat.pop[ids]),
        cat.Rt[ids][rows, primary],
    ])
    return ids, F


def diversify(ids, scores, cat, imp_k_primary, top_n=20, penalty=1.2):
    """
    Greedy re-ranking (like "MMR"): after we pick a course for skill k, courses for the SAME skill
    lose `penalty` of their score per course already picked. Result: a broader, more useful list.

    HOW WE CHOSE penalty=1.2 (tuned on VALIDATION learners, never on test):
        penalty  recall@10  gap-coverage@10  distinct skills in top 8
          0.45     0.496        0.665              4.8
          0.80     0.485        0.690              5.2
          1.20     0.476        0.711              5.5   <- chosen: small loss of expert agreement,
          2.00     0.467        0.735              5.9      clearly broader, more useful paths
    """
    ids = list(ids)
    scores = np.asarray(scores, float).copy()
    primary = {c: p for c, p in zip(ids, imp_k_primary)}
    taken, chosen = {}, []
    pool = set(range(len(ids)))
    while pool and len(chosen) < top_n:
        best, best_s = None, -1e18
        for j in pool:
            s = scores[j] / (1.0 + penalty * taken.get(primary[ids[j]], 0))
            if s > best_s:
                best, best_s = j, s
        pool.remove(best)
        chosen.append(ids[best])
        taken[primary[ids[best]]] = taken.get(primary[ids[best]], 0) + 1
    return chosen


def primary_skill(cat, ids, gap_est, req, imp):
    """Main skill of each course for THIS learner (the one with the most value)."""
    theta_hat = req - gap_est
    gain = gain_matrix(cat, theta_hat, req)
    contrib = cat.Rt[ids] * imp[None, :] * gain[ids]
    return contrib.argmax(1)


def order_as_path(chosen, cat, gap_est, req, imp):
    """
    Turn a ranked list into a LEARNING PATH:
      - skills with the biggest (importance-weighted) gap come first
      - inside one skill: beginner course before advanced (prerequisite order)
    Returns list of (course_id, primary_skill_index).
    """
    prim = primary_skill(cat, chosen, gap_est, req, imp)
    items = list(zip(chosen, prim))
    severity = gap_est * imp
    items.sort(key=lambda t: (-severity[t[1]], t[1], cat.lf[t[0]], -cat.quality[t[0]]))
    return items


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------
def dcg(grades):
    grades = np.asarray(grades, float)
    return float(((2 ** grades - 1) / np.log2(np.arange(2, len(grades) + 2))).sum())


def ndcg_at_k(ranked, votes, k):
    """NDCG: 1.0 = the expert-approved courses are at the very top, in the best order."""
    gains = [votes.get(c, 0) for c in ranked[:k]]
    ideal = sorted(votes.values(), reverse=True)[:k]
    return dcg(gains) / dcg(ideal) if ideal else 0.0


def recall_at_k(ranked, votes, k, min_votes=2):
    """Recall@K: of the courses >= min_votes experts chose, what share are in our top K?"""
    rel = {c for c, v in votes.items() if v >= min_votes}
    if not rel:
        return np.nan
    return len(rel.intersection(ranked[:k])) / len(rel)
