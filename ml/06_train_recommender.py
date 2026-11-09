"""
06_train_recommender.py  --  "WHICH COURSES SHOULD THIS PERSON TAKE?"  (AI model #3: the RANKER)
===============================================================================================
Two sub-steps:

  A) PREDICTED GAPS FOR EVERYONE, WITHOUT CHEATING.
     The recommender must work with the gap model's ESTIMATES (it will never know true gaps in
     real life). But the gap model was trained on the training learners, so its predictions for
     THEM are unrealistically good. Fix: out-of-fold (OOF) prediction - each training learner's
     gap is predicted by a model that never saw that learner (5-fold, split by learner).
     Test learners get predictions from the saved model, which never saw them either.

  B) LEARNING TO RANK.
     For each training learner we build a shortlist of 80 candidate courses and describe each one
     with 15 numbers (see recommender.CAND_FEATURES). The label is how many of the 3 experts picked
     it (0,1,2,3). A gradient-boosting regressor learns "what makes experts choose a course".
     At recommendation time we sort the shortlist by the model's prediction.
"""
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from features import FEATURE_NAMES, build_features, history_counts
from recommender import CAND_FEATURES, Catalogue, candidate_table, diversify, ndcg_at_k, primary_skill

HERE = Path(__file__).parent
PROC, REP, MOD = HERE / "data/processed", HERE / "reports", HERE / "models"
SEED = 42


def predicted_gaps(L, R):
    """Part A: gap estimates for every learner and skill, honest (OOF for train/val, held-out model for test)."""
    N, K = L["req"].shape
    hist_cnt = history_counts(L["hist"], R)
    X3 = build_features(L["role_idx"], L["exp"], L["edu"], L["req"], L["quiz"], L["selfr"], L["sup"], hist_cnt)
    core = L["imp"] >= 1.0
    person = np.repeat(np.arange(N)[:, None], K, 1)
    rows = np.where(core.ravel())[0]
    X = X3.reshape(N * K, -1)[rows]
    y = L["gap"].ravel()[rows]
    who = person.ravel()[rows]
    split = L["split"][who]
    cat_idx = [FEATURE_NAMES.index("role_idx"), FEATURE_NAMES.index("comp_idx")]

    pred = np.zeros(N * K)
    out = np.zeros(len(rows))
    dev = np.where(split != "test")[0]                       # train + val rows
    for f, (a, b) in enumerate(GroupKFold(n_splits=5).split(dev, groups=who[dev])):
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=24, l2_regularization=1.0,
                                          categorical_features=cat_idx, random_state=SEED)
        m.fit(X[dev[a]], y[dev[a]])
        out[dev[b]] = m.predict(X[dev[b]])
        print(f"  OOF fold {f + 1}/5 done")
    test = np.where(split == "test")[0]
    final = joblib.load(MOD / "gap_regressor.joblib")        # trained on train split only
    out[test] = final.predict(X[test])
    pred[rows] = np.clip(out, 0, 4)
    return pred.reshape(N, K)


def build_rank_data(ids_of_learners, L, cat, pred_gap):
    """Part B: for each learner -> shortlist + features + expert-vote labels."""
    Xs, ys, groups = [], [], []
    for i in ids_of_learners:
        ids, F = candidate_table(cat, pred_gap[i], L["req"][i], L["imp"][i], L["hist"][i])
        if len(ids) == 0:
            continue
        votes = L["expert_votes"][i]
        Xs.append(F)
        ys.append([votes.get(int(c), 0) for c in ids])
        groups.append(np.full(len(ids), i))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(groups)


def eval_ndcg(model, learner_ids, L, cat, pred_gap, k=10):
    """Average NDCG@k of 'rank by model, then diversify' on a set of learners."""
    vals = []
    for i in learner_ids:
        ids, F = candidate_table(cat, pred_gap[i], L["req"][i], L["imp"][i], L["hist"][i])
        if len(ids) == 0:
            continue
        sc = model.predict(F)
        prim = primary_skill(cat, ids, pred_gap[i], L["req"][i], L["imp"][i])
        ranked = diversify(ids, sc, cat, prim, top_n=k)
        vals.append(ndcg_at_k(ranked, L["expert_votes"][i], k))
    return float(np.mean(vals))


def main():
    L = pickle.load(open(PROC / "learners.pkl", "rb"))
    R = np.load(PROC / "course_comp_matrix.npy")
    df = pd.read_csv(PROC / "catalogue_tagged.csv")
    C = len(df)

    # popularity = how many TRAINING learners completed each course (this is also the baseline's ranking)
    pop = np.zeros(C)
    for i in np.where(L["split"] == "train")[0]:
        for c in L["hist"][i]:
            pop[c] += 1
    np.save(PROC / "popularity.npy", pop)
    cat = Catalogue(df, R, pop)

    print("A) predicting gaps without cheating...")
    pred_gap = predicted_gaps(L, R)
    np.save(PROC / "pred_gap.npy", pred_gap)

    print("B) building ranking data...")
    tr_ids = np.where(L["split"] == "train")[0]
    va_ids = np.where(L["split"] == "val")[0]
    Xtr, ytr, _ = build_rank_data(tr_ids, L, cat, pred_gap)
    print(f"   {len(ytr)} (learner, course) training rows, {int((ytr > 0).sum())} expert-chosen")

    configs = [
        dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0),
        dict(max_iter=350, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=2.0),
        dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=5.0, min_samples_leaf=40),
    ]
    best, best_s, trials = None, -1, []
    for cfg in configs:
        m = HistGradientBoostingRegressor(random_state=SEED, **cfg).fit(Xtr, ytr)
        s = eval_ndcg(m, va_ids, L, cat, pred_gap)
        trials.append({"config": cfg, "val_ndcg@10": round(s, 4)})
        print(f"   config {cfg} -> validation NDCG@10 = {s:.4f}")
        if s > best_s:
            best, best_s = m, s

    joblib.dump(best, MOD / "ranker.joblib")
    json.dump({"trials": trials, "best_val_ndcg@10": round(best_s, 4), "candidate_features": CAND_FEATURES,
               "training_rows": int(len(ytr)), "positive_rows": int((ytr > 0).sum())},
              open(REP / "ranker_report.json", "w"), indent=2)
    print("Saved ranker. Best validation NDCG@10:", round(best_s, 4))


if __name__ == "__main__":
    main()
