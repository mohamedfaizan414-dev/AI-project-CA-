"""
07_evaluate.py  --  THE EXAM  (graded on test learners who were NEVER used in training)
========================================================================================
We compare 6 ways of recommending courses on the same ~800 test learners, against the
expert paths (3 simulated experts, "relevant" = chosen by >= 2 of them):

   1. Random                          - lower bound ("is anything better than luck?")
   2. Popularity                      - THE REQUIRED BASELINE: most-completed courses overall
   3. Popularity within job skills    - a stronger baseline: most popular courses that at least
                                        touch the learner's role skills
   4. Ours - heuristic                - coverage score from the PREDICTED gaps (no learned ranker)
   5. Ours - learned ranker (FINAL)   - the full system
   6. Reference: same system but given the TRUE gaps - shows how much is lost to gap-inference errors

Metrics:
   Recall@K   share of expert-chosen courses found in our top K
   NDCG@K     rewards putting the courses MORE experts agreed on higher up (1.0 = perfect)
   Gap coverage@10  an outcome metric: share of the learner's real (importance-weighted) gap for
                    which at least one recommended course is useful (right skill AND right level)
We also give 95% bootstrap confidence intervals, so a lucky split cannot fool us.
"""
import json
import pickle
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender import Catalogue, candidate_table, diversify, ndcg_at_k, primary_skill, recall_at_k

HERE = Path(__file__).parent
PROC, REP, MOD = HERE / "data/processed", HERE / "reports", HERE / "models"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)
KS = [5, 10, 20]

# website colour theme (forest + sand) reused in figures
FOREST, MOSS, SAND, CLAY, INK = "#1F4D3A", "#6B8F71", "#D9C9A3", "#B5835A", "#1B2A22"


def gap_coverage(ranked, theta, req, imp, G, cat, k=10):
    """Outcome metric: which of the learner's real gaps get at least one USEFUL course in the top-k?"""
    gap = np.maximum(req - theta, 0) * imp
    if gap.sum() == 0:
        return np.nan
    covered = np.zeros(len(theta), bool)
    for c in ranked[:k]:
        useful = (G[c] > 0) & (cat.lf[c] <= theta + 1.0) & (cat.lt[c] > theta)
        covered |= useful
    return float((gap * covered).sum() / gap.sum())


def main():
    L = pickle.load(open(PROC / "learners.pkl", "rb"))
    R = np.load(PROC / "course_comp_matrix.npy")
    G = (np.load(PROC / "course_gold_rules.npy") >= 0.75).astype(np.float32)
    pop = np.load(PROC / "popularity.npy")
    pred_gap = np.load(PROC / "pred_gap.npy")
    df = pd.read_csv(PROC / "catalogue_tagged.csv")
    cat = Catalogue(df, R, pop)
    ranker = joblib.load(MOD / "ranker.joblib")
    C = len(df)
    test_ids = np.where(L["split"] == "test")[0]
    print(f"Evaluating on {len(test_ids)} test learners, {C} courses")

    pop_order = np.argsort(-pop)                                  # global popularity ranking (computed on TRAIN learners)

    def top_excluding(order, done, n=20):
        return [int(c) for c in order if c not in done][:n]

    def sys_random(i):
        done = set(L["hist"][i])
        perm = RNG.permutation(C)
        return top_excluding(perm, done)

    def sys_pop(i):
        return top_excluding(pop_order, set(L["hist"][i]))

    def sys_pop_role(i):
        # courses whose skills overlap the learner's role-critical skills, ordered by popularity
        core_skill = (L["imp"][i] >= 1.0).astype(float)
        touches = (cat.Rt * core_skill[None, :]).max(1) >= 0.5
        order = [c for c in pop_order if touches[c]]
        return top_excluding(order, set(L["hist"][i]))

    def run_ours(i, gap, use_ranker):
        ids, F = candidate_table(cat, gap, L["req"][i], L["imp"][i], L["hist"][i])
        if len(ids) == 0:
            return []
        sc = ranker.predict(F) if use_ranker else F[:, 0]         # F[:,0] is the plain coverage score
        prim = primary_skill(cat, ids, gap, L["req"][i], L["imp"][i])
        return diversify(ids, sc, cat, prim, top_n=20)

    systems = {
        "Random": sys_random,
        "Popularity (baseline)": sys_pop,
        "Popularity within job skills": sys_pop_role,
        "Ours: heuristic": lambda i: run_ours(i, pred_gap[i], False),
        "Ours: learned ranker (final)": lambda i: run_ours(i, pred_gap[i], True),
        "Reference: ranker with TRUE gaps": lambda i: run_ours(i, L["gap"][i], True),
    }

    per_learner = {s: {m: [] for m in [f"recall@{k}" for k in KS] + [f"ndcg@{k}" for k in KS] + ["gapcov@10"]} for s in systems}
    for i in test_ids:
        votes = L["expert_votes"][i]
        for name, fn in systems.items():
            ranked = fn(i)
            for k in KS:
                per_learner[name][f"recall@{k}"].append(recall_at_k(ranked, votes, k))
                per_learner[name][f"ndcg@{k}"].append(ndcg_at_k(ranked, votes, k))
            per_learner[name]["gapcov@10"].append(gap_coverage(ranked, L["theta"][i], L["req"][i], L["imp"][i], G, cat, 10))

    def boot_ci(x, n=1000):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        means = [RNG.choice(x, len(x)).mean() for _ in range(n)]
        return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

    table = {}
    for name, ms in per_learner.items():
        table[name] = {m: {"mean": round(float(np.nanmean(v)), 4), "ci95": boot_ci(v)} for m, v in ms.items()}

    # paired bootstrap: is OUR final system really better than the popularity baseline?
    def paired(a, b, metric):
        d = np.asarray(per_learner[a][metric], float) - np.asarray(per_learner[b][metric], float)
        d = d[~np.isnan(d)]
        boots = [RNG.choice(d, len(d)).mean() for _ in range(1000)]
        return {"mean_difference": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
    final, base = "Ours: learned ranker (final)", "Popularity (baseline)"
    paired_tests = {m: paired(final, base, m) for m in ["recall@10", "ndcg@10", "gapcov@10"]}

    report = {"test_learners": int(len(test_ids)), "systems": table, "final_vs_popularity_paired": paired_tests,
              "definition": "Relevant course = chosen by >= 2 of 3 simulated experts; NDCG gain = number of expert votes"}
    json.dump(report, open(REP / "recsys_report.json", "w"), indent=2)

    # ------------------------------------------------------------------ print a tidy table
    print(f"\n{'System':<36}" + "".join(f"{m:>11}" for m in ["recall@5", "recall@10", "recall@20", "ndcg@10", "gapcov@10"]))
    for name, t in table.items():
        print(f"{name:<36}" + "".join(f"{t[m]['mean']:>11.3f}" for m in ["recall@5", "recall@10", "recall@20", "ndcg@10", "gapcov@10"]))
    print("\nFinal vs popularity (paired):", json.dumps(paired_tests))

    make_figures(table)


def make_figures(table):
    """Slide-ready PNGs in reports/figures (same forest/sand theme as the website)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    # 1) recommender comparison
    names = ["Random", "Popularity (baseline)", "Popularity within job skills", "Ours: heuristic", "Ours: learned ranker (final)"]
    cols = [SAND, CLAY, "#C9B88F", MOSS, FOREST]
    metrics = ["recall@10", "ndcg@10", "gapcov@10"]
    labels = ["Recall@10", "NDCG@10", "Gap coverage@10"]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="#F6F0E2")
    ax.set_facecolor("#F6F0E2")
    w = 0.16
    for j, (n, c) in enumerate(zip(names, cols)):
        vals = [table[n][m]["mean"] for m in metrics]
        ax.bar(np.arange(3) + (j - 2) * w, vals, w, color=c, label=n)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("score (higher is better)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Recommendation quality on unseen test learners", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "recommender_comparison.png", dpi=170)
    plt.close(fig)

    # 2) gap-model ablation + 3) confusion matrix
    g = json.load(open(REP / "gap_model_report.json"))
    ab = g["ablation"]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F6F0E2")
    ax.set_facecolor("#F6F0E2")
    ax.barh([a["information_used"] for a in ab][::-1], [a["validation_macro_f1"] for a in ab][::-1], color=MOSS)
    ax.set_xlabel("gap classification macro-F1 (validation)")
    ax.set_title("What each information source adds to the gap model", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "gap_ablation.png", dpi=170)
    plt.close(fig)

    cm = np.array(g["confusion_matrix_test"])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor="#F6F0E2")
    ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", color=INK if cmn[i, j] < 0.6 else "white")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["None", "Moderate", "Critical"]); ax.set_yticklabels(["None", "Moderate", "Critical"])
    ax.set_xlabel("predicted gap"); ax.set_ylabel("true gap")
    ax.set_title("Gap model: confusion matrix (test)", color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "gap_confusion.png", dpi=170)
    plt.close(fig)

    # 4) cleaning funnel
    c = json.load(open(REP / "cleaning_report.json"))["steps"]
    steps = [s for s in c if s["removed"] > 0]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor="#F6F0E2")
    ax.set_facecolor("#F6F0E2")
    ax.barh([s["step"] for s in steps][::-1], [s["removed"] for s in steps][::-1], color=CLAY)
    ax.set_xlabel("rows removed")
    ax.set_title("Dataset cleaning: what each step removed", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "cleaning_funnel.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
