"""
04_simulate_learners.py  --  MAKING THE LEARNERS (because real ones are secret)
===============================================================================
HONEST NOTE (say this to your teacher before he asks!):
  No public dataset of real officials, their assessments and their expert-approved
  learning paths exists - iGOT data belongs to the government. The problem statement
  itself says "student-sourced public data". So we SIMULATE learners, but we do it with
  explicit, realistic rules, and the simulation is *separate* from the models we train.
  Real course data (6,988 courses) is real; the people are simulated.
  When MoSPI data becomes available, replace this file's output with real tables and
  steps 05-08 run unchanged.

What we simulate for each learner
  1. Who they are  : role, years of experience, education.
  2. The TRUTH     : a hidden true proficiency (1-5) for every competency.
                     The model NEVER sees this. It must be inferred.
  3. What we SEE   : noisy assessment data, like in real life:
        * quiz score        (accurate-ish, but only for ~55% of competencies)
        * self-rating       (biased: weak people over-rate themselves - Dunning-Kruger)
        * supervisor rating (only ~40% of competencies, small noise)
  4. History       : courses they already completed (partly popularity-driven, "herding").
  5. Expert panel  : 3 simulated experts each pick the 10 best courses for that learner
                     using the TRUE gaps. These are the "expert learning paths" that
                     the recommender is graded against.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from framework import COMP_IDS, COMPETENCIES, ROLES, ROLE_IDS, role_requirements

HERE = Path(__file__).parent
PROC = HERE / "data/processed"
RNG = np.random.default_rng(2026)
K = len(COMP_IDS)
N_LEARNERS = 4000
N_EXPERTS, PATH_LEN = 3, 10

# ----------------------------------------------------------------------------
# load the catalogue produced by step 03
# ----------------------------------------------------------------------------
cat = pd.read_csv(PROC / "catalogue_tagged.csv")
R = np.load(PROC / "course_comp_matrix.npy")        # learned scores  (used by OUR recommender)
G = np.load(PROC / "course_gold_rules.npy")         # strict rules    (used by the EXPERTS)
G = (G >= 0.75).astype(np.float32)                  # experts think in yes/no: "does it teach k?"
C = len(cat)
lf = cat["level_from"].to_numpy(float)              # level a course starts at
lt = cat["level_to"].to_numpy(float)                # level it takes you to
hours = cat["hours"].to_numpy(float)
quality = cat["quality"].to_numpy(float)

# ----------------------------------------------------------------------------
# Course "popularity propensity": how likely a random learner is to pick the course.
# Udemy gives REAL subscriber counts. Coursera/edX do not, so we simulate a propensity
# (slightly higher for better-rated courses). We use a z-score and a softmax so that a few
# courses become very popular - the classic long-tail seen in every online platform.
# ----------------------------------------------------------------------------
z = np.zeros(C)
udemy = (cat["provider"] == "Udemy").to_numpy()
subs = np.log1p(cat["subscribers"].fillna(0).to_numpy(float))
z[udemy] = (subs[udemy] - subs[udemy].mean()) / (subs[udemy].std() + 1e-9)
z[~udemy] = RNG.normal(0, 1, (~udemy).sum()) + 0.8 * (quality[~udemy] - 0.5)
pop_weight = np.exp(1.1 * z)
pop_weight /= pop_weight.sum()

# ----------------------------------------------------------------------------
# competency "personality": which hidden talents drive which competencies
# (so that someone good at SQL is probably also good at Python: related skills move together).
# ----------------------------------------------------------------------------
FACTORS = ["quant", "digital", "people", "domain"]
LOAD = np.zeros((K, 4))
for k, c in enumerate(COMPETENCIES):
    if c["group"] == "Domain":
        LOAD[k] = [0.45, 0.05, 0.0, 0.50]
    elif c["group"] == "Functional":
        LOAD[k] = [0.25, 0.55, 0.0, 0.10]
    else:
        LOAD[k] = [0.0, 0.0, 0.6, 0.25]
# a few competencies lean on specific talents
for cid, f in [("quant_foundations", 0), ("python_analysis", 1), ("r_analysis", 0), ("machine_learning", 0),
               ("leadership", 2), ("communication", 2), ("geospatial", 1)]:
    LOAD[COMP_IDS.index(cid), f] += 0.2

role_weights = np.array([0.14, 0.10, 0.17, 0.10, 0.12, 0.05, 0.07, 0.06, 0.05, 0.08, 0.06])
role_weights /= role_weights.sum()


def gap_class(gap):
    """0 = no gap (<0.5 level), 1 = moderate (0.5-1.5), 2 = critical (>=1.5)."""
    return np.where(gap >= 1.5, 2, np.where(gap >= 0.5, 1, 0))


def make_learner(i):
    role_idx = RNG.choice(len(ROLES), p=role_weights)
    role = ROLES[role_idx]
    req, imp = role_requirements(role["id"])
    req, imp = np.array(req, float), np.array(imp, float)
    core = imp >= 1.0

    # 1) who they are
    exp = float(np.clip(RNG.normal(1.5 + 3.2 * role["seniority"], 3.0), 0, 30))
    edu = int(RNG.choice([0, 1, 2], p=[0.45, 0.35, 0.20]))      # 0 graduate, 1 postgraduate, 2 stats/maths/CS degree
    f = RNG.normal(0, 1, 4)                                       # hidden talents (quant, digital, people, domain)

    # 2) TRUE proficiency per competency
    #    core skills: centred on what the role requires minus a "still learning" shortfall that shrinks with experience.
    #    non-core skills: everybody has some basic level (~2).
    shortfall = 1.55 - 0.07 * min(exp, 15)
    centre = np.where(core, req - shortfall, 2.0)
    edu_bonus = 0.25 * (edu >= 1) * (LOAD[:, 0] > 0.3) + 0.25 * (edu == 2) * (LOAD[:, 1] > 0.3)
    theta = centre + 0.85 * (LOAD @ f) + edu_bonus + RNG.normal(0, 0.55, K)
    theta = np.clip(theta, 1, 5)

    # 3) history: courses already completed (and they slightly raise proficiency)
    n_hist = int(min(RNG.poisson(1.2 + 0.2 * exp), 12))
    gap0 = np.maximum(req - theta, 0) * imp
    comp_p = np.exp(1.0 * gap0) * (0.4 + imp)            # people tend to study what they lack and what their job needs
    comp_p /= comp_p.sum()
    hist = []
    for _ in range(n_hist):
        if RNG.random() < 0.35:                          # 35%: "herding" - just pick a popular course
            c = RNG.choice(C, p=pop_weight)
        else:                                            # 65%: pick a course for a skill they need
            k = RNG.choice(K, p=comp_p)
            cand = np.where(G[:, k] > 0)[0]
            if len(cand) == 0:
                continue
            w = pop_weight[cand]
            c = RNG.choice(cand, p=w / w.sum())
        if c not in hist:
            hist.append(int(c))
    for c in hist:                                       # learning happened: bump related skills a little
        theta = np.clip(theta + 0.25 * G[c] * (theta < lt[c]), 1, 5)

    # 4) what we OBSERVE
    quiz = np.clip(theta + RNG.normal(0, 0.55, K), 1, 5)
    quiz[RNG.random(K) > 0.55] = np.nan                  # only ~55% of competencies were tested
    overconf = RNG.normal(0.0, 0.30)                     # some people are just more confident
    bias = 0.55 - 0.28 * (theta - 1)                     # Dunning-Kruger: weak -> +0.55, expert -> -0.57
    selfr = np.clip(theta + bias + overconf + RNG.normal(0, 0.55, K), 1, 5)
    selfr[RNG.random(K) > 0.85] = np.nan                 # a few skipped
    sup = np.clip(theta + RNG.normal(0, 0.50, K), 1, 5)
    sup[RNG.random(K) > 0.40] = np.nan                   # supervisors rate ~40%

    gap = np.maximum(req - theta, 0)
    return dict(role_idx=role_idx, exp=exp, edu=edu, req=req, imp=imp, theta=theta, gap=gap,
                gap_cls=gap_class(gap), quiz=quiz, selfr=selfr, sup=sup, hist=hist)


def expert_scores(theta, req, imp, completed):
    """
    The 'true value' of every course for a learner, from an expert's point of view.
    For every course and skill we ask: how much of the learner's REAL gap would it close?
        gain = (how far the course takes you, capped at what the job needs) - (where you are now / course starts)
    Courses that start far above the learner's level (>1 level) are useless -> skipped.
    """
    th, rq = theta[None, :], req[None, :]
    start = np.maximum(lf[:, None], th)                      # you can only gain above where you are
    end = np.minimum(lt[:, None], rq)                        # no point exceeding what the job needs
    gain = np.clip(end - start, 0, 2) / 2.0
    gain *= (lf[:, None] <= th + 1.0)                        # not too advanced
    s = (G * imp[None, :] * gain).sum(axis=1)                # sum over the skills the course teaches
    s = s * (0.85 + 0.3 * quality) / (hours ** 0.15)         # experts like good ratings, and are mildly time-aware
    s[list(completed)] = -1                                  # never recommend what they already did
    return s


def expert_path(base, sigma):
    """
    One expert: noisy version of the true scores, then pick the top PATH_LEN with a diversity rule
    (at most 3 courses for the same main skill) - experts don't give 10 SQL courses.
    """
    sc = base + RNG.normal(0, sigma, len(base)) * (base > 0)
    order = np.argsort(-sc)
    chosen, per_skill = [], {}
    for c in order:
        if sc[c] <= 0:
            break
        main = int(np.argmax(G[c])) if G[c].any() else -1
        if per_skill.get(main, 0) >= 3:
            continue
        chosen.append(int(c))
        per_skill[main] = per_skill.get(main, 0) + 1
        if len(chosen) == PATH_LEN:
            break
    return chosen


def main():
    print(f"Simulating {N_LEARNERS} learners against {C} real courses...")
    L = [make_learner(i) for i in range(N_LEARNERS)]
    out = {k: np.array([l[k] for l in L]) for k in
           ["role_idx", "exp", "edu", "req", "imp", "theta", "gap", "gap_cls", "quiz", "selfr", "sup"]}
    out["hist"] = [l["hist"] for l in L]

    # expert panel
    experts, grades = [], []
    for l in L:
        base = expert_scores(l["theta"], l["req"], l["imp"], l["hist"])
        sigma = 0.12 * max(base.max(), 1e-6)                  # experts disagree a bit (12% of the top score)
        picks = [expert_path(base, sigma) for _ in range(N_EXPERTS)]
        votes = {}
        for p in picks:
            for c in p:
                votes[c] = votes.get(c, 0) + 1                # grade = how many experts chose it (1..3)
        experts.append(votes)
    out["expert_votes"] = experts

    # train / validation / test split BY LEARNER (never mix a learner across splits)
    perm = RNG.permutation(N_LEARNERS)
    n_tr, n_va = int(0.6 * N_LEARNERS), int(0.2 * N_LEARNERS)
    out["split"] = np.empty(N_LEARNERS, dtype=object)
    out["split"][perm[:n_tr]] = "train"
    out["split"][perm[n_tr:n_tr + n_va]] = "val"
    out["split"][perm[n_tr + n_va:]] = "test"
    out["pop_weight"] = pop_weight

    with open(PROC / "learners.pkl", "wb") as fh:
        pickle.dump(out, fh)

    # --- quick sanity numbers
    core = out["imp"] >= 1.0
    cls = out["gap_cls"][core]
    print("Core-skill gap classes  none/moderate/critical:",
          [round(float((cls == c).mean()), 3) for c in range(3)])
    print("Avg expert votes agreeing (courses chosen by >=2 experts):",
          round(float(np.mean([sum(v >= 2 for v in e.values()) for e in experts])), 2), "per learner")
    print("Avg completed courses:", round(float(np.mean([len(h) for h in out['hist']])), 2))
    print("Share of competencies with a quiz score:", round(float(np.mean(~np.isnan(out['quiz']))), 3))


if __name__ == "__main__":
    main()
