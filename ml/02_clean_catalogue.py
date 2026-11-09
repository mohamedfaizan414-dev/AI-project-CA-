"""
02_clean_catalogue.py  --  DATASET CLEANING
============================================
Input : three REAL public course datasets (data/raw/)
          * coursera.csv  (3,522 courses)  Kaggle "Coursera Course Dataset"
          * edx.csv       (720 courses)    Kaggle "edX Courses"
          * udemy.csv     (3,678 courses)  Kaggle "Udemy Courses" (has subscriber counts)
Output: data/processed/catalogue_clean.csv  -> ONE tidy table, same columns for all
        reports/cleaning_report.json         -> how many rows each step removed

Think of this as washing vegetables before cooking: raw data has mud
(duplicates, junk text, wrong labels). If we feed mud to an AI model, it
learns mud. Every step below fixes one specific problem that we FOUND by
looking at the data (see the README for the evidence).
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).parent
RAW, OUT, REP = HERE / "data/raw", HERE / "data/processed", HERE / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

LOG = []  # we append one record per cleaning step so we can show a "funnel" later


def log(step, before, after, note=""):
    """Remember how many rows a step removed (this proves we cleaned on purpose)."""
    LOG.append({"step": step, "rows_before": int(before), "rows_after": int(after),
                "removed": int(before - after), "note": note})
    print(f"  {step:<42} {before:>6} -> {after:<6} ({before - after} removed)")


# ----------------------------------------------------------------------------
# STEP 1: load the three files and rename columns to ONE common schema
# ----------------------------------------------------------------------------
def load_all():
    c = pd.read_csv(RAW / "coursera.csv")
    e = pd.read_csv(RAW / "edx.csv")
    u = pd.read_csv(RAW / "udemy.csv")

    coursera = pd.DataFrame({
        "provider": "Coursera",
        "title": c["Course Name"], "url": c["Course URL"],
        "institution": c["University"], "level_raw": c["Difficulty Level"],
        "rating_raw": c["Course Rating"], "description": c["Course Description"],
        "skills_text": c["Skills"], "hours_raw": np.nan, "subscribers": np.nan,
    })
    edx = pd.DataFrame({
        "provider": "edX",
        "title": e["Name"], "url": e["Link"],
        "institution": e["University"], "level_raw": e["Difficulty Level"],
        "rating_raw": np.nan,                                   # edX has no ratings
        "description": e["About"].fillna("") + ". " + e["Course Description"].fillna(""),
        "skills_text": "", "hours_raw": np.nan, "subscribers": np.nan,
    })
    # Udemy: we drop the "Musical Instruments" subject right away - guitar lessons
    # have nothing to do with statistics, and keeping them would only add noise.
    u_before = len(u)
    u = u[u["subject"] != "Musical Instruments"]
    log("udemy: drop out-of-domain subject (music)", u_before, len(u))
    udemy = pd.DataFrame({
        "provider": "Udemy",
        "title": u["course_title"], "url": u["url"],
        "institution": "Udemy instructors", "level_raw": u["level"],
        "rating_raw": np.nan,                                   # only review COUNTS exist
        "description": u["subject"],                            # Udemy has no description
        "skills_text": "", "hours_raw": u["content_duration"].replace(0, np.nan),
        "subscribers": u["num_subscribers"],
    })
    df = pd.concat([coursera, edx, udemy], ignore_index=True)
    log("load + unify 3 datasets", len(df), len(df), "common columns")
    return df


# ----------------------------------------------------------------------------
# STEP 2: clean the text
# ----------------------------------------------------------------------------
COUPON = re.compile(r"[^.]*\b(coupon|promo code|use the following)\b[^.]*\.?", re.I)
HTML = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+")


def clean_text(s):
    """Make one piece of text tidy: no HTML, no links, no coupon spam, no weird spaces."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\ufffd", "-")             # broken-character marker -> a dash
    s = unicodedata.normalize("NFKC", s)     # fixes things like fancy quotes / odd characters
    s = HTML.sub(" ", s)                     # remove <b> and other HTML tags
    s = URL.sub(" ", s)                      # remove web links
    s = COUPON.sub(" ", s)                   # remove sentences about coupon codes
    s = re.sub(r"\s+", " ", s)               # many spaces/newlines -> one space
    return s.strip()


# ----------------------------------------------------------------------------
# STEP 3: standard level names + rating numbers + course hours
# ----------------------------------------------------------------------------
ADV = re.compile(r"\b(advanced|expert|capstone|masterclass|deep dive|professional certificate)\b", re.I)
BEG = re.compile(r"\b(intro|introduction|basics?|fundamentals?|beginner|foundations?|getting started|essentials?|101)\b", re.I)


def normalise_level(row):
    """
    Everything becomes Beginner / Intermediate / Advanced.
    Returns (level, was_it_guessed). Honest labelling: we keep a flag when we guess.
    """
    raw = str(row["level_raw"]).strip().lower()
    if raw in ("beginner", "beginner level"):
        return "Beginner", False
    if raw in ("intermediate", "intermediate level", "conversant"):   # Coursera's 'Conversant' ~ intermediate
        return "Intermediate", False
    if raw in ("advanced", "expert level"):
        return "Advanced", False
    # 'Not Calibrated' (Coursera) and 'All Levels' (Udemy): guess from the title words
    if ADV.search(row["title"]):
        return "Advanced", True
    if BEG.search(row["title"]):
        return "Beginner", True
    return ("Beginner" if raw == "all levels" else "Intermediate"), True


HOURS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-\s*\d+\s*)?(hours?|hrs?)\b", re.I)
WEEKS = re.compile(r"(\d+)\s*(?:-\s*\d+\s*)?weeks?\b", re.I)
DEFAULT_HOURS = {"Beginner": 8, "Intermediate": 14, "Advanced": 20}   # last-resort assumption


def estimate_hours(row):
    """
    Course length matters for building a learning path with a time budget.
    1) Udemy gives real hours. 2) Otherwise look for '10 hours' or '4 weeks' in the text.
    3) Otherwise use a typical value for the level (and flag it as imputed).
    """
    if not np.isnan(row["hours_raw"]):
        return float(row["hours_raw"]), False
    m = HOURS.search(row["description"])
    if m and 0.5 <= float(m.group(1)) <= 120:
        return float(m.group(1)), False
    m = WEEKS.search(row["description"])
    if m and 1 <= int(m.group(1)) <= 20:
        return float(int(m.group(1)) * 4), False           # assume ~4 hours of study per week
    return float(DEFAULT_HOURS[row["level"]]), True


def main():
    print("Cleaning catalogue...")
    df = load_all()

    # ---- text cleaning
    for col in ["title", "description", "skills_text", "institution"]:
        df[col] = df[col].map(clean_text)
    log("clean text (html, links, coupons, spaces)", len(df), len(df))

    # ---- drop rows that are unusable
    n = len(df)
    df = df[df["title"].str.len() >= 4]
    log("drop empty/very short titles", n, len(df))

    n = len(df)
    # Coursera/edX need a real description (>=40 characters); Udemy only has a title so it is exempt.
    df = df[(df["provider"] == "Udemy") | (df["description"].str.len() >= 40)]
    log("drop courses with no real description", n, len(df))

    # ---- keep English only: if more than 10% of characters are not plain letters, it is probably Hindi/Chinese/etc.
    n = len(df)
    def non_ascii_share(t):
        return sum(ord(ch) > 127 for ch in t) / max(1, len(t))
    df = df[(df["title"] + " " + df["description"]).map(non_ascii_share) <= 0.10]
    log("keep English-language courses only", n, len(df))

    # ---- remove duplicates (three kinds, from easy to clever)
    n = len(df)
    df = df.drop_duplicates(subset=["url"])
    log("dedupe: same URL", n, len(df))

    n = len(df)
    df["_key"] = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df = df.drop_duplicates(subset=["provider", "_key"])
    log("dedupe: same title in same provider", n, len(df))

    # Near-duplicates ("Python for Everybody" vs "Python For Everybody - Part 1"):
    # turn each title into numbers (TF-IDF) and compare. Cosine similarity >= 0.92 = same course.
    n = len(df)
    df = df.reset_index(drop=True)
    tv = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(df["_key"])
    drop = set()
    for start in range(0, len(df), 1000):                    # do it in chunks so memory stays small
        sims = (tv[start:start + 1000] @ tv.T).toarray()
        for i in range(sims.shape[0]):
            gi = start + i
            if gi in drop:
                continue
            # only look at LATER rows, so the first copy is the one we keep
            dup = np.where(sims[i, gi + 1:] >= 0.92)[0] + gi + 1
            drop.update(dup.tolist())
    df = df.drop(index=list(drop)).reset_index(drop=True)
    log("dedupe: near-identical titles (TF-IDF >= 0.92)", n, len(df))

    # ---- standard level, rating, hours
    lv = df.apply(normalise_level, axis=1, result_type="expand")
    df["level"], df["level_inferred"] = lv[0], lv[1]

    df["rating"] = pd.to_numeric(df["rating_raw"], errors="coerce")   # 'Not Calibrated' -> NaN
    df.loc[~df["rating"].between(1, 5), "rating"] = np.nan            # impossible values -> NaN
    log("parse ratings ('Not Calibrated' -> missing)", len(df), len(df),
        f"{int(df['rating'].isna().sum())} courses have no rating")

    hr = df.apply(estimate_hours, axis=1, result_type="expand")
    df["hours"], df["hours_imputed"] = hr[0], hr[1]
    log("estimate course hours", len(df), len(df), f"{int(df['hours_imputed'].sum())} imputed from level defaults")

    # ---- the text the AI will read: title (x2 so it counts more) + skills + description
    df["text"] = (df["title"] + ". " + df["title"] + ". " + df["skills_text"] + ". " +
                  df["description"].str.slice(0, 1500)).str.strip()

    df["course_id"] = ["C%05d" % i for i in range(len(df))]
    cols = ["course_id", "provider", "title", "institution", "url", "level", "level_inferred",
            "rating", "hours", "hours_imputed", "subscribers", "skills_text", "description", "text"]
    df[cols].to_csv(OUT / "catalogue_clean.csv", index=False)

    summary = {
        "final_rows": int(len(df)),
        "by_provider": df["provider"].value_counts().to_dict(),
        "by_level": df["level"].value_counts().to_dict(),
        "level_inferred_share": round(float(df["level_inferred"].mean()), 3),
    }
    (REP / "cleaning_report.json").write_text(json.dumps({"steps": LOG, "summary": summary}, indent=2))
    print("Done:", summary)


if __name__ == "__main__":
    main()
