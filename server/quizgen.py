"""
quizgen.py  --  MAKE MULTIPLE-CHOICE QUESTIONS FROM ANY LEARNING MATERIAL
==========================================================================
No paid AI service is used. Everything is classic, explainable NLP:

  1. READ the text with spaCy (a free NLP library). It splits sentences and tags every word
     (noun, verb, adjective...) - this is called part-of-speech (POS) tagging.
  2. RANK sentences by importance with TEXTRANK: a sentence that is similar to many other
     sentences is "central" (same idea as Google's PageRank, but for sentences).
  3. FIND KEY TERMS: noun phrases (like "sampling error") scored by TF-IDF
     = common in THIS document but not an everyday phrase.
  4. BUILD one question from each important sentence:
        - "definition" style : the sentence looks like "X is / refers to ..." -> ask which term fits.
        - "fill-in-the-blank": hide the best key term inside the sentence.
  5. BUILD DISTRACTORS (the wrong options): other key terms that appear in SIMILAR sentences
     as the right answer (co-occurrence) but NOT in this sentence. Plausible-but-wrong = a good MCQ.
"""
import io
import random
import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DEFINITION = re.compile(r"^(?P<term>[A-Z][A-Za-z0-9\-/ ]{2,45}?)\s+(?:is|are|refers to|means|is defined as|can be defined as)\s+(?P<rest>.{25,})$")
# phrases that make poor answers (too generic)
GENERIC = set("""way ways thing things time times lot number part kind type example examples case cases result results
 use fact order level levels form process help work value values one two three people area areas set system side""".split())


@lru_cache(maxsize=1)
def get_nlp():
    """Load spaCy once (loading takes ~1 second, so we cache it)."""
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError as e:
        raise RuntimeError("spaCy model missing. Run:  python -m spacy download en_core_web_sm") from e


def extract_text(filename, data: bytes) -> str:
    """Read an uploaded file (.txt .md .pdf) and return plain text."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def analyse(text):
    """
    Returns a list of sentences; each sentence = dict(text, terms=[(key, surface, start, end), ...]).
    A "term" is a noun phrase (1-3 words) with the leading 'the/a/this...' removed.
    """
    nlp = get_nlp()
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()[:60000]            # safety limit
    out = []
    for sent in nlp(text).sents:
        words = [t for t in sent if not t.is_space]
        if not (8 <= len(words) <= 45):
            continue
        terms = []
        for ch in sent.noun_chunks:
            toks = [t for t in ch if t.pos_ not in ("DET", "PRON", "NUM", "PUNCT", "SYM", "ADV") and t.is_alpha]
            while toks and toks[0].pos_ == "ADJ" and toks[0].lower_ in ("such", "many", "more", "most", "other", "same", "own", "several", "each", "different", "various"):
                toks = toks[1:]
            if not toks or len(toks) > 3 or toks[-1].pos_ not in ("NOUN", "PROPN"):
                continue
            surface = sent.doc.text[toks[0].idx: toks[-1].idx + len(toks[-1])]
            if len(surface) < 4 or toks[-1].lemma_.lower() in GENERIC:
                continue
            # key = lowercase words with the LAST word as a lemma ("samples" -> "sample") so singular/plural merge
            key = " ".join([t.lower_ for t in toks[:-1]] + [toks[-1].lemma_.lower()])
            terms.append((key, surface, toks[0].idx - sent.start_char, toks[-1].idx + len(toks[-1]) - sent.start_char))
        out.append({"text": sent.text.strip(), "terms": terms})
    return out


def textrank(sentences, iters=40, d=0.85):
    """PageRank over a sentence-similarity graph -> importance score per sentence."""
    n = len(sentences)
    if n < 2:
        return np.ones(n)
    X = TfidfVectorizer(stop_words="english").fit_transform([s["text"] for s in sentences])
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0)
    row = sim.sum(1, keepdims=True)
    P = np.divide(sim, row, out=np.zeros_like(sim), where=row > 0)       # each row sums to 1 = "chance of jumping"
    r = np.ones(n) / n
    for _ in range(iters):
        r = (1 - d) / n + d * (P.T @ r)
    return r


def generate_mcqs(text, n_questions=8, seed=7):
    rng = random.Random(seed)
    sents = analyse(text)
    if len(sents) < 4:
        return {"questions": [], "note": "Need at least a few full sentences of text to build questions."}
    rank = textrank(sents)

    # ---- term statistics: tf (how often) and df (in how many sentences)
    keys = sorted({k for s in sents for (k, _, _, _) in s["terms"]})
    if len(keys) < 6:
        return {"questions": [], "note": "Not enough distinct key terms found in this material."}
    kidx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(sents), len(keys)))
    for si, s in enumerate(sents):
        for (k, _, _, _) in s["terms"]:
            M[si, kidx[k]] += 1
    tf, df = M.sum(0), (M > 0).sum(0)
    idf = np.log((1 + len(sents)) / (1 + df)) + 1
    length_bonus = np.array([1 + 0.3 * (len(k.split()) - 1) for k in keys])
    tscore = tf * idf * length_bonus * np.where(df >= 2, 1.0, 0.3)       # terms seen in 2+ sentences are real topics
    # co-occurrence similarity: terms are similar if they appear in the same sentences
    Mb = (M > 0).astype(float)
    nrm = np.linalg.norm(Mb, axis=0) + 1e-9
    co = (Mb.T @ Mb) / np.outer(nrm, nrm)
    # distractors come from terms that appear in 2+ sentences (real topics of the document, not one-off phrases)
    pool = [int(i) for i in np.argsort(-tscore) if df[i] >= 2][:80]
    if len(pool) < 8:
        pool = [int(i) for i in np.argsort(-tscore)[:80]]

    questions, used_keys, picked = [], set(), []
    for si in np.argsort(-rank):
        s = sents[si]
        words = set(s["text"].lower().split())
        if any(len(words & set(p.lower().split())) / len(words | set(p.lower().split())) > 0.55 for p in picked):
            continue                                                       # too similar to a question we already have
        cands = [t for t in s["terms"] if t[0] not in used_keys and tscore[kidx[t[0]]] > 0]
        if not cands:
            continue
        key, surface, a, b = max(cands, key=lambda t: tscore[kidx[t[0]]])
        # ---- distractors: similar terms that are NOT in this sentence
        s_keys = {t[0] for t in s["terms"]}
        ai = kidx[key]
        dist = []
        for j in sorted(pool, key=lambda j: -co[ai, j]):
            k2 = keys[j]
            if k2 == key or k2 in s_keys or key in k2 or k2 in key:
                continue
            if re.search(r"\b" + re.escape(k2) + r"s?\b", s["text"], flags=re.I):
                continue                                                   # the option must not appear in the question itself
            if abs(len(k2.split()) - len(key.split())) > 1:
                continue
            dist.append(k2)
            if len(dist) == 3:
                break
        if len(dist) < 3:
            continue

        def style(w):                                                      # match capitalisation of the right answer
            return w[:1].upper() + w[1:] if surface[:1].isupper() else w
        options = [surface] + [style(d) for d in dist]
        dm = DEFINITION.match(s["text"])
        if dm and key.split()[-1] in dm.group("term").lower():
            qtype = "definition"
            rest = dm.group("rest").rstrip(".")
            question = f"Which term matches this description: “{rest[0].upper() + rest[1:]}”?"
        else:
            qtype = "fill-in-the-blank"
            question = s["text"][:a] + "_____" + s["text"][b:]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options[o] for o in order]
        questions.append({
            "type": qtype, "question": question, "options": shuffled, "answer_index": shuffled.index(surface),
            "explanation": f"From the material: “{s['text']}”",
            "difficulty": "easy" if rank[si] > np.median(rank) else "medium",
        })
        used_keys.add(key)
        picked.append(s["text"])
        if len(questions) >= n_questions:
            break
    top_terms = [keys[i] for i in np.argsort(-tscore)[:12]]
    return {"questions": questions, "sentences_analysed": len(sents), "key_terms": top_terms}
