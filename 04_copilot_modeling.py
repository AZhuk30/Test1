"""
04_copilot_modeling.py — Modeling and evaluation for Complaint Intelligence.

Run after 02_copilot_data_prep.ipynb. Needs data/complaints_model_ready.parquet.

Two tracks and three follow-up experiments:

  TRACK A — Retrieval comparison (the project's core hypothesis)
      BM25 (lexical baseline), TF-IDF cosine, dense embeddings (two models),
      and a hybrid of BM25 + dense via reciprocal rank fusion. Scored with
      Precision@5, Precision@10, and MRR over 26 analyst-style queries.

  EXPERIMENT 1 — Chunking ablation
      Holds the query set fixed and varies only whether the retriever sees the
      whole complaint or just its opening, isolating the effect of truncation
      from the effect of changing the query set.

  EXPERIMENT 2 — Paired significance tests
      All systems run on the SAME queries, so per-query differences are paired.
      A paired bootstrap on those differences is the correct test and is far
      more sensitive than checking whether two marginal intervals overlap.

  TRACK B — Product classification (diagnostic, not the deployed task)
      Majority-class baseline, Complement Naive Bayes, TF-IDF + logistic
      regression, TF-IDF + linear SVC, embeddings + logistic regression, and a
      sparse+dense union. Stratified 80/20 split, 5-fold stratified CV for
      tuning. Macro-F1 is the primary metric given 31:1 class imbalance.

  EXPERIMENT 3 — Classification pooling
      Mean vs max vs concatenated pooling of a complaint's chunk vectors, to
      test whether the embedding/TF-IDF gap is a pooling artifact or a
      representational capacity limit.

LEAKAGE NOTE
    The `embedding_input` column produced by the data-prep notebook is prefixed
    with "Product: ... | Issue: ...", i.e. it contains both the classification
    target and the field used to define retrieval relevance. Embedding it
    produced a spurious macro-F1 of 0.971 versus 0.646 on raw narrative text.
    Every model here reads `narrative_clean` only, and assert_no_label_leakage()
    raises if a label-bearing column is ever passed to the encoder.

RUNTIME
    First run encodes ~95k chunks with two models (~10-15 min on CPU), then
    caches to data/chunks_*.npy. Later runs are ~10 min, dominated by the grid
    searches. Set QUICK = True for a fast 8,000-row single-model pass.

OUTPUTS (data/)
    modeling_retrieval.csv          Track A comparison, with bootstrap CIs
    modeling_ablation_chunking.csv  Experiment 1
    modeling_paired_tests.csv       Experiment 2
    modeling_classification.csv     Track B comparison
    modeling_pooling.csv            Experiment 3
    modeling_confusion_pairs.csv    top misclassification pairs
    modeling_top_features.csv       interpretability: top terms per class
"""

import re
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import normalize
from scipy.sparse import hstack, csr_matrix

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_STATE = 42
DATA_DIR = Path("data")
TEXT_COL = "narrative_clean"      # label-free text — the ONLY text any model sees
LABEL_COL = "product"

QUICK = False                     # True = 8k rows, one embedding model
N_BOOT = 5000                     # bootstrap resamples
K_VALUES = [5, 10]

# 160-word chunks with 40-word overlap sit inside both models' token limits, so
# the two models see identical text.
CHUNK_WORDS, CHUNK_OVERLAP = 160, 40

EMBED_MODELS = {
    # key: (huggingface id, query instruction prefix)
    "MiniLM-L6": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "BGE-small": ("BAAI/bge-small-en-v1.5",
                  "Represent this sentence for searching relevant passages: "),
}
if QUICK:
    EMBED_MODELS = {"MiniLM-L6": EMBED_MODELS["MiniLM-L6"]}

# Each query is paired with the CFPB issue label(s) a complaint must carry to
# count as relevant. This proxies human relevance judgments: imperfect, because
# consumers mislabel their own complaints (Bastani et al., 2019), but applied
# identically to every retriever, so the comparison stays fair.
EVAL_QUERIES = [
    ("A debt collector keeps calling me about a debt that is not mine",
     ["Attempts to collect debt not owed"]),
    ("I am being chased for a debt I already paid off",
     ["Attempts to collect debt not owed", "Written notification about debt"]),
    ("The collection agency never sent me written validation of the debt",
     ["Written notification about debt"]),
    ("A collector threatened to sue me or garnish my wages",
     ["Took or threatened to take negative or legal action"]),
    ("Debt collectors calling my workplace and my relatives",
     ["Communication tactics",
      "Threatened to contact someone or share information improperly"]),
    ("The collector lied about who they were and what I owed",
     ["False statements or representation"]),
    ("My bank charged me overdraft fees I did not authorize",
     ["Problem caused by your funds being low",
      "Problem with a lender or other company charging your account",
      "Managing an account"]),
    ("The bank closed my account without warning or explanation",
     ["Closing an account", "Closing your account"]),
    ("My account was frozen and I cannot access my own money",
     ["Managing an account", "Problem caused by your funds being low"]),
    ("There are fraudulent charges on my credit card",
     ["Problem with a purchase shown on your statement"]),
    ("I was charged fees and interest I never agreed to",
     ["Fees or interest", "Charged fees or interest you didn't expect"]),
    ("My credit card application was denied without a clear reason",
     ["Getting a credit card"]),
    ("A merchant refuses to refund me and the card issuer denied my dispute",
     ["Problem with a purchase shown on your statement",
      "Problem with a company's investigation into an existing problem"]),
    ("My mortgage payment was not applied correctly",
     ["Trouble during payment process"]),
    ("I am behind on my mortgage and cannot get a loan modification",
     ["Struggling to pay mortgage"]),
    ("Problems with the closing process on my home loan",
     ["Closing on a mortgage",
      "Applying for a mortgage or refinancing an existing mortgage"]),
    ("Problems getting my student loan payments counted for forgiveness",
     ["Dealing with your lender or servicer"]),
    ("I cannot afford my monthly loan payments and need help",
     ["Struggling to repay your loan", "Struggling to pay your loan"]),
    ("Someone opened an account in my name and my credit report is wrong",
     ["Incorrect information on your report"]),
    ("The credit bureau ignored my dispute about an error on my report",
     ["Problem with a company's investigation into an existing problem"]),
    ("A company pulled my credit report without my permission",
     ["Improper use of your report"]),
    ("My money transfer never arrived and the company will not refund me",
     ["Fraud or scam", "Money was not available when promised"]),
    ("I cannot access the funds in my digital wallet app",
     ["Trouble accessing funds in your mobile or digital wallet"]),
    ("My car was repossessed without proper notice",
     ["Repossession"]),
    ("The dealership misled me about my auto loan terms",
     ["Getting a loan or lease", "Managing the loan or lease"]),
    ("There were unauthorized transactions on my prepaid card",
     ["Problem with a purchase or transfer", "Trouble using the card"]),
]

# ══════════════════════════════════════════════════════════════════════════════
# Load and validate
# ══════════════════════════════════════════════════════════════════════════════
print("Loading model-ready complaints...")
df = pd.read_parquet(DATA_DIR / "complaints_model_ready.parquet")
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)

# Collapse classes under 200 complaints into "Other" so stratified splitting and
# macro-F1 stay stable. A guard rather than an active transform on this corpus.
counts = df[LABEL_COL].value_counts()
df[LABEL_COL] = df[LABEL_COL].where(~df[LABEL_COL].isin(counts[counts < 200].index),
                                    "Other")
if QUICK:
    df = df.sample(8000, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  {len(df):,} complaints, {df[LABEL_COL].nunique()} product classes")
print(df[LABEL_COL].value_counts().to_string())


def assert_no_label_leakage(texts, labels, n_check=500):
    """Fail loudly if the text passed to a model contains its own label."""
    hits = sum(1 for t, l in list(zip(texts, labels))[:n_check]
               if str(l).lower()[:20] in str(t).lower()[:200])
    if hits > n_check * 0.5:
        raise RuntimeError(
            f"LEAKAGE: {hits}/{n_check} texts contain their own product label. "
            f"TEXT_COL must be the raw narrative, not a metadata-prefixed field "
            f"such as `embedding_input`.")
    print(f"  Leakage check passed ({hits}/{n_check} incidental matches).")


assert_no_label_leakage(df[TEXT_COL].astype(str).tolist(), df[LABEL_COL].tolist())

# ══════════════════════════════════════════════════════════════════════════════
# Chunking
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nChunking narratives ({CHUNK_WORDS}-word windows, {CHUNK_OVERLAP} overlap)...")


def chunk_words(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    w = str(text).split()
    if len(w) <= size:
        return [" ".join(w)] if w else [""]
    step = size - overlap
    return [" ".join(w[i:i + size]) for i in range(0, len(w), step)
            if len(w[i:i + size]) > overlap // 2]


chunk_texts, chunk_owner, is_first_chunk = [], [], []
for doc_i, t in enumerate(df[TEXT_COL].astype(str)):
    for j, c in enumerate(chunk_words(t)):
        chunk_texts.append(c)
        chunk_owner.append(doc_i)
        is_first_chunk.append(j == 0)
chunk_owner = np.asarray(chunk_owner)
is_first_chunk = np.asarray(is_first_chunk)
print(f"  {len(df):,} complaints -> {len(chunk_texts):,} chunks "
      f"({len(chunk_texts) / len(df):.2f} per complaint)")

# ══════════════════════════════════════════════════════════════════════════════
# Embeddings (cached per model)
# ══════════════════════════════════════════════════════════════════════════════
print("\nEmbedding chunks...")


def embed_chunks(model_key):
    hf_id, _ = EMBED_MODELS[model_key]
    cache = DATA_DIR / f"chunks_{model_key}{'_quick' if QUICK else ''}.npy"
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(chunk_texts):
            print(f"  [{model_key}] loaded cache {emb.shape}")
            return emb
        print(f"  [{model_key}] cache size mismatch — re-encoding.")
    print(f"  [{model_key}] encoding {len(chunk_texts):,} chunks...")
    m = SentenceTransformer(hf_id)
    emb = m.encode(chunk_texts, batch_size=128, show_progress_bar=True,
                   convert_to_numpy=True, normalize_embeddings=True)
    np.save(cache, emb)
    return emb


chunk_emb = {k: embed_chunks(k) for k in EMBED_MODELS}
st_models = {k: SentenceTransformer(v[0]) for k, v in EMBED_MODELS.items()}
BEST_DENSE = "BGE-small" if "BGE-small" in chunk_emb else "MiniLM-L6"

# ══════════════════════════════════════════════════════════════════════════════
# Retrieval machinery
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding lexical indexes...")
t0 = time.time()
tokenized = [re.findall(r"[a-z0-9']+", t.lower()) for t in df[TEXT_COL].astype(str)]
bm25 = BM25Okapi(tokenized)
tfidf_ret = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2)
X_ret = tfidf_ret.fit_transform(df[TEXT_COL].astype(str))
print(f"  built in {time.time() - t0:.1f}s")


def bm25_rank(q):
    return np.argsort(bm25.get_scores(re.findall(r"[a-z0-9']+", q.lower())))[::-1]


def tfidf_rank(q):
    return np.argsort((X_ret @ tfidf_ret.transform([q]).T).toarray().ravel())[::-1]


def make_dense_rank(model_key, first_chunk_only=False):
    """Score each complaint by its BEST-matching chunk (max-pool).

    first_chunk_only=True simulates the pre-chunking behaviour, where the model
    silently truncated anything past its token limit.
    """
    ce, prefix = chunk_emb[model_key], EMBED_MODELS[model_key][1]
    st = st_models[model_key]
    mask = is_first_chunk if first_chunk_only else np.ones(len(ce), dtype=bool)
    ce_use, owner_use = ce[mask], chunk_owner[mask]

    def rank(q):
        qv = st.encode([prefix + q], convert_to_numpy=True,
                       normalize_embeddings=True).ravel()
        doc = np.full(len(df), -1.0, dtype=np.float32)
        np.maximum.at(doc, owner_use, ce_use @ qv)
        return np.argsort(doc)[::-1]
    return rank


def rrf(rankings, k=60, depth=200):
    """Reciprocal rank fusion — combines rankings without score calibration."""
    scores = {}
    for r in rankings:
        for rank_i, idx in enumerate(r[:depth], 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank_i)
    return np.array(sorted(scores, key=scores.get, reverse=True))


def per_query_metrics(rank_fn):
    """One row per query. Recall@k is deliberately omitted: relevant sets number
    in the thousands, so recall at k<=10 is bounded near zero and carries no
    information about ranking quality."""
    rows = []
    for q, issues in EVAL_QUERIES:
        rel = df["issue"].isin(issues).values
        if rel.sum() == 0:
            print(f"  ⚠ no relevant docs: {q[:55]}")
            continue
        r = rank_fn(q)
        row = {f"P@{k}": rel[r[:k]].sum() / k for k in K_VALUES}
        rr = 0.0
        for i, ix in enumerate(r[:100], 1):
            if rel[ix]:
                rr = 1.0 / i
                break
        row["MRR"] = rr
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean()
                      for _ in range(n_boot)])
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Bootstrap the mean of per-query differences (a - b), resampling queries
    and keeping both systems' scores for a query together."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return d.mean(), lo, hi, min(p, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TRACK A — Retrieval comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("TRACK A — RETRIEVAL COMPARISON")
print("═" * 70)

dense_full = {k: make_dense_rank(k) for k in chunk_emb}
_best = dense_full[BEST_DENSE]

systems = {"BM25 (lexical baseline)": bm25_rank, "TF-IDF cosine": tfidf_rank}
for k in chunk_emb:
    systems[f"Dense: {k}"] = dense_full[k]
systems[f"Hybrid RRF (BM25 + {BEST_DENSE})"] = lambda q: rrf([bm25_rank(q), _best(q)])

pq_cache, rows = {}, []
for name, fn in systems.items():
    print(f"\nEvaluating {name}...")
    pq = per_query_metrics(fn)
    pq_cache[name] = pq
    rec = {"system": name, "n_queries": len(pq)}
    for m in pq.columns:
        lo, hi = bootstrap_ci(pq[m])
        rec[m] = pq[m].mean()
        rec[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    rows.append(rec)
    print(f"  P@5={rec['P@5']:.3f} {rec['P@5_ci']}  "
          f"P@10={rec['P@10']:.3f}  MRR={rec['MRR']:.3f} {rec['MRR_ci']}")

retrieval = pd.DataFrame(rows).set_index("system")
retrieval.to_csv(DATA_DIR / "modeling_retrieval.csv")
print("\nTRACK A RESULTS")
print(retrieval[["P@5", "P@5_ci", "P@10", "MRR", "MRR_ci"]].round(3).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Chunking ablation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("EXPERIMENT 1 — CHUNKING ABLATION (query set held fixed)")
print("═" * 70)

abl_rows = []
for k in chunk_emb:
    for label, first_only in [("first chunk only (truncated)", True),
                              ("all chunks", False)]:
        pq = (per_query_metrics(make_dense_rank(k, first_chunk_only=True))
              if first_only else pq_cache[f"Dense: {k}"])
        abl_rows.append({"system": f"{k} — {label}", **pq.mean().to_dict()})
        print(f"  {k} — {label:30s} P@5={pq['P@5'].mean():.3f}  "
              f"MRR={pq['MRR'].mean():.3f}")

ablation = pd.DataFrame(abl_rows).set_index("system")
ablation.to_csv(DATA_DIR / "modeling_ablation_chunking.csv")
print("\nEffect of chunking alone (same queries, same model):")
for k in chunk_emb:
    a = ablation.loc[f"{k} — first chunk only (truncated)"]
    b = ablation.loc[f"{k} — all chunks"]
    print(f"  {k}: P@5 {a['P@5']:.3f} -> {b['P@5']:.3f} ({b['P@5'] - a['P@5']:+.3f}), "
          f"MRR {a['MRR']:.3f} -> {b['MRR']:.3f} ({b['MRR'] - a['MRR']:+.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Paired significance tests
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("EXPERIMENT 2 — PAIRED BOOTSTRAP TESTS")
print("═" * 70)

bm25_name = "BM25 (lexical baseline)"
dense_name = f"Dense: {BEST_DENSE}"
hybrid_name = f"Hybrid RRF (BM25 + {BEST_DENSE})"
comparisons = [(dense_name, bm25_name), (hybrid_name, bm25_name),
               (hybrid_name, dense_name)]

test_rows = []
for a_name, b_name in comparisons:
    for metric in ["P@5", "P@10", "MRR"]:
        a, b = pq_cache[a_name][metric], pq_cache[b_name][metric]
        diff, lo, hi, p = paired_bootstrap(a, b)
        wins = int((a.values > b.values).sum())
        losses = int((a.values < b.values).sum())
        ties = int((a.values == b.values).sum())
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        test_rows.append({"comparison": f"{a_name} vs {b_name}", "metric": metric,
                          "mean_diff": diff, "ci_low": lo, "ci_high": hi,
                          "p_value": p, "significant_95": sig,
                          "win_loss_tie": f"{wins}/{losses}/{ties}"})
        print(f"  {a_name} vs {b_name} [{metric}]: {diff:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] p={p:.3f} sig={sig}  W/L/T={wins}/{losses}/{ties}")

pd.DataFrame(test_rows).to_csv(DATA_DIR / "modeling_paired_tests.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# TRACK B — Product classification
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("TRACK B — PRODUCT CLASSIFICATION (diagnostic)")
print("═" * 70)

# np.asarray(...astype(str)) is required: parquet loads this column as an
# Arrow-backed extension array, which sklearn cannot index during CV.
y = np.asarray(df[LABEL_COL].astype(str))
i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.20,
                              stratify=y, random_state=RANDOM_STATE)
y_tr, y_te = y[i_tr], y[i_te]
txt_tr = df[TEXT_COL].astype(str).values[i_tr]
txt_te = df[TEXT_COL].astype(str).values[i_te]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Stratified rather than time-based: this classifier diagnoses representational
# quality, not forecasting. Class proportions must match across splits for
# macro-F1 to be comparable.

tfidf_clf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2)
Xs_tr = tfidf_clf.fit_transform(txt_tr)
Xs_te = tfidf_clf.transform(txt_te)


def pool_chunks(model_key, how="mean"):
    """Complaint-level vectors from chunk vectors."""
    ce = chunk_emb[model_key]
    if how == "mean":
        acc = np.zeros((len(df), ce.shape[1]), dtype=np.float32)
        np.add.at(acc, chunk_owner, ce)
        n = np.bincount(chunk_owner, minlength=len(df)).reshape(-1, 1)
        return normalize(acc / np.maximum(n, 1))
    acc = np.full((len(df), ce.shape[1]), -1e9, dtype=np.float32)
    np.maximum.at(acc, chunk_owner, ce)
    return normalize(acc)


doc_emb = pool_chunks(BEST_DENSE, "mean")
Xd_tr, Xd_te = doc_emb[i_tr], doc_emb[i_te]
Xu_tr = hstack([Xs_tr, csr_matrix(Xd_tr)]).tocsr()
Xu_te = hstack([Xs_te, csr_matrix(Xd_te)]).tocsr()


def score(name, tr_pred, te_pred, extra=None):
    r = {"model": name,
         "train_accuracy": accuracy_score(y_tr, tr_pred),
         "test_accuracy": accuracy_score(y_te, te_pred),
         "train_macro_f1": f1_score(y_tr, tr_pred, average="macro"),
         "test_macro_f1": f1_score(y_te, te_pred, average="macro"),
         "test_weighted_f1": f1_score(y_te, te_pred, average="weighted")}
    if extra:
        r.update(extra)
    return r


def tune(est, grid, Xtr, Xte, name):
    g = GridSearchCV(est, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    g.fit(Xtr, y_tr)
    b = g.best_estimator_
    print(f"  {name}: best={g.best_params_}, CV macro-F1={g.best_score_:.3f}")
    return score(name, b.predict(Xtr), b.predict(Xte),
                 {"cv_macro_f1": g.best_score_,
                  "best_params": str(g.best_params_)}), b


results, fitted = [], {}

print("\n[1/6] Majority-class baseline...")
dm = DummyClassifier(strategy="most_frequent").fit(Xs_tr, y_tr)
results.append(score("Majority-class baseline", dm.predict(Xs_tr), dm.predict(Xs_te)))
fitted["Majority-class baseline"] = dm.predict(Xs_te)

print("[2/6] Complement Naive Bayes (TF-IDF)...")
r, m = tune(ComplementNB(), {"alpha": [0.1, 0.5, 1.0]}, Xs_tr, Xs_te,
            "ComplementNB + TF-IDF")
results.append(r); fitted[r["model"]] = m.predict(Xs_te)

print("[3/6] Logistic regression (TF-IDF)...")
r, lr = tune(LogisticRegression(max_iter=2000, class_weight="balanced"),
             {"C": [1.0, 10.0]}, Xs_tr, Xs_te, "TF-IDF + LogisticRegression")
results.append(r); fitted[r["model"]] = lr.predict(Xs_te)

print("[4/6] Linear SVC (TF-IDF)...")
r, m = tune(LinearSVC(class_weight="balanced"), {"C": [0.1, 0.5, 1.0]},
            Xs_tr, Xs_te, "TF-IDF + LinearSVC")
results.append(r); fitted[r["model"]] = m.predict(Xs_te)

print(f"[5/6] Logistic regression ({BEST_DENSE} embeddings)...")
r, m = tune(LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"C": [1.0, 10.0]}, Xd_tr, Xd_te,
            f"{BEST_DENSE} embeddings + LogisticRegression")
results.append(r); fitted[r["model"]] = m.predict(Xd_te)

print("[6/6] Union: TF-IDF + embeddings...")
r, m = tune(LinearSVC(class_weight="balanced"), {"C": [0.1, 0.5]},
            Xu_tr, Xu_te, "Union (TF-IDF + embeddings) + LinearSVC")
results.append(r); fitted[r["model"]] = m.predict(Xu_te)

classification = pd.DataFrame(results).set_index("model")
classification.to_csv(DATA_DIR / "modeling_classification.csv")
print("\nTRACK B RESULTS")
print(classification.round(3).to_string())

# --- Interpretability and error analysis --------------------------------------
feat = np.array(tfidf_clf.get_feature_names_out())
pd.DataFrame.from_dict(
    {c: feat[np.argsort(lr.coef_[i])[::-1][:10]].tolist()
     for i, c in enumerate(lr.classes_)}, orient="index"
).to_csv(DATA_DIR / "modeling_top_features.csv")

best_name = classification["test_macro_f1"].idxmax()
best_pred = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_te, best_pred))

labels = sorted(set(y))
cm = confusion_matrix(y_te, best_pred, labels=labels)
pairs = sorted([(cm[i, j], cm[i, j] / cm[i].sum(), a, b)
                for i, a in enumerate(labels) for j, b in enumerate(labels)
                if i != j and cm[i, j] > 0], reverse=True)
pd.DataFrame(pairs[:15], columns=["n_errors", "pct_of_true_class",
                                  "true_product", "predicted_product"]
             ).to_csv(DATA_DIR / "modeling_confusion_pairs.csv", index=False)
print("TOP CONFUSION PAIRS")
for n, pct, a, b in pairs[:8]:
    print(f"  {n:>4} ({pct:.0%})  {a[:36]:38s} → {b[:36]}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Classification pooling
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("EXPERIMENT 3 — CLASSIFICATION POOLING (mean vs max vs concat)")
print("═" * 70)

mean_pool = doc_emb
max_pool = pool_chunks(BEST_DENSE, "max")
pool_rows = []
for name, X in [("mean-pool", mean_pool), ("max-pool", max_pool),
                ("concat [mean|max]", np.hstack([mean_pool, max_pool]))]:
    g = GridSearchCV(LogisticRegression(max_iter=2000, class_weight="balanced"),
                     {"C": [1.0, 10.0]}, scoring="f1_macro", cv=cv, n_jobs=-1)
    g.fit(X[i_tr], y_tr)
    pred = g.best_estimator_.predict(X[i_te])
    pool_rows.append({"pooling": name, "dims": X.shape[1],
                      "cv_macro_f1": g.best_score_,
                      "test_accuracy": accuracy_score(y_te, pred),
                      "test_macro_f1": f1_score(y_te, pred, average="macro")})
    print(f"  {name:20s} dims={X.shape[1]:>4}  "
          f"test macro-F1={pool_rows[-1]['test_macro_f1']:.3f}")

pooling = pd.DataFrame(pool_rows).set_index("pooling")
pooling.to_csv(DATA_DIR / "modeling_pooling.csv")

tfidf_ref = classification.loc["TF-IDF + LinearSVC", "test_macro_f1"]
best_pool = pooling["test_macro_f1"].max()
print(f"\n  TF-IDF + LinearSVC reference: {tfidf_ref:.3f}")
print(f"  Best embedding pooling:       {best_pool:.3f}  "
      f"(gap {best_pool - tfidf_ref:+.3f})")
if best_pool > pooling.loc["mean-pool", "test_macro_f1"] + 0.02:
    print("  -> Pooling hypothesis SUPPORTED: mean-pooling was diluting signal.")
else:
    print("  -> Pooling hypothesis NOT supported: the gap is a capacity limit, "
          "not a pooling artifact.")

print("\n✅ Done. Seven modeling_*.csv files written to data/.")
