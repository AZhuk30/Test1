"""
retrieval.py — Shared retrieval layer for Complaint Intelligence.

Single source of truth for how complaints are embedded and searched. Both
03_copilot_rag.py and app.py import from here, so the deployed artifact can
never drift away from the configuration that 04_copilot_modeling.py evaluated.

CONFIGURATION AND WHY
    Corpus:  data/complaints_model_ready.parquet (cleaned and deduplicated).
             Using the raw file would let near-identical complaints appear as
             separate supporting citations, which is exactly what a
             citation-backed tool must not do.
    Text:    narrative_clean — the label-free narrative. Never `embedding_input`,
             which is prefixed with "Product: ... | Issue: ..." and leaks the
             category into the vector (see the leakage note in 04).
    Model:   BAAI/bge-small-en-v1.5. Beat all-MiniLM-L6-v2 and BM25 on the
             26-query evaluation (P@5 0.592 vs 0.531 and 0.431).
             Queries need the instruction prefix below; documents do not.
    Pooling: one vector per complaint, encoded from the full narrative. The
             chunking ablation in 04 found no benefit from splitting long
             narratives (P@5 0.600 truncated vs 0.592 chunked), so the simpler
             document-level index is used here.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")

# Full corpus, used locally and for every reported evaluation number.
FULL_CORPUS_PATH = DATA_DIR / "complaints_model_ready.parquet"
FULL_EMB_PATH = DATA_DIR / "embeddings_bge_docs.npy"

# Small committed subset, used only where the full embedding cache is absent —
# in practice, Streamlit Cloud, where encoding 44k narratives on boot would
# exceed the timeout and the memory limit. Built by make_demo_subset.py.
DEMO_CORPUS_PATH = DATA_DIR / "complaints_demo.parquet"
DEMO_EMB_PATH = DATA_DIR / "embeddings_demo.npy"


def resolve_dataset() -> tuple[Path, Path, bool]:
    """Return (corpus_path, embedding_path, is_demo).

    Prefers the full corpus whenever its embedding cache exists, so local runs
    and all evaluation use the complete data. Falls back to the demo subset
    only when the full cache is missing but the demo files are present.
    """
    if FULL_CORPUS_PATH.exists() and FULL_EMB_PATH.exists():
        return FULL_CORPUS_PATH, FULL_EMB_PATH, False
    if DEMO_CORPUS_PATH.exists() and DEMO_EMB_PATH.exists():
        return DEMO_CORPUS_PATH, DEMO_EMB_PATH, True
    # Nothing cached: fall back to the full corpus and encode it.
    return FULL_CORPUS_PATH, FULL_EMB_PATH, False


CORPUS_PATH, EMB_PATH, IS_DEMO = resolve_dataset()

TEXT_COL = "narrative_clean"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Columns surfaced alongside each retrieved complaint, when present.
DISPLAY_COLS = ["complaint_id", "date_received", "product", "sub_product",
                "issue", "sub_issue", "company", "state", "company_response",
                "timely"]


def load_corpus() -> pd.DataFrame:
    """Load the cleaned, deduplicated corpus used for all evaluation."""
    corpus_path, _, _ = resolve_dataset()
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run 02_copilot_data_prep.ipynb first, "
            f"or make_demo_subset.py to build the smaller demo corpus.")
    df = pd.read_parquet(corpus_path)
    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"],
                                             errors="coerce", utc=True)
    return df


def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def build_or_load_embeddings(df: pd.DataFrame, model: SentenceTransformer,
                             progress: bool = True) -> np.ndarray:
    """Return unit-normalised document vectors, encoding once and caching."""
    _, emb_path, _ = resolve_dataset()
    if emb_path.exists():
        emb = np.load(emb_path)
        if len(emb) == len(df):
            return emb
        print("  Embedding cache size mismatch — re-encoding.")
    texts = df[TEXT_COL].astype(str).tolist()
    emb = model.encode(texts, batch_size=128, show_progress_bar=progress,
                       convert_to_numpy=True, normalize_embeddings=True)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    return emb


def retrieve(query: str, df: pd.DataFrame, embeddings: np.ndarray,
             model: SentenceTransformer, top_k: int = 5,
             product_filter: str | None = None) -> pd.DataFrame:
    """Return the top_k most semantically similar complaints, best first."""
    qv = model.encode([QUERY_PREFIX + query], convert_to_numpy=True,
                      normalize_embeddings=True).ravel()
    scores = embeddings @ qv                      # unit vectors -> cosine
    if product_filter and product_filter != "All":
        scores = np.where((df["product"] == product_filter).values, scores, -1.0)
    top_k = min(top_k, len(df))
    top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    out = df.iloc[top_idx].copy()
    out["similarity"] = scores[top_idx]
    return out[out["similarity"] > 0]


def build_context(retrieved: pd.DataFrame, char_limit: int = 1200) -> str:
    """Format retrieved complaints as numbered evidence for the generator."""
    blocks = []
    for i, (_, row) in enumerate(retrieved.iterrows(), 1):
        header = (f"[{i}] Product: {row.get('product', 'Unknown')} | "
                  f"Issue: {row.get('issue', 'Unknown')} | "
                  f"Company: {row.get('company', 'Unknown')}")
        blocks.append(f"{header}\n{str(row[TEXT_COL])[:char_limit]}")
    return "\n\n".join(blocks)
