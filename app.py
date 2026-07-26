"""
app.py — Complaint Intelligence dashboard.

Run: streamlit run app.py

Shares its retrieval layer with 03_copilot_rag.py via retrieval.py, so the
deployed tool runs the configuration evaluated in 04_copilot_modeling.py:
BGE-small embeddings over the cleaned, deduplicated corpus.

Answer generation uses Google Gemini and requires GEMINI_API_KEY. Without it the
app still performs semantic search and shows source complaints, but produces no
written answer — it does not fabricate a summary, because an ungrounded summary
presented next to real citations would be misleading.
"""

import os

import pandas as pd
import streamlit as st

import retrieval
import gemini_client

GEMINI_MODEL = gemini_client.MODEL

# On Gemini 3 models this budget covers internal reasoning tokens as well as the
# visible answer, so keep it generous or responses come back empty.
MAX_OUTPUT_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are a consumer-complaint analyst supporting compliance and regulatory "
    "review. Answer using ONLY the numbered complaints provided. Support every "
    "factual claim with a bracketed citation such as [2], naming the SPECIFIC "
    "complaint that claim rests on — do not list every complaint after every "
    "sentence. Never use outside knowledge. If the complaints do not answer "
    "the question, say so plainly. Be concise: one short paragraph, then up "
    "to four bullet points."
)

st.set_page_config(page_title="Complaint Intelligence", page_icon="🔍",
                   layout="wide")


# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return retrieval.load_model()


@st.cache_data
def get_corpus():
    return retrieval.load_corpus()


@st.cache_data
def get_embeddings(n_rows: int):
    """n_rows busts the cache if the corpus changes."""
    return retrieval.build_or_load_embeddings(get_corpus(), get_model(),
                                              progress=False)


df = get_corpus()
model = get_model()

if not retrieval.EMB_PATH.exists():
    st.info("Building the semantic index on first load. This takes a few "
            "minutes and only happens once.")
embeddings = get_embeddings(len(df))

def _load_api_key() -> bool:
    """Find the key in Streamlit secrets (Cloud) or the environment (local).

    Streamlit Cloud has no shell, so the key is set under Settings -> Secrets.
    Copying it into the environment lets gemini_client stay unaware of which
    it came from.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return True
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    if key:
        os.environ["GEMINI_API_KEY"] = key
        return True
    return False


HAS_KEY = _load_api_key()


# ── Answer generation ────────────────────────────────────────────────────────
def generate_answer(question: str, retrieved: pd.DataFrame) -> dict:
    """Delegates to gemini_client, which retries transient 503/429 failures."""
    text = gemini_client.generate(
        prompt=f"Complaints:\n{retrieval.build_context(retrieved)}"
               f"\n\nQuestion: {question}",
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        verbose=False,             # retries would otherwise print to the console
    )

    cited = gemini_client.extract_citations(text)
    valid = set(range(1, len(retrieved) + 1))
    return {"text": text,
            "cited": [c for c in cited if c in valid],
            "invalid": [c for c in cited if c not in valid]}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Complaint Intelligence")
    st.caption(f"**{len(df):,}** complaints indexed")
    st.divider()
    product_filter = st.selectbox(
        "Filter by product",
        ["All"] + sorted(df["product"].dropna().unique().tolist()))
    top_k = st.slider("Complaints to retrieve", 3, 10, 5)
    st.divider()
    if retrieval.IS_DEMO:
        st.caption("Demo corpus \u2014 a stratified sample of the full "
                   "44,234-complaint dataset. Reported evaluation results use "
                   "the complete corpus.")
    st.caption(f"Embeddings: `{retrieval.EMBED_MODEL}`")
    if HAS_KEY:
        st.caption(f"Answers: `{GEMINI_MODEL}`")
    else:
        st.warning("No API key found — search works, but no written answers. "
                   "Set GEMINI_API_KEY locally, or add it under Settings → "
                   "Secrets on Streamlit Cloud.")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🔍 Complaint Intelligence")
st.caption("Ask a question about consumer complaints and get an answer grounded "
           "in specific, traceable complaint narratives.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Complaints", f"{len(df):,}")
c2.metric("Products", df["product"].nunique())
c3.metric("Companies", df["company"].nunique())
c4.metric("States", df["state"].nunique() if "state" in df.columns else "—")
st.divider()

# ── Query ────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "What problems are consumers reporting with debt collectors?",
    "Why are consumers disputing credit card charges?",
    "What goes wrong during the mortgage payment process?",
    "What issues do consumers face with money transfer apps?",
]

if "query" not in st.session_state:
    st.session_state.query = ""

st.subheader("Ask a question")
cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(ex[:32] + "…", key=f"ex_{i}", use_container_width=True):
        st.session_state.query = ex

query = st.text_input("Your question:", value=st.session_state.query,
                      placeholder="e.g. What are consumers saying about "
                                  "credit reporting errors?")

if query:
    with st.spinner("Searching complaints..."):
        retrieved = retrieval.retrieve(query, df, embeddings, model,
                                       top_k=top_k,
                                       product_filter=product_filter)

    if retrieved.empty:
        st.warning("No matching complaints found. Try rephrasing, or widen the "
                   "product filter.")
    else:
        if HAS_KEY:
            with st.spinner("Reading the complaints and writing an answer..."):
                try:
                    result = generate_answer(query, retrieved)
                    st.markdown("### Answer")
                    st.markdown(result["text"])
                    if result["cited"]:
                        st.caption("Cited complaints: " +
                                   ", ".join(f"[{c}]" for c in result["cited"]))
                    if result["invalid"]:
                        st.warning(
                            f"The answer cited {result['invalid']}, which is "
                            f"outside the retrieved set. Treat those claims as "
                            f"unverified.")
                except Exception as e:
                    st.error(f"Answer generation failed: {e}")
                    st.caption("Source complaints are still shown below.")
        else:
            st.info("Set GEMINI_API_KEY to generate a written answer. The "
                    "retrieved source complaints are shown below.")

        st.markdown(f"### Source complaints ({len(retrieved)})")
        st.caption("Numbered to match the citations above.")
        for i, (_, row) in enumerate(retrieved.iterrows(), 1):
            with st.expander(
                f"[{i}] {row.get('company', 'Unknown')} — "
                f"{row.get('product', 'Unknown')} — {row.get('issue', 'Unknown')} "
                f"(similarity {row['similarity']:.2f})"
            ):
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"**State:** {row.get('state', 'N/A')}")
                m2.markdown(f"**Company response:** "
                            f"{row.get('company_response', 'N/A')}")
                m3.markdown(f"**Timely:** {row.get('timely', 'N/A')}")
                if pd.notna(row.get("date_received", None)):
                    st.caption(f"Received {row['date_received']:%B %d, %Y}")
                st.markdown("**Narrative:**")
                st.write(str(row[retrieval.TEXT_COL]))

st.divider()

# ── Trends ───────────────────────────────────────────────────────────────────
st.subheader("📊 Complaint trends")
t1, t2 = st.columns(2)
with t1:
    st.markdown("**Top products**")
    st.bar_chart(df["product"].value_counts().head(8))
with t2:
    st.markdown("**Top issues**")
    st.bar_chart(df["issue"].value_counts().head(8))

st.markdown("**Top companies by complaint volume**")
st.bar_chart(df["company"].value_counts().head(10))

if "date_received" in df.columns and df["date_received"].notna().any():
    st.markdown("**Complaints over time**")
    st.line_chart(df.set_index("date_received").resample("W").size())
