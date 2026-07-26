"""
03_copilot_rag.py — Citation-backed question answering over CFPB complaints.

Run after 02_copilot_data_prep.ipynb. Retrieval configuration lives in
retrieval.py, which app.py also imports, so the command-line pipeline and the
deployed artifact always run the same evaluated setup.

This script generates an answer grounded in retrieved complaints and requires
every claim to carry a bracketed citation pointing at one of them. Answers
without a traceable source are the failure mode that motivated the whole
approach (Lewis et al., 2020; Ji et al., 2023), so the prompt forbids using
outside knowledge and the output is checked for unresolvable citations.

Answer generation uses the Google Gen AI SDK (`google-genai`). The older
`google-generativeai` package is deprecated — uninstall it if present, since
the two collide.

Usage:
    export GEMINI_API_KEY=...
    python 03_copilot_rag.py
    python 03_copilot_rag.py --question "Why are consumers disputing auto loans?"
"""

import os
import re
import argparse

import retrieval
import gemini_client

# Model name, token budget, and retry policy live in gemini_client.py.
# Legacy note: gemini-3.5-flash is the current general-purpose default. Switch to a Pro model
# if answers need stronger multi-step reasoning.
GEMINI_MODEL = "gemini-3.5-flash"
TOP_K = 5

# On Gemini 3 models this budget is shared between internal reasoning tokens and
# visible output, so it is set well above the length of answer we actually want.
# Set it too low and reasoning consumes everything, returning empty text.
MAX_OUTPUT_TOKENS = 2048

SYSTEM_PROMPT = (
    "You are a consumer-complaint analyst supporting compliance and regulatory "
    "review. Answer using ONLY the numbered complaints provided. Follow these "
    "rules exactly:\n"
    "1. Support every factual claim with a bracketed citation, e.g. [2], "
    "naming the complaint it came from. A sentence with no citation is not "
    "acceptable unless it is an explicit statement about what the evidence "
    "does not show.\n"
    "1a. Cite the SPECIFIC complaint each claim rests on. Do not list every "
    "complaint after every sentence. Cite two or more only when a claim "
    "genuinely draws on several, and prefer naming the one clearest source.\n"
    "2. Never use outside knowledge, and never infer beyond what the "
    "complaints state.\n"
    "3. If the complaints do not answer the question, say so plainly and "
    "describe what they do cover instead.\n"
    "4. Be concise: at most one short paragraph, then up to four bullet points."
)


def has_api_key() -> bool:
    return gemini_client.has_api_key()


def generate(question: str, context: str) -> str:
    """Call Gemini with the retrieved complaints as the only evidence.

    Delegates to gemini_client, which retries transient 503/429 failures and
    falls back to a second model if the primary stays unavailable.
    """
    return gemini_client.generate(
        prompt=f"Complaints:\n{context}\n\nQuestion: {question}",
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,           # low: this is a factual grounding task
    )


def answer(question: str, df, embeddings, model, top_k: int = TOP_K) -> dict:
    """Retrieve evidence, generate a grounded answer, and validate citations."""
    retrieved = retrieval.retrieve(question, df, embeddings, model, top_k=top_k)

    if retrieved.empty:
        return {"question": question, "answer": "No matching complaints found.",
                "sources": [], "citations_used": [], "invalid_citations": []}

    if not has_api_key():
        listing = "\n".join(
            f"[{i}] {r.get('product', '?')} — {r.get('issue', '?')} "
            f"({r.get('company', '?')})"
            for i, (_, r) in enumerate(retrieved.iterrows(), 1))
        return {"question": question,
                "answer": ("GEMINI_API_KEY not set, so no answer was generated. "
                           f"Retrieved complaints:\n{listing}"),
                "sources": retrieved.to_dict("records"),
                "citations_used": [], "invalid_citations": []}

    text = generate(question, retrieval.build_context(retrieved))

    # A citation pointing outside the retrieved set means the answer is not
    # traceable, which defeats the purpose. Surface it rather than hide it.
    cited = gemini_client.extract_citations(text)
    valid = set(range(1, len(retrieved) + 1))
    return {"question": question, "answer": text,
            "sources": retrieved.to_dict("records"),
            "citations_used": [c for c in cited if c in valid],
            "invalid_citations": [c for c in cited if c not in valid]}


DEFAULT_QUESTIONS = [
    "What problems are consumers reporting with debt collectors?",
    "Why are consumers disputing credit card charges?",
    "What goes wrong during the mortgage payment process?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    print("Loading corpus...")
    df = retrieval.load_corpus()
    print(f"  {len(df):,} complaints")

    print(f"Loading {retrieval.EMBED_MODEL}...")
    model = retrieval.load_model()

    print("Preparing embeddings (first run encodes and caches)...")
    embeddings = retrieval.build_or_load_embeddings(df, model)
    print(f"  {embeddings.shape}")

    if has_api_key():
        print(f"\nAnswer generation: ON ({GEMINI_MODEL})")
    else:
        print("\nAnswer generation: OFF — set GEMINI_API_KEY to enable")

    questions = [args.question] if args.question else DEFAULT_QUESTIONS
    for q in questions:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        result = answer(q, df, embeddings, model, top_k=args.top_k)
        print(f"\n{result['answer']}\n")
        print("Sources:")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s.get('company', '?')} — {s.get('product', '?')} "
                  f"— {s.get('issue', '?')} (similarity {s['similarity']:.3f})")
        if result["invalid_citations"]:
            print(f"  ⚠ answer cited complaints that were not retrieved: "
                  f"{result['invalid_citations']}")

    print("\n✅ Done. Run `streamlit run app.py` for the dashboard.")


if __name__ == "__main__":
    main()
