"""
gemini_client.py — Shared answer-generation layer.

Single source of truth for which Gemini model is used and how failures are
handled. 03_copilot_rag.py, app.py, and test_pipeline.py all call through here,
so the model name lives in one place and cannot drift between the CLI, the
deployed app, and the tests.

WHY THE RETRY LOGIC EXISTS
    Google's Flash models return 503 UNAVAILABLE under high demand. A single
    transient 503 previously aborted a whole test run, and would crash the
    Streamlit app mid-question. Requests are now retried with exponential
    backoff, and if the primary model stays unavailable the request falls back
    to a second model before giving up.

    429 (rate limit) is also retried, since the free tier allows roughly ten
    requests per minute and a burst of questions will hit it.
"""

import os
import re
import time

MODEL = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.6-flash"   # tried once if MODEL stays unavailable

# On Gemini 3 models this budget covers internal reasoning tokens as well as the
# visible answer, so keep it well above the length of answer actually wanted.
# Too low and reasoning consumes it all, returning empty text.
MAX_OUTPUT_TOKENS = 2048

MAX_ATTEMPTS = 4
RETRYABLE_CODES = {429, 500, 502, 503, 504}


def has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY"))


def _status_code(exc) -> int | None:
    """Pull the HTTP status out of a google-genai error, defensively."""
    for attr in ("code", "status_code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


def generate(prompt: str, system_instruction: str,
             max_output_tokens: int = MAX_OUTPUT_TOKENS,
             temperature: float = 0.2,
             model: str | None = None,
             verbose: bool = True) -> str:
    """Generate text, retrying transient failures and falling back if needed.

    Raises RuntimeError on empty output, and re-raises the underlying API error
    if every attempt fails.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()          # reads GEMINI_API_KEY from the environment
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    models_to_try = [model or MODEL]
    if not model and FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None
    for model_name in models_to_try:
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = client.models.generate_content(
                    model=model_name, contents=prompt, config=config)
                text = resp.text
                if not text:
                    raise RuntimeError(
                        "Gemini returned no text. The token budget was likely "
                        "consumed by internal reasoning, or the response was "
                        "filtered. Try raising MAX_OUTPUT_TOKENS.")
                return text

            except RuntimeError:
                raise                      # empty output is not retryable

            except Exception as e:         # google.genai.errors.APIError et al.
                last_error = e
                code = _status_code(e)
                final_attempt = attempt == MAX_ATTEMPTS - 1
                if code not in RETRYABLE_CODES or final_attempt:
                    break
                wait = 2 ** attempt        # 1s, 2s, 4s
                if verbose:
                    print(f"      {code} from {model_name} — retrying in "
                          f"{wait}s (attempt {attempt + 2}/{MAX_ATTEMPTS})")
                time.sleep(wait)

        if verbose and model_name != models_to_try[-1]:
            print(f"      {model_name} unavailable — falling back to "
                  f"{models_to_try[-1]}")

    raise last_error


# ── Citation parsing ─────────────────────────────────────────────────────────
# Models cite in two formats: one bracket per source ("[1] [3]") and grouped
# ("[1, 2, 3]"). An earlier regex only matched the first, so a fully-cited
# answer was scored as having no citations at all. This handles both.
CITATION_RE = re.compile(r"\[([\d,\s]+)\]")


def extract_citations(text: str) -> list[int]:
    """Return every complaint number cited in the text, sorted and deduplicated."""
    found = set()
    for group in CITATION_RE.findall(text):
        for n in re.findall(r"\d+", group):
            found.add(int(n))
    return sorted(found)
