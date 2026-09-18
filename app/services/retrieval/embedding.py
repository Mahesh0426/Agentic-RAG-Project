import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


# ── Model initialisation ───────────────────────────────────────────────────────
# reusable function is to test whether Gemini embeddings are working.
def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None


# Fallback model - it will load a local embedding model if Gemini is unavailable.
def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


# Lazy Initialization -  This function chooses which embedding model your application will use.
def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Dimension Public helpers ─────────────────────────────────────────────────────────────
# It will tell application the dimension of the active embedding model.
def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────
# This function is used to embed texts in batches of 50 eg :
# Chunks 1-50 -> Gemini , Chunks 51-100 -> Gemini, Chunks 101-120 -> Gemini

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Exponential backoff: 1 s → 2 s → 4 s → 8 s (4 attempts total)
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────
# this function is used to embed a single query
def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


# ── Bulk Text Ingestion ────────────────────────────────────────────────────────
# Converts multiple document chunks into vector embeddings (used during ingestion)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts into dense vectors in batches of BATCH_SIZE.
    Returns a list of vectors matching the order of input texts.
    """
    # 1. Ensure the active embedding model (Gemini or fallback) is initialized
    _init()
    
    all_embeddings: list[list[float]] = []
    
    # 2. Slice texts into smaller batches (e.g. 0-50, 50-100) to avoid API timeout/payload limits
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        
        # 3. Logfire span traces duration, active model, and progress for this specific batch
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            # 4. Embed the current batch (with retry if rate limited) and append results
            all_embeddings.extend(_embed_batch(batch))
            
    # 5. Return all generated vector embeddings
    return all_embeddings