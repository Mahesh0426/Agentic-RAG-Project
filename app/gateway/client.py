# ============================================================
# LLM GATEWAY CLIENT (PORTKEY)
# Purpose: Manages communication with LLM providers through the Portkey AI Gateway.
# Provides automatic fallback between models, response caching, request retries,
# and centralized observability metadata for LangChain and LangGraph nodes.
# ============================================================
import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from app.config import settings


# ============================================================
# CONFIGURATION: GATEWAY_CONFIG
# Purpose: Defines routing strategy, caching mode, retries, and fallback model targets.
#   - Fallback: primary @rag/llama-3.3-70b-versatile → @brag/llama-3.1-8b-instant on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
# ============================================================
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-120b"}},
    ]
}

# Saved config ID (pc-...) if configured in environment, otherwise None (to prevent inline_config_blocked)
ACTIVE_PORTKEY_CONFIG = settings.PORTKEY_CONFIG_ID if settings.PORTKEY_CONFIG_ID else None

# Standalone Portkey native client instance configured with gateway rules
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    **({"config": ACTIVE_PORTKEY_CONFIG} if ACTIVE_PORTKEY_CONFIG else {})
)


# ============================================================
# FUNCTION: get_langchain_llm
# Purpose: Factory function returning a Portkey-proxied ChatOpenAI instance
#          compatible with LangChain and LangGraph pipeline nodes.
# ============================================================
def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    # 1. Build Portkey header payload
    header_kwargs = {
        "api_key": settings.PORTKEY_API_KEY,
        "metadata": {
            "feature": feature,
            "_user": "rag-system",
            "environment": "production"
        }
    }
    if ACTIVE_PORTKEY_CONFIG:
        header_kwargs["config"] = ACTIVE_PORTKEY_CONFIG

    # 2. Instantiate ChatOpenAI pointed at Portkey's reverse-proxy URL instead of OpenAI
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        # Reference the virtual model slug registered in Portkey
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,
        # Attach custom Portkey headers containing auth and tracing metadata
        default_headers=createHeaders(**header_kwargs)
    )


# ============================================================
# FUNCTION: extract_cache_status
# Purpose: Defensively inspect HTTP response objects to determine
#          whether Portkey served a cached response or made a live LLM call.
# ============================================================
def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    # 1. Iterate through common underlying response attribute names across client versions
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        # 2. If the raw response object exists, inspect its HTTP headers
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            # 3. If the cache status header is present, return uppercase value (e.g. 'HIT' or 'MISS')
            if status:
                return status.upper()

    # 4. Default to 'MISS' if headers cannot be resolved or header was omitted
    return "MISS"