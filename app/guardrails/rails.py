#The purpose of rails.py is to act as a security and conversational gatekeeper 
# for the application before any query enters the main RAG pipeline.
import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


# ============================================================
# FUNCTION: initialize_rails
# Purpose: Build and cache the NeMo Guardrails singleton at app startup
# ============================================================
def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    # 1. Configure the gatekeeper LLM (fast, lightweight model with temperature 0 for deterministic checks)
    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # 2. Parse Colang flows (intent & dialogue rules) and YAML system instructions
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    # 3. Instantiate the rails engine and assign to singleton variable
    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (llama-3.1-8b-instant).")


# ============================================================
# FUNCTION: guard
# Purpose: Inspect incoming user queries for off-topic questions, jailbreaks,
#          or greetings, and intercept them before reaching the RAG pipeline.
# ============================================================
def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    # 1. Guard check: ensure the rails singleton is ready
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    # 2. Trace execution and send user message to NeMo Rails for classification
    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # 3. Extract the text response from NeMo result (handles dict or string)
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        # 4. Check if any known rail indicator substring is present in the response
        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        # 5. If a rail triggered, intercept and return the canned response directly
        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        # 6. If no rails fired, let the message proceed into the RAG pipeline
        logfire.info("✅ Guardrails passed.")
        return False, None