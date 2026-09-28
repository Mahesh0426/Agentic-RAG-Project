
# ============================================================
# AGENT NODE: RESPONDER (GENERATE)
# Purpose: LangGraph node responsible for final response generation.
# Formulates prompt context from conversation history and retrieved documents,
# invokes the Portkey AI Gateway, detects response cache status (HIT/MISS),
# and returns state updates for the agent pipeline.
# ============================================================
import logfire
from app.agents.state import AgentState
from app.config import settings
from app.gateway.client import extract_cache_status, portkey_client


# ============================================================
# FUNCTION: generate_node
# Purpose: Synthesizes final response via Portkey Gateway based on query type (RAG vs Conversational)
# ============================================================
def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    # 1. Format previous conversation turns (excluding the latest message) into readable chat history
    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    # 2. Extract the latest user question from the conversation messages
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    # 3. Branch prompt construction based on query intent:
    if query == "CONVERSATIONAL":
        # Branch A: Pure dialogue/conversational flow (no technical document retrieval needed)
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        # Branch B: Technical RAG flow — concatenate retrieved chunks within token/character limits
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    # 4. Invoke LLM through Portkey Gateway within a Logfire tracing span
    with logfire.span("✍️ LLM Synthesis"):
        try:
            # 5. Call Portkey Chat Completions API with low temperature for focused synthesis
            response = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            # 6. Extract the generated message text
            content = response.choices[0].message.content

            # 7. Check if response was served directly from Portkey's cache
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            # 8. Update plan steps and UI status indicator based on cache result
            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            # 9. Return state dictionary to LangGraph to update pipeline state
            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            # 10. Log errors and re-raise to fail cleanly
            logfire.error(f"LLM Generation failed: {e}")
            raise e