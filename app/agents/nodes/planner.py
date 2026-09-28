# ============================================================
# AGENT NODE: PLANNER
# Purpose: LangGraph routing node that inspects full conversation context
# and determines whether to answer conversationally from memory or initiate
# technical document retrieval by generating an optimized search query.
# ============================================================
from app.agents.state import AgentState
from app.gateway.client import get_langchain_llm
from app.config import settings
from langchain_groq import ChatGroq
import logfire

#Port key backend LLM fallback  + retry + cache 
llm = get_langchain_llm(feature="planner")

# llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0)


# ============================================================
# FUNCTION: planner_node
# Purpose: Classifies query intent and sets pipeline plan in LangGraph state
# ============================================================
def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # 1. Build the past conversation history string (excluding the latest message).
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    # 2. Extract ONLY the latest message to evaluate.
    # If the list is empty, safely fallback to an empty string "".
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    # 3. Construct planning prompt with conversation context and routing rules
    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    
    # 4. Invoke LLM within Logfire tracing span to make the routing decision
    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")
        
    # 5. Branch based on planner decision:
    # Case A: Pure conversational intent / memory lookup — skip vector retrieval
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
        
    # Case B: Technical query — record refined search query for retrieval node
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }