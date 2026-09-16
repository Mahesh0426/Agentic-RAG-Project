import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY= os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT= os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION= "Production_Grade_Rag"
    
    GROQ_API_KEY= os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY= os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL= "openai/gpt-oss-120b"
    
    
settings = Settings()
    