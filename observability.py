import os
from dotenv import load_dotenv
from langfuse import Langfuse

# 1. Load your API keys from the local .env file
load_dotenv()

# 2. Extract configuration (with standard fallbacks to default hosting values)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# 3. Initialize the global Langfuse client instance
# (This instance is what you are importing into your retriever and chain files)
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

# 4. Optional: Run an active health check on app startup to catch bad keys immediately
try:
    langfuse.auth_check()
    print("✅ Langfuse observability client authenticated successfully!")
except Exception as e:
    print(f"❌ Langfuse authentication failed! Check your credentials. Error: {e}")