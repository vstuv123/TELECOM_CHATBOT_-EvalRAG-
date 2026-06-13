import os
import sys
import types
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset

# ==========================================
# COMPATIBILITY PATCH
# ==========================================
import langchain_google_vertexai
legacy_path = "langchain_community.chat_models.vertexai"
if legacy_path not in sys.modules:
    fake_module = types.ModuleType(legacy_path)
    fake_module.ChatVertexAI = langchain_google_vertexai.ChatVertexAI
    sys.modules[legacy_path] = fake_module

# ==========================================
# LOAD ENVIRONMENT
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# ==========================================
# IMPORTS
# ==========================================
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate

from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

os.environ["GROQ_API_KEY"] = os.getenv("CHATBOT_GROQ_API_KEY")

# ==========================================
# FIX: ASYNC COMPATIBLE EMBEDDING INTERFACE
# ==========================================
class RagasEmbeddingWrapper:
    """Implements both sync and async methods required by different Ragas versions."""
    def __init__(self, langchain_embeddings):
        self.lc_embeds = langchain_embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.lc_embeds.embed_documents(texts)

    # FIX: Add the async variants that Ragas 0.4.x awaits internally
    async def aembed_text(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.lc_embeds.embed_documents(texts)


print("Initializing LangChain Models...")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Build base model and encapsulate inside our plain Python type wrapper
base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
local_embeddings = RagasEmbeddingWrapper(base_embeddings)

print("Building a 1-item mock dataset...")
mock_samples = {
    "question": ["How do I troubleshoot a completely blank signal bar on my phone?"],
    "answer": ["You should try turning your airplane mode on for 10 seconds and then turn it back off to re-register."],
    "contexts": [["Step 2 - Rule out a temporary network glitch. Toggle airplane mode on for 10 seconds then off. This forces the device to re-register on the network."]],
    "ground_truth": ["Toggle airplane mode on for 10 seconds and then off to force a network re-registration."]
}
dataset = Dataset.from_dict(mock_samples)

print("Launching Ragas evaluation...")

# Initialize metrics by attaching structural model requirements
faithfulness_metric = Faithfulness(llm=groq_llm)

answer_relevancy_metric = AnswerRelevancy(llm=groq_llm)
answer_relevancy_metric.embeddings = local_embeddings

context_precision_metric = ContextPrecision(llm=groq_llm)

context_recall_metric = ContextRecall(llm=groq_llm)

answer_correctness_metric = AnswerCorrectness(llm=groq_llm)
answer_correctness_metric.embeddings = local_embeddings

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness_metric,
        answer_relevancy_metric,
        context_precision_metric,
        context_recall_metric,
        answer_correctness_metric,
    ],
)


print("\n=== SUCCESS: RAGAS METRICS RESULTS ===\n")
print(type(result))
print(result.keys() if isinstance(result, dict) else "Result is not a dict")

# Convert it to a dictionary first
metrics_dict = result.to_pandas().mean(numeric_only=True).to_dict()

# Now you can easily get any key safely
faithfulness_score = metrics_dict.get("faithfulness", 0.0)
print(f"Extracted Faithfulness: {faithfulness_score:.4f}")

answer_relevancy_score = metrics_dict.get("answer_relevancy", 0.0)
print(f"Extracted Answer Relevancy: {answer_relevancy_score:.4f}")

print(result)

# =========================================================
# CONSOLIDATE RESULTS AND EXPORT TO CSV
# =========================================================
print("\nConsolidating all batch results into a structured DataFrame...")

try:
    # 1. Convert each Ragas result object to a standard Pandas DataFrame
    df1 = result.to_pandas()
    
    # 3. Clean up metric column structures if any batch had a 429/503 skip
    metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']
    for col in metric_cols:
        if col in df1.columns:
            # Fill skipped/failed metric evaluations with 0.0 instead of blank NaN
            df1[col] = df1[col].fillna(0.0)
    
    # 4. Display a structured row-by-row look on your terminal screen
    print("\n" + "="*80)
    print("STRUCTURED EVALUATION REPORT OVERVIEW (FIRST 5 ROWS)")
    print("="*80)
    # Shows columns for clear readability
    preview_cols = ['question', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    available_cols = [c for c in preview_cols if c in df1.columns]
    print(df1[available_cols].head(5).to_string())
    print("="*80)

except Exception as e:
    print(f"\n❌ Error building final DataFrame or saving CSV: {e}")