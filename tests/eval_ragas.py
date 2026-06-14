import json
import os
import sys
import types
import time
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from langchain_core.callbacks import BaseCallbackHandler
import time

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

# Ensure the tests folder exists
TESTS_DIR = ROOT_DIR / "tests"

# Define path for cache dataset file
# Change 'cached_rag_dataset.json' to use a specific name if you want to bypass the dynamic lookup
CACHE_FILE_PATH = TESTS_DIR / "cached_rag_dataset.json"

# FIX: Set the primary variable to your fresh Key 1 immediately at startup
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY_1")
print(f"Using GROQ_API_KEY_1 for initial setup: {os.getenv('GROQ_API_KEY')}...")

if not os.getenv("GROQ_API_KEY_1") or not os.getenv("GROQ_API_KEY_2"):
    raise ValueError("CRITICAL ERROR: One or both GROQ_API_KEYs were not found! Check your .env file.")

# ==========================================
# IMPORTS
# ==========================================
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from rag_chain import build_chain, _format_docs
from ragas import evaluate
# FIX: Import RunConfig directly from the root of ragas
from ragas import RunConfig

from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

# -------------------------------------------------------------
# 🟢 PLACE THE DIAGNOSTIC CODE BLOCK RIGHT HERE
# -------------------------------------------------------------
import groq

# Save the original initialization method of the Groq client
_original_groq_init = groq.Groq.__init__

def diagnostic_groq_init(self, *args, **kwargs):
    # Retrieve the API key that the Groq library is actively loading
    active_key = kwargs.get("api_key") or os.environ.get("GROQ_API_KEY", "")
    masked_key = f"{active_key[:5]}...{active_key[-5:]}" if len(active_key) > 10 else "MISSING"
    
    print(f"\n📡 [NETWORK CALL] Groq Client initialized for a metric request using key: {masked_key}")
    
    # Call the real constructor to let the request proceed normally
    _original_groq_init(self, *args, **kwargs)

# Overwrite the initialization method with our diagnostic checker
groq.Groq.__init__ = diagnostic_groq_init
# -------------------------------------------------------------

# ==========================================
# ASYNC COMPATIBLE EMBEDDING INTERFACE
# ==========================================
class RagasEmbeddingWrapper:
    """Implements both sync and async methods required by Ragas 0.4.x."""
    def __init__(self, langchain_embeddings):
        self.lc_embeds = langchain_embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.lc_embeds.embed_documents(texts)

    async def aembed_text(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self.lc_embeds.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.lc_embeds.embed_documents(texts)


# ==========================================
# LOAD TARGET EVALUATION DATASET
# ==========================================
DATASET_PATH = ROOT_DIR / "datasets" / "eval_dataset.json"
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"Loaded {len(eval_data)} evaluation samples.")

# ==========================================
# BUILD RETRIEVAL & JUDGE ENGINES
# ==========================================
print("Initializing Chatbot Chain...")
chain = build_chain()

class GroqRateLimiter(BaseCallbackHandler):
    """Forces a sleep pause before every single LLM call to prevent 429 TPM errors."""
    def __init__(self, delay_seconds: float = 30.0):
        self.delay_seconds = delay_seconds

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Pause execution to let the rolling Groq TPM capacity clear up
        print(f"   [Rate Limiter] Cooling down for {self.delay_seconds} seconds...")
        time.sleep(self.delay_seconds)


print("Initializing LangChain Judge Models...")

# Create the rate limiter instance (30 seconds is very safe for the 6k TPM limit)
rate_limiter = GroqRateLimiter(delay_seconds=30.0)

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    max_retries=5,              # Automatically retry up to 5 times if it fails
    timeout=60,                 # Wait up to 60 seconds for a response before timing out
    callbacks=[rate_limiter]  # <-- This tells LangChain to sleep on every request
)

base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
local_embeddings = RagasEmbeddingWrapper(base_embeddings)

# Configure Metric Objects
faithfulness_metric = Faithfulness(llm=groq_llm)

answer_relevancy_metric = AnswerRelevancy(llm=groq_llm)
answer_relevancy_metric.embeddings = local_embeddings

context_precision_metric = ContextPrecision(llm=groq_llm)

context_recall_metric = ContextRecall(llm=groq_llm)

answer_correctness_metric = AnswerCorrectness(llm=groq_llm)
answer_correctness_metric.embeddings = local_embeddings

# ==========================================
# CHECK CACHE OR EXECUTE PIPELINE EXTRACTION
# ==========================================
all_samples = []

if CACHE_FILE_PATH.exists():
    print(f"\n[CACHE FOUND] Loading pre-generated dataset from: {CACHE_FILE_PATH}")
    with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    print(f"Successfully loaded {len(all_samples)} samples from cache. Skipping LLM generation.")
else:
    print("\n[NO CACHE FOUND] Running pipeline extraction across evaluation dataset...")
    print("Initializing Chatbot Chain...")
    chain = build_chain()

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for idx, item in enumerate(eval_data):
        question = item["question"]
        print(f" [{idx + 1}/{len(eval_data)}] Extracting context & generating answer...")
        
        docs = chain.retrieve_context(question)
        contexts = [doc.page_content for doc in docs]
        
        if not docs:
            from rag_chain import FALLBACK_RESPONSE
            answer = FALLBACK_RESPONSE
        else:
            answer = chain.chain.invoke({
                "context": _format_docs(docs),
                "question": question,
            })
            
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])
        time.sleep(2.0)

    # Build the complete data tracking list
    all_samples = [
        {
            "question": questions[i],
            "answer": answers[i],
            "contexts": contexts_list[i],
            "ground_truth": ground_truths[i],
        }
        for i in range(len(questions))
    ]
    
    # Save the file to tests directory so it can be reused later
    with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=4, ensure_ascii=False)
    print(f"\n[CACHE SAVED] Generated dataset saved to: {CACHE_FILE_PATH}")

# ==========================================
# DUAL-KEY SPLIT EVALUATION WORKFLOW
# ==========================================
# Calculate the exact midpoint split index
midpoint = len(all_samples) // 2
batch_1_samples = all_samples[:7]
batch_2_samples = all_samples[7:14]
batch_3_samples = all_samples[14:21]

# Construct safe thread configs
groq_safe_config = RunConfig(max_workers=1, timeout=300)
metrics_list = [
    faithfulness_metric,
    answer_relevancy_metric,
    context_precision_metric,
    context_recall_metric,
    answer_correctness_metric,
]

# --- RUN BATCH 1 (USING KEY 1) ---
print(f"\n[BATCH 1] Swapping engine to GROQ_API_KEY_1 ({len(batch_1_samples)} samples)...")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY_1")
dataset_1 = Dataset.from_list(batch_1_samples)

result_1 = evaluate(
    dataset=dataset_1,
    metrics=metrics_list,
    run_config=groq_safe_config
)

print("\n✅ BATCH 1 EVALUATION FINISHED CLEANLY.")

# --- RUN BATCH 2 (USING KEY 2) ---
print(f"\n[BATCH 2] Swapping engine to GROQ_API_KEY_2 ({len(batch_2_samples)} samples)...")
os.environ["GROQ_API_KEY"] = os.getenv("CHATBOT_GROQ_API_KEY")  # <-- Use the second key for this batch

# FORCE RE-INITIALIZATION OF THE LLM CLIENT WITH THE NEW KEY
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    max_retries=5,
    timeout=60,
    callbacks=[rate_limiter] 
)
# Re-assign the refreshed LLM to all metrics so they read the new key
for metric in metrics_list:
    metric.llm = groq_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = local_embeddings  # Re-assign embeddings if the metric uses them

dataset_2 = Dataset.from_list(batch_2_samples)

result_2 = evaluate(
    dataset=dataset_2,
    metrics=metrics_list,
    run_config=groq_safe_config
)

print("\n✅ BATCH 2 EVALUATION FINISHED CLEANLY.")

# --- RUN BATCH 3 (USING KEY 3) ---
print(f"\n[BATCH 3] Swapping engine to Chatbot GROQ_API_KEY ({len(batch_3_samples)} samples)...")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY_2")  # This should be the same as Key 1, but it forces the client to re-read the env variable

# FORCE RE-INITIALIZATION OF THE LLM CLIENT WITH THE NEW KEY
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    max_retries=5,
    timeout=60,
    callbacks=[rate_limiter] 
)
# Re-assign the refreshed LLM to all metrics so they read the new key
for metric in metrics_list:
    metric.llm = groq_llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = local_embeddings  # Re-assign embeddings if the metric uses them

dataset_3 = Dataset.from_list(batch_3_samples)

result_3 = evaluate(
    dataset=dataset_3,
    metrics=metrics_list,
    run_config=groq_safe_config
)

print("\n✅ BATCH 3 EVALUATION FINISHED CLEANLY.")

# ==========================================
# UNIFIED MATHEMATICAL AGGREGATION
# ==========================================
print("\nCalculating weighted averages across both keys...")
final_results = {}
len_1 = len(batch_1_samples)
len_2 = len(batch_2_samples)
len_3 = len(batch_3_samples)
total_len = len_1 + len_2 + len_3

result_1_dict = result_1.to_pandas().mean(numeric_only=True).to_dict()  # Convert to dict for easier access
result_2_dict = result_2.to_pandas().mean(numeric_only=True).to_dict()  # Convert to dict for easier access
result_3_dict = result_3.to_pandas().mean(numeric_only=True).to_dict()  # Convert to dict for easier access

# CORRECT - .scores extracts the internal dictionary from the Ragas object
for key in result_1_dict.keys():
    # Calculate weighted mean to account for uneven odd splits (10 vs 11 samples)
    val_1 = result_1_dict.get(key) or 0.0
    val_2 = result_2_dict.get(key) or 0.0
    val_3 = result_3_dict.get(key) or 0.0

    final_results[key] = ((val_1 * len_1) + (val_2 * len_2) + (val_3 * len_3)) / total_len

print("\n=== COMPLETE RATE-LIMITED RAGAS RESULTS ===")
for metric_name, score in final_results.items():
    print(f"{metric_name}: {score:.4f}")


# =========================================================
# CONSOLIDATE RESULTS AND EXPORT TO CSV
# =========================================================
import pandas as pd
print("\nConsolidating all batch results into a structured DataFrame...")

try:
    # 1. Convert each Ragas result object to a standard Pandas DataFrame
    df1 = result_1.to_pandas()
    df2 = result_2.to_pandas()
    df3 = result_3.to_pandas()
    
    # 2. Merge all 3 chunks into one master table
    df_final = pd.concat([df1, df2, df3], ignore_index=True)
    
    # 3. Clean up metric column structures if any batch had a 429/503 skip
    metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']
    for col in metric_cols:
        if col in df_final.columns:
            # Fill skipped/failed metric evaluations with 0.0 instead of blank NaN
            df_final[col] = df_final[col].fillna(0.0)
    
    # 4. Display a structured row-by-row look on your terminal screen
    print("\n" + "="*80)
    print("STRUCTURED EVALUATION REPORT OVERVIEW (FIRST 5 ROWS)")
    print("="*80)
    # Shows columns for clear readability
    preview_cols = ['question', 'faithfulness', 'answer_relevancy', 'answer_correctness']
    available_cols = [c for c in preview_cols if c in df_final.columns]
    print(df_final[available_cols].head(5).to_string())
    print("="*80)
    
    # 5. Define export path inside your tests folder
    csv_filename = TESTS_DIR / "final_ragas_evaluation_report.csv"
    
    # 6. Save data cleanly to a CSV file
    df_final.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"\n💾 [SUCCESS] Entire dataset exported safely! File saved to:\n👉 {csv_filename}")

except Exception as e:
    print(f"\n❌ Error building final DataFrame or saving CSV: {e}")


# =========================================================
# CRITICAL PHASE 3: REGRESSION GATE CONTROL
# =========================================================
print("\nChecking system quality gate thresholds...")

# 1. Define strict production acceptance benchmarks (0.0 to 1.0 scale)
FAITHFULNESS_THRESHOLD = 0.85      # 85% minimum factual groundedness
ANSWER_RELEVANCY_THRESHOLD = 0.80  # 80% minimum alignment to user intent

failed_gate = False
failure_reasons = []

# 2. Extract and check the actual unified metrics
current_faithfulness = final_results.get("faithfulness", 0.0)
current_relevancy = final_results.get("answer_relevancy", 0.0)

if current_faithfulness < FAITHFULNESS_THRESHOLD:
    failed_gate = True
    failure_reasons.append(
        f"❌ QUALITY REGRESSION: Faithfulness score dropped to {current_faithfulness:.4f} "
        f"(Target Requirement: >= {FAITHFULNESS_THRESHOLD})"
    )

if current_relevancy < ANSWER_RELEVANCY_THRESHOLD:
    failed_gate = True
    failure_reasons.append(
        f"❌ QUALITY REGRESSION: Answer Relevancy score dropped to {current_relevancy:.4f} "
        f"(Target Requirement: >= {ANSWER_RELEVANCY_THRESHOLD})"
    )

# 3. Intercept execution to drive CI/CD build success or blockage
print("\n" + "="*50)
print("             🛡️ REGRESSION GATE ANALYSIS            ")
print("="*50)

if failed_gate:
    print("🛑 BUILD INTERRUPTED: Your latest adjustments breached safety bars!")
    for reason in failure_reasons:
        print(reason)
    print("\nAction Needed: Revert or tune your system prompt / chunk files before merging.")
    print("="*50 + "\n")
    sys.exit(1)  # Hard OS failure code blocks the branch or GitHub Action pull request merge
else:
    print("✅ BUILD COMPLIANT! All performance metrics meet active safety constraints.")
    print(f"🔹 Verified Faithfulness: {current_faithfulness:.4f}")
    print(f"🔹 Verified Answer Relevancy: {current_relevancy:.4f}")
    print("\nStatus: Safe to merge pull request into your core main production branch.")
    print("="*50 + "\n")
    sys.exit(0)  # Standard clean code lets development pipelines advance normally