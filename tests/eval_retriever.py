import json
import sys
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "chroma_store"
DATASET_PATH = ROOT / "datasets" / "eval_dataset.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAMES = ["faq", "tickets", "guides"]
RETRIEVE_PER_COLLECTION = 8
TOP_K = 5

# =========================================================
# REGRESSION GATING OPTIMIZATION TARGETS
# =========================================================
MIN_RECALL_GAIN = 0.045      # Must be at least 4.5% better (absolute gain)
MIN_TOP1_GAIN = 0.150       # Must be at least 15.0% better (absolute gain)
MIN_MRR_GAIN = 0.100        # Must be at least 0.100 points better (absolute gain)

@lru_cache(maxsize=1)
def get_reranker():
    # Loads exactly once per process when first called
    return CrossEncoder("BAAI/bge-reranker-base")


def load_dataset() -> List[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_source_label(doc) -> str:
    source = doc.metadata.get("source", "unknown").lower()
    if source == "faq":
        return f"FAQ-{doc.metadata.get('faq_id', 'unknown')}"
    if source == "ticket":
        return f"{doc.metadata.get('ticket_id', 'unknown')}"
    if source in {"guide", "guides"}:
        return f"Guide page {doc.metadata.get('chunk_index', 'unknown')}"
    return source.upper()


def build_stores():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    stores = {}
    for name in COLLECTION_NAMES:
        stores[name] = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return stores


def retrieve_candidates(stores, query: str):
    results = []
    for store in stores.values():
        candidates = store.similarity_search_with_score(query, k=RETRIEVE_PER_COLLECTION)
        results.extend(candidates)
    results.sort(key=lambda item: item[1])
    return results


def reciprocal_rank(expected: List[str], retrieved: List[str]) -> float:
    for idx, label in enumerate(retrieved, start=1):
        if label in expected:
            return 1 / idx
    return 0.0


def evaluate_without_reranker(stores):
    data = load_dataset()
    recall_hits = 0
    top1_hits = 0
    mrr_sum = 0

    for item in data:
        query = item["question"]
        expected = item["expected_sources"]
        results = retrieve_candidates(stores, query)[:TOP_K]

        retrieved_labels = [get_source_label(doc) for doc, _ in results]

        if any(label in expected for label in retrieved_labels):
            recall_hits += 1

        if retrieved_labels and retrieved_labels[0] in expected:
            top1_hits += 1
        mrr_sum += reciprocal_rank(expected, retrieved_labels)

    total = len(data)
    return {
        "recall": recall_hits / total,
        "top1": top1_hits / total,
        "mrr": mrr_sum / total,
    }


def evaluate_with_reranker(stores):
    data = load_dataset()
    reranker = get_reranker()
    recall_hits = 0
    top1_hits = 0
    mrr_sum = 0

    for item in data:
        query = item["question"]
        expected = item["expected_sources"]
        candidates = retrieve_candidates(stores, query)

        docs = [doc for doc, _ in candidates]
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked_labels = [get_source_label(doc) for doc, _ in ranked[:TOP_K]]

        if any(label in expected for label in reranked_labels):
            recall_hits += 1
        if reranked_labels and reranked_labels[0] in expected:
            top1_hits += 1
        mrr_sum += reciprocal_rank(expected, reranked_labels)

    total = len(data)
    return {
        "recall": recall_hits / total,
        "top1": top1_hits / total,
        "mrr": mrr_sum / total,
    }


if __name__ == "__main__":
    stores = build_stores()
    baseline = evaluate_without_reranker(stores)
    reranked = evaluate_with_reranker(stores)

    print("\n=== MULTI-COLLECTION RERANKER EVALUATION ===\n")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Collections: {', '.join(COLLECTION_NAMES)}")
    print(f"Retrieve per collection: {RETRIEVE_PER_COLLECTION}")
    print(f"Eval top-K: {TOP_K}\n")

    print("WITHOUT RERANKER")
    print(f"Recall@{TOP_K}: {baseline['recall']:.2%}")
    print(f"Top-1 Accuracy: {baseline['top1']:.2%}")
    print(f"MRR: {baseline['mrr']:.4f}\n")

    print("WITH RERANKER")
    print(f"Recall@{TOP_K}: {reranked['recall']:.2%}")
    print(f"Top-1 Accuracy: {reranked['top1']:.2%}")
    print(f"MRR: {reranked['mrr']:.4f}")

     # =========================================================
    # RERANKER PERFORMANCE OPTIMIZATION GATING
    # =========================================================
    print("Checking retrieval optimization uplift requirements...")
    
    actual_recall_gain = reranked['recall'] - baseline['recall']
    actual_top1_gain = reranked['top1'] - baseline['top1']
    actual_mrr_gain = reranked['mrr'] - baseline['mrr']
    
    failed_optimization_gate = False
    failure_messages = []

    if actual_recall_gain < MIN_RECALL_GAIN:
        failed_optimization_gate = True
        failure_messages.append(
            f"❌ RECALL GAIN INSUFFICIENT: Got +{actual_recall_gain:.2%}, expected >= +{MIN_RECALL_GAIN:.1%}"
        )
        
    if actual_top1_gain < MIN_TOP1_GAIN:
        failed_optimization_gate = True
        failure_messages.append(
            f"❌ TOP-1 GAIN INSUFFICIENT: Got +{actual_top1_gain:.2%}, expected >= +{MIN_TOP1_GAIN:.1%}"
        )
        
    if actual_mrr_gain < MIN_MRR_GAIN:
        failed_optimization_gate = True
        failure_messages.append(
            f"❌ MRR GAIN INSUFFICIENT: Got +{actual_mrr_gain:.4f}, expected >= +{MIN_MRR_GAIN:.3f}"
        )

    print("\n" + "="*60)
    print("             🛡️ RETRIEVAL OPTIMIZATION ANALYSIS             ")
    print("="*60)

    if failed_optimization_gate:
        print("🛑 BUILD REJECTED: Reranker failed to provide the necessary quality uplift!")
        for error in failure_messages:
            print(error)
        print("\nAction: Refuse merge. Check cross-encoder model setup or chunk indexing.")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print("✅ BUILD APPROVED: Reranker meets production scaling performance targets.")
        print(f"🔹 Verified Recall Boost: +{actual_recall_gain:.2%}")
        print(f"🔹 Verified Top-1 Boost: +{actual_top1_gain:.2%}")
        print(f"🔹 Verified MRR Boost: +{actual_mrr_gain:.4f}")
        print("="*60 + "\n")
        sys.exit(0)
"# Triggering RAG Quality Gate CI Pipeline Run" 
"# Triggering RAG Quality Gate CI Pipeline Run" 
