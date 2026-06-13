"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from models import get_reranker, EMBED_MODEL
from observability import langfuse
from langfuse.decorators import observe, langfuse_context

CHROMA_DIR  = "chroma_store"
DISTANCE_THRESHOLD = 0.95  # Tune this based on your data and needs. Lower = more relevant but fewer results; Higher = more results but less relevant.

def _format_doc(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown").lower()
    if source == "faq":
        cite = f"FAQ #{doc.metadata.get('faq_id', 'unknown')}"
    elif source == "ticket":
        cite = f"Ticket #{doc.metadata.get('ticket_id', 'unknown')}"
    elif source in {"guide", "guides"}:
        cite = f"Guide chunk {doc.metadata.get('chunk_index', 'unknown')}"
    else:
        cite = source.upper()

    return cite

def build_retriever(
    k_faq: int = 7,
    k_tickets: int = 7,
    k_guides: int = 7,
    distance_threshold: float = DISTANCE_THRESHOLD,
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # <-- Force Cosine Distance
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # <-- Force Cosine Distance
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # <-- Force Cosine Distance
    )

    # Modern @observe tag tells Langfuse this function is an internal execution Span step
    @observe(as_type="span", name="retrieval_process")
    def retrieve(input_data: dict) -> list[Document]:
        query = input_data.get("query")

        # Log basic metadata onto the span context natively
        langfuse_context.update_current_observation(input={"query": query})

        # --- NESTED SPAN 1: CHROMA VECTOR SEARCH ---
        @observe(as_type="span", name="chroma_vector_search")
        def run_vector_search(search_query):
            scored_results = []
            for store, k in [(faq_store, k_faq), (tickets_store, k_tickets), (guides_store, k_guides)]:
                scored_results.extend(store.similarity_search_with_score(search_query, k=k))

            filtered = [(doc, score) for doc, score in scored_results if score <= distance_threshold]
            filtered.sort(key=lambda item: item[1])
            filtered_docs = [doc for doc, _ in filtered]
            
            langfuse_context.update_current_observation(
                output={
                    "num_docs_retrieved": len(filtered_docs),
                    "documents": [{"cite": _format_doc(doc), "distance": score} for doc, score in filtered]
                }
            )
            return filtered_docs, filtered

        # Execute search step
        filtered_docs, filtered = run_vector_search(query)

        # Early return if sparse results
        if len(filtered_docs) < 2:
            langfuse_context.update_current_observation(output={"final_docs_returned": len(filtered_docs)})
            return filtered_docs

        # --- NESTED SPAN 2: CROSS-ENCODER RERANKER ---
        @observe(as_type="span", name="cross_encoder_rerank")
        def run_reranker(search_query, candidate_docs):
            pairs = [(search_query, doc.page_content) for doc in candidate_docs]
            scores = get_reranker().predict(pairs)
            
            ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in ranked]
            if len(final_docs) > 5:
                final_docs = final_docs[:5]

            langfuse_context.update_current_observation(
                output={
                    "ranked_documents": [
                        {
                            "rank": idx + 1,
                            "cite": _format_doc(doc),
                            "score": float(score),
                        }
                        for idx, (doc, score)
                        in enumerate(ranked)
                    ],
                }
            )
            return final_docs

        # Execute reranking step
        final_docs = run_reranker(query, filtered_docs)

        langfuse_context.update_current_observation(output={"final_docs_returned": len(final_docs)})
        return final_docs
    
    return RunnableLambda(retrieve)
