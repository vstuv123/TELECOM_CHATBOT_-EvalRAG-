"""Evaluate ticket-only retrieval quality using top-3 recall.

This script tests 10 hand-crafted (question, expected_ticket_id) pairs against
the persisted 'tickets' Chroma collection in chroma_store.
"""
import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from models import EMBED_MODEL

CHROMA_DIR = "chroma_store"
TICKET_COLLECTION = "tickets"
TOP_K = 3

TICKET_EVAL_PAIRS: List[Tuple[str, str]] = [
    (
        "My phone shows SIM not provisioned after a software update.",
        "TK-005",
    ),
    (
        "I was charged twice for the same $45 monthly plan payment.",
        "TK-006",
    ),
    (
        "I got a huge charge on my Spain trip even though I had a roaming bundle.",
        "TK-004",
    ),
    (
        "My mobile internet stopped working after I switched from 3G to 4G mode.",
        "TK-001",
    ),
    (
        "I can’t complete eSIM activation on my new iPhone; the QR scan keeps failing.",
        "TK-009",
    ),
    (
        "Hotspot is blocked on my unlimited plan even though tethering should be allowed.",
        "TK-014",
    ),
    (
        "I cannot download my itemised bill PDF; the download button just spins.",
        "TK-010",
    ),
    (
        "All incoming calls go straight to voicemail even though my phone shows full signal.",
        "TK-007",
    ),
    (
        "Password reset email for the MyTelecom app never arrives.",
        "TK-013",
    ),
    (
        "I have international roaming enabled, but I have no service in Tokyo.",
        "TK-012",
    ),
]


def build_ticket_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name=TICKET_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def evaluate_topk_recall(ticket_store: Chroma, top_k: int = TOP_K) -> None:
    correct = 0
    results = []

    for question, expected_ticket_id in TICKET_EVAL_PAIRS:
        retrieved = ticket_store.similarity_search_with_score(question, k=top_k)
        retrieved_ids = [doc.metadata.get("ticket_id", "unknown") for doc, _ in retrieved]
        in_topk = expected_ticket_id in retrieved_ids
        if in_topk:
            correct += 1

        results.append({
            "question": question,
            "expected": expected_ticket_id,
            "retrieved": retrieved_ids,
            "top_k_match": in_topk,
            "scores": [score for _, score in retrieved],
        })

    recall = correct / len(TICKET_EVAL_PAIRS)

    print("Ticket Retrieval Evaluation")
    print("============================")
    print(f"Top-{top_k} recall: {correct}/{len(TICKET_EVAL_PAIRS)} = {recall:.2%}\n")

    for idx, row in enumerate(results, start=1):
        print(f"{idx}. Question: {row['question']}")
        print(f"   Expected ticket: {row['expected']}")
        print(f"   Retrieved top-{top_k}: {row['retrieved']}")
        print(f"   Scores: {row['scores']}")
        print(f"   Match: {'YES' if row['top_k_match'] else 'NO'}\n")


if __name__ == "__main__":
    print("Using the existing persisted 'tickets' collection in chroma_store.")
    ticket_store = build_ticket_store()
    evaluate_topk_recall(ticket_store)
