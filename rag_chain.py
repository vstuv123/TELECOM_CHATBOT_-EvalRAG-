"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""
from inspect import trace
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from observability import langfuse  # Global verified client instance
from retriever import build_retriever
import re
from langfuse.decorators import observe, langfuse_context

# Define common phrases your model uses when refusing to answer out-of-context queries
REFUSAL_PHRASES = [
    "i don't know", 
    "not enough information", 
    "context does not contain",
    "cannot answer confidently",
    "I don't have access to",
    "I can't assist with that",
    "outside the telecom scope",
    "I can't help with that",
    "isn't covered in any of the provided information",
    "there's no information related",
    "I can't answer",
    "I can't help with that",
    "context doesn't have information"
]

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from FAQ entries, resolved support tickets, and guide document chunks.

When you answer, cite each fact by source and identifier, for example:
- [FAQ #123]
- [Ticket #TK-005]
- [Guide chunk 12]

If the context does not contain enough information to answer confidently, say so clearly and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""

def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").lower()
        if source == "faq":
            cite = f"FAQ #{doc.metadata.get('faq_id', 'unknown')}"
        elif source == "ticket":
            cite = f"Ticket #{doc.metadata.get('ticket_id', 'unknown')}"
        elif source in {"guide", "guides"}:
            cite = f"Guide chunk {doc.metadata.get('chunk_index', 'unknown')}"
        else:
            cite = source.upper()

        sections.append(f"[{cite}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

def calculate_citation_metrics(docs: list[Document], answer: str) -> dict:
    """
    Computes production-grade citation coverage and adherence metrics.
    """
    if not docs:
        return {"has_citation": 0.0, "citation_accuracy": 0.0}
        
    # Extract all cited source markers from the text (e.g., 'faq #123', 'ticket #tk-005')
    # Matches patterns like [FAQ #123] or [Ticket #TK-005] case-insensitively
    # Updated regex to support hyphens, numbers, spaces, and case-insensitivity
    found_citations = set(re.findall(r'\[(faq\s*#\d+|ticket\s*#[\w\-]+|guide\s*chunk\s*\d+)\]', answer.lower()))
    
    # Map the actual retrieved source tokens from your vectors
    retrieved_citations = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown").lower()
        if source == "faq":
            retrieved_citations.add(f"faq #{doc.metadata.get('faq_id', 'unknown')}".lower())
        elif source == "ticket":
            retrieved_citations.add(f"ticket #{doc.metadata.get('ticket_id', 'unknown')}".lower())
        elif source in {"guide", "guides"}:
            retrieved_citations.add(f"guide chunk {doc.metadata.get('chunk_index', 'unknown')}".lower())

    # Calculate metrics
    has_citation = 1.0 if len(found_citations) > 0 else 0.0
    
    # Out of the retrieved documents, how many were actually cited?
    cited_retrieved = found_citations.intersection(retrieved_citations)
    citation_accuracy = len(cited_retrieved) / len(retrieved_citations) if retrieved_citations else 0.0
    
    return {
        "has_citation": has_citation,
        "citation_accuracy": citation_accuracy,
        "total_citations_found": len(found_citations),
        "hallucinated_citations_count": len(found_citations - retrieved_citations) # Citations LLM made up
    }


FALLBACK_RESPONSE = (
    "I'm sorry, I don't know the answer based on the available information. "
    "Please call 611 or use the MyTelecom app for assistance."
)

def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # CRITICAL: stream_usage=True tells Groq to return usage tokens at the end of the stream loop
    llm = ChatGroq(
        model="qwen/qwen3-32b", 
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2, 
    )

    # Note: We execute chain steps deliberately to extract prompt text & token structures manually
    chain = prompt | llm | StrOutputParser()

    class RAGChainWithFallback:
        def __init__(self, chain, retriever, prompt_template, llm):
            self.chain = chain
            self.retriever = retriever
            self.prompt_template = prompt_template
            self.llm = llm

        @observe(name="rag_request")  # Creates the root trace using modern SDK OpenTelemetry schemas
        def invoke(self, question: str) -> str:
            start_total = time.time()
            
            # 1. Update the root trace inputs natively using context binding
            langfuse_context.update_current_trace(
                input={"question": question}
            )
            
            # 2. Measure Retriever Latency
            start_retrieval = time.time()
            docs = self.retriever.invoke({
                "query": question, 
                "trace_id": langfuse_context.get_current_trace_id()
            })
            retrieval_latency = (time.time() - start_retrieval) * 1000

            # 3. Process context strings
            context_str = _format_docs(docs)

            # --- NESTED SPAN 3: DEDICATED CONTEXT STATS MONITORING ---
            @observe(as_type="span", name="context_stats")
            def log_context_stats(docs_list, formatted_text):
                langfuse_context.update_current_observation(
                    output={
                        "retrieved_docs_count": len(docs_list),
                        "context_characters_length": len(formatted_text)
                    }
                )
            
            # Execute stats collection step
            log_context_stats(docs, context_str)

            # Update the context string once retrieved so your cloud judge can scan it
            langfuse_context.update_current_trace(
                input={"question": question, "context": context_str},
            )

            # Defensive Short-Circuit: Handle empty context scenario early
            if not docs:
                langfuse_context.update_current_trace(output=FALLBACK_RESPONSE)
                langfuse_context.score_current_observation(
                    name="no_retrieval_failure",
                    value=1.0,
                    comment="Short-circuit fallback: Chroma DB returned 0 documents under the distance threshold."
                )
                return FALLBACK_RESPONSE

            # Format complete prompt text
            formatted_prompt = self.prompt_template.format(context=context_str, question=question)
            prompt_value = self.prompt_template.invoke({"context": context_str, "question": question})

            # =================================================================
            # MODERN SOLUTION: Nested Generation wrapper with Exception Safety
            # =================================================================
            @observe(as_type="generation", name="qwen_generation")
            def execute_llm_call(prompt_input):
                # Set baseline model input tags
                langfuse_context.update_current_observation(
                    model="qwen-2.5-32b",
                    input={
                        "prompt": formatted_prompt,
                        "context": context_str
                    }
                )
                
                try:
                    start_llm = time.time()
                    llm_response = self.llm.invoke(prompt_input)
                    answer_content = llm_response.content
                    
                    llm_latency = (time.time() - start_llm) * 1000
                    total_latency = (time.time() - start_total) * 1000
                    
                    # Extract Groq token counters safely
                    usage_data = llm_response.response_metadata.get("token_usage", {})
                    
                    # Finalize generation outputs and parameters upon successful run
                    langfuse_context.update_current_observation(
                        output=answer_content,
                        usage={
                            "input": usage_data.get("prompt_tokens", 0),
                            "output": usage_data.get("completion_tokens", 0),
                            "total": usage_data.get("total_tokens", 0),
                        },
                        metadata={
                            "retrieval_ms": retrieval_latency,
                            "llm_ms": llm_latency,
                            "total_latency_ms": total_latency
                        }
                    )
                    return answer_content

                except Exception as llm_exception:
                    # Catch and log hardware model processing failures explicitly inside the generation tree block
                    langfuse_context.update_current_observation(
                        level="ERROR",
                        output=f"Model Invocation Error: {str(llm_exception)}"
                    )
                    # Re-raise the exception up to the main function block for orchestration tracking
                    raise llm_exception

            # 4. Execute the protected generation execution step
            try:
                answer = execute_llm_call(prompt_value)
                
                # Finalize trace root output parameters
                langfuse_context.update_current_trace(output=answer)

                # 5. Compute and upload production evaluation metrics
                citation_data = calculate_citation_metrics(docs, answer)
            
                # A. Log overall Citation Presence (Yes/No)
                langfuse_context.score_current_observation(
                    name="citation_presence",
                    value=citation_data["has_citation"],
                    comment=f"Found {citation_data['total_citations_found']} absolute citations inside text string."
                )
            
                # B. Log exact Citation Accuracy (Adherence to context)
                langfuse_context.score_current_observation(
                    name="citation_accuracy",
                    value=citation_data["citation_accuracy"],
                    comment=f"Model cited {citation_data['total_citations_found']} documents. Hallucinated: {citation_data['hallucinated_citations_count']}"
                )

                # C. Log Document Retrieval State (Success bar flag)
                langfuse_context.score_current_observation(
                    name="no_retrieval_failure",
                    value=0.0,
                    comment="Chroma DB successfully returned relevant documentation tracks."
                )

                # D. Log Alignment Guardrails (Scanning for defensive refusal strings)
                response_lower = answer.lower()
                is_refusal = 1.0 if any(phrase.lower() in response_lower for phrase in REFUSAL_PHRASES) else 0.0
                langfuse_context.score_current_observation(
                    name="llm_defensive_refusal",
                    value=is_refusal,
                    comment="The LLM explicitly refused to answer due to missing context or out-of-domain query."
                )
                
                return answer

            except Exception as system_error:
                # 6. Global Catch-All: Mark the root system execution trace as entirely failed
                langfuse_context.score_current_observation(
                    name="system_failure",
                    value=1.0,
                    comment=f"Pipeline Execution Blocked. Details: {str(system_error)}"
                )
                langfuse_context.update_current_trace(output=FALLBACK_RESPONSE)

                return FALLBACK_RESPONSE
            finally:
                # Make sure to flush the error trace as well so it registers
                langfuse_context.flush()


        @observe(name="rag_stream_request")  # This tag creates the trace using modern SDK OpenTelemetry schemas
        def stream(self, question: str):
            start_total = time.time()
            
            # 1. Update the root trace inputs natively using context binding
            langfuse_context.update_current_trace(
                input={"question": question}
            )
            
            # Retrieval Processing
            start_retrieval = time.time()

            # Pass the context-bound trace down to your retriever file smoothly
            docs = self.retriever.invoke({"query": question, "trace": langfuse_context.get_current_trace_id()})
            
            retrieval_latency = (time.time() - start_retrieval) * 1000
            context_str = _format_docs(docs)

            # --- NESTED SPAN 3: DEDICATED CONTEXT STATS MONITORING ---
            @observe(as_type="span", name="context_stats")
            def log_context_stats(docs_list, formatted_text):
                langfuse_context.update_current_observation(
                    output={
                        "retrieved_docs_count": len(docs_list),
                        "context_characters_length": len(formatted_text)
                    }
                )
            
            # Execute stats collection step
            log_context_stats(docs, context_str)

            # Update the context string once retrieved so your cloud judge can scan it
            langfuse_context.update_current_trace(
                input={"question": question, "context": context_str},
            )
            
            if not docs:
                langfuse_context.update_current_trace(output=FALLBACK_RESPONSE)
                langfuse_context.score_current_observation(
                    name="no_retrieval_failure",
                    value=1.0,
                    comment="Chroma DB returned 0 documents under the distance threshold."
                )
                yield FALLBACK_RESPONSE
                return

            # Format complete prompts before entering execution blocks
            formatted_prompt = self.prompt_template.format(context=context_str, question=question)
            prompt_value = self.prompt_template.invoke({"context": context_str, "question": question})

            # =================================================================
            # MODERN SOLUTION: Isolated Generation Span Function with Full Try/Except
            # =================================================================
            @observe(as_type="generation", name="qwen_stream_generation")
            def execute_llm_stream(prompt_input):
                # Set initial generation input details natively
                langfuse_context.update_current_observation(
                    model="qwen-2.5-32b",
                    input={
                        "prompt": formatted_prompt,
                        "context": context_str
                    }
                )

                try:
                    start_llm = time.time()
                    final_accumulated_message = None 
                    full_response = ""
                
                    # Consume and stream tokens directly from the model
                    for chunk in self.llm.stream(prompt_input):
                        if chunk.content:
                            full_response += chunk.content
                            yield chunk.content
                    
                        # FIX: Indentation correction. This must accumulate chunk by chunk INSIDE the loop!
                        if final_accumulated_message is None:
                            final_accumulated_message = chunk
                        else:
                            final_accumulated_message += chunk

                    llm_latency = (time.time() - start_llm) * 1000
                    total_latency = (time.time() - start_total) * 1000
                
                    input_tokens, output_tokens, total_tokens = 0, 0, 0
                    if final_accumulated_message and hasattr(final_accumulated_message, 'usage_metadata') and final_accumulated_message.usage_metadata:
                        usage = final_accumulated_message.usage_metadata
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                
                    # Natively finalize generation inputs, output texts, token counts, and latencies
                    langfuse_context.update_current_observation(
                        output=full_response,
                        usage={"input": input_tokens, "output": output_tokens, "total": total_tokens},
                        metadata={"retrieval_ms": retrieval_latency, "llm_ms": llm_latency, "total_latency_ms": total_latency}
                    )
                
                    # Save string outside local closure scope for parent trace access
                    self._last_stream_output = full_response

                except Exception as llm_error:
                    # Catch and flag raw model dropouts or connection terminations inside the generation block
                    langfuse_context.update_current_observation(
                        level="ERROR",
                        output=f"LLM Stream Interrupted: {str(llm_error)}"
                    )
                    raise llm_error

            # 2. Main execution block with a global catch-all structure matching your invoke style
            try:
                # Iterate over the inner generator safely
                for chunk_content in execute_llm_stream(prompt_value):
                    yield chunk_content
                
                # 3. Finalize trace root output parameters upon clean loop resolution
                full_response = getattr(self, "_last_stream_output", "")
                langfuse_context.update_current_trace(output=full_response)

                # Calculate the production metrics
                citation_data = calculate_citation_metrics(docs, full_response)
            
                # 4. Log deterministic quality scores via context API
                langfuse_context.score_current_observation(
                    name="citation_presence",
                    value=citation_data["has_citation"],
                    comment=f"Found {citation_data['total_citations_found']} absolute citations inside text string."
                )
            
                langfuse_context.score_current_observation(
                    name="citation_accuracy",
                    value=citation_data["citation_accuracy"],
                    comment=f"Model cited {citation_data['total_citations_found']} documents. Hallucinated: {citation_data['hallucinated_citations_count']}"
                )

                langfuse_context.score_current_observation(
                    name="no_retrieval_failure",
                    value=0.0,
                    comment="Chroma DB successfully returned relevant data contexts."
                )
            
                # Check for defensive alignment phrases
                response_lower = full_response.lower()
                is_refusal = 1.0 if any(phrase.lower() in response_lower for phrase in REFUSAL_PHRASES) else 0.0
                langfuse_context.score_current_observation(
                    name="llm_defensive_refusal",
                    value=is_refusal,
                    comment="The LLM explicitly refused to answer due to missing context or out-of-domain query."
                )
    
            except Exception as system_error:
                # Global Catch-All: Marks the entire root system execution trace as failed
                langfuse_context.score_current_observation(
                    name="system_failure",
                    value=1.0,
                    comment=f"Global Stream Wrapper Runtime Exception: {str(system_error)}"
                )
                langfuse_context.update_current_trace(output=FALLBACK_RESPONSE)

                # Make sure to flush the error trace as well so it registers
                langfuse_context.flush()
                yield FALLBACK_RESPONSE
            finally:
                # =========================================================
                # 🚀 CRITICAL FIX: FORCE IMMEDIATELY FLUSH FLIGHT DATA
                # =========================================================
                # This explicitly locks the trace container state and triggers the cloud AI judge instantly!
                langfuse_context.flush()

    return RAGChainWithFallback(chain, retriever, prompt, llm)