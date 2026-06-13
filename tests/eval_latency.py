import os
import sys
import datetime
from langfuse import Langfuse

# =========================================================
# PRODUCTION LATENCY THRESHOLDS (MAX LIMITS)
# =========================================================
MAX_P50_LATENCY_SECONDS = 10.0  # Fail build if 50% of users wait > 10 seconds
MAX_P95_LATENCY_SECONDS = 25.0  # Fail build if 5% of users wait > 25 seconds
MAX_P99_LATENCY_SECONDS = 28.0  # Fail build if 1% of users wait > 28 seconds

def verify_production_latency():
    print("Connecting to Langfuse API Platform...")
    
    # Initialize the client (Reads standard keys from your environment)
    langfuse = Langfuse()
    
    # Define timeframe lookup window (Look at all traces from the last 24 hours)
    one_day_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    
    print("Fetching historical trace metrics from the past 24 hours...")
    try:
        # Pull your live production streaming trace entries
        traces = langfuse.get_traces(
            name="rag_stream_request",
            from_timestamp=one_day_ago,
            limit=100
        )
    except Exception as e:
        print(f"❌ API Connection Error: Failed to reach Langfuse. Details: {e}")
        sys.exit(1)

    # Extract execution durations (Langfuse tracks duration in seconds)
    durations = [trace.duration for trace in traces.data if trace.duration is not None]
    
    if not durations:
        print("⚠️ No live production trace history located for 'rag_stream_request' in the last 24 hours.")
        print("Skipping latency gate execution due to insufficient dataset history.")
        sys.exit(0)

    # Sort durations from fastest to slowest to calculate percentiles
    durations.sort()
    total_traces = len(durations)
    
    # Calculate index positions for P50 and P95
    p50_idx = int(total_traces * 0.50)
    p95_idx = int(total_traces * 0.95)
    p99_idx = int(total_traces * 0.99)
    
    p50_latency = durations[p50_idx]
    p95_latency = durations[min(p95_idx, total_traces - 1)]
    p99_latency = durations[min(p99_idx, total_traces - 1)]

    print("\n" + "="*60)
    print("             🛡️ PRODUCTION LATENCY METRIC REPORT            ")
    print("="*60)
    print(f"🔹 Total Live Traces Evaluated: {total_traces}")
    print(f"🔹 Current P50 (Median) Delay:  {p50_latency:.2f}s")
    print(f"🔹 Current P95 (Tail) Delay:    {p95_latency:.2f}s")
    print(f"🔹 Current P99 (Tail) Delay:    {p99_latency:.2f}s")
    print(f"🔹 Target P50 Maximum Cap:      {MAX_P50_LATENCY_SECONDS:.2f}s")
    print(f"🔹 Target P95 Maximum Cap:      {MAX_P95_LATENCY_SECONDS:.2f}s")
    print(f"🔹 Target P99 Maximum Cap:      {MAX_P99_LATENCY_SECONDS:.2f}s")
    print("="*60)

    # Trigger Regression Gate Blockade
    if p50_latency > MAX_P50_LATENCY_SECONDS:
        print(f"🛑 BUILD REJECTED: Production P50 median latency ({p50_latency:.2f}s) exceeds your {MAX_P50_LATENCY_SECONDS}s safety ceiling!")
        print("Reason: Chatbot responses are too slow for the average user experience.")
        print("="*60 + "\n")
        sys.exit(1)
    if p95_latency > MAX_P95_LATENCY_SECONDS:
        print(f"🛑 BUILD REJECTED: Production P95 tail latency ({p95_latency:.2f}s) exceeds your {MAX_P95_LATENCY_SECONDS}s safety ceiling!")
        print("Reason: Chatbot responses are streaming too slowly for our unluckiest users.")
        print("="*60 + "\n")
        sys.exit(1)
    if p99_latency > MAX_P99_LATENCY_SECONDS:
        print(f"🛑 BUILD REJECTED: Production P99 tail latency ({p99_latency:.2f}s) exceeds your {MAX_P99_LATENCY_SECONDS}s safety ceiling!")
        print("Reason: Chatbot responses are streaming too slowly for our unluckiest users.")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print("✅ BUILD COMPLIANT: Live system application speed is within safe operational margins.")
        print("="*60 + "\n")
        sys.exit(0)

if __name__ == "__main__":
    verify_production_latency()
