"""
Core entry point for the real RAG support pipeline (Hiver SDE Assignment).
Exposes run_support_agent() for evaluation harnesses and interactive use.
"""
from rag.pipeline import SupportAgentPipeline, get_pipeline, run_support_agent

__all__ = ["run_support_agent", "SupportAgentPipeline", "get_pipeline"]

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="GWR Support Agent - Interactive Query & Speed Test")
    parser.add_argument(
        "--message", "-m",
        type=str,
        default="How do I claim compensation for this delay?",
        help="Customer message query"
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        default="Customer: @GWRHelp my train from Paddington to Reading was delayed by 45 minutes.",
        help="Preceding conversation context (optional)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Top-K historical examples to retrieve"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("GWR SUPPORT AGENT - INTERACTIVE TEST & SPEED BENCHMARK")
    print("=" * 65)
    print(f"Context: {args.context}")
    print(f"Message: {args.message}")
    print(f"Top-K:   {args.top_k}\n")
    print("Running pipeline...")

    start_time = time.perf_counter()
    result = run_support_agent(
        conversation_context=args.context,
        customer_message=args.message,
        top_k=args.top_k
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    print("\n" + "=" * 65)
    print("RESULT")
    print("=" * 65)
    print(f"Intent:     {result['intent']}")
    print(f"Decision:   {result['decision']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Latency:    {latency_ms:.1f} ms ({latency_ms / 1000:.2f} s)")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nRetrieved Examples Count: {len(result['retrieved_examples'])}")
    for i, ex in enumerate(result['retrieved_examples'], start=1):
        print(f"  [{i}] Thread: {ex['thread_id']} (Score: {ex['score']:.3f})")
    print("=" * 65)

