"""
Core entry point for the real RAG support pipeline (Hiver SDE Assignment).
Exposes run_support_agent() for evaluation harnesses and interactive use.
"""
from rag.pipeline import SupportAgentPipeline, get_pipeline, run_support_agent

__all__ = ["run_support_agent", "SupportAgentPipeline", "get_pipeline"]

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("GWR SUPPORT AGENT - INTERACTIVE TEST")
    print("=" * 60)

    test_context = "Customer: @GWRHelp my train from Paddington to Reading was delayed by 45 minutes."
    test_msg = "How do I claim compensation for this delay?"

    print(f"Context: {test_context}")
    print(f"Message: {test_msg}\n")
    print("Running pipeline...")

    result = run_support_agent(
        conversation_context=test_context,
        customer_message=test_msg,
        top_k=3
    )

    print("\nResult:")
    print(f"Intent:     {result['intent']}")
    print(f"Decision:   {result['decision']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Response:   {result['response']}")
    print(f"\nRetrieved Examples Count: {len(result['retrieved_examples'])}")
    for i, ex in enumerate(result['retrieved_examples'], start=1):
        print(f"  [{i}] Thread: {ex['thread_id']} (Score: {ex['score']:.3f})")
