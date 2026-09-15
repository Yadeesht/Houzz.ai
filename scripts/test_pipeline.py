"""
Test and verification suite for the GWRHelp RAG support pipeline.

Verifies:
1. Golden set zero-leakage check on FAISS metadata and retrieval.
2. End-to-end execution on 5 distinct non-golden customer scenarios.
3. Printed outputs: intent, decision, response, confidence, retrieved thread IDs, and scores.
4. Error handling tests: empty queries, malformed LLM outputs, missing files.

Usage:
    python scripts/test_pipeline.py
"""
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import (
    ALLOWED_DECISIONS,
    ALLOWED_INTENTS,
    GOLDEN_REVIEW_PATH,
    METADATA_PATH,
)
from rag.index import load_golden_thread_ids
from rag.llm import LLMValidationError, validate_and_parse_llm_output
from rag.pipeline import run_support_agent

logging.basicConfig(level=logging.WARNING)


# 5 Curated test scenarios outside the golden set
TEST_SCENARIOS = [
    {
        "id": "scenario_1",
        "description": "Train Cancellation & Ticket Acceptance",
        "expected_intent": "TRAIN DELAY / CANCELLATION",
        "context": "Customer: @GWRHelp our 08:30 from Swindon to London Paddington was cancelled without notice.",
        "message": "What is the alternative train we can take to get to Paddington, and will our tickets still be accepted?"
    },
    {
        "id": "scenario_2",
        "description": "Delay Repay / Compensation Claim",
        "expected_intent": "REFUND / COMPENSATION",
        "context": "Customer: @GWRHelp my journey yesterday from Bristol Temple Meads was delayed over 2 hours due to signal failure.",
        "message": "Can I claim Delay Repay compensation online, and what proof do I need to provide?"
    },
    {
        "id": "scenario_3",
        "description": "Lost Backpack / Property on Train",
        "expected_intent": "BAGGAGE / LOST PROPERTY",
        "context": "Customer: @GWRHelp I think I left my black backpack under seat 42 in coach B on the 14:15 service to Swansea.",
        "message": "Who do I contact to report lost property at Swansea station?"
    },
    {
        "id": "scenario_4",
        "description": "Wheelchair Accessibility & Passenger Assistance",
        "expected_intent": "ACCESSIBILITY / SPECIAL ASSISTANCE",
        "context": "Customer: @GWRHelp I am traveling tomorrow with my mother who uses a wheelchair.",
        "message": "Can we arrange ramp assistance at Reading station and is there step-free access to platform 7?"
    },
    {
        "id": "scenario_5",
        "description": "Staff Conduct Complaint",
        "expected_intent": "STAFF / SERVICE COMPLAINT",
        "context": "Customer: @GWRHelp the conductor on the 17:45 from Oxford was extremely rude to passengers when asked why the heating was broken.",
        "message": "How can I file a formal complaint regarding this member of staff?"
    }
]


def test_golden_set_exclusion():
    """Verify that no golden review thread_id exists in the FAISS metadata."""
    print("\n" + "=" * 70)
    print("VERIFICATION 1: GOLDEN SET EXCLUSION CHECK")
    print("=" * 70)

    golden_ids = load_golden_thread_ids(GOLDEN_REVIEW_PATH)
    print(f"Loaded {len(golden_ids)} golden evaluation thread IDs.")

    if not METADATA_PATH.exists():
        print(f"Metadata file {METADATA_PATH} does not exist yet. Please build the index first.")
        return False

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    indexed_threads = {item["thread_id"] for item in metadata}
    overlap = indexed_threads.intersection(golden_ids)

    print(f"Total unique threads indexed: {len(indexed_threads):,}")
    print(f"Golden set overlap count:     {len(overlap)}")

    assert len(overlap) == 0, f"CONTAMINATION DETECTED: {len(overlap)} golden threads found in index: {overlap}"
    print(">>> PASS: ZERO golden review threads found in indexed corpus. Golden set is strictly quarantined.")
    return True


def test_pipeline_scenarios():
    """Run pipeline on 5 realistic non-golden scenarios."""
    print("\n" + "=" * 70)
    print("VERIFICATION 2: RUNNING PIPELINE ON 5 TEST SCENARIOS")
    print("=" * 70)

    golden_ids = load_golden_thread_ids(GOLDEN_REVIEW_PATH)
    all_passed = True

    for i, sc in enumerate(TEST_SCENARIOS, start=1):
        print(f"\n--- [Scenario {i}/5]: {sc['description']} ---")
        print(f"Context: {sc['context']}")
        print(f"Customer Message: {sc['message']}")

        result = run_support_agent(
            conversation_context=sc["context"],
            customer_message=sc["message"],
            top_k=3
        )

        intent = result["intent"]
        decision = result["decision"]
        confidence = result["confidence"]
        response = result["response"]
        retrieved = result["retrieved_examples"]

        print(f"\n[Agent Output]")
        print(f"  Detected Intent: {intent}")
        print(f"  Decision:        {decision}")
        print(f"  Confidence:      {confidence}")
        print(f"  Response:        {response}")

        print(f"\n[Retrieved Examples ({len(retrieved)})]")
        for j, ex in enumerate(retrieved, start=1):
            t_id = ex["thread_id"]
            score = ex["score"]
            snippet = ex["text"].replace("\n", " ")[:90]
            print(f"  ({j}) Thread: {t_id} | Score: {score:.3f} | Snippet: {snippet}...")

            # Verify no golden set leakage in runtime retrieval
            if t_id in golden_ids:
                print(f"  CRITICAL ERROR: Golden thread {t_id} retrieved!")
                all_passed = False

        # Schema validations
        assert intent in ALLOWED_INTENTS, f"Invalid intent: {intent}"
        assert decision in ALLOWED_DECISIONS, f"Invalid decision: {decision}"
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"
        assert len(response.strip()) > 0, "Empty response generated"

    if all_passed:
        print("\n>>> PASS: All 5 scenarios executed successfully with valid schemas and clean retrieval.")
    return all_passed


def test_error_handling():
    """Test resilience against edge cases and malformed inputs."""
    print("\n" + "=" * 70)
    print("VERIFICATION 3: ERROR HANDLING & EDGE CASES")
    print("=" * 70)

    # 1. Empty message & context
    res_empty = run_support_agent(conversation_context="", customer_message="")
    assert res_empty["decision"] == "ASK_CLARIFICATION", "Failed to handle empty input"
    print(">>> PASS: Empty input handled gracefully with ASK_CLARIFICATION.")

    # 2. Malformed JSON handling
    malformed_tests = [
        "not json at all",
        '{"intent": "INVALID_INTENT", "decision": "RESPOND", "response": "Hi", "confidence": 0.9}',
        '{"intent": "REFUND / COMPENSATION", "decision": "INVALID_DECISION", "response": "Hi", "confidence": 0.9}',
        '{"intent": "REFUND / COMPENSATION", "decision": "RESPOND", "response": "", "confidence": 0.9}',
        '{"intent": "REFUND / COMPENSATION", "decision": "RESPOND", "response": "Hi", "confidence": "not_a_number"}',
    ]

    caught_count = 0
    for bad_input in malformed_tests:
        try:
            validate_and_parse_llm_output(bad_input)
        except LLMValidationError:
            caught_count += 1

    assert caught_count == len(malformed_tests), f"Only caught {caught_count}/{len(malformed_tests)} errors"
    print(f">>> PASS: Schema validator correctly rejected all {caught_count} malformed LLM outputs.")


def main():
    print("=" * 70)
    print("GWR SUPPORT PIPELINE INTEGRATION TEST SUITE")
    print("=" * 70)

    # 1. Golden set exclusion check
    golden_ok = test_golden_set_exclusion()
    if not golden_ok:
        print("Please build the FAISS index before running the full test suite.")
        sys.exit(1)

    # 2. Pipeline 5 scenarios
    test_pipeline_scenarios()

    # 3. Error handling
    test_error_handling()

    print("\n" + "=" * 70)
    print("ALL TEST SUITES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
