"""
Evaluation script using LLM-as-a-Judge for the GWRHelp RAG support pipeline.
Supports multi-turn sequential replay, initial first-turn triage, and conversational slice evaluation.
Saves all detailed turn results, retrieved contexts, and judge metrics to a CSV in the Data/ folder.

Usage:
    python core/eval.py [--input-file Data/gwr_golden_review.csv] [--samples 25] [--eval-mode replay]
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.myapproach import run_support_agent
from core.prompt import LLMJudge
from rag.config import DATA_DIR, GOLDEN_REVIEW_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GWR RAG pipeline using LLM-as-a-Judge")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=GOLDEN_REVIEW_PATH,
        help="Path to input golden dataset CSV"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DATA_DIR / "evaluation_results.csv",
        help="Path to save evaluation CSV"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of threads to evaluate (0 for all rows)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved historical examples per query"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["replay", "first_turn"],
        default="replay",
        help=(
            "Evaluation mode: "
            "'replay' sequentially replays the thread turn-by-turn with dynamic AI context; "
            "'first_turn' evaluates only the initial customer root inquiry."
        )
    )
    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
        help="In replay mode, use real historical GWR responses in context instead of the AI's generated response."
    )
    return parser.parse_args()



def parse_conversation_turns(conversation: str) -> List[Tuple[str, str]]:
    """
    Parse a conversation string into chronological list of (speaker, text) tuples.
    """
    if not conversation or pd.isna(conversation):
        return []

    lines = str(conversation).splitlines()
    turns = []
    current_speaker = None
    current_text = []

    turn_header_re = re.compile(r"^(Customer|GWRHelp|[A-Za-z0-9_]+):\s*(.*)", re.IGNORECASE)

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        match = turn_header_re.match(line_clean)
        if match:
            if current_speaker and current_text:
                turns.append((current_speaker, " ".join(current_text).strip()))
                current_text = []

            spk = match.group(1).capitalize()
            speaker = "Customer" if spk.lower() == "customer" else "GWRHelp"
            current_speaker = speaker
            text_part = match.group(2).strip()
            if text_part:
                current_text.append(text_part)
        else:
            if current_speaker:
                current_text.append(line_clean)

    if current_speaker and current_text:
        turns.append((current_speaker, " ".join(current_text).strip()))

    return turns


def group_consecutive_speaker_blocks(turns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Merge consecutive messages from the same speaker into a single coherent turn.
    Also strips Twitter 'nan' artifacts.
    For example:
        Customer: Tweet 1
        Customer: Tweet 2 (continuation)
        GWRHelp:  Reply part 1
        GWRHelp:  Reply part 2
    becomes:
        Customer: Tweet 1 Tweet 2
        GWRHelp:  Reply part 1 Reply part 2
    """
    if not turns:
        return []

    cleaned_turns = [
        (spk, txt.strip())
        for spk, txt in turns
        if txt and txt.strip().lower() != "nan"
    ]

    if not cleaned_turns:
        return []

    blocks = []
    current_spk, current_txt = cleaned_turns[0]
    merged_texts = [current_txt]

    for spk, txt in cleaned_turns[1:]:
        if spk == current_spk:
            merged_texts.append(txt)
        else:
            blocks.append((current_spk, " ".join(merged_texts)))
            current_spk = spk
            merged_texts = [txt]

    if merged_texts:
        blocks.append((current_spk, " ".join(merged_texts)))

    return blocks


def extract_dialogue_pairs(blocks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Extract alternating (Customer Batch, GWRHelp Batch) pairs for sequential replay.
    """
    pairs = []
    i = 0
    while i < len(blocks):
        if blocks[i][0] == "Customer":
            cust_msg = blocks[i][1]
            gt_resp = ""
            if i + 1 < len(blocks) and blocks[i + 1][0] == "GWRHelp":
                gt_resp = blocks[i + 1][1]
                i += 2
            else:
                i += 1
            pairs.append((cust_msg, gt_resp))
        else:
            i += 1
    return pairs


def extract_first_turn_components(turns: List[Tuple[str, str]]) -> Tuple[str, str, str]:
    """
    Extract the initial customer root message (context="") and the first real GWRHelp reply.
    """
    blocks = group_consecutive_speaker_blocks(turns)
    if not blocks:
        return "", "", ""

    first_cust_idx = next((i for i, (s, _) in enumerate(blocks) if s == "Customer"), -1)
    if first_cust_idx == -1:
        return "", "", ""

    cust_msg = blocks[first_cust_idx][1]
    gt_resp = ""
    for i in range(first_cust_idx + 1, len(blocks)):
        if blocks[i][0] == "GWRHelp":
            gt_resp = blocks[i][1]
            break

    return "", cust_msg, gt_resp



def format_retrieved_context(retrieved_examples: List[Dict]) -> str:
    """Format full retrieved context text into readable string."""
    blocks = []
    for i, ex in enumerate(retrieved_examples, start=1):
        t_id = ex.get("thread_id", "Unknown")
        sc = ex.get("score", 0.0)
        txt = ex.get("text", "").strip()
        blocks.append(f"[Example {i} | Thread: {t_id} | Score: {sc:.3f}]\n{txt}")
    return "\n\n".join(blocks) if blocks else "None"


def main():
    args = parse_args()

    print("=" * 75)
    print("GWR SUPPORT PIPELINE: MULTI-TURN LLM-AS-A-JUDGE EVALUATION")
    print("=" * 75)
    print(f"Input CSV:          {args.input_file}")
    print(f"Output CSV:         {args.output_file}")
    print(f"Evaluation Mode:    {args.eval_mode.upper()}")
    print(f"Target Threads:     {args.samples if args.samples > 0 else 'ALL'}")
    print(f"Top-K Retrieval:    {args.top_k}")
    print(f"Teacher Forcing:    {args.teacher_forcing}")
    print("=" * 75)

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found at: {args.input_file}")

    # Load dataset
    df_raw = pd.read_csv(args.input_file)
    logger.info("Loaded input dataset with %d rows.", len(df_raw))

    # Detect conversation text column (supports both 'full_conversation' and 'data')
    conv_col = None
    for col_candidate in ["full_conversation", "data", "conversation"]:
        if col_candidate in df_raw.columns:
            conv_col = col_candidate
            break

    if not conv_col:
        raise KeyError(
            f"Input CSV at {args.input_file} must contain a conversation text column "
            f"(either 'full_conversation' or 'data'). Found columns: {list(df_raw.columns)}"
        )

    # Keep rows with usable labels and conversations
    df = df_raw[df_raw["intent"].notna() & df_raw[conv_col].notna()].copy()
    if conv_col != "full_conversation":
        df["full_conversation"] = df[conv_col]
    logger.info("Rows with valid intent and conversation (from '%s'): %d", conv_col, len(df))

    # Sampling threads
    if args.samples > 0 and len(df) > args.samples:
        df_eval = df.sample(n=args.samples, random_state=args.random_seed).copy()
    else:
        df_eval = df.copy()

    logger.info("Proceeding with %d threads to evaluate.", len(df_eval))

    # Initialize Judge
    judge = LLMJudge()

    evaluation_records = []
    start_time = time.time()
    total_threads = len(df_eval)

    for thread_num, (_, row) in enumerate(df_eval.iterrows(), start=1):
        thread_id = str(row.get("thread_id", f"Row_{thread_num}"))
        ground_truth_intent = str(row.get("intent", "")).strip()
        full_conversation = str(row.get("full_conversation", "")).strip()

        raw_turns = parse_conversation_turns(full_conversation)
        blocks = group_consecutive_speaker_blocks(raw_turns)

        print(f"\n[{thread_num}/{total_threads}] === THREAD: {thread_id} | Golden Intent: {ground_truth_intent} ===")

        if args.eval_mode == "replay":
            # Extract alternating dialogue turns
            dialogue_turns = extract_dialogue_pairs(blocks)
            if not dialogue_turns:
                logger.warning("No valid customer dialogue turns found in %s", thread_id)
                continue

            running_context: List[str] = []
            num_turns_in_thread = len(dialogue_turns)

            for turn_idx, (cust_msg, gt_resp) in enumerate(dialogue_turns, start=1):
                context_str = "\n".join(running_context)

                print(f"\n  -- [Turn {turn_idx}/{num_turns_in_thread}] --")
                print(f"  Customer Message: {cust_msg[:75]}...")
                if context_str:
                    print(f"  Context Turns:    {len(running_context)//2} preceding exchanges")

                # 1. Run RAG Pipeline
                try:
                    pipeline_out = run_support_agent(
                        conversation_context=context_str,
                        customer_message=cust_msg,
                        top_k=args.top_k
                    )
                except Exception as e:
                    logger.error("Pipeline failure on %s (Turn %d): %s", thread_id, turn_idx, e)
                    continue

                pred_intent = pipeline_out["intent"]
                pred_decision = pipeline_out["decision"]
                pred_confidence = pipeline_out["confidence"]
                gen_response = pipeline_out["response"]
                retrieved_examples = pipeline_out["retrieved_examples"]

                intent_matched = (pred_intent.lower() == ground_truth_intent.lower())

                # 2. Run LLM Judge
                try:
                    judge_out = judge.evaluate(
                        customer_message=cust_msg,
                        conversation_context=context_str,
                        retrieved_examples=retrieved_examples,
                        ground_truth_intent=ground_truth_intent,
                        predicted_intent=pred_intent,
                        ground_truth_response=gt_resp,
                        generated_response=gen_response
                    )
                    retrieval_score = judge_out["retrieval_score"]
                    response_score = judge_out["response_score"]
                    retrieval_reason = judge_out["retrieval_reasoning"]
                    response_reason = judge_out["response_reasoning"]
                    print(f"  Predicted Intent: {pred_intent} ({'MATCH' if intent_matched else 'MISMATCH'})")
                    print(f"  Retrieval Utility Score: {retrieval_score}% | Response Quality Score: {response_score}%")
                except Exception as e:
                    logger.error("Judge evaluation failed for %s (Turn %d): %s", thread_id, turn_idx, e)
                    retrieval_score = None
                    response_score = None
                    retrieval_reason = f"Judge evaluation failed: {e}"
                    response_reason = f"Judge evaluation failed: {e}"
                    print(f"  Predicted Intent: {pred_intent} ({'MATCH' if intent_matched else 'MISMATCH'})")
                    print(f"  Judge Evaluation: FAILED (Recorded as None)")

                # Extract retrieved metadata
                retrieved_ids = [ex.get("thread_id", "") for ex in retrieved_examples]
                retrieval_scores = [round(ex.get("score", 0.0), 3) for ex in retrieved_examples]
                retrieved_context_text = format_retrieved_context(retrieved_examples)

                record = {
                    "thread_id": thread_id,
                    "turn_number": turn_idx,
                    "total_thread_turns": num_turns_in_thread,
                    "eval_mode": "replay",
                    "customer_message": cust_msg,
                    "conversation_context": context_str,
                    "retrieved_context": retrieved_context_text,
                    "retrieved_thread_ids": ";".join(retrieved_ids),
                    "retrieval_scores": ";".join(str(s) for s in retrieval_scores),
                    "ground_truth_intent": ground_truth_intent,
                    "predicted_intent": pred_intent,
                    "intent_match": intent_matched,
                    "predicted_decision": pred_decision,
                    "predicted_confidence": pred_confidence,
                    "ground_truth_response": gt_resp,
                    "generated_response": gen_response,
                    "retrieval_score_percent": retrieval_score,
                    "retrieval_reasoning": retrieval_reason,
                    "response_score_percent": response_score,
                    "response_reasoning": response_reason,
                }
                evaluation_records.append(record)

                # Update context for the next turn
                running_context.append(f"Customer: {cust_msg}")
                if args.teacher_forcing and gt_resp:
                    running_context.append(f"GWRHelp: {gt_resp}")
                else:
                    running_context.append(f"GWRHelp: {gen_response}")

                time.sleep(1.0)

        else:
            # Single-turn initial triage (first_turn mode)
            context, customer_msg, gt_response = extract_first_turn_components(raw_turns)
            if not customer_msg:
                continue


            print(f"  Customer Message: {customer_msg[:80]}...")

            try:
                pipeline_out = run_support_agent(
                    conversation_context=context,
                    customer_message=customer_msg,
                    top_k=args.top_k
                )
            except Exception as e:
                logger.error("Pipeline failure on %s: %s", thread_id, e)
                continue

            pred_intent = pipeline_out["intent"]
            pred_decision = pipeline_out["decision"]
            pred_confidence = pipeline_out["confidence"]
            gen_response = pipeline_out["response"]
            retrieved_examples = pipeline_out["retrieved_examples"]

            intent_matched = (pred_intent.lower() == ground_truth_intent.lower())

            try:
                judge_out = judge.evaluate(
                    customer_message=customer_msg,
                    conversation_context=context,
                    retrieved_examples=retrieved_examples,
                    ground_truth_intent=ground_truth_intent,
                    predicted_intent=pred_intent,
                    ground_truth_response=gt_response,
                    generated_response=gen_response
                )
                retrieval_score = judge_out["retrieval_score"]
                response_score = judge_out["response_score"]
                retrieval_reason = judge_out["retrieval_reasoning"]
                response_reason = judge_out["response_reasoning"]
                print(f"  Predicted Intent: {pred_intent} ({'MATCH' if intent_matched else 'MISMATCH'})")
                print(f"  Retrieval Utility Score: {retrieval_score}% | Response Quality Score: {response_score}%")
            except Exception as e:
                logger.error("Judge evaluation failed for %s: %s", thread_id, e)
                retrieval_score = None
                response_score = None
                retrieval_reason = f"Judge evaluation failed: {e}"
                response_reason = f"Judge evaluation failed: {e}"
                print(f"  Predicted Intent: {pred_intent} ({'MATCH' if intent_matched else 'MISMATCH'})")
                print(f"  Judge Evaluation: FAILED (Recorded as None)")

            retrieved_ids = [ex.get("thread_id", "") for ex in retrieved_examples]
            retrieval_scores = [round(ex.get("score", 0.0), 3) for ex in retrieved_examples]
            retrieved_context_text = format_retrieved_context(retrieved_examples)

            record = {
                "thread_id": thread_id,
                "turn_number": 1,
                "total_thread_turns": 1,
                "eval_mode": args.eval_mode,
                "customer_message": customer_msg,
                "conversation_context": context,
                "retrieved_context": retrieved_context_text,
                "retrieved_thread_ids": ";".join(retrieved_ids),
                "retrieval_scores": ";".join(str(s) for s in retrieval_scores),
                "ground_truth_intent": ground_truth_intent,
                "predicted_intent": pred_intent,
                "intent_match": intent_matched,
                "predicted_decision": pred_decision,
                "predicted_confidence": pred_confidence,
                "ground_truth_response": gt_response,
                "generated_response": gen_response,
                "retrieval_score_percent": retrieval_score,
                "retrieval_reasoning": retrieval_reason,
                "response_score_percent": response_score,
                "response_reasoning": response_reason,
            }
            evaluation_records.append(record)
            time.sleep(1.0)

    # Save Results
    results_df = pd.DataFrame(evaluation_records)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_file, index=False)

    elapsed = time.time() - start_time

    # Summary Metrics
    num_eval = len(results_df)
    if num_eval > 0:
        overall_acc = (results_df["intent_match"].sum() / num_eval) * 100.0

        valid_retrieval = results_df["retrieval_score_percent"].dropna()
        valid_response = results_df["response_score_percent"].dropna()

        avg_retrieval = valid_retrieval.mean() if len(valid_retrieval) > 0 else 0.0
        avg_response = valid_response.mean() if len(valid_response) > 0 else 0.0

        print("\n" + "=" * 75)
        print("MULTI-TURN EVALUATION SUMMARY RESULTS")
        print("=" * 75)
        print(f"Total Threads Processed:            {total_threads}")
        print(f"Total Turns Evaluated:              {num_eval}")
        print(f"Overall Intent Accuracy:            {overall_acc:.1f}%")

        if args.eval_mode == "replay" and "turn_number" in results_df.columns:
            turn1_df = results_df[results_df["turn_number"] == 1]
            later_df = results_df[results_df["turn_number"] > 1]
            if len(turn1_df) > 0:
                t1_acc = (turn1_df["intent_match"].sum() / len(turn1_df)) * 100.0
                print(f"  - Turn 1 (Initial Contact) Acc:   {t1_acc:.1f}% ({len(turn1_df)} turns)")
            if len(later_df) > 0:
                tl_acc = (later_df["intent_match"].sum() / len(later_df)) * 100.0
                print(f"  - Turn 2+ (Follow-up Retention):  {tl_acc:.1f}% ({len(later_df)} turns)")

        print(f"Successfully Judged Turns:          {len(valid_retrieval)}/{num_eval}")
        print(f"Average Retrieval Relevance/Utility: {avg_retrieval:.1f}%")
        print(f"Average Response Quality:           {avg_response:.1f}%")
        print(f"Total Time Taken:                   {elapsed:.1f}s ({elapsed/num_eval:.2f}s per turn)")
        print(f"Detailed Results Saved To:          {args.output_file}")
        print("=" * 75)


if __name__ == "__main__":
    main()
