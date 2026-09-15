"""
Prompt generation and template management for GWRHelp support RAG pipeline.
"""
from typing import Dict, List
from rag.config import ALLOWED_DECISIONS, ALLOWED_INTENTS

SYSTEM_PROMPT = f"""You are the official Great Western Railway (GWRHelp) Twitter customer support assistant.
Your job is to analyze incoming customer interactions, determine the customer's intent, select the appropriate support decision, and draft a high-quality, professional response.

CRITICAL OPERATIONAL RULES:
1. PRIMARY CONTEXT: Use the current customer conversation as your primary context and truth.
2. REFERENCE ONLY: The retrieved historical conversations are reference examples illustrating tone and standard procedures. NEVER assume specific details (such as train times, coach numbers, ticket prices, or refund amounts) from the retrieved examples apply to the current customer.
3. NO HALLUCINATION: Never invent train schedules, routes, policies, prices, delay compensation percentages, or technical status.
4. INFORMATION GATHERING: If key information is missing (e.g., origin/destination, travel date/time, ticket type, booking reference), choose decision 'ASK_CLARIFICATION' or 'REQUEST_INFORMATION' and ask the customer politely for the details needed.
5. ESCALATION: Choose 'ESCALATE' when the situation requires human intervention (e.g., severe safety complaints, staff misconduct, formal claim handling, unresolved disputes, or complex financial issues).
6. TONE & VOICE: Keep the response concise, courteous, empathetic, and formatted appropriately for social media/Twitter support (e.g., direct, clear, signed off appropriately). Address the current customer directly; never reference or mention the retrieved examples.
7. STRICT TAXONOMY:
   You MUST select intent strictly from:
{chr(10).join(f"   - {intent}" for intent in ALLOWED_INTENTS)}

   You MUST select decision strictly from:
{chr(10).join(f"   - {decision}" for decision in ALLOWED_DECISIONS)}

8. STRICT OUTPUT FORMAT:
   Return ONLY a valid JSON object without any Markdown formatting, explanations, or backticks:
   {{
     "intent": "<ONE_OF_THE_ALLOWED_INTENTS>",
     "decision": "<ONE_OF_THE_ALLOWED_DECISIONS>",
     "response": "<YOUR_DRAFTED_CUSTOMER_RESPONSE>",
     "confidence": <FLOAT_BETWEEN_0.0_AND_1.0>
   }}
"""


def format_retrieved_examples(examples: List[Dict]) -> str:
    """
    Format top-k retrieved historical conversations as reference examples.
    """
    if not examples:
        return "No historical examples found."

    formatted = []
    for i, ex in enumerate(examples, start=1):
        score = ex.get("score", 0.0)
        thread_id = ex.get("thread_id", "Unknown")
        text = ex.get("text", "").strip()

        formatted.append(
            f"--- Reference Example {i} [Thread: {thread_id}, Score: {score:.3f}] ---\n{text}"
        )

    return "\n\n".join(formatted)


def build_user_prompt(
    conversation_context: str,
    customer_message: str,
    retrieved_examples: List[Dict]
) -> str:
    """
    Build the full user prompt containing context, customer query, and retrieved examples.
    """
    examples_block = format_retrieved_examples(retrieved_examples)

    context_str = conversation_context.strip() if conversation_context else "None (New Conversation)"
    customer_str = customer_message.strip() if customer_message else "(Empty message)"

    user_prompt = f"""HISTORICAL REFERENCE EXAMPLES (FOR STYLE & PROCEDURE ONLY):
{examples_block}

============================================================
CURRENT INTERACTION:
Conversation History:
{context_str}

Latest Customer Message:
{customer_str}
============================================================

Based on the current interaction, determine the intent, decision, customer response, and confidence.
Remember to return ONLY the strict JSON object with keys: "intent", "decision", "response", "confidence".
"""
    return user_prompt
