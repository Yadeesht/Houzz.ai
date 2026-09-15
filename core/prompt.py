"""
LLM-as-a-Judge prompt templates and evaluator for RAG support pipeline.
Evaluates context retrieval relevance/utility and response generation quality.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

JUDGE_SYSTEM_PROMPT = """You are an expert AI evaluator and judge for customer support systems (specifically Great Western Railway - GWRHelp).
Your mission is to rigorously and objectively evaluate a Customer Support RAG pipeline on two dimensions:

1. RETRIEVAL RELEVANCE & UTILITY (Score: 0 to 100%):
   - Assess whether the retrieved historical conversations are semantically relevant to the customer's specific problem.
   - Check if the retrieved examples provide useful reference templates, tone, or operational guidance.
   - A score of 100 means the retrieved examples are highly relevant and directly useful.
   - A score below 50 means the retrieved examples are off-topic or unhelpful.

2. RESPONSE GENERATION QUALITY (Score: 0 to 100%):
   - Assess the newly generated RAG response against the customer's message, conversation context, and the ground-truth human agent response.
   - Evaluate correctness, empathy, professionalism, conciseness, absence of hallucinated facts/policies, and alignment with GWR brand tone.
   - A score of 90-100 means the response is stellar, accurate, helpful, and polite.
   - A score of 60-80 means acceptable with minor flaws or slightly generic.
   - A score below 50 means incorrect, hallucinated, rude, or completely unhelpful.

CRITICAL INSTRUCTION:
You MUST return ONLY a valid JSON object without markdown or code fences:
{
  "retrieval_score": <integer 0-100>,
  "retrieval_reasoning": "<concise explanation of retrieval relevance and usefulness>",
  "response_score": <integer 0-100>,
  "response_reasoning": "<concise evaluation of the generated response accuracy, tone, and helpfulness>"
}
"""


def build_judge_prompt(
    customer_message: str,
    conversation_context: str,
    retrieved_examples: List[Dict],
    ground_truth_intent: str,
    predicted_intent: str,
    ground_truth_response: str,
    generated_response: str
) -> str:
    """
    Format evaluation inputs for the LLM judge.
    """
    retrieved_text_blocks = []
    for i, ex in enumerate(retrieved_examples, start=1):
        t_id = ex.get("thread_id", "Unknown")
        score = ex.get("score", 0.0)
        snippet = ex.get("text", "").strip()
        retrieved_text_blocks.append(
            f"[Example {i} - Thread: {t_id}, Similarity: {score:.3f}]\n{snippet}"
        )

    retrieved_formatted = (
        "\n\n".join(retrieved_text_blocks)
        if retrieved_text_blocks
        else "No examples retrieved."
    )

    prompt = f"""EVALUATION CASE:

[CURRENT CUSTOMER INQUIRY]
Conversation History: {conversation_context or "None"}
Customer Message: {customer_message}

[INTENT COMPARISON]
Ground Truth Intent: {ground_truth_intent or "Unknown"}
Predicted Intent:    {predicted_intent}

[RETRIEVED CONTEXT / EXAMPLES FROM FAISS]
{retrieved_formatted}

[AGENT RESPONSES]
Reference Human Agent Response (Ground Truth):
{ground_truth_response or "None available"}

Generated RAG Pipeline Response:
{generated_response}

Please evaluate both (1) Retrieval Relevance & Utility and (2) Response Generation Quality.
Return ONLY the JSON object with keys: "retrieval_score", "retrieval_reasoning", "response_score", "response_reasoning".
"""
    return prompt


class LLMJudge:
    """
    LLM Judge utilizing Gemini via direct HTTP request to evaluate RAG outputs.
    """

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, timeout: int = 30):
        self.model = model
        self.timeout = timeout
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Please add your API key: GEMINI_API_KEY or GOOGLE_API_KEY must be set in your environment or .env file."
            )

    def evaluate(
        self,
        customer_message: str,
        conversation_context: str,
        retrieved_examples: List[Dict],
        ground_truth_intent: str,
        predicted_intent: str,
        ground_truth_response: str,
        generated_response: str
    ) -> Dict[str, Any]:
        """
        Invoke the LLM Judge and parse the JSON evaluation.
        """
        user_prompt = build_judge_prompt(
            customer_message=customer_message,
            conversation_context=conversation_context,
            retrieved_examples=retrieved_examples,
            ground_truth_intent=ground_truth_intent,
            predicted_intent=predicted_intent,
            ground_truth_response=ground_truth_response,
            generated_response=generated_response
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json"
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        max_retries = 5
        raw_text = None

        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    break
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                # Retry on rate limit (429) or temporary server unavailable (500, 502, 503, 504)
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    sleep_time = min(25.0, 3.0 * (1.8 ** (attempt - 1)))
                    logger.warning(
                        "[LLM Judge] HTTP %d (%s). Sleeping %.1fs before retrying (Attempt %d/%d)...",
                        e.code, e.reason, sleep_time, attempt, max_retries
                    )
                    import time
                    time.sleep(sleep_time)
                    continue
                raise RuntimeError(f"Judge API HTTP {e.code} error: {err_body or e.reason}")
            except Exception as e:
                if attempt < max_retries:
                    sleep_time = min(25.0, 3.0 * (1.8 ** (attempt - 1)))
                    logger.warning(
                        "[LLM Judge] %s. Sleeping %.1fs before retrying (Attempt %d/%d)...",
                        e, sleep_time, attempt, max_retries
                    )
                    import time
                    time.sleep(sleep_time)
                    continue
                raise RuntimeError(f"Judge API call failed after {max_retries} attempts: {e}")

        if raw_text is None:
            raise RuntimeError("Judge API returned empty response after all retry attempts.")

        # Clean and parse JSON
        cleaned = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse judge JSON: %s. Output:\n%s", e, raw_text)
            raise ValueError(f"Judge returned non-JSON output: {raw_text}")

        ret_score = int(parsed["retrieval_score"])
        resp_score = int(parsed["response_score"])

        return {
            "retrieval_score": ret_score,
            "retrieval_reasoning": str(parsed.get("retrieval_reasoning", "")).strip(),
            "response_score": resp_score,
            "response_reasoning": str(parsed.get("response_reasoning", "")).strip(),
        }

