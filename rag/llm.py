"""
Gemini LLM client using direct HTTP request (gemini-3.1-flash-lite) and strict JSON schema validation.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from rag.config import ALLOWED_DECISIONS, ALLOWED_INTENTS
from rag.prompt import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


class LLMValidationError(Exception):
    """Raised when LLM response does not conform to the required JSON schema or taxonomy."""
    pass


def clean_json_response(raw_text: str) -> str:
    """
    Extract clean JSON string from raw LLM output, removing any accidental markdown blocks.
    """
    text = raw_text.strip()

    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Find outermost JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1].strip()

    return text


def validate_and_parse_llm_output(raw_output: str) -> Dict[str, Any]:
    """
    Parse raw LLM output and validate against taxonomy, decision options,
    confidence range, and response content.
    """
    cleaned = clean_json_response(raw_output)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMValidationError(f"Failed to parse JSON from LLM output: {e}\nRaw output:\n{raw_output}")

    if not isinstance(parsed, dict):
        raise LLMValidationError(f"LLM output must be a JSON object, got {type(parsed).__name__}")

    # Required keys
    required_keys = ["intent", "decision", "response", "confidence"]
    for k in required_keys:
        if k not in parsed:
            raise LLMValidationError(f"Missing required key '{k}' in LLM response: {parsed}")

    intent = str(parsed["intent"]).strip()
    decision = str(parsed["decision"]).strip()
    response = str(parsed["response"]).strip()

    # Validate intent against fixed taxonomy
    matched_intent = next((i for i in ALLOWED_INTENTS if i.lower() == intent.lower()), None)
    if not matched_intent:
        raise LLMValidationError(f"Invalid intent '{intent}'. Must be one of: {ALLOWED_INTENTS}")
    intent = matched_intent

    # Validate decision against allowed options
    matched_decision = next((d for d in ALLOWED_DECISIONS if d.lower() == decision.lower()), None)
    if not matched_decision:
        raise LLMValidationError(f"Invalid decision '{decision}'. Must be one of: {ALLOWED_DECISIONS}")
    decision = matched_decision

    # Validate response
    if not response:
        raise LLMValidationError("Response drafted by LLM is empty.")

    # Validate confidence
    try:
        confidence = float(parsed["confidence"])
    except (ValueError, TypeError):
        raise LLMValidationError(f"Confidence value '{parsed.get('confidence')}' is not numeric.")

    if 0.0 <= confidence <= 100.0 and confidence > 1.0:
        confidence = confidence / 100.0

    if not (0.0 <= confidence <= 1.0):
        raise LLMValidationError(f"Confidence {confidence} is outside valid range [0.0, 1.0]")

    return {
        "intent": intent,
        "decision": decision,
        "response": response,
        "confidence": round(confidence, 3),
    }


class GeminiSupportLLM:
    """
    Simple Gemini client using HTTP POST via urllib.request.
    Defaults to gemini-3.1-flash-lite.
    """

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 30):
        self.model = model
        self.timeout = timeout
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Please add your API key: GEMINI_API_KEY or GOOGLE_API_KEY must be set in your environment or .env file."
            )

    def generate(
        self,
        conversation_context: str,
        customer_message: str,
        retrieved_examples: list
    ) -> Dict[str, Any]:
        """
        Call Gemini API and return strictly validated structured output.
        """
        user_prompt = build_user_prompt(
            conversation_context=conversation_context,
            customer_message=customer_message,
            retrieved_examples=retrieved_examples
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
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
        raw_output = None

        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_output = data["candidates"][0]["content"]["parts"][0]["text"]
                    break
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    sleep_time = min(25.0, 3.0 * (1.8 ** (attempt - 1)))
                    logger.warning(
                        "[Gemini API] HTTP %d (%s). Sleeping %.1fs before retry (Attempt %d/%d)...",
                        e.code, e.reason, sleep_time, attempt, max_retries
                    )
                    import time
                    time.sleep(sleep_time)
                    continue
                raise RuntimeError(f"Gemini API HTTP {e.code} error: {err_body or e.reason}")
            except Exception as e:
                if attempt < max_retries:
                    sleep_time = min(25.0, 3.0 * (1.8 ** (attempt - 1)))
                    logger.warning(
                        "[Gemini API] %s. Sleeping %.1fs before retry (Attempt %d/%d)...",
                        e, sleep_time, attempt, max_retries
                    )
                    import time
                    time.sleep(sleep_time)
                    continue
                raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}")

        if raw_output is None:
            raise RuntimeError("Gemini API returned empty response after all retries.")

        # Validate structured JSON
        return validate_and_parse_llm_output(raw_output)



# Alias for compatibility with pipeline
SupportLLM = GeminiSupportLLM
