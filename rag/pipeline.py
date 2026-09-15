"""
End-to-end RAG support agent pipeline for GWR customer service.
"""
import logging
from typing import Any, Dict, Optional

from rag.config import DEFAULT_TOP_K
from rag.llm import SupportLLM
from rag.retriever import Retriever, build_retrieval_query

logger = logging.getLogger(__name__)


class SupportAgentPipeline:
    """
    RAG support pipeline coordinating query construction, semantic retrieval,
    and single-call structured LLM response generation.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[SupportLLM] = None
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or SupportLLM()

    def run(
        self,
        conversation_context: str = "",
        customer_message: str = "",
        top_k: int = DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        Execute the support agent pipeline for an incoming customer interaction.

        Args:
            conversation_context: Chronological preceding conversation context.
            customer_message: Current customer inquiry.
            top_k: Number of retrieved historical examples.

        Returns:
            Dict containing:
                - intent: Classified intent from allowed taxonomy.
                - decision: Action decision from allowed decisions.
                - response: Drafted customer support message.
                - confidence: Numeric confidence in [0.0, 1.0].
                - retrieved_examples: Top-k historical reference objects.
        """
        ctx = str(conversation_context or "").strip()
        msg = str(customer_message or "").strip()

        if not msg and not ctx:
            return {
                "intent": "OTHER",
                "decision": "ASK_CLARIFICATION",
                "response": "Hello! How can we help you with your journey today?",
                "confidence": 1.0,
                "retrieved_examples": []
            }

        # 1. Build dense retrieval query
        retrieval_query = build_retrieval_query(ctx, msg)

        # 2. Semantic retrieval from FAISS
        try:
            retrieved_examples = self.retriever.retrieve(
                query=retrieval_query,
                top_k=top_k,
                deduplicate_threads=True
            )
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            retrieved_examples = []

        # 3. LLM generation with structured JSON validation
        llm_output = self.llm.generate(
            conversation_context=ctx,
            customer_message=msg,
            retrieved_examples=retrieved_examples
        )

        return {
            "intent": llm_output["intent"],
            "decision": llm_output["decision"],
            "response": llm_output["response"],
            "confidence": llm_output["confidence"],
            "retrieved_examples": retrieved_examples
        }


# Global singleton pipeline instance for lightweight reuse
_GLOBAL_PIPELINE: Optional[SupportAgentPipeline] = None


def get_pipeline() -> SupportAgentPipeline:
    """Get or initialize the global pipeline singleton."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = SupportAgentPipeline()
    return _GLOBAL_PIPELINE


def run_support_agent(
    conversation_context: str = "",
    customer_message: str = "",
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """
    Main external API function to process a customer interaction.

    Usage:
        result = run_support_agent(
            conversation_context="Customer: My train was delayed.",
            customer_message="Can I claim Delay Repay?"
        )
    """
    pipeline = get_pipeline()
    return pipeline.run(
        conversation_context=conversation_context,
        customer_message=customer_message,
        top_k=top_k
    )
