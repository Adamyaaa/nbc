from app.core.llm import LLMClient
from app.models.schemas import EvaluationResult

class LLMJudge:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(self, question: str, expected_answer: str, retrieved_context: str, generated_answer: str) -> EvaluationResult:
        """Uses an LLM to evaluate the generated answer against the expected answer and context."""
        
        system_instruction = "You are an impartial, expert evaluator of AI systems. You must evaluate the AI's response and the retrieved context based on Correctness, Relevance, Groundedness, Context Precision, and Context Recall."
        
        prompt = f"""Evaluate the following generated answer and retrieval context based on these inputs:

QUESTION: {question}
EXPECTED ANSWER: {expected_answer}
RETRIEVED CONTEXT: {retrieved_context}
GENERATED ANSWER: {generated_answer}

Provide a score from 0.0 to 1.0 for each of the following criteria:
1. Correctness: Does the generated answer factually match the expected answer?
2. Relevance: Does the generated answer directly address the user's question without unnecessary filler?
3. Groundedness: Is the generated answer fully supported by the retrieved context? (0.0 if hallucinated).
4. Context Precision: Was the retrieved context highly focused and relevant to the question, without too much noisy/useless information?
5. Context Recall: Did the retrieved context contain ALL the necessary facts required to form the expected answer?

Also provide a brief explanation for your scores."""

        return self.llm_client.generate_structured(
            prompt=prompt,
            schema=EvaluationResult,
            system_instruction=system_instruction
        )
