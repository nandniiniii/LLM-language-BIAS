# Self-consistency engine
# Core logic: sample the small model N times, measure agreement
# If agreement is high → trust it. If low → escalate.

from src.models import call_small_model, call_large_model
from src.config import N_SAMPLES, CONSISTENCY_THRESHOLD


def get_majority_answer(answers: list[str]) -> str:
    """Return the most common answer from a list of samples."""
    return max(set(answers), key=answers.count)


def compute_confidence(answers: list[str]) -> float:
    """
    Confidence = fraction of answers that match the majority.
    Example: [A, A, A, B, A] → majority=A, confidence=4/5=0.8
    """
    majority = get_majority_answer(answers)
    return answers.count(majority) / len(answers)


def route_query(prompt: str) -> dict:
    """
    Main routing function.
    1. Sample small model N times
    2. Compute confidence
    3. If confident → return small model answer
    4. If not confident → escalate to large model
    """
    # Step 1: Sample small model
    answers = [call_small_model(prompt) for _ in range(N_SAMPLES)]

    # Step 2: Compute confidence
    confidence = compute_confidence(answers)
    majority_answer = get_majority_answer(answers)

    # Step 3: Route decision
    if confidence >= (CONSISTENCY_THRESHOLD / N_SAMPLES):
        return {
            "answer": majority_answer,
            "model_used": "small",
            "confidence": confidence,
            "escalated": False,
            "all_samples": answers
        }
    else:
        # Step 4: Escalate
        large_answer = call_large_model(prompt)
        return {
            "answer": large_answer,
            "model_used": "large",
            "confidence": confidence,
            "escalated": True,
            "all_samples": answers
        }