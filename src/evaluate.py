from datasets import load_dataset
import pandas as pd
from typing import Any


def compute_metrics(results: list[dict[str, Any]]) -> dict:
    """
    Computes all research metrics from routing results.
    Overconfidence = confident but wrong — our key metric.
    """
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    escalated = sum(1 for r in results if r["escalated"])

    overconfident = sum(
        1 for r in results
        if not r["escalated"] and not r["is_correct"]
    )

    return {
        "total": total,
        "accuracy": correct / total,
        "escalation_rate": escalated / total,
        "overconfidence_rate": overconfident / total,
        "small_model_used": (total - escalated) / total,
    }


def save_results(results: list[dict], path: str):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")