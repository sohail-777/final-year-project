from transformers import pipeline
from typing import List, Tuple, Dict
import threading

# Lazy-load pattern to avoid expensive load during import in some contexts
_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                # model: tabularisai/multilingual-sentiment-analysis
                _pipeline = pipeline(
                    "text-classification", model="tabularisai/multilingual-sentiment-analysis")
    return _pipeline


def analyze_sentiment(texts: List[str]) -> Tuple[Dict[str, int], str]:
    """
    Analyze a list of texts and return (counts, overall_label)
    counts: dict with keys 'POSITIVE','NEGATIVE','NEUTRAL' and integer counts
    overall_label: the label with the highest count (string)
    """
    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")

    pipe = _get_pipeline()
    # Pipeline returns a list of dicts: {'label': 'LABEL', 'score': float}
    results = pipe(texts)

    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for r in results:
        label = str(r.get("label", "")).upper()
        if "NEG" in label:
            counts["NEGATIVE"] += 1
        elif "NEU" in label:
            counts["NEUTRAL"] += 1
        else:
            counts["POSITIVE"] += 1

    overall = max(counts, key=counts.get) if sum(
        counts.values()) > 0 else "NEUTRAL"
    return counts, overall
