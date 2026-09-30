"""
Guardrails binary evaluation.
Sends each test input to the live /query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
"""


import time
import copy
import requests
import logfire

# FastAPI endpoint for sending queries to the agent
API_URL = "http://localhost:8000/query"


def _is_blocked(response_json: dict) -> bool:
    """
    Inspects the response payload to determine if the input/output guardrail intercepted the request.
    Checks the 'thought_process' list for any step indicating 'guardrails fired'.
    """
    # Extract thought process logs, defaulting to an empty list if absent
    tp = response_json.get("thought_process") or []
    # Return True if any thought step mentions that guardrails were triggered
    return any("guardrails fired" in step.lower() for step in tp)


def run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list:
    """
    Executes guardrails test cases against the live FastAPI endpoint.
    
    For each test case:
      1. Dispatches prompt to the live API endpoint.
      2. Detects whether guardrails blocked or allowed the request.
      3. Compares actual behavior with expected behavior.
      4. Assigns confusion matrix label: TP, TN, FP, or FN.

    Args:
        guardrails_samples: List of test cases containing id, input prompt, and expected_blocked.
        progress_callback: Optional callable (index, total, text) to emit progress updates to UI.

    Returns:
        list: Enriched test sample dictionaries containing 'actual_blocked' and 'result' (TP/TN/FP/FN).
    """
    # Make a deep copy to ensure caller's original test dataset is not modified in-place
    samples = copy.deepcopy(guardrails_samples)
    n = len(samples)

    with logfire.span("🛡️ Eval — Guardrails Tests", total=n):
        for i, sample in enumerate(samples):
            # Notify external UI or progress bar if callback is provided
            if progress_callback:
                progress_callback(i, n, sample["input"])

            # Trace individual test execution in Logfire
            with logfire.span(
                f"🛡️ Test {sample['id']}",
                input_text=sample["input"][:80],
                expected_blocked=sample["expected_blocked"],
            ):
                try:
                    # Submit query to the live FastAPI /query route
                    resp = requests.post(
                        API_URL,
                        json={"q": sample["input"], "thread_id": f"guardrail_eval_{i}"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    # Check whether guardrails blocked this response
                    blocked = _is_blocked(resp.json())

                except requests.exceptions.ConnectionError:
                    # Server is offline or unreachable
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    blocked = False

                except Exception as e:
                    # Catch-all for network timeouts or unexpected response structures
                    logfire.error(f"❌ Guardrails test error: {e}")
                    blocked = False

                expected = sample["expected_blocked"]
                sample["actual_blocked"] = blocked

                # Classify into confusion matrix categories:
                # - TP (True Positive): Malicious input correctly blocked
                # - FN (False Negative): Malicious input slipped through (security risk)
                # - TN (True Negative): Safe input correctly allowed
                # - FP (False Positive): Safe input incorrectly blocked (over-blocking)
                if expected and blocked:
                    sample["result"] = "TP"
                elif expected and not blocked:
                    sample["result"] = "FN"
                elif not expected and not blocked:
                    sample["result"] = "TN"
                else:
                    sample["result"] = "FP"

                # Log individual test outcome to Logfire
                logfire.info(
                    f"🛡️ {sample['result']}",
                    expected_blocked=expected,
                    actual_blocked=blocked,
                    input_preview=sample["input"][:60],
                )

            # Brief sleep between API requests to prevent request throttling
            time.sleep(2)

    return samples


def compute_guardrails_metrics(results: list) -> dict:
    """
    Computes binary evaluation metrics (Precision, Recall, Accuracy) from scored test results.

    Confusion Matrix definitions:
      - Precision = TP / (TP + FP) -> Out of all blocked requests, how many were truly harmful?
      - Recall    = TP / (TP + FN) -> Out of all harmful requests, how many were successfully caught?
      - Accuracy  = (TP + TN) / Total -> Overall percentage of correct decisions made.

    Args:
        results: List of scored sample dictionaries containing 'result' (TP, TN, FP, FN).

    Returns:
        dict: Summary statistics including counts and rounded metric scores.
    """
    # Count confusion matrix occurrences
    tp = sum(1 for r in results if r["result"] == "TP")
    tn = sum(1 for r in results if r["result"] == "TN")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")

    # Calculate metrics with zero-division guards
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "total": len(results),
        "correct": tp + tn,
    }

