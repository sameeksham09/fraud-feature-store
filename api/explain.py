"""
Explainability module for fraud scoring.

Uses SHAP TreeExplainer when `shap` is installed, otherwise falls back to
XGBoost global feature importances weighted by the current feature values.
Also generates human-readable reasons from domain thresholds.

This is what makes the project stand out: every /score response tells
*why* a transaction was flagged, not just *how much*.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from features import HIGH_AMOUNT_THRESHOLD, SUSPICIOUS_AMOUNT_THRESHOLD

# ── Optional SHAP ─────────────────────────────────────────────────────────
try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Module-level explainer (initialized once at startup via init_explainer)
_explainer = None
_explainer_feature_names: Optional[List[str]] = None


def init_explainer(model, feature_names: List[str]) -> bool:
    """
    Initialize the SHAP TreeExplainer for the given XGBoost model.
    Call once during API startup (api/main.py lifespan). Returns True if
    SHAP is available and the explainer was created, False otherwise.
    """
    global _explainer, _explainer_feature_names
    if not SHAP_AVAILABLE:
        print("ℹ️  shap not installed — falling back to importance-based explanations")
        print("   Install with: pip install shap>=0.44.0")
        return False
    try:
        _explainer = shap.TreeExplainer(model)
        _explainer_feature_names = feature_names
        print(f"✅ SHAP TreeExplainer initialized ({len(feature_names)} features)")
        return True
    except Exception as exc:
        print(f"⚠️  SHAP explainer init failed, using fallback: {exc}")
        _explainer = None
        return False


def _shap_values(model, X: np.ndarray) -> Optional[np.ndarray]:
    """Return per-feature SHAP values for a single row X (1, n_features)."""
    global _explainer
    if _explainer is None:
        return None
    try:
        vals = _explainer.shap_values(X)
        # XGBoost binary classifier: shap_values may be list [class0, class1]
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        # Newer shap returns Explanation object
        if hasattr(vals, "values"):
            vals = vals.values  # type: ignore
        vals = np.asarray(vals)
        if vals.ndim == 2:
            return vals[0]  # (1, n) → (n,)
        return vals
    except Exception as exc:
        print(f"⚠️  SHAP inference failed, using fallback: {exc}")
        return None


def _fallback_contributions(
    model, features: Dict[str, float], feature_order: List[str]
) -> Dict[str, float]:
    """
    Fast fallback when SHAP is unavailable.
    Weights global feature_importances_ by normalized feature activation.
    Normalization maps each feature to roughly [0, 1] based on expected
    ranges from the synthetic generator (models/train.py).
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_order):
        # uniform fallback if model has no importances
        importances = np.ones(len(feature_order)) / len(feature_order)
    else:
        s = float(np.sum(importances))
        if s > 0:
            importances = importances / s  # normalize to sum=1

    # Expected maxima for normalization (from generate_synthetic_data)
    maxima = {
        "txn_count_5min": 20.0,
        "total_amount_5min": 5000.0,
        "avg_amount_5min": 2500.0,
        "max_amount_5min": 5000.0,
        "is_foreign": 1.0,
        "is_high_amount": 1.0,
        "is_suspicious": 1.0,
    }

    contributions: Dict[str, float] = {}
    for idx, feat in enumerate(feature_order):
        val = float(features.get(feat, 0.0))
        norm = min(val / maxima.get(feat, 1.0), 1.0) if maxima.get(feat, 1) != 0 else 0.0
        # Center flags so 0 contributes negatively, 1 positively
        if feat in ("is_foreign", "is_high_amount", "is_suspicious"):
            norm = (val * 2) - 1  # 0→-1, 1→+1  then scaled
            # scale flag contribution down slightly so velocity features compete
            norm *= 0.6
        contributions[feat] = float(importances[idx] * norm)
    return contributions


def _human_reasons(
    features: Dict[str, float],
    fraud_prob: float,
    merchant_country: Optional[str] = None,
    amount: Optional[float] = None,
) -> List[str]:
    """Generate ordered human-readable reasons for the prediction."""
    reasons: List[str] = []

    is_foreign = bool(features.get("is_foreign", 0))
    is_high = bool(features.get("is_high_amount", 0))
    is_suspicious = bool(features.get("is_suspicious", 0))
    txn_count = float(features.get("txn_count_5min", 0))
    total_amt = float(features.get("total_amount_5min", 0))
    avg_amt = float(features.get("avg_amount_5min", 0))
    max_amt = float(features.get("max_amount_5min", 0))

    # Rule-based reasons — ordered by severity
    if is_suspicious:
        country = merchant_country or "foreign"
        reasons.append(
            f"Foreign transaction exceeds suspicious threshold "
            f"(${SUSPICIOUS_AMOUNT_THRESHOLD:.0f}) — country={country}"
        )
    elif is_foreign:
        country = merchant_country or "non-US"
        reasons.append(f"Foreign merchant country ({country})")

    if is_high:
        amt = amount if amount is not None else max_amt or avg_amt
        reasons.append(
            f"High-value transaction (${amt:.2f} > ${HIGH_AMOUNT_THRESHOLD:.0f} threshold)"
        )

    if txn_count > 10:
        reasons.append(f"High velocity: {txn_count:.0f} transactions in last 5 minutes")
    elif txn_count > 5:
        reasons.append(f"Elevated velocity: {txn_count:.0f} transactions in last 5 minutes")

    if max_amt > 1000:
        reasons.append(f"Peak amount in window is high (${max_amt:.2f})")

    if total_amt > 5000:
        reasons.append(f"Large total volume in window (${total_amt:.2f} in 5 min)")

    if not reasons:
        if fraud_prob > 0.4:
            reasons.append("Model flagged transaction — pattern matches historical fraud")
        else:
            reasons.append("No strong fraud indicators in this transaction")

    return reasons


def explain_prediction(
    model,
    features: Dict[str, float],
    feature_order: List[str],
    fraud_prob: float,
    merchant_country: Optional[str] = None,
    amount: Optional[float] = None,
) -> Dict:
    """
    Compute per-feature contributions and human reasons.

    Returns dict with:
      - shap_values: Dict[str, float]
      - top_contributors: List[{feature, feature_value, shap_value}]
      - reasons: List[str]
      - method: "shap" | "importance_fallback"
    """
    X = np.array([[float(features.get(f, 0.0)) for f in feature_order]], dtype=float)

    shap_vals = _shap_values(model, X)
    if shap_vals is not None:
        contributions = {feat: float(shap_vals[i]) for i, feat in enumerate(feature_order)}
        method = "shap"
    else:
        contributions = _fallback_contributions(model, features, feature_order)
        method = "importance_fallback"

    # Top 3 contributors by absolute SHAP/contribution
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_contributors = [
        {
            "feature": feat,
            "feature_value": float(features.get(feat, 0.0)),
            "contribution": round(float(val), 4),
            "direction": "increases risk" if val > 0 else "decreases risk",
        }
        for feat, val in ranked[:3]
    ]

    shap_values_rounded = {k: round(float(v), 4) for k, v in contributions.items()}
    reasons = _human_reasons(features, fraud_prob, merchant_country, amount)

    return {
        "shap_values": shap_values_rounded,
        "top_contributors": top_contributors,
        "reasons": reasons,
        "method": method,
    }
