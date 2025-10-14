import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, chi2
from typing import Dict, List, Optional, Any


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class BiasMetrics:
    """Lightweight utility for detecting bias in dataset columns."""

    @staticmethod
    def calculate_demographic_parity(
        df: pd.DataFrame,
        sensitive_col: str,
        target_col: str,
        positive_outcome: Any = 1
    ) -> Dict[str, Any]:
        """Calculate demographic parity - equal selection rate across groups (80% rule)."""
        results = {
            "metric": "demographic_parity",
            "groups": {},
            "disparate_impact": None,
            "bias_detected": False,
            "details": ""
        }

        try:
            group_stats = {}
            for group in df[sensitive_col].unique():
                group_data = df[df[sensitive_col] == group]
                total = len(group_data)
                positive = int(
                    (group_data[target_col] == positive_outcome).sum())
                selection_rate = positive / total if total > 0 else 0.0

                group_stats[str(group)] = {
                    "count": int(total),
                    "positive_outcomes": positive,
                    "selection_rate": float(selection_rate)
                }

            results["groups"] = group_stats

            rates = [g["selection_rate"]
                     for g in group_stats.values() if g["selection_rate"] > 0]
            if rates and len(rates) > 1:
                disparate_impact = min(rates) / max(rates)
                results["disparate_impact"] = float(disparate_impact)

                if disparate_impact < 0.8:
                    results["bias_detected"] = True
                    results["details"] = f"Disparate impact {disparate_impact:.3f} < 0.8"
                else:
                    results[
                        "details"] = f"Disparate impact {disparate_impact:.3f} >= 0.8 (acceptable)"

        except Exception as e:
            results["error"] = str(e)

        return convert_numpy_types(results)

    @staticmethod
    def calculate_statistical_parity(
        df: pd.DataFrame,
        sensitive_col: str,
        target_col: str
    ) -> Dict[str, Any]:
        """Chi-square test for independence between sensitive attribute and target."""
        results = {
            "metric": "statistical_parity",
            "chi2_statistic": None,
            "p_value": None,
            "bias_detected": False
        }

        try:
            contingency_table = pd.crosstab(df[sensitive_col], df[target_col])
            chi2_stat, p_value, dof, _ = chi2_contingency(contingency_table)

            results["chi2_statistic"] = float(chi2_stat)
            results["p_value"] = float(p_value)
            results["degrees_of_freedom"] = int(dof)

            if p_value < 0.05:
                results["bias_detected"] = True
                results["details"] = f"Significant association (p={p_value:.4f})"
            else:
                results["details"] = f"No significant association (p={p_value:.4f})"

        except Exception as e:
            results["error"] = str(e)

        return convert_numpy_types(results)

    @staticmethod
    def calculate_representation_bias(
        df: pd.DataFrame,
        sensitive_col: str
    ) -> Dict[str, Any]:
        """Check for severe representation imbalance across groups."""
        results = {
            "metric": "representation_bias",
            "actual_distribution": {},
            "bias_detected": False
        }

        try:
            value_counts = df[sensitive_col].value_counts()
            total = len(df)

            actual_dist = {}
            for value, count in value_counts.items():
                actual_dist[str(value)] = {
                    "count": int(count),
                    "proportion": float(count / total)
                }

            results["actual_distribution"] = actual_dist

            proportions = [v["proportion"] for v in actual_dist.values()]
            if proportions:
                max_prop = max(proportions)
                min_prop = min(proportions)

                if min_prop > 0:
                    imbalance_ratio = max_prop / min_prop
                    results["imbalance_ratio"] = float(imbalance_ratio)

                    if imbalance_ratio > 5:
                        results["bias_detected"] = True
                        results["details"] = f"Severe imbalance: ratio = {imbalance_ratio:.2f}"
                    else:
                        results["details"] = f"Acceptable imbalance: ratio = {imbalance_ratio:.2f}"

        except Exception as e:
            results["error"] = str(e)

        return convert_numpy_types(results)

    @staticmethod
    def calculate_distribution_similarity(
        df: pd.DataFrame,
        sensitive_col: str,
        feature_col: str
    ) -> Dict[str, Any]:
        """Compare feature distributions across sensitive groups using KS test."""
        results = {
            "metric": "distribution_similarity",
            "feature_column": feature_col,
            "comparisons": [],
            "max_ks_statistic": None,
            "bias_detected": False
        }

        try:
            if not pd.api.types.is_numeric_dtype(df[feature_col]):
                results["error"] = "Feature must be numeric for KS test"
                return convert_numpy_types(results)

            groups = df[sensitive_col].unique()
            max_ks = 0.0

            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    data1 = df[df[sensitive_col] ==
                               group1][feature_col].dropna()
                    data2 = df[df[sensitive_col] ==
                               group2][feature_col].dropna()

                    if len(data1) > 0 and len(data2) > 0:
                        ks_stat, p_value = ks_2samp(data1, data2)

                        results["comparisons"].append({
                            "group1": str(group1),
                            "group2": str(group2),
                            "ks_statistic": float(ks_stat),
                            "p_value": float(p_value),
                            "different_distributions": bool(p_value < 0.05)
                        })
                        max_ks = max(max_ks, ks_stat)

            results["max_ks_statistic"] = float(max_ks) if max_ks > 0 else None

            if max_ks > 0.3 or any(c["different_distributions"] for c in results["comparisons"]):
                results["bias_detected"] = True
                results["details"] = f"Distribution differences (max KS={max_ks:.3f})"

        except Exception as e:
            results["error"] = str(e)

        return convert_numpy_types(results)

    @staticmethod
    def run_comprehensive_bias_analysis(
        df: pd.DataFrame,
        sensitive_col: str,
        target_col: str,
        prediction_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run essential bias detection tests and return results."""
        results = {
            "sensitive_column": sensitive_col,
            "target_column": target_col,
            "total_samples": int(len(df)),
            "analyses": {},
            "overall_bias_detected": False
        }

        # 1. Demographic Parity
        results["analyses"]["demographic_parity"] = BiasMetrics.calculate_demographic_parity(
            df, sensitive_col, target_col
        )

        # 2. Statistical Parity (Chi-square)
        results["analyses"]["statistical_parity"] = BiasMetrics.calculate_statistical_parity(
            df, sensitive_col, target_col
        )

        # 3. Representation Bias
        results["analyses"]["representation_bias"] = BiasMetrics.calculate_representation_bias(
            df, sensitive_col
        )

        # 4. Distribution Similarity (only for first 2 numeric features to keep it light)
        if feature_cols:
            distribution_analyses = []
            for feature_col in feature_cols[:2]:  # Limit to 2 features
                if feature_col in df.columns and pd.api.types.is_numeric_dtype(df[feature_col]):
                    dist_analysis = BiasMetrics.calculate_distribution_similarity(
                        df, sensitive_col, feature_col
                    )
                    distribution_analyses.append(dist_analysis)

            if distribution_analyses:
                results["analyses"]["distribution_similarity"] = distribution_analyses

        # Check if any analysis detected bias
        for analysis_name, analysis_result in results["analyses"].items():
            if isinstance(analysis_result, list):
                if any(a.get("bias_detected", False) for a in analysis_result):
                    results["overall_bias_detected"] = True
                    break
            elif analysis_result.get("bias_detected", False):
                results["overall_bias_detected"] = True
                break

        return convert_numpy_types(results)
