import pandas as pd
from typing import List, Optional
from app.schemas.dataset_schema import TestResult
from scipy.stats import skew, kurtosis


def run_data_quality_tests(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    sensitive_column: Optional[str] = None
) -> List[TestResult]:
    results: List[TestResult] = []

    # 1 Missing values
    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100
    max_missing_col = missing_percent.idxmax()
    status = "pass"
    suggestion = None
    if missing_percent.max() > 20:
        status = "warning"
        suggestion = f"Column '{max_missing_col}' has {missing_percent[max_missing_col]:.1f}% missing values."

    results.append(TestResult(
        test_name="missing_values",
        status=status,
        details=missing_percent.to_dict(),
        suggestion=suggestion
    ))

    # 2 Duplicate rows
    dup_count = df.duplicated().sum()

    results.append(TestResult(
        test_name="duplicate_rows",
        status="pass" if dup_count == 0 else "warning",
        details={"duplicate_count": int(dup_count)},
        suggestion="Remove duplicate rows" if dup_count > 0 else None
    ))

    # 3 Constant columns
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]

    results.append(TestResult(
        test_name="constant_columns",
        status="pass" if not const_cols else "warning",
        details={"constant_columns": const_cols},
        suggestion=f"Remove constant columns: {', '.join(const_cols)}" if const_cols else None
    ))

    # 4 Column cardinality
    high_card_cols = [col for col in df.columns if df[col].nunique() > 100]
    results.append(TestResult(
        test_name="high_cardinality_columns",
        status="pass" if not high_card_cols else "warning",
        details={"high_cardinality_columns": high_card_cols},
        suggestion=f"Consider encoding high cardinality columns: {', '.join(high_card_cols)}" if high_card_cols else None
    ))

    # 5 Numeric column stats
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        # describe() gives a dict keyed by column -> stat -> value
        stats = df[numeric_cols].describe().to_dict()
        results.append(TestResult(
            test_name="numeric_stats",
            status="pass",
            details=stats
        ))

        # 6 Skewness & Kurtosis
        skewness = {}
        kurtosis_vals = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                skewness[col] = None
                kurtosis_vals[col] = None
                continue
            try:
                skewness[col] = float(skew(series))
            except Exception:
                skewness[col] = None
            try:
                kurtosis_vals[col] = float(kurtosis(series))
            except Exception:
                kurtosis_vals[col] = None

        results.append(TestResult(
            test_name="skewness",
            status="pass",
            details=skewness
        ))
        results.append(TestResult(
            test_name="kurtosis",
            status="pass",
            details=kurtosis_vals
        ))

    # 7 Fairness / Disparity test (if target and sensitive columns are provided)
    if target_column and sensitive_column:
        if target_column in df.columns and sensitive_column in df.columns:

            # Compute group means
            group_means = df.groupby(sensitive_column)[
                target_column].mean().to_dict()

            if len(group_means) >= 2:
                min_group = min(group_means, key=group_means.get)
                max_group = max(group_means, key=group_means.get)
                disparity_ratio = group_means[min_group] / \
                    (group_means[max_group] + 1e-6)

                status = "pass" if disparity_ratio >= 0.8 else "warning"
                suggestion = None if status == "pass" else f"Target distribution differs across {sensitive_column} groups"

                results.append(TestResult(
                    test_name="fairness_disparity",
                    status=status,
                    details={
                        "group_means": group_means,
                        "disparity_ratio": disparity_ratio
                    },
                    suggestion=suggestion
                ))
            else:
                # Not enough groups to compare fairness
                results.append(TestResult(
                    test_name="fairness_disparity",
                    status="pass",
                    details={
                        "message": "Only one group present, cannot compute disparity."}
                ))
        else:
            results.append(TestResult(
                test_name="fairness_disparity",
                status="warning",
                details={"error": "Target or sensitive column not found"}
            ))

    return results
