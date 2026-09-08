from scipy.stats import chi2_contingency, fisher_exact


def chi_square_test(category_1, category_2):
    """
    Perform a Chi-square test of independence.
    """
    data = {
        "category_1": category_1,
        "category_2": category_2
    }

    import pandas as pd

    df = pd.DataFrame(data).dropna()

    contingency_table = pd.crosstab(
        df["category_1"],
        df["category_2"]
    )

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError(
            "Chi-square test requires at least two categories "
            "in each variable."
        )

    statistic, p_value, degrees_of_freedom, expected = chi2_contingency(
        contingency_table
    )

    return {
        "test": "Chi-square test",
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
        "contingency_table": contingency_table,
        "expected_frequencies": expected,
        "sample_size": len(df)
    }


def fisher_exact_test(category_1, category_2):
    """
    Perform Fisher's exact test.

    Requires a 2x2 contingency table.
    """
    import pandas as pd

    df = pd.DataFrame({
        "category_1": category_1,
        "category_2": category_2
    }).dropna()

    contingency_table = pd.crosstab(
        df["category_1"],
        df["category_2"]
    )

    if contingency_table.shape != (2, 2):
        raise ValueError(
            "Fisher's exact test requires a 2x2 contingency table."
        )

    statistic, p_value = fisher_exact(contingency_table)

    return {
        "test": "Fisher's exact test",
        "statistic": statistic,
        "p_value": p_value,
        "contingency_table": contingency_table,
        "sample_size": len(df)
    }