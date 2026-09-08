from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from .utils import clean_categorical_numeric, get_groups


def anova_test(category, numeric):
    """
    Perform one-way ANOVA.

    Requires at least three groups.
    """
    category, numeric = clean_categorical_numeric(category, numeric)

    groups = get_groups(category, numeric)

    if len(groups) < 3:
        raise ValueError(
            "ANOVA requires at least three groups."
        )

    group_values = list(groups.values())

    if any(len(group) < 2 for group in group_values):
        raise ValueError(
            "Each group must contain at least 2 observations."
        )

    statistic, p_value = f_oneway(*group_values)

    return {
        "test": "One-way ANOVA",
        "statistic": statistic,
        "p_value": p_value,
        "number_of_groups": len(groups),
        "sample_size": len(category)
    }


def kruskal_test(category, numeric):
    """
    Perform Kruskal-Wallis H test.

    Requires at least three groups.
    """
    category, numeric = clean_categorical_numeric(category, numeric)

    groups = get_groups(category, numeric)

    if len(groups) < 3:
        raise ValueError(
            "Kruskal-Wallis test requires at least three groups."
        )

    group_values = list(groups.values())

    if any(len(group) < 2 for group in group_values):
        raise ValueError(
            "Each group must contain at least 2 observations."
        )

    statistic, p_value = kruskal(*group_values)

    return {
        "test": "Kruskal-Wallis test",
        "statistic": statistic,
        "p_value": p_value,
        "number_of_groups": len(groups),
        "sample_size": len(category)
    }


def tukey_test(category, numeric, alpha=0.05):
    """
    Perform Tukey HSD post-hoc test.

    Used after a significant ANOVA result.
    """
    category, numeric = clean_categorical_numeric(category, numeric)

    if category.nunique() < 3:
        raise ValueError(
            "Tukey HSD requires at least three groups."
        )

    result = pairwise_tukeyhsd(
        endog=numeric,
        groups=category,
        alpha=alpha
    )

    return {
        "test": "Tukey HSD",
        "result": result
    }