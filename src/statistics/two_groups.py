import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

from .utils import clean_categorical_numeric, get_groups


def welch_t_test(category, numeric):
    """
    Perform Welch's independent two-sample t-test.

    The categorical variable must contain exactly two groups.
    """
    category, numeric = clean_categorical_numeric(category, numeric)

    groups = get_groups(category, numeric)

    if len(groups) != 2:
        raise ValueError(
            "Welch's t-test requires exactly two groups."
        )

    group_names = list(groups.keys())

    group_1 = groups[group_names[0]]
    group_2 = groups[group_names[1]]

    if len(group_1) < 2 or len(group_2) < 2:
        raise ValueError(
            "Each group must contain at least 2 observations."
        )

    statistic, p_value = ttest_ind(
        group_1,
        group_2,
        equal_var=False
    )

    mean_1 = group_1.mean()
    mean_2 = group_2.mean()

    return {
        "test": "Welch's t-test",
        "group_1": group_names[0],
        "group_2": group_names[1],
        "mean_1": mean_1,
        "mean_2": mean_2,
        "mean_difference": mean_1 - mean_2,
        "statistic": statistic,
        "p_value": p_value,
        "n_1": len(group_1),
        "n_2": len(group_2)
    }


def mann_whitney_test(category, numeric):
    """
    Perform Mann-Whitney U test.

    The categorical variable must contain exactly two groups.
    """
    category, numeric = clean_categorical_numeric(category, numeric)

    groups = get_groups(category, numeric)

    if len(groups) != 2:
        raise ValueError(
            "Mann-Whitney U test requires exactly two groups."
        )

    group_names = list(groups.keys())

    group_1 = groups[group_names[0]]
    group_2 = groups[group_names[1]]

    if len(group_1) < 2 or len(group_2) < 2:
        raise ValueError(
            "Each group must contain at least 2 observations."
        )

    statistic, p_value = mannwhitneyu(
        group_1,
        group_2,
        alternative="two-sided"
    )

    return {
        "test": "Mann-Whitney U test",
        "group_1": group_names[0],
        "group_2": group_names[1],
        "statistic": statistic,
        "p_value": p_value,
        "n_1": len(group_1),
        "n_2": len(group_2)
    }