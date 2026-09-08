from scipy.stats import pearsonr, spearmanr

from .utils import clean_numeric_pair


def pearson_test(x, y):
    """
    Perform Pearson correlation test.

    Returns:
        dict containing correlation coefficient and p-value.
    """
    x, y = clean_numeric_pair(x, y)

    if len(x) < 3:
        raise ValueError("At least 3 valid observations are required.")

    statistic, p_value = pearsonr(x, y)

    return {
        "test": "Pearson correlation",
        "statistic": statistic,
        "p_value": p_value,
        "sample_size": len(x)
    }


def spearman_test(x, y):
    """
    Perform Spearman rank correlation test.

    Returns:
        dict containing correlation coefficient and p-value.
    """
    x, y = clean_numeric_pair(x, y)

    if len(x) < 3:
        raise ValueError("At least 3 valid observations are required.")

    statistic, p_value = spearmanr(x, y)

    return {
        "test": "Spearman correlation",
        "statistic": statistic,
        "p_value": p_value,
        "sample_size": len(x)
    }