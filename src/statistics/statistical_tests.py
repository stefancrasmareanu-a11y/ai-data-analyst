import streamlit as st

from .correlations import pearson_test, spearman_test
from .two_groups import welch_t_test, mann_whitney_test
from .multiple_groups import anova_test, kruskal_test, tukey_test
from .categorical_tests import chi_square_test, fisher_exact_test
import pandas as pd


def run_statistical_test(df, variable_1, variable_2, test_name):
    """
    Run a statistical test based on the selected test name.
    """

    if test_name == "Pearson correlation":
        return pearson_test(
            df[variable_1],
            df[variable_2]
        )

    elif test_name == "Spearman correlation":
        return spearman_test(
            df[variable_1],
            df[variable_2]
        )

    elif test_name == "Welch's t-test":
        if pd.api.types.is_numeric_dtype(df[variable_1]):
        # Pass categorical variable first, then numeric variable
            return welch_t_test(df[variable_2], df[variable_1])
        else:
        # Pass categorical variable first, then numeric variable
            return welch_t_test(df[variable_1], df[variable_2])

    elif test_name == "Mann-Whitney U test":
        if pd.api.types.is_numeric_dtype(df[variable_2]):
        # Pass categorical variable first, then numeric variable
            return mann_whitney_test(
                df[variable_1],
                df[variable_2]
                )
        else:
            return mann_whitney_test(
                df[variable_2],
                df[variable_1]
                )

    elif test_name == "One-way ANOVA":
        return anova_test(
            df[variable_1],
            df[variable_2]
        )

    elif test_name == "Kruskal-Wallis test":
        return kruskal_test(
            df[variable_1],
            df[variable_2]
        )

    elif test_name == "Tukey HSD":
        return tukey_test(
            df[variable_1],
            df[variable_2]
        )

    elif test_name == "Chi-square test":
        return chi_square_test(
            df[variable_1],
            df[variable_2]
        )

    elif test_name == "Fisher's exact test":
        return fisher_exact_test(
            df[variable_1],
            df[variable_2]
        )

    else:
        raise ValueError(f"Unknown statistical test: {test_name}")