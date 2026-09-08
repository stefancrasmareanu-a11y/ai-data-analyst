import pandas as pd


def get_variable_type(series):
    """
    Determine whether a pandas Series is numeric or categorical.
    """
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def clean_numeric_pair(x, y):
    """
    Remove missing values from two numeric variables.

    Returns:
        x_clean, y_clean
    """
    data = pd.DataFrame({
        "x": x,
        "y": y
    }).dropna()

    return data["x"], data["y"]


def clean_categorical_numeric(category, numeric):
    """
    Remove rows with missing categorical or numeric values.

    Returns:
        category_clean, numeric_clean
    """
    data = pd.DataFrame({
        "category": category,
        "numeric": numeric
    }).dropna()

    return data["category"], data["numeric"]


def check_group_count(category, min_groups=2):
    """
    Check the number of unique groups in a categorical variable.
    """
    return category.nunique() >= min_groups


def get_groups(category, numeric):
    """
    Split numeric values into groups according to a categorical variable.

    Returns:
        dictionary {group_name: numeric_values}
    """
    data = pd.DataFrame({
        "category": category,
        "numeric": numeric
    }).dropna()

    groups = {}

    for name, group in data.groupby("category"):
        groups[name] = group["numeric"]

    return groups