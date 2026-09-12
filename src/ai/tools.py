import pandas as pd
# from scipy.stats import pearsonr, spearmanr
from src.statistics.recommendations import recommend_test
from src.statistics.statistical_tests import run_statistical_test
from scipy.stats import pearsonr, spearmanr

DATASET_METADATA_TOOL = {
    "type": "function",
    "name": "get_dataset_metadata",
    "description": (
        "Get basic information about the currently uploaded dataset, "
        "including the number of rows, number of columns, column names, "
        "and data types."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
}

GET_COLUMN_STATISTICS_TOOL = {
    "type": "function",
    "name": "get_column_statistics",
    "description": (
        "Calculate descriptive statistics for a column "
        "in the currently uploaded dataset."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": "The name of the column to analyze."
            }
        },
        "required": ["column"],
        "additionalProperties": False
    }
}

GET_COLUMN_VALUES_TOOL = {
    "type": "function",
    "name": "get_column_values",
    "description": (
        "Return the unique values and their frequencies for a column "
        "in the currently uploaded dataset. Use this tool when the user "
        "asks what values, categories, or possible values a column contains."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": (
                    "The name of the column whose unique values "
                    "should be retrieved."
                )
            }
        },
        "required": ["column"],
        "additionalProperties": False
    }
}

GET_CATEGORICAL_SUMMARY_TOOL = {
    "type": "function",
    "name": "get_categorical_summary",
    "description": (
       "Return frequency information for a categorical column. Use this tool when the user "
        "asks to summarize a given categorical column."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": (
                    "The name of the column whose unique values "
                    "should be retrieved."
                )
            }
        },
        "required": ["column"],
        "additionalProperties": False
    }
}

RECOMMEND_STATISTICAL_TEST_TOOL = {
    "type": "function",
    "name": "recommend_statistical_test",
    "description": (
        "Recommend appropriate statistical tests for two variables "
        "in the currently uploaded dataset. Use this tool when the "
        "user asks which statistical test should be used to determine "
        "a relationship, difference, or association between two variables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "variable_1": {
                "type": "string",
                "description": (
                    "The name of the first variable."
                )
            },
            "variable_2": {
                "type": "string",
                "description": (
                    "The name of the second variable."
                )
            }
        },
        "required": [
            "variable_1",
            "variable_2"
        ],
        "additionalProperties": False
    }
}

RUN_STATISTICAL_ANALYSIS_TOOL = {
    "type": "function",
    "name": "run_statistical_analysis",
    "description": (
        "Run a specific statistical test on two variables in the "
        "currently uploaded dataset. Use this tool when the user "
        "explicitly asks to run a statistical test or when a specific "
        "test has already been selected or recommended."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "variable_1": {
                "type": "string",
                "description": (
                    "The name of the first variable."
                )
            },
            "variable_2": {
                "type": "string",
                "description": (
                    "The name of the second variable."
                )
            },
            "test_name": {
                "type": "string",
                "enum": [
                    "Pearson correlation",
                    "Spearman correlation",
                    "Welch's t-test",
                    "Mann-Whitney U test",
                    "One-way ANOVA",
                    "Kruskal-Wallis test",
                    "Tukey HSD",
                    "Chi-square test",
                    "Fisher's exact test"
                ],
                "description": (
                    "The exact statistical test to run."
                )
            }
        },
        "required": [
            "variable_1",
            "variable_2",
            "test_name"
        ],
        "additionalProperties": False
    }
}

CALCULATE_CORRELATION_TOOL = {
    "type": "function",
    "name": "calculate_correlation",
    "description": (
        "Calculate the Pearson or Spearman correlation "
        "between two numeric columns."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column_1": {
                "type": "string",
                "description": "First numeric column."
            },
            "column_2": {
                "type": "string",
                "description": "Second numeric column."
            },
            "method": {
                "type": "string",
                "enum": ["pearson", "spearman"],
                "description": "Correlation method to use."
            }
        },
        "required": [
            "column_1",
            "column_2",
            "method"
        ],
        "additionalProperties": False
    }
}

def get_dataset_metadata(df):
    metadata = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {
            column: str(df[column].dtype)
            for column in df.columns
        }
    }
    
    return metadata

def get_column_statistics(df, column):
    """
    Return descriptive statistics for a column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the dataset."
        )

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):

        statistics = {
            "column": column,
            "data_type": "numeric",
            "count": int(series.count()),
            "missing_values": int(series.isna().sum()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "standard_deviation": float(series.std()),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "25th_percentile": float(series.quantile(0.25)),
            "75th_percentile": float(series.quantile(0.75))
        }

    else:

        value_counts = series.value_counts(dropna=False)

        statistics = {
            "column": column,
            "data_type": "categorical",
            "count": int(series.count()),
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique()),
            "most_common_value": str(value_counts.index[0])
            if len(value_counts) > 0 else None,
            "most_common_count": int(value_counts.iloc[0])
            if len(value_counts) > 0 else 0
        }

    return statistics

def get_column_values(df, column):
    """
    Return unique values and their frequencies for a column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the dataset."
        )

    series = df[column]

    value_counts = series.value_counts(dropna=False)

    values = []

    for value, count in value_counts.items():

        if pd.isna(value):
            value = "Missing"
        else:
            value = str(value)

        values.append({
            "value": value,
            "count": int(count)
        })

    return {
        "column": column,
        "unique_values": int(series.nunique()),
        "values": values
    }

def calculate_correlation(df, column_1, column_2, method="pearson"):
    """
    Calculate Pearson or Spearman correlation between two numeric columns.
    """

    if column_1 not in df.columns:
        raise ValueError(
            f"Column '{column_1}' does not exist."
        )

    if column_2 not in df.columns:
        raise ValueError(
            f"Column '{column_2}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(df[column_1]):
        raise ValueError(
            f"Column '{column_1}' must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(df[column_2]):
        raise ValueError(
            f"Column '{column_2}' must be numeric."
        )

    data = df[[column_1, column_2]].dropna()

    if len(data) < 3:
        raise ValueError(
            "At least 3 valid observations are required."
        )

    if method.lower() == "pearson":

        statistic, p_value = pearsonr(
            data[column_1],
            data[column_2]
        )

    elif method.lower() == "spearman":

        statistic, p_value = spearmanr(
            data[column_1],
            data[column_2]
        )

    else:

        raise ValueError(
            "Method must be 'pearson' or 'spearman'."
        )

    return {
        "column_1": column_1,
        "column_2": column_2,
        "method": method,
        "correlation": float(statistic),
        "p_value": float(p_value),
        "sample_size": len(data)
    }

def get_categorical_summary(df, column):
    """
    Return frequency information for a categorical column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is numeric, not categorical."
        )

    counts = df[column].value_counts(dropna=False)

    result = []

    for value, count in counts.items():

        if pd.isna(value):
            value = "Missing"

        result.append({
            "category": str(value),
            "count": int(count),
            "percentage": float(
                count / len(df) * 100
            )
        })

    return {
        "column": column,
        "number_of_categories": int(
            df[column].nunique()
        ),
        "categories": result
    }

def recommend_statistical_test(df, variable_1, variable_2):
    """
    Recommend appropriate statistical tests for two variables.
    """

    recommendation = recommend_test(
        df,
        variable_1,
        variable_2
    )

    return recommendation

def run_statistical_analysis(
    df,
    variable_1,
    variable_2,
    test_name
):
    """
    Run a selected statistical test.
    """

    result = run_statistical_test(
        df,
        variable_1,
        variable_2,
        test_name
    )

    return result