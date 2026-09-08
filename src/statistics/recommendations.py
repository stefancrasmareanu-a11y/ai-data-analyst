from .utils import get_variable_type
import streamlit as st
#from .statistics.statistics_ui import display_test_recommendations
from .statistical_tests import run_statistical_test

def display_test_result(result):
    """
    Display the result of a statistical test.
    """

    st.subheader("Test Result")

    st.write(f"**Test:** {result['test']}")

    if "statistic" in result:
        st.write(
            f"**Test statistic:** {result['statistic']:.4f}"
        )

    if "p_value" in result:
        st.write(
            f"**p-value:** {result['p_value']:.4f}"
        )

    if "sample_size" in result:
        st.write(
            f"**Sample size:** {result['sample_size']}"
        )

    # Specific information for two-group tests
    if "group_1" in result:

        st.write(
            f"**Group 1:** {result['group_1']}"
        )

        st.write(
            f"**Group 2:** {result['group_2']}"
        )

        if "mean_1" in result:
            st.write(
                f"**Mean ({result['group_1']}):** "
                f"{result['mean_1']:.4f}"
            )

        if "mean_2" in result:
            st.write(
                f"**Mean ({result['group_2']}):** "
                f"{result['mean_2']:.4f}"
            )

        if "mean_difference" in result:
            st.write(
                f"**Mean difference:** "
                f"{result['mean_difference']:.4f}"
            )

    # Chi-square
    if "degrees_of_freedom" in result:
        st.write(
            f"**Degrees of freedom:** "
            f"{result['degrees_of_freedom']}"
        )

    if "contingency_table" in result:

        st.write("**Contingency table**")

        st.dataframe(
            result["contingency_table"]
        )


def display_test_recommendations(df):
    """
    Display statistical test recommendations and
    allow the user to run a selected test.
    """
    
    all_columns = df.columns.tolist()
    
    col1, col2 = st.columns(2)

    with col1:
        variable_1 = st.selectbox(
            "Select first variable",
            all_columns,
            key="first variable"
        )

    with col2:
        variable_2 = st.selectbox(
            "Select second variable",
            all_columns,
            index=1,
            key="second variable"
            )
        
    recommendation = recommend_test(df, variable_1, variable_2)

    st.write(recommendation["recommended_tests"])

    st.info(recommendation["reason"])  

    st.info(
        "### Statistical Test Recommendation\n\n"
        + recommendation["reason"]
    )

    st.write("**Recommended tests:**")

    tests = recommendation["recommended_tests"]
    

    for i, test_name in enumerate(tests):
        

        if st.button(
            f"Run {test_name}",
            key=f"run_test_{variable_1}_{variable_2}_{i}"
        ):
                
            result = run_statistical_test(
                df,
                variable_1,
                variable_2,
                test_name
                )

            display_test_result(result)


def recommend_test(df, variable_1, variable_2):
    """
    Recommend appropriate statistical tests based on
    variable types and number of groups.
    """
    x = df[variable_1]
    y = df[variable_2]

    type_1 = get_variable_type(x)
    type_2 = get_variable_type(y)

    # Numeric vs Numeric
    if type_1 == "numeric" and type_2 == "numeric":
        return {
            "relationship": "numeric-numeric",
            "recommended_tests": [
                "Pearson correlation",
                "Spearman correlation"
            ],
            "reason": (
                "Both variables are numeric. "
                "Pearson measures linear association, while "
                "Spearman measures monotonic association."
            )
        }

    # Categorical vs Numeric
    if type_1 == "categorical" and type_2 == "numeric":
        return _recommend_categorical_numeric(x,y)

    # Numeric vs Categorical
    if type_1 == "numeric" and type_2 == "categorical":
        return _recommend_categorical_numeric(y,x)

    # Categorical vs Categorical
    if type_1 == "categorical" and type_2 == "categorical":
        return _recommend_categorical_categorical(x, y)

    raise ValueError("Unable to determine variable types.")


def _recommend_categorical_numeric(category, numeric):
    """
    Recommend tests for categorical vs numeric variables.
    """

    number_of_groups = category.dropna().nunique()

    if number_of_groups == 2:
        return {
            "relationship": "categorical-numeric",
            "number_of_groups": 2,
            "recommended_tests": [
                "Welch's t-test",
                "Mann-Whitney U test"
            ],
            "reason": (
                "The categorical variable contains two groups. "
                "Welch's t-test compares group means, while "
                "Mann-Whitney U provides a non-parametric alternative."
            )
        }

    if number_of_groups >= 3:
        return {
            "relationship": "categorical-numeric",
            "number_of_groups": number_of_groups,
            "recommended_tests": [
                "One-way ANOVA",
                "Kruskal-Wallis test"
            ],
            "reason": (
                "The categorical variable contains three or more groups. "
                "ANOVA compares group means, while Kruskal-Wallis "
                "provides a non-parametric alternative."
            )
        }

    raise ValueError(
        "The categorical variable must contain at least two groups."
    )


def _recommend_categorical_categorical(category_1, category_2):
    """
    Recommend tests for two categorical variables.
    """

    groups_1 = category_1.dropna().nunique()
    groups_2 = category_2.dropna().nunique()

    if groups_1 < 2 or groups_2 < 2:
        raise ValueError(
            "Both categorical variables must contain at least two categories."
        )

    recommended_tests = ["Chi-square test"]

    if groups_1 == 2 and groups_2 == 2:
        recommended_tests.append("Fisher's exact test")

    return {
        "relationship": "categorical-categorical",
        "recommended_tests": recommended_tests,
        "reason": (
            "Both variables are categorical. "
            "Chi-square tests whether they are statistically independent. "
            "For a 2x2 table with small expected frequencies, "
            "Fisher's exact test can be used."
        )
    }