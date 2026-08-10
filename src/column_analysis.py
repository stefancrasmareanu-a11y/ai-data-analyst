import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def column_explorer(df):

    column = st.selectbox(
        "Select a column",
        df.columns, 
        key="column_explorer"
    )

    series = df[column]

    st.write(f"**Column:** {column}")

    # Missing values
    missing = (
        series.isna().sum()
        + (
            series.fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
    )

    # Numeric column
    if pd.api.types.is_numeric_dtype(series):

        st.write("**Type:** Numeric")
        st.write(f"**Missing Values:** {missing}")
        st.write(f"**Unique Values:** {series.nunique()}")

        st.write(f"**Minimum:** {series.min()}")
        st.write(f"**Maximum:** {series.max()}")
        st.write(f"**Mean:** {series.mean():.2f}")
        st.write(f"**Median:** {series.median():.2f}")
        st.write(f"**Standard Deviation:** {series.std():.2f}")

    # Datetime column
    elif pd.api.types.is_datetime64_any_dtype(series):

        st.write("**Type:** Datetime")
        st.write(f"**Missing Values:** {missing}")
        st.write(f"**Unique Values:** {series.nunique()}")

        st.write(f"**Earliest:** {series.min()}")
        st.write(f"**Latest:** {series.max()}")

    # Categorical/Text column
    else:

        st.write("**Type:** Categorical")
        st.write(f"**Missing Values:** {missing}")
        st.write(f"**Unique Values:** {series.nunique()}")

        value_counts = series.value_counts(dropna=True)

        if not value_counts.empty:
            st.write(f"**Top Value:** {value_counts.index[0]}")
            st.write(f"**Frequency:** {value_counts.iloc[0]}")
                   

def column_analysis(df):
    """
    Performs an analysis of the selected column.
    Numeric columns:
        - Histogram
        - Boxplot
        - Summary statistics
    Categorical columns:
        - Summary statistics
        - Frequency table
        - Bar chart
    """

    column = st.selectbox(
        "Select a column",
        df.columns,
        key="column_analysis"
    )

    series = df[column]

    # Remove missing values for statistics/plots
    clean_series = series.dropna()

    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    if pd.api.types.is_numeric_dtype(series):

        st.write("### 📈 Distribution Analysis")

        q1 = clean_series.quantile(0.25)
        q2 = clean_series.quantile(0.50)
        q3 = clean_series.quantile(0.75)

        stats = pd.DataFrame({
            "Statistic": [
                "Mean",
                "Median",
                "Standard Deviation",
                "Minimum",
                "25% Quartile",
                "50% Quartile",
                "75% Quartile",
                "Maximum"
            ],
            "Value": [
                round(clean_series.mean(), 2),
                round(clean_series.median(), 2),
                round(clean_series.std(), 2),
                round(clean_series.min(), 2),
                round(q1, 2),
                round(q2, 2),
                round(q3, 2),
                round(clean_series.max(), 2)
            ]
        })

        st.dataframe(
            stats,
            hide_index=True,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write("#### Histogram")

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.hist(clean_series, bins=30)

            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

        with col2:

            st.write("#### Boxplot")

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.boxplot(clean_series, vert=False)

            ax.set_xlabel(column)

            st.pyplot(fig)

    # =====================================================
    # CATEGORICAL COLUMNS
    # =====================================================

    else:

        st.write("### 📝 Categorical Analysis")

        unique_values = clean_series.nunique()

        value_counts = clean_series.value_counts()

        summary = pd.DataFrame({
            "Statistic": [
                "Unique Values",
                "Most Frequent Value",
                "Frequency"
            ],
            "Value": [
                unique_values,
                value_counts.index[0] if not value_counts.empty else "N/A",
                value_counts.iloc[0] if not value_counts.empty else 0
            ]
        })

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        st.write("#### Frequency Table")

        frequency_table = (
            value_counts
            .rename_axis(column)
            .reset_index(name="Count")
        )

        frequency_table["Percentage"] = (
            frequency_table["Count"]
            / frequency_table["Count"].sum()
            * 100
        ).round(2)

        st.dataframe(
            frequency_table,
            hide_index=True,
            use_container_width=True
        )

        st.write("#### Top Categories")

        top_categories = value_counts.head(10)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            top_categories.index.astype(str),
            top_categories.values
        )

        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"Top 10 values in '{column}'")

        plt.xticks(rotation=45, ha="right")

        st.pyplot(fig)