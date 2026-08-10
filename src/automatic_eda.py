import pandas as pd
import streamlit as st


def automatic_eda_summary(df):
    """
    Automatically analyzes the dataset and provides
    a high-level EDA summary.
    """

    # --------------------------------------------------
    # Basic dataset information
    # --------------------------------------------------

    rows = df.shape[0]
    columns = df.shape[1]

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    missing_values = df.isna().sum().sum()

    # Also count empty strings / whitespace
    text_columns = df.select_dtypes(
        include=["object", "category"]
    )

    if not text_columns.empty:

        empty_strings = (
            text_columns
            .apply(
                lambda col:
                col.astype("string")
                .str.strip()
                .eq("")
                .sum()
            )
            .sum()
        )

        missing_values += empty_strings

    # --------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    # --------------------------------------------------
    # Constant columns
    # --------------------------------------------------

    constant_columns = [
        col
        for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    # --------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------

    st.write("### 📋 Dataset Overview")

    overview = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Numeric Columns",
            "Categorical Columns",
            "Missing Values",
            "Duplicate Rows",
            "Constant Columns"
        ],
        "Value": [
            f"{rows:,}",
            columns,
            len(numeric_columns),
            len(categorical_columns),
            f"{missing_values:,}",
            f"{duplicate_rows:,}",
            len(constant_columns)
        ]
    })

    st.dataframe(
        overview,
        hide_index=True,
        use_container_width=True
    )

    # --------------------------------------------------
    # Data Quality Assessment
    # --------------------------------------------------

    st.write("### 🧹 Data Quality")

    if missing_values == 0:

        st.success(
            "✅ No missing values detected."
        )

    else:

        st.warning(
            f"⚠️ {missing_values:,} "
            "missing or blank values detected."
        )

    if duplicate_rows == 0:

        st.success(
            "✅ No duplicate rows detected."
        )

    else:

        st.warning(
            f"⚠️ {duplicate_rows:,} "
            "duplicate rows detected."
        )

    if constant_columns:

        st.info(
            "ℹ️ Constant columns detected: "
            + ", ".join(constant_columns)
        )

    else:

        st.success(
            "✅ No constant columns detected."
        )

    # --------------------------------------------------
    # Numeric Analysis
    # --------------------------------------------------

    if numeric_columns:

        st.write("### 📈 Numeric Analysis")

        numeric_summary = []

        for col in numeric_columns:

            series = df[col].dropna()

            if series.empty:
                continue

            numeric_summary.append({
                "Column": col,
                "Mean": round(series.mean(), 2),
                "Median": round(series.median(), 2),
                "Std": round(series.std(), 2),
                "Min": round(series.min(), 2),
                "Max": round(series.max(), 2)
            })

        if numeric_summary:

            numeric_summary_df = pd.DataFrame(
                numeric_summary
            )

            st.dataframe(
                numeric_summary_df,
                hide_index=True,
                use_container_width=True
            )

    # --------------------------------------------------
    # Categorical Analysis
    # --------------------------------------------------

    if categorical_columns:

        st.write("### 📝 Categorical Analysis")

        categorical_summary = []

        for col in categorical_columns:

            series = df[col].dropna()

            if series.empty:
                continue

            value_counts = series.value_counts()

            categorical_summary.append({
                "Column": col,
                "Unique Values": series.nunique(),
                "Most Common": value_counts.index[0],
                "Frequency": value_counts.iloc[0]
            })

        if categorical_summary:

            categorical_summary_df = pd.DataFrame(
                categorical_summary
            )

            st.dataframe(
                categorical_summary_df,
                hide_index=True,
                use_container_width=True
            )

    # --------------------------------------------------
    # Numeric-Numeric Correlation Analysis
    # --------------------------------------------------

    strong_correlations = []

    if len(numeric_columns) >= 2:

        st.write("### 🔗 Strong Correlations")

        correlation_matrix = df[
            numeric_columns
        ].corr()

        for i in range(
            len(correlation_matrix.columns)
        ):

            for j in range(
                i + 1,
                len(correlation_matrix.columns)
            ):

                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]

                correlation = correlation_matrix.iloc[i, j]

                if (
                    pd.notna(correlation)
                    and abs(correlation) >= 0.7
                ):

                    strong_correlations.append({
                        "Column 1": col1,
                        "Column 2": col2,
                        "Correlation": round(
                            correlation,
                            2
                        )
                    })

        if strong_correlations:

            strong_corr_df = pd.DataFrame(
                strong_correlations
            ).sort_values(
                "Correlation",
                key=lambda x: x.abs(),
                ascending=False
            )

            st.dataframe(
                strong_corr_df,
                hide_index=True,
                use_container_width=True
            )

        else:

            st.success(
                "✅ No strong correlations "
                "(|r| ≥ 0.70) detected."
            )

    # --------------------------------------------------
    # Categorical-Numeric Relationships
    # --------------------------------------------------

    categorical_numeric_relationships = []

    if (
        categorical_columns
        and numeric_columns
    ):

        st.write(
            "### 📊 Categorical–Numeric Relationships"
        )

        for cat_col in categorical_columns:

            # Skip high-cardinality categorical columns
            if df[cat_col].nunique(
                dropna=True
            ) > 30:
                continue

            for num_col in numeric_columns:

                temp_df = df[
                    [cat_col, num_col]
                ].dropna()

                if temp_df.empty:
                    continue

                # Need at least two categories
                if temp_df[cat_col].nunique() < 2:
                    continue

                grouped = (
                    temp_df
                    .groupby(cat_col)[num_col]
                )

                group_counts = grouped.count()
                group_means = grouped.mean()

                # Ignore groups with very few observations
                if (group_counts < 5).any():
                    continue

                overall_mean = temp_df[num_col].mean()

                # Between-group sum of squares
                between_ss = (
                    group_counts
                    * (
                        group_means
                        - overall_mean
                    ) ** 2
                ).sum()

                # Total sum of squares
                total_ss = (
                    (
                        temp_df[num_col]
                        - overall_mean
                    ) ** 2
                ).sum()

                if total_ss == 0:
                    continue

                # Eta squared
                eta_squared = (
                    between_ss / total_ss
                )

                mean_difference = (
                    group_means.max()
                    - group_means.min()
                )

                highest_group = (
                    group_means.idxmax()
                )

                lowest_group = (
                    group_means.idxmin()
                )

                categorical_numeric_relationships.append({
                    "Category": cat_col,
                    "Numeric": num_col,
                    "η²": round(
                        eta_squared,
                        3
                    ),
                    "Highest Mean": (
                        f"{highest_group} "
                        f"({group_means.max():.2f})"
                    ),
                    "Lowest Mean": (
                        f"{lowest_group} "
                        f"({group_means.min():.2f})"
                    ),
                    "Mean Difference": round(
                        mean_difference,
                        2
                    )
                })

        # Display results
        if categorical_numeric_relationships:

            relationships_df = (
                pd.DataFrame(
                    categorical_numeric_relationships
                )
                .sort_values(
                    "η²",
                    ascending=False
                )
            )

            st.write(
                "**Strongest categorical–numeric relationships:**"
            )

            st.dataframe(
                relationships_df.head(10),
                hide_index=True,
                use_container_width=True
            )

        else:

            st.info(
                "No suitable categorical–numeric "
                "relationships were found."
            )

    # --------------------------------------------------
    # Automatic Conclusion
    # --------------------------------------------------

    st.write("### 💡 Automatic Summary")

    observations = []

    if missing_values > 0:

        observations.append(
            f"The dataset contains "
            f"{missing_values:,} "
            "missing or blank values."
        )

    if duplicate_rows > 0:

        observations.append(
            f"The dataset contains "
            f"{duplicate_rows:,} "
            "duplicate rows."
        )

    if constant_columns:

        observations.append(
            f"{len(constant_columns)} constant "
            "column(s) may not provide useful "
            "analytical information."
        )

    if strong_correlations:

        observations.append(
            f"{len(strong_correlations)} strong "
            "correlation(s) were detected "
            "between numeric variables."
        )

    # Strong categorical-numeric relationships
    strong_categorical_numeric = [
        relationship
        for relationship
        in categorical_numeric_relationships
        if relationship["η²"] >= 0.14
    ]

    if strong_categorical_numeric:

        observations.append(
            f"{len(strong_categorical_numeric)} strong "
            "categorical–numeric relationship(s) "
            "were detected."
        )

    if not observations:

        observations.append(
            "The dataset appears to have good "
            "basic data quality with no major "
            "issues detected by the automatic checks."
        )

    for observation in observations:

        st.write(
            f"• {observation}"
        )