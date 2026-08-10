import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def scatter_plot_explorer(df):
    """
    Explore relationships between numeric and categorical columns.

    Numeric vs Numeric:
        Scatter plot

    Categorical vs Numeric:
        Boxplot
    """

#    st.subheader("📊 Data Relationship Explorer")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if len(numeric_columns) < 2 and len(categorical_columns) == 0:
        st.info("Not enough suitable columns for analysis.")
        return

    analysis_type = st.radio(
        "Select analysis type",
        [
            "Numeric vs Numeric",
            "Categorical vs Numeric"
        ],
        key="relationship_analysis_type"
    )

    # =====================================================
    # NUMERIC VS NUMERIC
    # =====================================================

    if analysis_type == "Numeric vs Numeric":

        if len(numeric_columns) < 2:
            st.info(
                "You need at least two numeric columns "
                "for a scatter plot."
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox(
                "Select X-axis",
                numeric_columns,
                key="scatter_x"
            )

        with col2:
            y_column = st.selectbox(
                "Select Y-axis",
                numeric_columns,
                index=1,
                key="scatter_y"
            )

        plot_df = df[[x_column, y_column]].dropna()

        if plot_df.empty:
            st.warning(
                "There are no valid data points to display."
            )
            return

        correlation = plot_df[x_column].corr(
            plot_df[y_column]
        )

        st.write(
            f"**Pearson correlation:** `{correlation:.3f}`"
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            plot_df[x_column],
            plot_df[y_column],
            alpha=0.5
        )

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(
            f"{y_column} vs {x_column}"
        )

        st.pyplot(fig)


    # =====================================================
    # CATEGORICAL VS NUMERIC
    # =====================================================

    else:

        if len(categorical_columns) == 0:
            st.info(
                "No categorical columns were found."
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            category_column = st.selectbox(
                "Select category",
                categorical_columns,
                key="category_column"
            )

        with col2:
            numeric_column = st.selectbox(
                "Select numeric variable",
                numeric_columns,
                key="category_numeric_column"
            )

        # Sort categories alphabetically
        categories = sorted(
            df[category_column]
            .dropna()
            .astype(str)
            .unique()
        )

        plot_df = df[
            [category_column, numeric_column]
        ].dropna()

        plot_df[category_column] = (
            plot_df[category_column]
            .astype(str)
        )

        if plot_df.empty:
            st.warning(
                "There are no valid data points to display."
            )
            return

        # Keep only categories that exist in the data
        categories = [
            category
            for category in categories
            if category in plot_df[category_column].values
        ]

        data = [
            plot_df.loc[
                plot_df[category_column] == category,
                numeric_column
            ]
            for category in categories
        ]

        # -------------------------------------------------
        # Boxplot
        # -------------------------------------------------

        st.write(
            f"### {numeric_column} by {category_column}"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.boxplot(
            data,
            tick_labels=categories
        )

        ax.set_xlabel(category_column)
        ax.set_ylabel(numeric_column)

        ax.set_title(
            f"{numeric_column} by {category_column}"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        st.pyplot(fig)

        # -------------------------------------------------
        # Summary statistics
        # -------------------------------------------------

        st.write("### 📊 Category Summary")

        summary = (
            plot_df
            .groupby(category_column)[numeric_column]
            .agg(
                Count="count",
                Mean="mean",
                Median="median",
                Minimum="min",
                Maximum="max"
            )
            .reindex(categories)
            .round(2)
            .reset_index()
        )

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )