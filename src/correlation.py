import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_analysis(df):
    """
    Calculates and displays the correlation matrix
    and a heatmap for numeric columns.
    """

#   st.subheader("🔗 Correlation Analysis")

    # Get numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Check if there are enough numeric columns
    if numeric_df.shape[1] < 2:
        st.info(
            "Correlation analysis requires at least two numeric columns."
        )
        return

    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()

    # Display correlation matrix
    st.write("### 📊 Correlation Matrix")

    st.dataframe(
        correlation_matrix.round(2),
        use_container_width=True
    )

    # Display heatmap
    st.write("### 🌡️ Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)     