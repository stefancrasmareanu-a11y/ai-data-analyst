import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_summary(df, uploaded_file):
    """
    Displays a summary of the uploaded dataset.
    """

    numeric_cols = len(df.select_dtypes(include="number").columns)
    categorical_cols = len(df.select_dtypes(exclude="number").columns)

    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

#    st.subheader("📋 Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📄 File", uploaded_file.name)
        st.metric("📏 Rows", f"{len(df):,}")
        st.metric("📊 Columns", df.shape[1])

    with col2:
        st.metric("💾 Memory", f"{memory_mb:.2f} MB")
        st.metric("🔢 Numeric Columns", numeric_cols)
        st.metric("🔤 Categorical Columns", categorical_cols)
        
def detect_column_types(df):
    """
    Returns a dictionary
    mapping each column name to its inferred data type.
    """

    column_types = {}

    for column in df.columns:
        column_types[column] = str(df[column].dtype)

    return column_types