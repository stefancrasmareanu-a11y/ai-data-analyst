import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")

st.write("Version v0.0.1")

st.write("Welcome to the AI Data Analyst project!")

st.write("Upload your dataset to begin.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv"
)

def detect_column_types(df):
    """
    Returns a dictionary
    mapping each column name to its inferred data type.
    """

    column_types = {}

    for column in df.columns:
        column_types[column] = str(df[column].dtype)

    return column_types

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")
    
    st.subheader("Dataset preview ")
    
    st.write(df.head())

    #st.dataframe(df)
    
    st.subheader("Dataset Summary")

    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

    st.write("**Data Types** ")
    st.dataframe(df.dtypes.rename("Type"))

    
    #types=detect_column_types(df)
    
    #st.write("**Data types: ** ")

    #for column, dtype in types.items():
        #st.write(f"{column}: **{dtype}**")