import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_duplicates(df):
    column = st.selectbox(
    "Select a column",
    df.columns,
    key="check_duplicates"
    )

    if st.button("Check duplicates"):

        duplicates = df[df[column].duplicated(keep=False)]

        if duplicates.empty:
            st.success("No duplicates found.")
        else:
            st.warning(f"{len(duplicates)} rows contain duplicated values.")

            st.dataframe(duplicates)