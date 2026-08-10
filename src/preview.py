import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preview(df):
    st.subheader("Dataset preview ")
    
    st.write(df.head())

    #st.dataframe(df)
    
    st.subheader("Dataset Summary")

    types_df = (
        df.dtypes
        .astype(str)
        .reset_index()
    )

    types_df.columns = ["Column", "Type"]

    st.dataframe(types_df, use_container_width=True)