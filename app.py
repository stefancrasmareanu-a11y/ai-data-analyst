import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.summary import *
from src.preview import *
from src.missing_values import *
from src.duplicates import *
from src.column_analysis import *
from src.correlation import *
from src.scatter_plot import *
from src.automatic_eda import *

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")

st.write("Version v0.2.5")

st.write("Welcome to the AI Data Analyst project!")

st.write("Upload your dataset to begin.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv"
)

def trim_spaces(df):

    string_cols = df.select_dtypes(include="object").columns

    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())
    
    return df
    
def main():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully!")

        df = trim_spaces(df)
    
        with st.expander("📋 Dataset Summary"):
            
            dataset_summary(df, uploaded_file)
            
        with st.expander("📄 Preview dataset"):
    
            preview(df)
        
        with st.expander("🧹 Missing vales check"):
        
            if st.button("Check if your data constains blanks", key="run_blanks_report"):
    
                blanks_count = count_blanks(df)
    
                if blanks_count != 0:
                    st.warning(f"⚠️ Your data contains {blanks_count} blank cells!")

                    if st.checkbox("View blank cells report"):
                        report_df = blank_report_dataframe(df)

                        st.dataframe(report_df)

                else:
                    st.success("✅ Data does not contain blank cells!")
       
        with st.expander("🔁 Duplicate Detection"):
    
            check_duplicates(df)

        with st.expander("🔍 Column Explorer"):
            
            column_explorer(df)
        
        with st.expander("📈 Column Analysis"):
            
            column_analysis(df)
        
        with st.expander("🔗 Correlation Analysis"):
            
            correlation_analysis(df)
            
        with st.expander("📊 Data Relationship Explorer"):
            
            scatter_plot_explorer(df)
            
        with st.expander("🤖 Automatic EDA Summary"):
            
             if st.button("Run automatic EDA:", key="run_automatic_eda"):
                
                automatic_eda_summary(df)
            

main()