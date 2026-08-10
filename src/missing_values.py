import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_blanks(df):
    return sum(
        pd.isna(value)
        or (isinstance(value, str) and value.strip() == "")
        for row in df.index
        for col in df.columns
        for value in [df.at[row, col]]
    )

#def display_blanks_summary(df):
#    for column in df.columns:
#       nans = count_blanks(df, column)[0]
#       empty_spaces = count_blanks(df, column)[1]
#        st.write("Column " + column + " has " + str(nans+empty_spaces) +
#                " blank values ( " + str(nans )+ " NaNs, " + str(empty_spaces) + " empty cells)")
        
def find_blanks(df, output_file="blanks.txt"):
    """
    Writes the row index and column name of every blank cell
    (NaN, empty string, or whitespace-only string) to a file.
    """

    with open(output_file, "w") as f:
        f.write("Blank values found:\n")
        f.write("-" * 40 + "\n")

        for row in df.index:
            for col in df.columns:
                value = df.at[row, col]

                if (
                    pd.isna(value)
                    or (isinstance(value, str) and value.strip() == "")
                ):
                    f.write(f"Row: {row}, Column: {col}\n")   
                    
def blank_report_dataframe(df):

    report = []

    for row in df.index:
        for col in df.columns:
            value = df.at[row, col]

            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                report.append({
                    "Row": row,
                    "Column": col
                })

    return pd.DataFrame(report)