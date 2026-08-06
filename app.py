import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")

st.write("Version v0.1.0")

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

def detect_column_types(df):
    """
    Returns a dictionary
    mapping each column name to its inferred data type.
    """

    column_types = {}

    for column in df.columns:
        column_types[column] = str(df[column].dtype)

    return column_types

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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")
    
    df = trim_spaces(df)
    
    with st.expander("📋 Dataset Summary"):
        dataset_summary(df, uploaded_file)
    
    with st.expander("📄 Preview dataset"):
    
        preview(df)
        
    with st.expander("🧹 Missing vales check"):
    
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

    
    #types=detect_column_types(df)
    
    #st.write("**Data types: ** ")

    #for column, dtype in types.items():
        #st.write(f"{column}: **{dtype}**")