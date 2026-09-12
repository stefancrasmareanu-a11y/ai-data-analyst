import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ai.assistant import ask_llm
from src.automatic_eda import automatic_eda_summary
from src.column_analysis import column_explorer, column_analysis
from src.correlation import correlation_analysis
from src.duplicates import check_duplicates
from src.missing_values import count_blanks, blank_report_dataframe
from src.preview import preview
from src.scatter_plot import scatter_plot_explorer
from src.statistics.recommendations import display_test_recommendations
from src.summary import dataset_summary


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data
def load_data(file):
    """Load CSV file and cache the result."""
    return pd.read_csv(file)


@st.cache_data
def trim_spaces(df):
    """Trim whitespace from object/string columns."""
    df = df.copy()

    string_cols = df.select_dtypes(include="object").columns

    if len(string_cols) > 0:
        df[string_cols] = df[string_cols].apply(
            lambda col: col.str.strip()
        )

    return df


def initialize_session_state():
    """Initialize Streamlit session state variables."""

    defaults = {
        "missing_values_run": False,
        "duplicates_run": False,
        "column_explorer_run": False,
        "column_analysis_run": False,
        "correlation_run": False,
        "scatter_plot_run": False,
        "automatic_eda_run": False,
        "statistics_run": False,
        "chat_messages": [],
        "uploaded_file_name": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state():
    """Reset analysis buttons when a new dataset is uploaded."""

    st.session_state.missing_values_run = False
    st.session_state.duplicates_run = False
    st.session_state.column_explorer_run = False
    st.session_state.column_analysis_run = False
    st.session_state.correlation_run = False
    st.session_state.scatter_plot_run = False
    st.session_state.automatic_eda_run = False
    st.session_state.statistics_run = False

    # Clear chat when a new dataset is loaded
    st.session_state.chat_messages = []


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main():

    initialize_session_state()

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    st.title("📊 AI Data Analyst")
    st.caption("Version v0.2.5")
    st.write("Welcome! Upload your dataset to begin your analysis.")

    # -----------------------------------------------------
    # File upload
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv"
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        return

    # Detect new dataset
    if (
        st.session_state.uploaded_file_name
        != uploaded_file.name
    ):
        st.session_state.uploaded_file_name = uploaded_file.name
        reset_analysis_state()

    # -----------------------------------------------------
    # Load and preprocess dataset
    # -----------------------------------------------------

    df = load_data(uploaded_file)
    df = trim_spaces(df)

    st.success(
        f"File uploaded successfully: **{uploaded_file.name}**"
    )

    # -----------------------------------------------------
    # Dataset Summary
    # -----------------------------------------------------

    with st.expander("📋 Dataset Summary"):

        if st.button(
            "Show dataset summary",
            key="run_dataset_summary"
        ):
            dataset_summary(df, uploaded_file)

    # -----------------------------------------------------
    # Dataset Preview
    # -----------------------------------------------------

    with st.expander("📄 Preview Dataset"):

        if st.button(
            "Preview dataset",
            key="run_preview"
        ):
            preview(df)

    # -----------------------------------------------------
    # Missing Values
    # -----------------------------------------------------

    with st.expander("🧹 Missing Values Check"):

        if st.button(
            "Check if your data contains blanks",
            key="run_blanks_report"
        ):
            st.session_state.missing_values_run = True

        if st.session_state.missing_values_run:

            blanks_count = count_blanks(df)

            if blanks_count != 0:

                st.warning(
                    f"⚠️ Your data contains "
                    f"{blanks_count} blank cells!"
                )

                if st.checkbox(
                    "View blank cells report",
                    key="show_blank_report"
                ):
                    report_df = blank_report_dataframe(df)
                    st.dataframe(
                        report_df,
                        use_container_width=True
                    )

            else:
                st.success(
                    "✅ Data does not contain blank cells!"
                )

    # -----------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------

    with st.expander("🔁 Duplicate Detection"):

        if st.button(
            "Check for duplicates",
            key="run_duplicates"
        ):
            st.session_state.duplicates_run = True

        if st.session_state.duplicates_run:
            check_duplicates(df)

    # -----------------------------------------------------
    # Column Explorer
    # -----------------------------------------------------

    with st.expander("🔍 Column Explorer"):

        if st.button(
            "Explore columns",
            key="run_column_explorer"
        ):
            st.session_state.column_explorer_run = True

        if st.session_state.column_explorer_run:
            column_explorer(df)

    # -----------------------------------------------------
    # Column Analysis
    # -----------------------------------------------------

    with st.expander("📈 Column Analysis"):

        if st.button(
            "Run column analysis",
            key="run_column_analysis"
        ):
            st.session_state.column_analysis_run = True

        if st.session_state.column_analysis_run:
            column_analysis(df)

    # -----------------------------------------------------
    # Correlation Analysis
    # -----------------------------------------------------

    with st.expander("🔗 Correlation Analysis"):

        if st.button(
            "Run correlation analysis",
            key="run_correlation"
        ):
            st.session_state.correlation_run = True

        if st.session_state.correlation_run:
            correlation_analysis(df)

    # -----------------------------------------------------
    # Data Relationship Explorer
    # -----------------------------------------------------

    with st.expander("📊 Data Relationship Explorer"):

        if st.button(
            "Run relationship analysis",
            key="run_scatter_plot"
        ):
            st.session_state.scatter_plot_run = True

        if st.session_state.scatter_plot_run:
            scatter_plot_explorer(df)

    # -----------------------------------------------------
    # Automatic EDA
    # -----------------------------------------------------

    with st.expander("🤖 Automatic EDA Summary"):

        if st.button(
            "Run automatic EDA",
            key="run_automatic_eda"
        ):
            st.session_state.automatic_eda_run = True

        if st.session_state.automatic_eda_run:
            automatic_eda_summary(df)

    # -----------------------------------------------------
    # Statistical Testing
    # -----------------------------------------------------

    with st.expander("🧮 Statistical Testing"):

        if st.toggle(
            "Show Statistical Tests",
            key="show_statistical_tests"
        ):
            display_test_recommendations(df)

    # -----------------------------------------------------
    # AI Assistant
    # -----------------------------------------------------

    st.divider()

    st.header("🤖 AI Assistant")

    st.caption(
        "Ask questions about your uploaded dataset."
    )

    # Display previous chat messages
    for message in st.session_state.chat_messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_message = st.chat_input(
        "Ask something about your dataset..."
    )

    if user_message:

        # Store user message
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.write(user_message)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    response = ask_llm(
                        user_message,
                        df
                    )

                    st.write(response)

                    # Store response
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )

                except Exception as e:

                    error_message = (
                        "⚠️ An error occurred while "
                        f"contacting the AI model:\n\n{e}"
                    )

                    st.error(error_message)


# ---------------------------------------------------------
# Application entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()