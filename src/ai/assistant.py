import json
from openai import OpenAI
import os
import streamlit as st

from .prompts import SYSTEM_PROMPT
from .tools import *
from .client import client



tools_list = [
    DATASET_METADATA_TOOL,
    GET_COLUMN_STATISTICS_TOOL,
    GET_COLUMN_VALUES_TOOL,
    CALCULATE_CORRELATION_TOOL,
    GET_CATEGORICAL_SUMMARY_TOOL,
    RECOMMEND_STATISTICAL_TEST_TOOL,
    RUN_STATISTICAL_ANALYSIS_TOOL
]

#client = OpenAI(
#    api_key=os.getenv("OPENAI_API_KEY")
#)

TOOL_FUNCTIONS = {
    "get_dataset_metadata": get_dataset_metadata,
    "get_column_statistics": get_column_statistics,
    "get_column_values": get_column_values,
    "calculate_correlation": calculate_correlation,
    "get_categorical_summary": get_categorical_summary,
    "recommend_statistical_test": recommend_statistical_test,
    "run_statistical_analysis": run_statistical_analysis,
}

def ask_llm(user_message, df):
    # 1. Send the initial request
    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=SYSTEM_PROMPT,
        input=user_message,
        tools=tools_list
    )

    # 2. Loop until no new function call tools are invoked
    while True:
        tool_outputs = []

        for item in response.output:
            if item.type == "function_call":
                function = TOOL_FUNCTIONS[item.name]
                tool_args = json.loads(item.arguments) if item.arguments else {}

                # Execute function safely
                if item.name == "get_dataset_metadata":
                    result = function(df)
                else:
                    result = function(df, **tool_args)

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                })

        # If tools were invoked, send their outputs back to the model
        if tool_outputs:
            current_id = response.id  # Capture the latest response ID
            response = client.responses.create(
                model="gpt-5.6-luna",
                instructions=SYSTEM_PROMPT,
                previous_response_id=current_id,  # Link to the preceding call
                input=tool_outputs,
                tools=tools_list
            )
        else:
            # Model produced its final response text
            break

    return response.output_text