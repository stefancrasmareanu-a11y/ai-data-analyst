import json

from openai import OpenAI
import os

from .prompts import SYSTEM_PROMPT
from .tools import (
    DATASET_METADATA_TOOL,
    get_dataset_metadata
)
from .client import client



#client = OpenAI(
#    api_key=os.getenv("OPENAI_API_KEY")
#)


def ask_llm(user_message, df):

    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=SYSTEM_PROMPT,
        input=user_message,
        tools=[DATASET_METADATA_TOOL]
    )

    tool_outputs = []

    for item in response.output:

        if item.type == "function_call":

            if item.name == "get_dataset_metadata":

                result = get_dataset_metadata(df)

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                })

    if tool_outputs:

        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=SYSTEM_PROMPT,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=[DATASET_METADATA_TOOL]
        )

    return response.output_text