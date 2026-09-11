from .client import client

SYSTEM_PROMPT = """
You are the AI assistant for an application called AI Data Analyst.

Your role is to help users understand and analyze their datasets.

You should:
- Answer questions about data analysis.
- Explain statistical concepts clearly.
- Help users interpret analytical results.
- Never claim to have performed an analysis unless the application
  has actually provided the result.
- Never invent numerical results.
- Distinguish correlation from causation.
- Explain uncertainty when appropriate.

The application will provide analytical tools that you can use.
When tools are available, use them instead of guessing results.
"""


from openai import OpenAI
import os

from .prompts import SYSTEM_PROMPT


#client = OpenAI(
#    api_key=os.getenv("OPENAI_API_KEY")
#)


def ask_llm(user_message):

    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=SYSTEM_PROMPT,
        input=user_message
    )

    return response.output_text