# feedback_handler_agent.py
# Deals with both positive and negative customer feedback.
# Positive => thank-you response
# Negative => extracts the issue, creates a ticket, sends empathy response

import re
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)
from database import generate_ticket_id, create_ticket

POSITIVE_PROMPT = """You are a professional banking customer support agent.
The customer has sent a positive message expressing satisfaction or gratitude.

Generate a warm, personalized thank-you response. Keep it brief (1-2 sentences).
If you can identify the specific issue they mentioned, reference it in your response.

Customer message: {message}

Respond ONLY with the thank-you message. Do not add any prefixes like "Response:" etc."""

NEGATIVE_PROMPT = """You are a professional banking customer support agent.
The customer has sent a message expressing a complaint or reporting an unresolved issue.

Extract a brief description of the issue from their message (one short phrase).
Also, try to identify the customer's name if mentioned; otherwise use "Valued Customer".

Respond in this exact JSON format (no markdown, no code blocks):
{{"issue_description": "brief description of issue", "customer_name": "name or Valued Customer"}}

Customer message: {message}"""

EMPATHY_PROMPT = """You are a professional banking customer support agent.
Generate an empathetic apology response for a customer who reported this issue: {issue_description}

The ticket number assigned is #{ticket_id}.

Requirements:
- Be empathetic and professional
- Mention the ticket number
- Assure follow-up
- Keep it to 1-2 sentences

Respond ONLY with the empathy message."""


class FeedbackHandlerAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            max_completion_tokens=4096,
        )

    def handle_positive(self, message: str) -> dict:
        """Generates a thank-you response for happy customers."""
        prompt = ChatPromptTemplate.from_template(POSITIVE_PROMPT)
        chain = prompt | self.llm
        rendered = prompt.format(message=message)
        result = chain.invoke({"message": message})
        text = result.content.strip()

        return {
            "classification": "positive_feedback",
            "response": text,
            "action": "Acknowledged positive feedback",
            "prompt_traces": [
                {"step": "positive_response", "prompt": rendered, "completion": text}
            ],
        }

    def handle_negative(self, message: str) -> dict:
        """Extracts the issue, creates a new support ticket, and responds empathetically."""
        traces = []

        # step 1 - figure out what the issue is
        extract_prompt = ChatPromptTemplate.from_template(NEGATIVE_PROMPT)
        chain = extract_prompt | self.llm
        rendered = extract_prompt.format(message=message)
        extraction = chain.invoke({"message": message})
        traces.append({
            "step": "extract_issue",
            "prompt": rendered,
            "completion": extraction.content.strip(),
        })

        try:
            parsed = json.loads(extraction.content.strip())
            issue = parsed.get("issue_description", message)
            name = parsed.get("customer_name", "Valued Customer")
        except (json.JSONDecodeError, KeyError):
            issue = message
            name = "Valued Customer"

        # step 2 - create the ticket
        tid = generate_ticket_id()
        create_ticket(tid, name, issue)

        # step 3 - generate the empathy response with the new ticket number
        emp_prompt = ChatPromptTemplate.from_template(EMPATHY_PROMPT)
        emp_chain = emp_prompt | self.llm
        rendered_emp = emp_prompt.format(issue_description=issue, ticket_id=tid)
        emp_result = emp_chain.invoke({"issue_description": issue, "ticket_id": tid})
        traces.append({
            "step": "empathy_response",
            "prompt": rendered_emp,
            "completion": emp_result.content.strip(),
        })

        return {
            "classification": "negative_feedback",
            "response": emp_result.content.strip(),
            "ticket_id": tid,
            "customer_name": name,
            "issue_description": issue,
            "action": f"Created ticket #{tid} for: {issue}",
            "prompt_traces": traces,
        }
