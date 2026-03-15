# query_handler_agent.py
# Handles ticket status lookups. Tries regex first for the ticket number,
# falls back to LLM extraction if regex doesn't find anything.

import re
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)
from database import get_ticket_status

EXTRACT_TICKET_PROMPT = """You are a banking support assistant.
Extract the ticket number from the following customer message.

The ticket number is a 6-digit number. Return ONLY the 6-digit number, nothing else.
If no ticket number is found, respond with "NONE".

Customer message: {message}"""


class QueryHandlerAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            max_completion_tokens=4096,
        )

    def _extract_ticket_number(self, message):
        """Quick extraction - just returns the number (or None)."""
        num, _ = self._extract_with_trace(message)
        return num

    def _extract_with_trace(self, message):
        """Tries regex first, then LLM. Returns (ticket_number, traces)."""
        traces = []

        # regex is faster and doesn't cost an API call
        match = re.search(r'\b(\d{6})\b', message)
        if match:
            traces.append({"step": "extract_ticket", "prompt": "(regex)", "completion": match.group(1)})
            return match.group(1), traces

        # no regex match - ask the LLM
        prompt = ChatPromptTemplate.from_template(EXTRACT_TICKET_PROMPT)
        chain = prompt | self.llm
        rendered = prompt.format(message=message)
        result = chain.invoke({"message": message})
        extracted = result.content.strip()
        traces.append({"step": "extract_ticket", "prompt": rendered, "completion": extracted})

        if extracted != "NONE" and re.match(r'^\d{6}$', extracted):
            return extracted, traces
        return None, traces

    def handle_query(self, message: str) -> dict:
        """Looks up a ticket and returns the status to the user."""
        ticket_num, traces = self._extract_with_trace(message)

        if not ticket_num:
            return {
                "classification": "query",
                "response": ("I couldn't find a ticket number in your message. "
                             "Could you please provide your 6-digit ticket number?"),
                "ticket_id": None,
                "action": "Failed to extract ticket number",
                "prompt_traces": traces,
            }

        ticket = get_ticket_status(ticket_num)

        if not ticket:
            return {
                "classification": "query",
                "response": (f"No ticket found with number #{ticket_num}. "
                             "Please verify the ticket number and try again."),
                "ticket_id": ticket_num,
                "action": f"Ticket #{ticket_num} not found in database",
                "prompt_traces": traces,
            }

        return {
            "classification": "query",
            "response": f"Your ticket #{ticket_num} is currently marked as: {ticket['status']}.",
            "ticket_id": ticket_num,
            "ticket_details": ticket,
            "action": f"Retrieved status for ticket #{ticket_num}: {ticket['status']}",
            "prompt_traces": traces,
        }
