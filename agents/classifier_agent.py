# classifier_agent.py
# Classifies user messages into one of 4 categories so the orchestrator
# knows which handler to pass the message to.

import re
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)

CLASSIFICATION_PROMPT = """You are a banking customer support classifier.

Analyze the following customer message and classify it into EXACTLY ONE of these categories:
1. positive_feedback  — The customer is expressing gratitude, satisfaction, or praise.
2. negative_feedback  — The customer is expressing a complaint, frustration, or reporting an unresolved issue.
3. query              — The customer is asking about the status of an existing support ticket (must reference a ticket number or ask about ticket status).
4. general            — The customer is sending a greeting, general question, or anything that does not fit the above categories.

Rules:
- If the message contains a ticket number and asks about its status, classify as "query".
- If the message expresses dissatisfaction or reports a problem, classify as "negative_feedback".
- If the message expresses thanks or satisfaction, classify as "positive_feedback".
- If the message is a greeting (e.g. "hello", "hi", "hey"), general conversation, or does not clearly fit the other categories, classify as "general".

Respond with ONLY the category name (one of: positive_feedback, negative_feedback, query, general).
Do NOT include any other text, explanation, or punctuation.

Customer message: {message}"""

VALID_CATEGORIES = {"positive_feedback", "negative_feedback", "query", "general"}


class ClassifierAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            max_completion_tokens=4096,
        )
        self.prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        self.chain = self.prompt | self.llm

    def classify(self, message: str) -> dict:
        """Run the message through the LLM classifier and return the result
        along with the prompt trace for debugging."""
        rendered_prompt = self.prompt.format(message=message)
        resp = self.chain.invoke({"message": message})
        raw = resp.content.strip()
        classification = raw.lower()

        # if the model returns something weird, try to figure it out from keywords
        if classification not in VALID_CATEGORIES:
            lower = message.lower()
            if any(w in lower for w in ["thank", "thanks", "great", "happy", "appreciate", "good"]):
                classification = "positive_feedback"
            elif any(w in lower for w in ["ticket", "status", "check", "update on"]):
                classification = "query"
            elif any(w in lower for w in ["hi", "hello", "hey", "good morning", "good evening"]):
                classification = "general"
            else:
                classification = "negative_feedback"

        # hard override: if the user clearly mentions a ticket number + keywords,
        # force it to "query" no matter what the LLM said
        has_ticket_num = bool(re.search(r'\b\d{6}\b', message))
        ticket_words = ["ticket", "status", "check", "update", "track"]
        if has_ticket_num and any(w in message.lower() for w in ticket_words):
            classification = "query"

        return {
            "classification": classification,
            "prompt_trace": rendered_prompt,
            "raw_completion": raw,
        }
