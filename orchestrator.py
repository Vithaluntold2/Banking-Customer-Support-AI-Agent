# orchestrator.py
# Central coordinator - takes a user message, classifies it, routes it
# to the right handler agent, and logs everything.

import json
import os
from datetime import datetime

from agents.classifier_agent import ClassifierAgent
from agents.feedback_handler_agent import FeedbackHandlerAgent
from agents.query_handler_agent import QueryHandlerAgent
from database import initialize_database
from config import LOG_DIR, LOG_FILE


class AgentOrchestrator:

    def __init__(self):
        initialize_database()
        self.classifier = ClassifierAgent()
        self.feedback_handler = FeedbackHandlerAgent()
        self.query_handler = QueryHandlerAgent()
        os.makedirs(LOG_DIR, exist_ok=True)

    def process_message(self, message: str) -> dict:
        """Main entry point. Classifies the message, routes to handler, logs it."""
        ts = datetime.now().isoformat()

        # classify first
        clf_result = self.classifier.classify(message)
        classification = clf_result["classification"]

        prompt_traces = [{
            "step": "classification",
            "prompt": clf_result["prompt_trace"],
            "completion": clf_result["raw_completion"],
        }]

        # route based on what the classifier says
        if classification == "positive_feedback":
            result = self.feedback_handler.handle_positive(message)
            path = "Classifier -> Positive Feedback Handler"

        elif classification == "negative_feedback":
            result = self.feedback_handler.handle_negative(message)
            path = "Classifier -> Negative Feedback Handler"

        elif classification == "query":
            result = self.query_handler.handle_query(message)
            path = "Classifier -> Query Handler"

        else:
            # general greetings / catch-all
            result = {
                "classification": "general",
                "response": (
                    "Hello! Welcome to our banking support. How can I help you today? You can:\n\n"
                    "- Share feedback about our services\n"
                    "- Report an issue you're facing\n"
                    "- Check the status of an existing ticket"
                ),
                "action": "Greeted customer",
            }
            path = "Classifier -> General Response"

        # merge traces from the handler into ours
        handler_traces = result.get("prompt_traces", [])
        prompt_traces.extend(handler_traces)
        result["prompt_traces"] = prompt_traces

        result["user_message"] = message
        result["agent_path"] = path
        result["timestamp"] = ts

        self._log(result)
        return result

    def _log(self, result):
        """Appends an interaction to the JSON log file."""
        entry = {
            "timestamp": result.get("timestamp"),
            "user_message": result.get("user_message"),
            "classification": result.get("classification"),
            "agent_path": result.get("agent_path"),
            "action": result.get("action"),
            "response": result.get("response"),
            "ticket_id": result.get("ticket_id"),
            "prompt_traces": result.get("prompt_traces", []),
            "user_feedback": None,
        }

        logs = self.get_logs()
        logs.append(entry)
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)

    def get_logs(self):
        """Reads the log file. Returns empty list if anything goes wrong."""
        if not os.path.exists(LOG_FILE):
            return []
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def clear_logs(self):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                json.dump([], f)

    def save_user_feedback(self, log_index, feedback):
        """Saves thumbs_up/thumbs_down on a specific log entry."""
        logs = self.get_logs()
        if 0 <= log_index < len(logs):
            logs[log_index]["user_feedback"] = feedback
            with open(LOG_FILE, "w") as f:
                json.dump(logs, f, indent=2)

    def get_feedback_stats(self):
        """Quick summary of user feedback across all interactions."""
        logs = self.get_logs()
        total = len(logs)
        up = sum(1 for l in logs if l.get("user_feedback") == "thumbs_up")
        down = sum(1 for l in logs if l.get("user_feedback") == "thumbs_down")
        rate = (up / (up + down) * 100) if (up + down) > 0 else 0
        return {
            "total": total,
            "thumbs_up": up,
            "thumbs_down": down,
            "no_feedback": total - up - down,
            "success_rate": rate,
        }
