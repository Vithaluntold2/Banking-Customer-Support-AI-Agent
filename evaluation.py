# evaluation.py
# Runs all 15 test cases through the pipeline, checks classification accuracy,
# and uses the LLM to score empathy + clarity of the responses.

from orchestrator import AgentOrchestrator
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)

# each tuple is (message, expected category)
TEST_CASES = [
    # positive
    ("Thanks for sorting out my net banking login issue.", "positive_feedback"),
    ("Great job resolving my credit card dispute!", "positive_feedback"),
    ("I really appreciate the help with my account.", "positive_feedback"),
    ("Your team was very helpful, thank you!", "positive_feedback"),
    ("Thanks for the quick resolution on my loan query.", "positive_feedback"),
    # negative
    ("My debit card replacement still hasn't arrived.", "negative_feedback"),
    ("I've been waiting for weeks and no one has responded.", "negative_feedback"),
    ("The mobile banking app keeps crashing when I try to transfer money.", "negative_feedback"),
    ("I was charged twice for the same transaction and no one is helping.", "negative_feedback"),
    ("Your customer service is terrible, I want to escalate.", "negative_feedback"),
    # queries
    ("Could you check the status of ticket 650932?", "query"),
    ("What's the update on my ticket #784520?", "query"),
    ("I want to know the status of ticket 901234.", "query"),
    ("Can you tell me if ticket 543210 has been resolved?", "query"),
    ("Please check ticket number 112233 for me.", "query"),
]


SCORING_PROMPT = """You are evaluating the quality of a banking customer support AI response.

Customer message: {message}
Classification: {classification}
AI Response: {response}

Score the response on two dimensions (1 = very poor, 5 = excellent):

1. EMPATHY: How empathetic, warm, and human does the response feel?
   - 5: Highly empathetic, acknowledges feelings, personalized
   - 3: Adequate but somewhat robotic
   - 1: Cold, dismissive, or inappropriate tone

2. CLARITY: How clear, informative, and well-structured is the response?
   - 5: Crystal clear, actionable, well-formatted
   - 3: Understandable but could be better
   - 1: Confusing, vague, or missing key information

Respond in EXACTLY this format (numbers only, one per line):
EMPATHY: <number>
CLARITY: <number>"""


def score_response_quality(response, classification, message):
    """Has the LLM rate a response for empathy and clarity (both 1-5)."""
    try:
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        prompt = ChatPromptTemplate.from_template(SCORING_PROMPT)
        chain = prompt | llm
        result = chain.invoke({
            "message": message,
            "classification": classification,
            "response": response,
        })

        text = result.content.strip()
        empathy, clarity = 3, 3  # defaults

        for line in text.split("\n"):
            line = line.strip().upper()
            if line.startswith("EMPATHY"):
                try:
                    empathy = max(1, min(5, int(line.split(":")[-1].strip())))
                except ValueError:
                    pass
            elif line.startswith("CLARITY"):
                try:
                    clarity = max(1, min(5, int(line.split(":")[-1].strip())))
                except ValueError:
                    pass

        return {"empathy_score": empathy, "clarity_score": clarity}
    except Exception:
        return {"empathy_score": 0, "clarity_score": 0}


def run_evaluation():
    """Runs all test cases, prints results, returns summary dict."""
    orch = AgentOrchestrator()
    total = len(TEST_CASES)
    correct = 0
    results = []

    print("=" * 70)
    print("  MULTI-AGENT SYSTEM EVALUATION")
    print("=" * 70)
    print()

    for i, (msg, expected) in enumerate(TEST_CASES, 1):
        try:
            result = orch.process_message(msg)
            actual = result["classification"]
            match = (actual == expected)
            if match:
                correct += 1

            quality = score_response_quality(result["response"], actual, msg)

            icon = "✅" if match else "❌"
            print(f"  {icon} Test {i:2d}: {msg[:50]}...")
            print(f"           Expected: {expected} | Got: {actual}")
            print(f"           Path: {result['agent_path']}")
            print(f"           Response: {result['response'][:80]}...")
            print(f"           Empathy: {quality['empathy_score']}/5 | Clarity: {quality['clarity_score']}/5")
            print()

            results.append({
                "message": msg, "expected": expected, "actual": actual,
                "correct": match, "response": result["response"],
                "agent_path": result["agent_path"],
                "empathy_score": quality["empathy_score"],
                "clarity_score": quality["clarity_score"],
            })
        except Exception as e:
            print(f"  ❌ Test {i:2d}: ERROR — {e}")
            results.append({
                "message": msg, "expected": expected, "actual": "ERROR",
                "correct": False, "response": str(e), "agent_path": "Error",
                "empathy_score": 0, "clarity_score": 0,
            })

    acc = (correct / total * 100) if total else 0
    scored = [r for r in results if r["empathy_score"] > 0]
    avg_emp = sum(r["empathy_score"] for r in scored) / len(scored) if scored else 0
    avg_cla = sum(r["clarity_score"] for r in scored) / len(scored) if scored else 0

    print("=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total test cases   : {total}")
    print(f"  Correct            : {correct}")
    print(f"  Incorrect          : {total - correct}")
    print(f"  Classification Acc : {acc:.1f}%")
    print(f"  Avg Empathy Score  : {avg_emp:.1f}/5")
    print(f"  Avg Clarity Score  : {avg_cla:.1f}/5")
    print()

    for cat in ["positive_feedback", "negative_feedback", "query"]:
        cat_tests = [r for r in results if r["expected"] == cat]
        cat_ok = sum(1 for r in cat_tests if r["correct"])
        ct = len(cat_tests)
        pct = (cat_ok / ct * 100) if ct else 0
        print(f"  {cat:20s}: {cat_ok}/{ct} ({pct:.0f}%)")

    print()
    print("=" * 70)

    return {
        "total": total, "correct": correct, "accuracy": acc,
        "avg_empathy": avg_emp, "avg_clarity": avg_cla, "results": results,
    }


if __name__ == "__main__":
    run_evaluation()
