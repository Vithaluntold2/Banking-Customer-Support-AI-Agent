# app.py - Streamlit frontend for the banking support chatbot

import streamlit as st
import pandas as pd
from datetime import datetime
from orchestrator import AgentOrchestrator
from database import get_all_tickets, initialize_database
from evaluation import run_evaluation, TEST_CASES


st.set_page_config(
    page_title="Banking Support AI Agent",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# some custom CSS for the badges and card look
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 8px; }
    details { border-radius: 10px !important; }

    .badge-positive { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 20px; font-size: .8rem; font-weight: 600; display: inline-block; }
    .badge-negative { background: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 20px; font-size: .8rem; font-weight: 600; display: inline-block; }
    .badge-query    { background: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 20px; font-size: .8rem; font-weight: 600; display: inline-block; }
    .badge-general  { background: #f3f4f6; color: #374151; padding: 4px 12px; border-radius: 20px; font-size: .8rem; font-weight: 600; display: inline-block; }
    .agent-path     { background: #f0f4ff; color: #4338ca; padding: 3px 10px; border-radius: 16px; font-size: .75rem; font-weight: 500; display: inline-block; margin-top: 4px; }

    .stButton > button { border-radius: 8px; font-weight: 600; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# session state init
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

orchestrator = st.session_state.orchestrator


def classification_badge(cls):
    badges = {
        "positive_feedback": ("Positive Feedback", "badge-positive"),
        "negative_feedback": ("Negative Feedback", "badge-negative"),
        "query": ("Query", "badge-query"),
        "general": ("General", "badge-general"),
    }
    label, css = badges.get(cls, ("Unknown", "badge-general"))
    return f'<span class="{css}">{label}</span>'


def agent_path_badge(path):
    return f'<span class="agent-path">{path}</span>'


# sidebar navigation
with st.sidebar:
    st.markdown("### :material/account_balance: Banking Support AI")
    st.divider()
    page = st.radio(
        "Navigate",
        [":material/chat: Chat", ":material/confirmation_number: Tickets",
         ":material/receipt_long: Logs", ":material/science: Evaluation",
         ":material/info: About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Try these messages:**")
    st.caption('"Thanks for resolving my credit card issue."')
    st.caption('"My debit card replacement still hasn\'t arrived."')
    st.caption('"Could you check the status of ticket 650932?"')


# ---- CHAT PAGE ----
if page == ":material/chat: Chat":
    st.markdown("### :material/support_agent: Customer Support Agent")
    st.caption("Send a message and watch the multi-agent system classify and respond.")

    # collect customer name once - used when creating tickets
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    cust_name = st.text_input(
        ":material/person: Your Name",
        value=st.session_state.customer_name,
        placeholder="Enter your name before chatting",
    )
    st.session_state.customer_name = cust_name.strip()
    st.divider()

    user_input = st.chat_input("Type your message here...")
    if user_input:
        name = st.session_state.customer_name or None
        with st.spinner("Processing through agent pipeline..."):
            result = orchestrator.process_message(user_input, customer_name=name)
        st.session_state.chat_history.append({"user": user_input, "result": result})

    for entry in st.session_state.chat_history:
        with st.chat_message("user", avatar=":material/person:"):
            st.write(entry["user"])

        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            r = entry["result"]
            left, right = st.columns([3, 1])
            with left:
                st.write(r["response"])
            with right:
                st.markdown(classification_badge(r["classification"]), unsafe_allow_html=True)
                st.markdown(agent_path_badge(r["agent_path"]), unsafe_allow_html=True)

            if r.get("ticket_id") and r["classification"] == "negative_feedback":
                st.info(f":material/confirmation_number: Ticket **#{r['ticket_id']}** created")

            if r.get("ticket_details"):
                with st.expander(":material/description: Ticket Details"):
                    d = r["ticket_details"]
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Ticket ID**\n\n`#{d['ticket_id']}`")
                    c1.markdown(f"**Customer**\n\n{d['customer_name']}")
                    c2.markdown(f"**Issue**\n\n{d['issue_description']}")
                    c2.markdown(f"**Status**\n\n`{d['status']}`")

            # feedback buttons
            idx = st.session_state.chat_history.index(entry)
            if idx not in st.session_state.feedback_given:
                fb = st.columns([1, 1, 8])
                with fb[0]:
                    if st.button("👍", key=f"up_{idx}", help="Helpful response"):
                        orchestrator.save_user_feedback(idx, "thumbs_up")
                        st.session_state.feedback_given.add(idx)
                        st.rerun()
                with fb[1]:
                    if st.button("👎", key=f"down_{idx}", help="Unhelpful response"):
                        orchestrator.save_user_feedback(idx, "thumbs_down")
                        st.session_state.feedback_given.add(idx)
                        st.rerun()
            else:
                st.caption(":material/check_circle: Feedback recorded")

    if st.session_state.chat_history:
        if st.button(":material/delete: Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ---- TICKETS PAGE ----
elif page == ":material/confirmation_number: Tickets":
    st.markdown("### :material/database: Support Tickets")
    st.caption("View and filter all tickets in the database.")
    st.divider()

    tickets = get_all_tickets()
    if tickets:
        df = pd.DataFrame(tickets)
        all_df = df.copy()

        statuses = ["All"] + sorted(df["status"].unique().tolist())
        sel = st.selectbox(":material/filter_alt: Filter by Status", statuses)
        if sel != "All":
            df = df[df["status"] == sel]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(all_df))
        c2.metric("Unresolved", len(all_df[all_df["status"] == "Unresolved"]))
        c3.metric("In Progress", len(all_df[all_df["status"] == "In Progress"]))
        c4.metric("Resolved", len(all_df[all_df["status"] == "Resolved"]))

        st.divider()
        st.dataframe(
            df,
            column_config={
                "ticket_id": st.column_config.TextColumn("Ticket ID"),
                "customer_name": st.column_config.TextColumn("Customer"),
                "issue_description": st.column_config.TextColumn("Issue"),
                "status": st.column_config.TextColumn("Status"),
                "created_at": st.column_config.TextColumn("Created"),
                "updated_at": st.column_config.TextColumn("Updated"),
            },
            hide_index=True, width="stretch",
        )
    else:
        st.info(":material/info: No tickets in the database yet.")


# ---- LOGS PAGE ----
elif page == ":material/receipt_long: Logs":
    st.markdown("### :material/bug_report: Interaction Logs & Debugging")
    st.caption("Prompt traces, classification outputs, and ticket actions.")
    st.divider()

    logs = orchestrator.get_logs()
    if logs:
        classes = [l.get("classification") for l in logs]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(logs))
        c2.metric("Positive", classes.count("positive_feedback"))
        c3.metric("Negative", classes.count("negative_feedback"))
        c4.metric("Queries", classes.count("query"))

        fb_stats = orchestrator.get_feedback_stats()
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("👍 Helpful", fb_stats["thumbs_up"])
        c6.metric("👎 Unhelpful", fb_stats["thumbs_down"])
        c7.metric("No Feedback", fb_stats["no_feedback"])
        has_fb = (fb_stats["thumbs_up"] + fb_stats["thumbs_down"]) > 0
        c8.metric("Success Rate", f"{fb_stats['success_rate']:.0f}%" if has_fb else "N/A")

        st.divider()

        icon_map = {
            "positive_feedback": ":material/thumb_up:",
            "negative_feedback": ":material/thumb_down:",
            "query": ":material/search:",
            "general": ":material/chat:",
        }

        for i, log in enumerate(reversed(logs), 1):
            icon = icon_map.get(log.get("classification"), ":material/chat:")
            with st.expander(
                f"{icon} #{len(logs) - i + 1} — {log.get('user_message', '')[:60]}",
                expanded=(i == 1),
            ):
                left, right = st.columns(2)
                left.markdown(f"**:material/schedule: Timestamp**\n\n`{log.get('timestamp')}`")
                left.markdown(f"**:material/chat: User Message**\n\n{log.get('user_message')}")
                left.markdown(f"**:material/label: Classification**\n\n{log.get('classification')}")
                right.markdown(f"**:material/route: Agent Path**\n\n{log.get('agent_path')}")
                right.markdown(f"**:material/bolt: Action**\n\n{log.get('action')}")
                right.markdown(f"**:material/reply: Response**\n\n{log.get('response')}")

                if log.get("ticket_id"):
                    st.markdown(f"**:material/confirmation_number: Ticket ID:** #{log['ticket_id']}")
                if log.get("user_feedback"):
                    fb_icon = "👍" if log["user_feedback"] == "thumbs_up" else "👎"
                    st.markdown(f"**User Feedback:** {fb_icon} {log['user_feedback']}")

                traces = log.get("prompt_traces", [])
                if traces:
                    st.markdown("---")
                    st.markdown("**:material/description: Prompt Traces**")
                    for ti, trace in enumerate(traces):
                        step_name = trace.get("step", f"Step {ti+1}")
                        with st.expander(f":material/code: {step_name}", expanded=False):
                            st.markdown("**Prompt sent to LLM:**")
                            st.code(trace.get("prompt", "(none)"), language="text")
                            st.markdown("**Raw LLM completion:**")
                            st.code(trace.get("completion", "(none)"), language="text")

        st.divider()
        if st.button(":material/delete: Clear Logs", type="secondary"):
            orchestrator.clear_logs()
            st.rerun()
    else:
        st.info(":material/info: No interactions logged yet. Try chatting first!")


# ---- EVALUATION PAGE ----
elif page == ":material/science: Evaluation":
    st.markdown("### :material/analytics: Model Evaluation")
    st.caption("Assess classification accuracy and response quality across test cases.")
    st.divider()

    st.markdown("**:material/list_alt: Test Cases**")
    test_df = pd.DataFrame(TEST_CASES, columns=["Message", "Expected Classification"])
    st.dataframe(test_df, hide_index=True, width="stretch")
    st.divider()

    if st.button(":material/play_arrow: Run Evaluation", type="primary"):
        with st.spinner("Running evaluation across all test cases..."):
            ev = run_evaluation()

        st.markdown("**:material/assessment: Results**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Tests", ev["total"])
        c2.metric("Correct", ev["correct"])
        c3.metric("Accuracy", f"{ev['accuracy']:.1f}%")

        c4, c5 = st.columns(2)
        c4.metric("Avg Empathy", f"{ev.get('avg_empathy', 0):.1f}/5")
        c5.metric("Avg Clarity", f"{ev.get('avg_clarity', 0):.1f}/5")
        st.divider()

        for r in ev["results"]:
            icon = ":material/check_circle:" if r["correct"] else ":material/cancel:"
            with st.expander(f"{icon} {r['message'][:60]}"):
                left, right = st.columns(2)
                left.markdown(f"**Expected:** `{r['expected']}`")
                left.markdown(f"**Got:** `{r['actual']}`")
                left.markdown(f"**Empathy:** {'⭐' * r.get('empathy_score', 0)} ({r.get('empathy_score', 0)}/5)")
                right.markdown(f"**Agent Path:** {r['agent_path']}")
                right.markdown(f"**Response:** {r['response'][:200]}")
                right.markdown(f"**Clarity:** {'⭐' * r.get('clarity_score', 0)} ({r.get('clarity_score', 0)}/5)")


# ---- ABOUT PAGE ----
elif page == ":material/info: About":
    st.markdown("### :material/architecture: System Architecture")
    st.divider()

    st.markdown("**Banking Customer Support AI Agent** — a multi-agent GenAI system for banking support workflows.")

    st.markdown("#### :material/account_tree: Agent Pipeline")
    st.markdown("""
    ```
    User Message
         │
         ▼
    ┌──────────────────┐
    │  Classifier Agent │  Categorizes input
    └────────┬─────────┘
             │
     ┌───────┼──────────────┬──────────────┐
     ▼       ▼              ▼              ▼
    ┌─────┐ ┌────────────┐ ┌───────────┐ ┌─────────┐
    │ Pos │ │  Neg Feed  │ │  Query    │ │ General │
    │ Feed│ │  Handler   │ │  Handler  │ │ Handler │
    └─────┘ └────────────┘ └───────────┘ └─────────┘
       │         │              │             │
       ▼         ▼              ▼             ▼
    Thank     Create         Lookup       Welcome
    You       Ticket +       Ticket       Message
    Message   Empathy        Status
    ```
    """)

    st.markdown("#### :material/table_chart: Agent Details")
    st.markdown("""
    | Agent | Trigger | Action |
    |-------|---------|--------|
    | **Classifier** | Every message | Routes to the correct handler |
    | **Positive Feedback Handler** | Praise / thanks | Warm thank-you response |
    | **Negative Feedback Handler** | Complaints | Creates ticket + empathy |
    | **Query Handler** | Ticket inquiries | Database status lookup |
    | **General Handler** | Greetings / other | Welcome message |
    """)

    st.markdown("#### :material/build: Tech Stack")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(":material/psychology: **LLM**\n\nAzure OpenAI\n\nGPT-5.2")
    c2.markdown(":material/storage: **Database**\n\nSQLite")
    c3.markdown(":material/web: **UI**\n\nStreamlit")
    c4.markdown(":material/description: **Logging**\n\nJSON-based")
