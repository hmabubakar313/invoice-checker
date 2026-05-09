import logging

from langgraph.graph import StateGraph, START, END
from core.services import tool as tools
from core.services.extract import extract_invoice_fields
from core.services.state import (
    InvoiceState,
    _llm_decide,
    extract_node,
    llm_decision_node,
    retrieve_node,
    retry_node,
    rules_node,
)
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("langgraph_checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)



REQUIRED_FIELDS = ("amount", "vendor", "date")


TOOL_REGISTRY = {
    "approve_invoice": lambda invoice_id, reason: tools.approve_invoice(invoice_id),
    "notify_manager": lambda invoice_id, reason: tools.notify_manager(invoice_id, reason),
}

def decide_and_act(document, text):
    fields = extract_invoice_fields(text)

    missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
    if missing:
        tool_name = "notify_manager"
        reason = f"Missing required fields: {', '.join(missing)}"
    else:
        tool_name, reason = _llm_decide(document.id, text, fields)

    TOOL_REGISTRY[tool_name](document.id, reason)
    _persist(document, tool_name, reason)

    return {"tool": tool_name, "reason": reason, "fields": fields}


def _persist(document, tool_name, reason):
    document.status = "approved" if tool_name == "approve_invoice" else "flagged"
    document.reason = reason
    document.save(update_fields=["status", "reason"])


def _build_graph():
    builder = StateGraph(InvoiceState)

    builder.add_node("extract", extract_node)
    builder.add_node("rules", rules_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("decision", llm_decision_node)
    builder.add_node("retry", retry_node)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "rules")
    builder.add_edge("retrieve", "decision")

    builder.add_conditional_edges(
        "rules",
        lambda state: "missing" if (state.get("decision") or {}).get("tool") == "notify_manager" else "continue",
        {"missing": END, "continue": "retrieve"},
    )

    builder.add_conditional_edges(
        "decision",
        lambda state: "retry" if "not sure" in state["decision"].get("reason", "").lower() else "done",
        {"retry": "retry", "done": END},
    )

    builder.add_conditional_edges(
        "retry",
        lambda state: "cap_hit" if state["retries"] >= 2 else "continue",
        {"cap_hit": END, "continue": "retrieve"},
    )
    
    return builder.compile(checkpointer=checkpointer)


graph = _build_graph()


def run_agent(invoice_id, text):
    initial_state = {
        "invoice_id": invoice_id,
        "text": text,
        "extracted_fields": {},
        "context": "",
        "decision": {},
        "retries": 0,
        "trace": [],
        "status": "running",
    }
    logging.info(f"Starting agent for invoice {invoice_id}")
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": f"invoice-{invoice_id}"}}
    )
    logging.info(f"Finished agent for invoice {invoice_id} with decision: {result.get('decision')}")
    return result
