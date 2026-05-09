from dataclasses import dataclass, field, fields
from typing import TypedDict, List, Dict, Any
import json
import logging

import requests
from langgraph.graph import StateGraph, START, END
from core.services.extract import extract_invoice_fields
from core.services.retrieve import retrieve


REQUIRED_FIELDS = ["amount", "vendor", "date"]


PROMPT_TEMPLATE = """
You are an invoice approval system. Required fields are already verified to exist;
your job now is to flag anything suspicious based on retrieved policies / past invoices.

Available tools:
- approve_invoice  — looks consistent with policy and prior invoices.
- notify_manager   — anything off: blocked vendor, duplicate, unusual amount, policy violation.

Extracted fields:
{fields}

Context:
{context}

Invoice:
{invoice}

Return ONLY JSON:
{{"tool": "approve_invoice" | "notify_manager", "reason": "<short>"}}
"""




class InvoiceState(TypedDict):
    invoice_id: int
    text: str
    extracted_fields: Dict[str, Any]
    context: str
    decision: Dict[str, Any]
    retries: int
    status: str
    trace: List[Dict[str, Any]]


def extract_node(state):
    fields = extract_invoice_fields(state["text"])
    logging.info(f"Extracted fields for invoice {state['invoice_id']}: {fields}")

    state["extracted_fields"] = fields
    state["trace"].append({
        "node": "extract_node",
        "output": fields,
    })

    return state


def rules_node(state):
    missing = [
        f for f in REQUIRED_FIELDS
        if not state["extracted_fields"].get(f)
    ]

    if missing:
        state["decision"] = {
            "tool": "notify_manager",
            "reason": f"Missing fields: {', '.join(missing)}"
        }

    state["trace"].append({
        "node": "rules_node",
        "output": {"missing": missing, "decision": state["decision"]},
    })

    return state


def retrieve_node(state):
    context = _retrieve_context(
        state["invoice_id"],
        state["text"]
    )
    logging.info(f"Retrieved context for invoice {state['invoice_id']}: {context}")

    state["context"] = context
    state["trace"].append({
        "node": "retrieve_node",
        "output": context,
    })

    return state


def llm_decision_node(state):
    tool_name, reason = _llm_decide(
        state["invoice_id"],
        state["text"],
        state["extracted_fields"],
        context=state["context"],
    )
    logging.info(f"LLM decision for invoice {state['invoice_id']}: tool={tool_name} reason={reason}")
    state["decision"] = {"tool": tool_name, "reason": reason}
    state["trace"].append({
        "node": "llm_decision_node",
        "output": state["decision"],
    })
    return state


def retry_node(state):
    state["retries"] += 1

    if state["retries"] >= 2:
        state["trace"].append({"node": "retry_node", "output": "cap_hit"})
        return state

    extra_context = _retrieve_context(state["invoice_id"], state["text"], k=10)
    logging.info(f"Extra context for invoice {state['invoice_id']} on retry {state['retries']}: {extra_context}")

    state["context"] += "\n" + extra_context
    state["trace"].append({"node": "retry_node", "output": extra_context})
    return state


def _llm_decide(invoice_id, text, fields, context=None):
    if context is None:
        context = _retrieve_context(invoice_id, text)

    prompt = PROMPT_TEMPLATE.format(
        fields=json.dumps(fields),
        context=context,
        invoice=text,
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
    )
    raw = response.json()["response"]

    decision = _parse_decision(raw)
    tool_name = decision.get("tool")
    reason = decision.get("reason", "")

    if tool_name not in ("approve_invoice", "notify_manager"):
        return "notify_manager", f"Could not parse agent decision: {raw[:200]}"

    if "not sure" in reason.lower():
        return "notify_manager", reason

    return tool_name, reason


def _retrieve_context(invoice_id, text, k=5):
    docs, ids = retrieve(text, k=k)
    # Drop chunks belonging to this same invoice — we already pass its full text.
    prefix = f"{invoice_id}_"
    external = [d for d, i in zip(docs, ids) if not i.startswith(prefix)]
    if not external:
        return "(no related context found)"
    return "\n---\n".join(external)


def _parse_decision(raw):
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
