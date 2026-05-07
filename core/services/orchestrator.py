import json
import requests

from core.services import tool as tools
from core.services.retrieve import retrieve
from core.services.extract import extract_invoice_fields


REQUIRED_FIELDS = ("amount", "vendor", "date")


TOOL_REGISTRY = {
    "approve_invoice": lambda invoice_id, reason: tools.approve_invoice(invoice_id),
    "notify_manager": lambda invoice_id, reason: tools.notify_manager(invoice_id, reason),
}


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


def _llm_decide(invoice_id, text, fields):
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

    if tool_name not in TOOL_REGISTRY:
        return "notify_manager", f"Could not parse agent decision: {raw[:200]}"

    if "not sure" in reason.lower():
        return "notify_manager", reason

    return tool_name, reason


def _persist(document, tool_name, reason):
    document.status = "approved" if tool_name == "approve_invoice" else "flagged"
    document.reason = reason
    document.save(update_fields=["status", "reason"])


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

