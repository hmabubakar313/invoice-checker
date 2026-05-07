import json
import requests


EXTRACTION_PROMPT = """Extract structured fields from this invoice text.

Return ONLY a JSON object. Use null (not "N/A", not a guess) for any field that is missing or unclear in the source. Do NOT invent values.

{{"amount": "<total amount, include currency if present>", "vendor": "<vendor / seller name>", "date": "<invoice date in YYYY-MM-DD if possible>"}}

Invoice text:
{text}
"""


def extract_invoice_fields(text):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": EXTRACTION_PROMPT.format(text=text),
            "stream": False,
        },
    )
    raw = response.json()["response"]
    return _parse_json(raw)


def _parse_json(raw):
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
