"""
Business Central + MCP Natural Language Settlement & Query Demo (Streamlit).
Self-sustained version with Voice-to-Text mic support for Streamlit Cloud deployment.
"""

import json
import os
import time

import pandas as pd
import requests
import streamlit as st

try:
    from streamlit_mic_recorder import speech_to_text
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False

from mcp_client import BusinessCentralMCPClient

st.set_page_config(page_title="Business Central AI Assistant", layout="wide")


# ============================================================ connection (cached once)
@st.cache_resource(show_spinner="Connecting to Business Central...")
def connect():
    client = BusinessCentralMCPClient()
    auth = client.authenticate()
    if not auth.get("success"):
        raise RuntimeError(f"Login failed: {auth.get('error')}")
    t = client.list_official_mcp_tools()
    if not t.get("success"):
        raise RuntimeError(f"Could not list MCP tools: {t.get('error')}")
    tools = t["tools"]
    names = [x["name"] for x in tools]
    ledger = next((n for n in names if "customerledger" in n.lower().replace("_", "")), None)
    if not ledger:
        raise RuntimeError(f"No customer ledger tool among: {names}")
    props = next(x for x in tools if x["name"] == ledger).get("inputSchema", {}).get("properties", {})
    return {
        "client": client,
        "tools": names,
        "ledger": ledger,
        "filter_key": next((k for k in props if "filter" in k.lower()), None),
        "top_key": next((k for k in props if k.lower().lstrip("$") in ("top", "pagesize", "limit", "maxrows")), None),
        "skip_key": next((k for k in props if k.lower().lstrip("$") in ("skip", "offset")), None),
    }


@st.cache_data(ttl=3600, show_spinner="Loading customer list from Business Central...")
def customer_master():
    return connect()["client"].fetch_live_customers() or {}


# ============================================================ helpers
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pick(row, *keys):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return None


def q(v):
    return "'" + str(v).replace("'", "''") + "'"


def norm(row, names):
    rem = num(pick(row, "RemainingAmount", "RemainingAmtLCY", "remainingAmount"))
    cust_no = pick(row, "CustomerNo", "customerNumber")
    return {
        "Entry": pick(row, "EntryNo", "entryNumber"),
        "Document": pick(row, "DocumentNo", "documentNumber"),
        "Type": pick(row, "DocumentType", "documentType") or "",
        "Customer No": cust_no,
        "Customer": pick(row, "CustomerName", "customerName") or names.get(str(cust_no or "").lower(), ""),
        "Posting Date": pick(row, "PostingDate", "postingDate"),
        "Due Date": pick(row, "DueDate", "dueDate"),
        "Amount": num(pick(row, "Amount", "AmountLCY", "amount")),
        "Remaining": rem,
        "Status": "OPEN" if (rem is not None and abs(rem) >= 0.005) else "closed",
    }


def fetch_rows(odata, max_rows):
    c = connect()
    args = {}
    if odata:
        if not c["filter_key"]:
            raise RuntimeError(f"{c['ledger']} has no filter argument.")
        args[c["filter_key"]] = odata
    if c["top_key"]:
        args[c["top_key"]] = max_rows
    names = customer_master()
    t0, rows, pages = time.time(), [], 0
    first_page = 0
    while True:
        if c["skip_key"] and pages:
            args[c["skip_key"]] = len(rows)
        res = c["client"].call_official_mcp_tool(c["ledger"], args)
        if not res.get("success"):
            raise RuntimeError(f"MCP error: {res.get('error')}")
        batch = []
        for part in (res["result"].get("result") or {}).get("content", []):
            try:
                data = json.loads(part.get("text", ""))
            except ValueError:
                continue
            raw = data.get("value", []) if isinstance(data, dict) else data
            batch = [norm(r, names) for r in raw if isinstance(r, dict)]
            break
        rows.extend(batch)
        pages += 1
        if not c["skip_key"] or not batch or len(rows) >= max_rows or pages >= 25:
            break
        if pages > 1 and len(batch) < first_page:
            break
        if pages == 1:
            first_page = len(batch)
    return rows[:max_rows], time.time() - t0


def resolve_customer(text):
    words = str(text).lower().split()
    return [(no, nm) for no, nm in customer_master().items()
            if all(w in f"{no} {nm}".lower() for w in words)]


# ============================================================ LLM: sentence -> intent
PROMPT = """Extract the user's request about Business Central customer ledger entries.
Return ONLY JSON: {"intent": "list"|"settle"|"other", "customer": string|null,
"open_only": true|false, "invoices_only": true|false, "document_no": string|null, "amount": number|null}
- "open", "unpaid", "outstanding", "pending", "due" => open_only true
- user asks about "invoices" (not all entries or payments) => invoices_only true
- customer: the customer name or number exactly as the user wrote it, else null
- settle/pay/close/apply an invoice => intent "settle"; document_no exactly as written
- amount only if the user states one; never invent values; use null when not stated"""


@st.cache_data(ttl=3600, show_spinner=False)
def understand(sentence):
    cfg = connect()["client"].config
    key = os.environ.get("GROQ_API_KEY") or cfg.get("groq_api_key") or cfg.get("GROQ_API_KEY")
    if not key and hasattr(st, "secrets"):
        key = st.secrets.get("groq_api_key") or st.secrets.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY in environment or Streamlit secrets.")

    model = os.environ.get("GROQ_MODEL") or cfg.get("GROQ_MODEL") or "openai/gpt-oss-120b"
    body = {"model": model, "temperature": 0, "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": sentence}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    for attempt in range(4):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
        if r.status_code == 429 and attempt < 3:
            time.sleep(min(float(r.headers.get("retry-after", 5) or 5), 30))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:300]}")
        return json.loads(r.json()["choices"][0]["message"]["content"])
    raise RuntimeError("Groq kept rate-limiting. Wait a minute and try again.")


# ============================================================ actions
def do_list(p, max_rows):
    parts, notes = [], []
    if p.get("customer"):
        matches = resolve_customer(p["customer"])
        if not matches:
            return {"level": "warning", "text": f"No customer matches '{p['customer']}' in the customer list."}
        if len(matches) > 10:
            return {"level": "warning",
                    "text": f"'{p['customer']}' matches {len(matches)} customers. Be more specific.",
                    "table": [{"Customer No": n, "Customer": m} for n, m in matches[:50]]}
        parts.append("(" + " or ".join(f"CustomerNo eq {q(n)}" for n, _ in matches) + ")")
        notes.append("Customer: " + ", ".join(f"{m} ({n})" for n, m in matches))
    if p.get("open_only"):
        parts.append("RemainingAmount ne 0")
    if p.get("invoices_only"):
        parts.append("DocumentType eq 'Invoice'")
    odata = " and ".join(parts) or None
    rows, secs = fetch_rows(odata, max_rows)
    if p.get("open_only"):
        rows = [r for r in rows if r["Status"] == "OPEN"]
    if p.get("invoices_only"):
        rows = [r for r in rows if str(r["Type"]).lower() == "invoice"]
    capped = len(rows) >= max_rows
    return {"level": "info", "filter": odata, "bc_seconds": secs, "table": rows, "notes": notes,
            "text": f"{len(rows)} entries{' (showing the first ' + str(max_rows) + ')' if capped else ''}."}


def lookup(ref, max_rows):
    rows, _ = fetch_rows(f"DocumentNo eq {q(ref)}", max_rows)
    if not rows and str(ref).isdigit():
        rows, _ = fetch_rows(f"EntryNo eq {int(ref)}", max_rows)
    return [r for r in rows if str(ref).lower() in (str(r["Document"]).lower(), str(r["Entry"]).lower())]


def do_settle(p, max_rows):
    ref = p.get("document_no")
    if not ref:
        return {"level": "warning", "text": "Which invoice? e.g. 'settle invoice <document no.> for 1'"}
    hits = lookup(ref, max_rows)
    if not hits:
        return {"level": "warning", "text": f"'{ref}' was not found in Business Central."}
    if len(hits) > 1:
        return {"level": "warning", "text": f"'{ref}' matches {len(hits)} entries. Use the entry number.", "table": hits}
    e = hits[0]
    if str(e["Type"]).lower() != "invoice":
        return {"level": "error", "text": f"Refused: {ref} is a {e['Type'] or 'non-invoice'} entry, not an invoice.", "table": [e]}
    if e["Status"] != "OPEN":
        return {"level": "error", "text": f"Refused: {ref} is already closed (remaining 0).", "table": [e]}
    outstanding = abs(e["Remaining"])
    amount = num(p.get("amount")) or outstanding
    if amount <= 0 or amount > outstanding + 0.005:
        return {"level": "error", "text": f"Refused: amount {amount:,.2f} must be between 0 and {outstanding:,.2f}.", "table": [e]}
    st.session_state.pending = {"entry": e, "amount": amount, "outstanding": outstanding}
    return {"level": "info", "text": "Preview below. Nothing has been posted yet.", "table": [e]}


def verify(last):
    e = last["entry"]
    rows, _ = fetch_rows(f"EntryNo eq {int(e['Entry'])}", 5)
    now = [r for r in rows if str(r["Entry"]) == str(e["Entry"])]
    if not now:
        return {"level": "error", "text": f"Could not read entry {e['Entry']} back from Business Central."}
    after = abs(now[0]["Remaining"] or 0)
    expected = max(last["before"] - last["amount"], 0)
    ok = abs(after - expected) < 0.01
    return {"level": "success" if ok else "error",
            "text": f"{'VERIFIED' if ok else 'NOT VERIFIED'}: re-read from Business Central, remaining "
                    f"{last['before']:,.2f} -> {after:,.2f} (expected {expected:,.2f}).",
            "table": now}


def post_pending():
    pend = st.session_state.pop("pending")
    e, amount = pend["entry"], pend["amount"]
    res = connect()["client"].apply_and_post_customer_payment(
        customer_number=str(e["Customer No"]), invoice_number=str(e["Document"]), amount=amount)
    if not res.get("success"):
        msg = f"Business Central did NOT complete it (step: {res.get('stage')}). " \
              f"Microsoft said: {res.get('error_from_microsoft') or res.get('error')}"
        if res.get("note"):
            msg += f"  {res['note']}"
        st.session_state.history.append({"prompt": f"Post {amount:,.2f} to {e['Document']}", "level": "error", "text": msg})
        return
    st.session_state.last = {"entry": e, "amount": amount, "before": pend["outstanding"]}
    st.session_state.history.append({"prompt": f"Post {amount:,.2f} to {e['Document']}", "level": "success",
                                     "text": f"Posted. Document no. from Business Central: {res.get('posting_number')}"})
    st.session_state.history.append({"prompt": "Verify", **verify(st.session_state.last)})


def cancel_pending():
    pend = st.session_state.pop("pending")
    st.session_state.history.append({"prompt": f"Cancel {pend['entry']['Document']}", "level": "info",
                                     "text": "Cancelled. Nothing was posted."})


def queue(text):
    st.session_state.queued = text


# ============================================================ UI
st.session_state.setdefault("history", [])

try:
    conn = connect()
    bc = conn["client"].bc_cfg
except Exception as ex:
    st.error(f"Could not connect: {ex}")
    st.stop()
with st.sidebar:
    st.subheader("Connection")
    st.write(f"**Environment:** {bc.get('environment')}")
    st.write(f"**Company:** {bc.get('company_name')}")
    st.write(f"**MCP tools:** {len(conn['tools'])}")
    with st.expander("Show MCP tools"):
        for n in conn["tools"]:
            st.caption(n)
    max_rows = st.slider("Max rows per query", 20, 500, 50, step=10)

    st.subheader("🎤 Voice Input")
    if HAS_VOICE:
        audio_text = speech_to_text(
            language='en',
            start_prompt="Click to Speak 🎙️",
            stop_prompt="Stop Listening ⏹️",
            key='voice_recorder'
        )
        if audio_text:
            st.success(f"Voice detected: '{audio_text}'")
            if st.button("Use Voice Command", type="primary", use_container_width=True):
                st.session_state.queued = audio_text
    else:
        st.info("Install `streamlit-mic-recorder` for mic support.")

    st.subheader("Try")
    for ex_prompt in ["show open invoices", "settle invoice T33HB2526/00313 for 1"]:
        st.button(ex_prompt, on_click=queue, args=(ex_prompt,), use_container_width=True)
    if st.session_state.get("last"):
        if st.button("Verify last settlement", use_container_width=True):
            st.session_state.history.append({"prompt": "Verify", **verify(st.session_state.last)})
    if st.button("Clear", use_container_width=True):
        st.session_state.history = []
        st.session_state.pop("pending", None)

st.title("Business Central AI Settlement Assistant")
st.caption("Ask in plain English or use microphone. The AI maps your intent; live queries run via Microsoft MCP.")

prompt = st.chat_input("e.g. show open invoices for <customer>  |  settle invoice <document no.> for 1") or st.session_state.pop("queued", None)
if prompt:
    entry = {"prompt": prompt}
    try:
        t0 = time.time()
        p = understand(prompt)
        entry["plan"], entry["llm_seconds"] = p, time.time() - t0
        if p.get("intent") == "list":
            entry.update(do_list(p, max_rows))
        elif p.get("intent") == "settle":
            st.session_state.pop("pending", None)
            entry.update(do_settle(p, max_rows))
        else:
            entry.update(level="warning", text="That isn't a ledger question or a settlement. Try 'show open invoices'.")
    except Exception as ex:
        entry.update(level="error", text=str(ex))
    st.session_state.history.append(entry)

for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["prompt"])
    with st.chat_message("assistant"):
        getattr(st, h.get("level", "info"))(h.get("text", ""))
        for n in h.get("notes", []):
            st.caption(n)
        if h.get("table"):
            st.dataframe(pd.DataFrame(h["table"]), use_container_width=True, hide_index=True,
                         column_config={"Amount": st.column_config.NumberColumn(format="%.2f"),
                                        "Remaining": st.column_config.NumberColumn(format="%.2f")})
        if h.get("plan") or h.get("filter"):
            timing = " · ".join(x for x in [
                f"AI {h['llm_seconds']:.1f}s" if h.get("llm_seconds") is not None else "",
                f"Business Central {h['bc_seconds']:.1f}s" if h.get("bc_seconds") is not None else ""] if x)
            with st.expander(f"How this was answered{'  (' + timing + ')' if timing else ''}"):
                st.write("**AI understood:**")
                st.json(h.get("plan", {}))
                st.write("**Filter sent to Business Central:**")
                st.code(h.get("filter") or "(none)")

pend = st.session_state.get("pending")
if pend:
    e = pend["entry"]
    with st.container(border=True):
        st.subheader("Confirm payment")
        st.write(f"Invoice **{e['Document']}**, customer **{e['Customer'] or e['Customer No']}** ({e['Customer No']})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Outstanding now", f"{pend['outstanding']:,.2f}")
        c2.metric("Pay", f"{pend['amount']:,.2f}")
        c3.metric("Expected after", f"{pend['outstanding'] - pend['amount']:,.2f}")
        st.caption(f"Posts a real payment through payment journal '{bc.get('payment_journal')}', then re-reads the invoice to verify.")
        b1, b2 = st.columns(2)
        b1.button("Post to Business Central", type="primary", on_click=post_pending, use_container_width=True)
        b2.button("Cancel", on_click=cancel_pending, use_container_width=True)
