"""
Live client for Microsoft Dynamics 365 Business Central.
Supports reading configuration from st.secrets (Streamlit Cloud) or local bc_config.json.
"""

import datetime
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BC_MCP_Client")

REQUIRED_BC = ("tenant_id", "environment", "company_name", "configuration_name")
REQUIRED_AUTH = ("client_id", "client_secret")
MCP_SESSION_EXPIRED = ("Session not found", "-32001")


def _mask(headers: Dict[str, str]) -> Dict[str, str]:
    h = dict(headers)
    if "Authorization" in h:
        h["Authorization"] = h["Authorization"][:15] + "..."
    return h


def _body(res: requests.Response) -> Any:
    try:
        return res.json()
    except ValueError:
        return res.text


class BusinessCentralMCPClient:
    def __init__(self, config_path: str = "bc_config.json"):
        self.config: Dict[str, Any] = {}

        # 1. Try reading from Streamlit Secrets first (Streamlit Cloud)
        if HAS_STREAMLIT and hasattr(st, "secrets") and len(st.secrets) > 0:
            try:
                self.config = {
                    "groq_api_key": st.secrets.get("groq_api_key") or st.secrets.get("GROQ_API_KEY"),
                    "business_central": dict(st.secrets.get("business_central", {})),
                    "auth": dict(st.secrets.get("auth", {})),
                }
            except Exception:
                pass

        # 2. Fall back to local bc_config.json
        if not self.config or not self.config.get("business_central"):
            if not os.path.isabs(config_path):
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)

        self.bc_cfg: Dict[str, Any] = self.config.get("business_central", {})
        self.auth_cfg: Dict[str, Any] = self.config.get("auth", {})

        self.mcp_url = self.bc_cfg.get("mcp_server_url") or "https://mcp.businesscentral.dynamics.com/"
        self.api_base = (
            f"https://api.businesscentral.dynamics.com/v2.0/"
            f"{self.bc_cfg.get('tenant_id')}/{self.bc_cfg.get('environment')}/api/v2.0"
        )

        self.access_token: Optional[str] = None
        self._token_expires_at = 0.0
        self.mcp_session_id: Optional[str] = None
        self._company_id: Optional[str] = None
        self._rpc_id = 0

    def authenticate(self) -> Dict[str, Any]:
        url = f"https://login.microsoftonline.com/{self.bc_cfg['tenant_id']}/oauth2/v2.0/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.auth_cfg["client_id"],
            "client_secret": self.auth_cfg["client_secret"],
            "scope": "https://api.businesscentral.dynamics.com/.default",
        }
        try:
            res = requests.post(url, data=payload, timeout=15)
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

        body = _body(res)
        if res.status_code != 200:
            self.access_token = None
            logger.error(f"Entra ID token error {res.status_code}: {res.text[:300]}")
            return {"success": False, "status_code": res.status_code, "error": body}

        self.access_token = body["access_token"]
        self._token_expires_at = time.time() + int(body.get("expires_in", 3600)) - 120
        logger.info("Authenticated with Microsoft Entra ID")
        return {"success": True}

    def _ensure_token(self) -> None:
        if not self.access_token or time.time() >= self._token_expires_at:
            res = self.authenticate()
            if not res.get("success"):
                raise RuntimeError(f"Authentication failed: {res.get('error')}")

    def _rest(self, method: str, path: str, **kw) -> requests.Response:
        self._ensure_token()
        url = path if path.startswith("http") else f"{self.api_base}/{path.lstrip('/')}"
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}
            if "json" in kw:
                headers["Content-Type"] = "application/json"
            res = requests.request(method, url, headers=headers, timeout=90, **kw)
            if res.status_code == 401 and attempt == 1:
                self.authenticate()
                continue
            return res
        return res

    def get_company_id(self) -> str:
        if self._company_id:
            return self._company_id
        res = self._rest("GET", "companies")
        if res.status_code != 200:
            raise RuntimeError(f"Could not list companies ({res.status_code}): {res.text[:500]}")
        companies = res.json().get("value", [])
        wanted = self.bc_cfg["company_name"].strip().lower()
        for c in companies:
            if wanted in (str(c.get("name", "")).lower(), str(c.get("displayName", "")).lower()):
                self._company_id = c["id"]
                return self._company_id
        names = [c.get("name") for c in companies]
        raise RuntimeError(f"Company '{self.bc_cfg['company_name']}' not found. Companies in this environment: {names}")

    def _company_path(self, sub: str) -> str:
        return f"companies({self.get_company_id()})/{sub.lstrip('/')}"

    def fetch_live_customers(self) -> Dict[str, str]:
        res = self._rest("GET", self._company_path("customers?$select=number,displayName"))
        if res.status_code != 200:
            logger.error(f"Customer list failed {res.status_code}: {res.text[:300]}")
            return {}
        return {
            str(c["number"]).lower(): c.get("displayName")
            for c in res.json().get("value", [])
            if c.get("number")
        }

    def _mcp_headers(self) -> Dict[str, str]:
        self._ensure_token()
        h = {
            "Authorization": f"Bearer {self.access_token}",
            "TenantId": self.bc_cfg["tenant_id"],
            "EnvironmentName": self.bc_cfg["environment"],
            "Company": self.bc_cfg["company_name"],
            "ConfigurationName": self.bc_cfg["configuration_name"],
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.mcp_session_id:
            h["Mcp-Session-Id"] = self.mcp_session_id
        return h

    @staticmethod
    def _parse_rpc(res: requests.Response) -> Optional[Dict[str, Any]]:
        text = res.text or ""
        if "text/event-stream" in res.headers.get("Content-Type", "") or text.lstrip().startswith(("event:", "data:")):
            msg = None
            for line in text.splitlines():
                if line.startswith("data:"):
                    try:
                        candidate = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    if isinstance(candidate, dict) and ("result" in candidate or "error" in candidate):
                        msg = candidate
            return msg
        try:
            data = res.json()
            return data if isinstance(data, dict) else None
        except ValueError:
            return None

    def _rpc_post(self, payload: Dict[str, Any]) -> requests.Response:
        return requests.post(self.mcp_url, headers=self._mcp_headers(), json=payload, timeout=60)

    def init_official_mcp_session(self) -> Dict[str, Any]:
        self.mcp_session_id = None
        self._rpc_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._rpc_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "bc-nl-connector", "version": "1.0"},
            },
        }
        try:
            res = self._rpc_post(payload)
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
        if res.status_code >= 400:
            return {"success": False, "status_code": res.status_code, "error": res.text[:1000]}

        self.mcp_session_id = res.headers.get("Mcp-Session-Id") or res.headers.get("mcp-session-id")
        try:
            self._rpc_post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except requests.RequestException as e:
            logger.warning(f"initialized notification failed: {e}")
        return {"success": True, "session_id": self.mcp_session_id, "server": self._parse_rpc(res)}

    def _rpc(self, method: str, params: Dict[str, Any], retry: bool = True) -> Dict[str, Any]:
        if not self.mcp_session_id:
            init = self.init_official_mcp_session()
            if not init.get("success"):
                return init

        self._rpc_id += 1
        payload = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params}
        headers = self._mcp_headers()
        try:
            res = self._rpc_post(payload)
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}

        expired = res.status_code == 404 or any(s in res.text for s in MCP_SESSION_EXPIRED)
        if retry and (expired or res.status_code == 401):
            if res.status_code == 401:
                self.authenticate()
            self.mcp_session_id = None
            return self._rpc(method, params, retry=False)

        if res.status_code >= 400:
            return {"success": False, "status_code": res.status_code, "error": res.text[:2000]}

        msg = self._parse_rpc(res)
        if msg is None:
            return {"success": False, "error": "MCP server returned no JSON-RPC message."}
        if "error" in msg:
            return {"success": False, "error": msg["error"], "result": msg}
        return {"success": True, "result": msg}

    def list_official_mcp_tools(self) -> Dict[str, Any]:
        tools: List[Dict[str, Any]] = []
        cursor = None
        while True:
            out = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            if not out.get("success"):
                return {"success": False, "error": out.get("error"), "tools": [], "count": 0}
            result = out["result"].get("result", {})
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return {"success": True, "count": len(tools), "tools": tools}

    def call_official_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = self._rpc("tools/call", {"name": tool_name, "arguments": arguments or {}})
        if out.get("success") and out["result"].get("result", {}).get("isError"):
            out["success"] = False
            out["error"] = out["result"]["result"].get("content")
        return out

    def _payment_journal(self) -> Dict[str, Any]:
        res = self._rest("GET", self._company_path("customerPaymentJournals"))
        if res.status_code != 200:
            raise RuntimeError(f"Could not list customer payment journals ({res.status_code}): {res.text[:500]}")
        journals = res.json().get("value", [])
        code = (self.bc_cfg.get("payment_journal") or "").strip().lower()
        if code:
            for j in journals:
                if str(j.get("code", "")).strip().lower() == code:
                    return j
            names = [j.get("code") for j in journals]
            raise RuntimeError(f"Configured journal '{self.bc_cfg.get('payment_journal')}' not found. Available: {names}")
        if not journals:
            raise RuntimeError("No customer payment journals in Business Central.")
        return journals[0]

    def apply_and_post_customer_payment(self, customer_number: str, invoice_number: str, amount: float) -> Dict[str, Any]:
        journal = self._payment_journal()
        jid = journal["id"]
        posting_date = datetime.date.today().isoformat()
        amt_float = -abs(float(amount))

        cust_res = self._rest("GET", self._company_path(f"customers?$filter=number eq '{customer_number}'&$select=id"))
        if cust_res.status_code != 200 or not cust_res.json().get("value"):
            return {"success": False, "stage": "find_customer", "error": f"Customer '{customer_number}' not found."}
        cust_id = cust_res.json()["value"][0]["id"]

        inv_res = self._rest("GET", self._company_path(f"salesInvoices?$filter=number eq '{invoice_number}'&$select=id,number,customerNumber"))
        if inv_res.status_code != 200 or not inv_res.json().get("value"):
            return {"success": False, "stage": "find_invoice", "error": f"Invoice '{invoice_number}' not found."}
        inv_data = inv_res.json()["value"][0]
        inv_id = inv_data["id"]

        line_payload = {
            "postingDate": posting_date,
            "customerId": cust_id,
            "customerNumber": customer_number,
            "amount": amt_float,
            "appliesToDocType": "Invoice",
            "appliesToDocNo": invoice_number,
            "appliesToInvoiceId": inv_id,
            "appliesToInvoiceNumber": invoice_number,
            "description": f"Settlement of {invoice_number}",
        }

        line_path = self._company_path(f"customerPaymentJournals({jid})/customerPayments")
        res = self._rest("POST", line_path, json=line_payload)
        if res.status_code not in (200, 201):
            return {"success": False, "stage": "create_line", "error": res.text, "error_from_microsoft": res.text}

        line_data = res.json()
        line_id = line_data["id"]

        post_path = self._company_path(f"customerPaymentJournals({jid})/customerPayments({line_id})/Microsoft.NAV.post")
        post_res = self._rest("POST", post_path)

        if post_res.status_code in (200, 204):
            post_num = post_res.headers.get("Location") or post_res.headers.get("location") or "Posted"
            return {"success": True, "posting_number": post_num, "line_id": line_id}

        err_text = post_res.text
        return {"success": False, "stage": "post", "error": err_text, "error_from_microsoft": err_text, "line_id": line_id}
