import os
import json
from typing import Any, Dict, List, Optional
from io import BytesIO

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Bank Categorization Review", layout="wide")

API_BASE_URL = os.getenv("RULE_BASE_API_URL", "http://localhost:8027")
RULE_BASE_PREFIX = "/gns-api/api/v2/rule-base"


def _api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{RULE_BASE_PREFIX}{path}"


def _safe_json(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {"raw_text": response.text}


def api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
    response = requests.get(_api_url(path), params=params or {}, timeout=timeout)
    payload = _safe_json(response)
    if response.status_code >= 400:
        raise RuntimeError(f"Request failed ({response.status_code}): {json.dumps(payload)[:400]}")
    return payload


def api_post_json(path: str, body: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    response = requests.post(_api_url(path), json=body, timeout=timeout)
    payload = _safe_json(response)
    if response.status_code >= 400:
        raise RuntimeError(f"Request failed ({response.status_code}): {json.dumps(payload)[:400]}")
    return payload


def api_put_json(path: str, body: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    response = requests.put(_api_url(path), json=body, timeout=timeout)
    payload = _safe_json(response)
    if response.status_code >= 400:
        raise RuntimeError(f"Request failed ({response.status_code}): {json.dumps(payload)[:400]}")
    return payload


def api_post_files(
    path: str,
    form_data: Dict[str, Any],
    files: List[tuple],
    timeout: int = 300,
) -> Dict[str, Any]:
    response = requests.post(_api_url(path), data=form_data, files=files, timeout=timeout)
    payload = _safe_json(response)
    if response.status_code >= 400:
        raise RuntimeError(f"Request failed ({response.status_code}): {json.dumps(payload)[:600]}")
    return payload


def load_clients() -> List[Dict[str, Any]]:
    data = api_get("/clients")
    clients = data.get("clients") or []
    return clients


def load_rules_for_client(client_id: str) -> List[Dict[str, Any]]:
    data = api_get(f"/{client_id}")
    return (data.get("data") or {}).get("rules") or []


def load_user_statement_runs(user_id: str) -> List[Dict[str, Any]]:
    data = api_get("/client-bank-statement-data", params={"user_id": user_id})
    return ((data.get("data") or {}).get("records") or [])


def run_categorization(client_id: str, user_id: str, statements: List[Any], check_register: Optional[Any]) -> Dict[str, Any]:
    files = []
    for file_obj in statements:
        files.append(("bank_statements", (file_obj.name, file_obj.getvalue(), "application/octet-stream")))

    if check_register is not None:
        files.append(("check_register", (check_register.name, check_register.getvalue(), "application/octet-stream")))

    return api_post_files(
        "/categorize-bank-statement",
        form_data={"client_id": client_id, "user_id": user_id},
        files=files,
    )


def is_rate_limit_message(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "429" in text
        or "quota" in text
        or "rate limit" in text
        or "resource exhausted" in text
        or "resource_exhausted" in text
    )


def normalize_column_name(name: Any) -> str:
    return " ".join(str(name or "").strip().lower().replace("_", " ").split())


def first_matching_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for candidate in candidates:
        for column in columns:
            if candidate in column:
                return column
    return None


def inspect_check_register(uploaded_file: Any) -> Dict[str, Any]:
    if uploaded_file is None:
        return {"error": "No file uploaded.", "preview": None}
    try:
        file_name = (uploaded_file.name or "").lower()
        file_bytes = uploaded_file.getvalue()
        if file_name.endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
    except Exception as exc:
        return {"error": f"Could not read check register file: {exc}", "preview": None}

    normalized_columns = [normalize_column_name(col) for col in df.columns.tolist()]
    check_column = first_matching_column(
        normalized_columns,
        ["check number", "check #", "check", "chk", "check no", "number"],
    )
    payee_column = first_matching_column(
        normalized_columns,
        ["payee name", "payee", "pay to", "vendor", "name"],
    )
    memo_column = first_matching_column(
        normalized_columns,
        ["memo", "description", "details", "notes", "memo description"],
    )
    detected = {
        "check_number": check_column,
        "payee": payee_column,
        "memo": memo_column,
    }
    missing = [label for label, value in detected.items() if value is None]
    preview = df.head(15)
    return {
        "error": None,
        "preview": preview,
        "detected_columns": detected,
        "missing_columns": missing,
        "column_names": df.columns.tolist(),
        "row_count": int(len(df.index)),
    }


def render_unknown_vendor_code_panel(
    client_id: str,
    unknown_vendor_codes: List[Dict[str, Any]],
    *,
    key_prefix: str,
) -> None:
    if not unknown_vendor_codes:
        return

    st.warning(
        f"Found {len(unknown_vendor_codes)} unknown vendor code pattern(s). "
        "You can add a mapping rule now so future rows are categorized correctly."
    )

    panel_rows = []
    for item in unknown_vendor_codes:
        panel_rows.append(
            {
                "Vendor Code": item.get("vendor_code"),
                "Rows": item.get("count"),
                "Sample Descriptions": " | ".join(item.get("sample_descriptions") or []),
            }
        )
    st.dataframe(pd.DataFrame(panel_rows), use_container_width=True, hide_index=True)

    options = [str(item.get("vendor_code") or "").strip() for item in unknown_vendor_codes if str(item.get("vendor_code") or "").strip()]
    if not options:
        return

    with st.expander("Quick action: create vendor code mapping rule", expanded=False):
        selected_code = st.selectbox(
            "Vendor code",
            options=options,
            key=f"{key_prefix}_vendor_code_select",
        )
        mapping_account_name = st.text_input(
            "Category name for this code",
            key=f"{key_prefix}_vendor_account_name",
        )
        mapping_account_id = st.text_input(
            "Account identifier for this code",
            key=f"{key_prefix}_vendor_account_id",
        )
        mapping_category_type = st.text_input(
            "Category type",
            value="Expense",
            key=f"{key_prefix}_vendor_category_type",
        )
        mapping_logic = st.text_area(
            "Reason for this mapping",
            value=f"Vendor code {selected_code} mapping created from review panel",
            key=f"{key_prefix}_vendor_logic",
        )
        default_trigger = f"DIRECT PAY {selected_code}"
        mapping_trigger_words = st.text_input(
            "Trigger words (comma separated)",
            value=default_trigger,
            key=f"{key_prefix}_vendor_triggers",
        )
        if st.button("Create mapping rule", key=f"{key_prefix}_create_vendor_rule"):
            if not mapping_account_name.strip() or not mapping_account_id.strip():
                st.error("Please enter both category name and account identifier.")
            else:
                try:
                    body = rule_payload_from_inputs(
                        trigger_text=mapping_trigger_words,
                        account_id=mapping_account_id,
                        account_name=mapping_account_name,
                        category_type=mapping_category_type,
                        logic=mapping_logic,
                        vendor_code=selected_code,
                        ach_vendor_only=True,
                    )
                    response = api_post_json(f"/{client_id}/rules", body)
                    st.success(f"Created mapping rule for vendor code {selected_code}.")
                    st.json(response)
                except Exception as exc:
                    st.error(str(exc))


def build_review_export_bytes(df: pd.DataFrame) -> Dict[str, bytes]:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Reviewed Transactions", index=False)
    excel_bytes = excel_buffer.getvalue()
    return {"csv": csv_bytes, "excel": excel_bytes}


def compute_rule_conflicts(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trigger_to_accounts: Dict[str, set] = {}
    for rule in rules:
        account = str(rule.get("account_name") or rule.get("category_type") or "Unknown").strip() or "Unknown"
        for trigger in rule.get("trigger_payee_contains") or []:
            trigger_text = " ".join(str(trigger or "").upper().split())
            if not trigger_text:
                continue
            if trigger_text not in trigger_to_accounts:
                trigger_to_accounts[trigger_text] = set()
            trigger_to_accounts[trigger_text].add(account)

    conflicts = []
    for trigger_text, accounts in trigger_to_accounts.items():
        if len(accounts) > 1:
            conflicts.append(
                {
                    "Trigger": trigger_text,
                    "Conflicting Categories": ", ".join(sorted(accounts)),
                }
            )
    return conflicts


def build_transactions_dataframe(transactions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for txn in transactions:
        credit_amount = txn.get("credit")
        debit_amount = txn.get("debit")
        amount = credit_amount if credit_amount not in (None, "") else debit_amount
        rows.append(
            {
                "Date": txn.get("date"),
                "Description": txn.get("description"),
                "Amount": amount,
                "Credit": credit_amount,
                "Debit": debit_amount,
                "Category": txn.get("account_name"),
                "Source": txn.get("categorization_source"),
                "Confidence": txn.get("confidence_score"),
                "Needs Review": bool(txn.get("needs_review_flag")),
                "Review Reason": txn.get("needs_review_reason"),
                "Rule Id": txn.get("matched_rule_id"),
                "Trigger": txn.get("matched_trigger_text"),
                "Vendor Code": txn.get("vendor_code"),
                "Matched Via Memo": bool(txn.get("matched_via_memo")),
                "Fingerprint": txn.get("transaction_fingerprint"),
            }
        )
    return pd.DataFrame(rows)


def rule_payload_from_inputs(
    trigger_text: str,
    account_id: str,
    account_name: str,
    category_type: str,
    logic: str,
    vendor_code: str,
    ach_vendor_only: bool,
) -> Dict[str, Any]:
    triggers = [part.strip() for part in trigger_text.split(",") if part.strip()]
    return {
        "trigger_payee_contains": triggers,
        "account_id": account_id.strip(),
        "account_name": account_name.strip(),
        "category_type": category_type.strip() or "Expense",
        "logic": logic.strip(),
        "vendor_code": vendor_code.strip() or None,
        "ach_vendor_only": ach_vendor_only,
    }


def ensure_state():
    if "clients" not in st.session_state:
        st.session_state["clients"] = []
    if "last_runs" not in st.session_state:
        st.session_state["last_runs"] = []
    if "selected_run_id" not in st.session_state:
        st.session_state["selected_run_id"] = None
    if "selected_client_id" not in st.session_state:
        st.session_state["selected_client_id"] = ""
    if "client_rules" not in st.session_state:
        st.session_state["client_rules"] = []


ensure_state()

st.title("Bank Categorization Review Workspace")
st.caption("Review results, fix categories, and save to the client rule engine.")

with st.sidebar:
    st.subheader("Connection")
    st.write(f"API address: `{API_BASE_URL}`")
    if st.button("Reload client list", use_container_width=True):
        try:
            st.session_state["clients"] = load_clients()
            st.success("Client list reloaded")
        except Exception as exc:
            st.error(str(exc))

if not st.session_state["clients"]:
    try:
        st.session_state["clients"] = load_clients()
    except Exception as exc:
        st.error(f"Could not load clients: {exc}")

clients = st.session_state["clients"]
if not clients:
    st.warning("No clients with rules were found.")
    st.stop()

client_label_to_id = {
    f"{c.get('client_name')} ({c.get('client_id')}) - rules: {c.get('rule_count', 0)}": c.get("client_id")
    for c in clients
}

selected_label = st.selectbox("Client", list(client_label_to_id.keys()))
selected_client_id = client_label_to_id[selected_label]
st.session_state["selected_client_id"] = selected_client_id

selected_client = next((c for c in clients if c.get("client_id") == selected_client_id), {})
gl_start = selected_client.get("gl_start_date")
gl_end = selected_client.get("gl_end_date")
if gl_start or gl_end:
    st.info(f"Rules were generated from general ledger dates: {gl_start or 'Unknown'} to {gl_end or 'Unknown'}")

run_tab, review_tab, rules_tab = st.tabs([
    "Run Categorization",
    "Review and Save",
    "Client Rules",
])

with run_tab:
    st.subheader("Run categorization")
    user_id = st.text_input("User identifier", value=st.session_state.get("user_id", "review-user"))
    st.session_state["user_id"] = user_id

    statements = st.file_uploader(
        "Upload one or more bank statement files",
        type=["pdf", "csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
    )
    check_register = st.file_uploader(
        "Optional check register file",
        type=["csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=False,
    )
    if check_register is not None:
        check_preview = inspect_check_register(check_register)
        with st.expander("Check register preview", expanded=True):
            if check_preview.get("error"):
                st.error(check_preview["error"])
            else:
                st.write(f"Rows in file: {check_preview.get('row_count', 0)}")
                detected = check_preview.get("detected_columns") or {}
                col1, col2, col3 = st.columns(3)
                col1.write(f"Check number column: `{detected.get('check_number') or 'Not detected'}`")
                col2.write(f"Payee column: `{detected.get('payee') or 'Not detected'}`")
                col3.write(f"Memo column: `{detected.get('memo') or 'Not detected'}`")
                missing = check_preview.get("missing_columns") or []
                if missing:
                    st.warning(
                        "Missing expected check register columns: "
                        + ", ".join(missing)
                        + ". Check matching may be less accurate."
                    )
                st.dataframe(check_preview.get("preview"), use_container_width=True, hide_index=True)

    if st.button("Run categorization now", type="primary"):
        if not statements:
            st.error("Please upload at least one bank statement file.")
        else:
            with st.spinner("Categorizing transactions. This can take a few minutes for larger files."):
                try:
                    result = run_categorization(selected_client_id, user_id, statements, check_register)
                    st.session_state["last_categorize_response"] = result

                    summary = result.get("summary") or {}
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total", summary.get("total_transactions", 0))
                    c2.metric("Matched", summary.get("matched_transactions", 0))
                    c3.metric("Ask My Accountant", summary.get("unmatched_transactions", 0))
                    ama_rate = float(result.get("ama_rate") or summary.get("ama_rate") or 0.0)
                    if ama_rate > 40:
                        st.warning(
                            f"Ask My Accountant is {ama_rate:.1f}%. "
                            "This is high, so rule refresh or vendor code mapping review is recommended."
                        )

                    warning_messages = result.get("warnings") or []
                    if warning_messages:
                        st.warning("Run warnings")
                        for warning_message in warning_messages:
                            st.write(f"- {warning_message}")

                    retry_count = int(result.get("rate_limit_retries_used") or 0)
                    if retry_count > 0:
                        st.info(
                            f"Processing slowed down by service limits. "
                            f"Automatic retries were used {retry_count} time(s)."
                        )

                    render_unknown_vendor_code_panel(
                        selected_client_id,
                        result.get("unknown_vendor_codes") or [],
                        key_prefix="run_panel",
                    )

                    run_rows = result.get("statement_runs") or []
                    if run_rows:
                        st.success(f"Saved {len(run_rows)} statement run record(s).")
                        run_df = pd.DataFrame(run_rows)
                        st.dataframe(run_df, use_container_width=True)
                        first_run = run_rows[0].get("run_id")
                        if first_run:
                            st.session_state["selected_run_id"] = first_run

                    st.session_state["last_runs"] = load_user_statement_runs(user_id)
                except Exception as exc:
                    if is_rate_limit_message(str(exc)):
                        st.warning(
                            "Processing is being delayed by service limits. "
                            "Please wait and try again. The system already retries automatically when possible."
                        )
                    else:
                        st.error(str(exc))

with review_tab:
    st.subheader("Review and save")
    user_id_review = st.text_input("User identifier for saved runs", value=st.session_state.get("user_id", "review-user"), key="review_user_id")

    col_a, col_b = st.columns([1, 1])
    if col_a.button("Load saved runs", use_container_width=True):
        try:
            st.session_state["last_runs"] = load_user_statement_runs(user_id_review)
            st.success(f"Loaded {len(st.session_state['last_runs'])} saved run(s)")
        except Exception as exc:
            st.error(str(exc))

    runs = [r for r in st.session_state.get("last_runs", []) if r.get("client_id") == selected_client_id]
    if not runs:
        st.info("No saved runs found for this client and user.")
    else:
        run_options = {
            f"{r.get('run_id')} | {r.get('client_name')} | {r.get('matched_transactions', 0)}/{r.get('total_transactions', 0)} matched": r
            for r in runs
        }

        default_label = next(iter(run_options.keys()))
        if st.session_state.get("selected_run_id"):
            for label, rec in run_options.items():
                if rec.get("run_id") == st.session_state["selected_run_id"]:
                    default_label = label
                    break

        selected_run_label = st.selectbox("Choose saved run", list(run_options.keys()), index=list(run_options.keys()).index(default_label))
        selected_run = run_options[selected_run_label]
        st.session_state["selected_run_id"] = selected_run.get("run_id")

        transactions = selected_run.get("transactions") or []
        if not transactions:
            st.warning("This run has no stored transaction detail.")
        else:
            summary = {
                "total": selected_run.get("total_transactions", len(transactions)),
                "matched": selected_run.get("matched_transactions", 0),
                "unmatched": selected_run.get("unmatched_transactions", 0),
            }
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", summary["total"])
            c2.metric("Matched", summary["matched"])
            c3.metric("Ask My Accountant", summary["unmatched"])
            selected_run_ama_rate = float(selected_run.get("ama_rate") or 0.0)
            if selected_run_ama_rate > 40:
                st.warning(
                    f"Ask My Accountant is {selected_run_ama_rate:.1f}% for this run. "
                    "Please review uncategorized rows and refresh rules if needed."
                )

            selected_run_warnings = selected_run.get("warnings") or []
            if selected_run_warnings:
                st.warning("Run warnings")
                for warning_message in selected_run_warnings:
                    st.write(f"- {warning_message}")

            selected_run_retry_count = int(selected_run.get("rate_limit_retries_used") or 0)
            if selected_run_retry_count > 0:
                st.info(
                    f"This run used automatic retry {selected_run_retry_count} time(s) due to service limits."
                )

            render_unknown_vendor_code_panel(
                selected_client_id,
                selected_run.get("unknown_vendor_codes") or [],
                key_prefix=f"review_panel_{selected_run.get('run_id')}",
            )

            df = build_transactions_dataframe(transactions)
            unique_sources = sorted([s for s in df["Source"].dropna().unique().tolist() if s])

            filter_cols = st.columns([2, 2, 2, 3])
            selected_sources = filter_cols[0].multiselect("Source filter", options=unique_sources, default=unique_sources)
            needs_review_only = filter_cols[1].checkbox("Needs review only")
            search_text = filter_cols[2].text_input("Search text")
            sort_field = filter_cols[3].selectbox("Sort by", options=["Date", "Description", "Category", "Source", "Confidence"])

            filtered_df = df.copy()
            if selected_sources:
                filtered_df = filtered_df[filtered_df["Source"].isin(selected_sources)]
            if needs_review_only:
                filtered_df = filtered_df[filtered_df["Needs Review"] == True]
            if search_text.strip():
                mask = filtered_df["Description"].astype(str).str.contains(search_text, case=False, na=False)
                filtered_df = filtered_df[mask]

            filtered_df = filtered_df.sort_values(by=sort_field, ascending=True)
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            export_bytes = build_review_export_bytes(filtered_df)
            export_cols = st.columns(2)
            export_cols[0].download_button(
                "Download filtered CSV",
                data=export_bytes["csv"],
                file_name=f"reviewed_run_{selected_run.get('run_id')}_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
            export_cols[1].download_button(
                "Download filtered spreadsheet",
                data=export_bytes["excel"],
                file_name=f"reviewed_run_{selected_run.get('run_id')}_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

            row_choices = [
                (idx, f"{idx + 1}. {row['Date']} | {str(row['Description'])[:70]} | {row['Category']}")
                for idx, row in filtered_df.reset_index(drop=False).iterrows()
            ]
            if not row_choices:
                st.info("No rows match the current filters.")
            else:
                chosen_label = st.selectbox("Choose one transaction row to review", [label for _, label in row_choices])
                chosen_position = next(pos for pos, label in row_choices if label == chosen_label)
                chosen_row = filtered_df.reset_index(drop=False).iloc[chosen_position]

                fingerprint = chosen_row.get("Fingerprint")
                full_row = next(
                    (txn for txn in transactions if (txn.get("transaction_fingerprint") or "") == fingerprint),
                    None,
                )
                if not full_row:
                    st.error("Could not find full transaction row for the selected fingerprint.")
                else:
                    detail_col, action_col = st.columns([1.2, 1])

                    with detail_col:
                        st.markdown("### Transaction details")
                        st.json(full_row)

                        matched_rule_id = (full_row.get("matched_rule_id") or "").strip()
                        if matched_rule_id:
                            if st.button("Load matching rule details"):
                                try:
                                    one_rule = api_get(f"/{selected_client_id}/rules/{matched_rule_id}")
                                    st.json(one_rule)
                                except Exception as exc:
                                    st.error(str(exc))

                    with action_col:
                        st.markdown("### Save action")
                        action = st.radio(
                            "Choose save behavior",
                            options=[
                                "Save once for this statement",
                                "Update existing client rule",
                                "Create new client rule",
                            ],
                        )

                        new_category = st.text_input(
                            "Category name",
                            value=str(full_row.get("account_name") or ""),
                            key=f"cat_{fingerprint}",
                        )
                        new_category_type = st.text_input(
                            "Category type",
                            value=str(full_row.get("category_type") or "Expense"),
                            key=f"cat_type_{fingerprint}",
                        )

                        default_trigger = str(full_row.get("matched_trigger_text") or full_row.get("description") or "")
                        trigger_text = st.text_input(
                            "Trigger words (comma separated)",
                            value=default_trigger,
                            key=f"trg_{fingerprint}",
                        )
                        account_id = st.text_input(
                            "Account identifier",
                            value=str(full_row.get("account_id") or ""),
                            key=f"accid_{fingerprint}",
                        )
                        logic = st.text_area(
                            "Rule explanation",
                            value=str(full_row.get("logic") or "Created from review screen"),
                            key=f"logic_{fingerprint}",
                        )
                        vendor_code = st.text_input(
                            "Optional 4 digit vendor code for ACH vendor payments",
                            value=str(full_row.get("vendor_code") or ""),
                            key=f"vcode_{fingerprint}",
                        )
                        ach_vendor_only = st.checkbox(
                            "Restrict this rule to ACH vendor payment rows",
                            value=bool(full_row.get("ach_vendor_only") or False),
                            key=f"achonly_{fingerprint}",
                        )
                        note = st.text_area("Optional note", value="", key=f"note_{fingerprint}")

                        if st.button("Save selected action", type="primary", key=f"save_action_{fingerprint}"):
                            try:
                                if action == "Save once for this statement":
                                    body = {
                                        "transaction_fingerprint": fingerprint,
                                        "account_name": new_category,
                                        "category_type": new_category_type,
                                        "reason": note,
                                        "user_id": user_id_review,
                                    }
                                    response = api_post_json(
                                        f"/{selected_client_id}/statement-runs/{selected_run.get('run_id')}/overrides",
                                        body,
                                    )
                                    st.success("Saved one-time override")
                                    st.json(response)

                                elif action == "Update existing client rule":
                                    if not matched_rule_id:
                                        st.error("This row has no matched rule id, so update existing rule is not available.")
                                    else:
                                        body = rule_payload_from_inputs(
                                            trigger_text=trigger_text,
                                            account_id=account_id,
                                            account_name=new_category,
                                            category_type=new_category_type,
                                            logic=logic,
                                            vendor_code=vendor_code,
                                            ach_vendor_only=ach_vendor_only,
                                        )
                                        response = api_put_json(f"/{selected_client_id}/rules/{matched_rule_id}", body)
                                        st.success("Updated existing rule")
                                        st.json(response)

                                else:
                                    body = rule_payload_from_inputs(
                                        trigger_text=trigger_text,
                                        account_id=account_id,
                                        account_name=new_category,
                                        category_type=new_category_type,
                                        logic=logic,
                                        vendor_code=vendor_code,
                                        ach_vendor_only=ach_vendor_only,
                                    )
                                    response = api_post_json(f"/{selected_client_id}/rules", body)
                                    st.success("Created new rule")
                                    st.json(response)

                                st.session_state["last_runs"] = load_user_statement_runs(user_id_review)
                            except Exception as exc:
                                st.error(str(exc))

                        if st.button("Reapply latest rules to this run", key=f"reapply_{fingerprint}"):
                            try:
                                response = api_post_json(
                                    f"/{selected_client_id}/statement-runs/{selected_run.get('run_id')}/reapply",
                                    {},
                                )
                                st.success("Reapplied latest rules to this run")
                                st.json(response.get("summary"))
                                st.session_state["last_runs"] = load_user_statement_runs(user_id_review)
                            except Exception as exc:
                                st.error(str(exc))

with rules_tab:
    st.subheader("Client rule list")

    if st.button("Load rules for selected client"):
        try:
            payload = api_get(f"/{selected_client_id}")
            data = payload.get("data") or {}
            rules = data.get("rules") or []
            st.session_state["client_rules"] = rules
            st.session_state["client_rule_conflicts"] = compute_rule_conflicts(rules)
        except Exception as exc:
            st.error(str(exc))

    rules = st.session_state.get("client_rules", [])
    if rules:
        rules_df = pd.DataFrame(
            [
                {
                    "Rule Id": r.get("rule_id"),
                    "Account": r.get("account_name"),
                    "Account Id": r.get("account_id"),
                    "Category Type": r.get("category_type"),
                    "Vendor Code": r.get("vendor_code"),
                    "ACH Vendor Only": r.get("ach_vendor_only"),
                    "Triggers": ", ".join(r.get("trigger_payee_contains") or []),
                    "Logic": r.get("logic"),
                }
                for r in rules
            ]
        )
        rule_filter_cols = st.columns([2, 2, 2])
        trigger_search_text = rule_filter_cols[0].text_input("Search trigger text")
        account_search_text = rule_filter_cols[1].text_input("Search account")
        all_category_types = sorted(
            [str(value) for value in rules_df["Category Type"].dropna().unique().tolist() if str(value).strip()]
        )
        selected_category_types = rule_filter_cols[2].multiselect(
            "Category type filter",
            options=all_category_types,
            default=all_category_types,
        )

        filtered_rules_df = rules_df.copy()
        if selected_category_types:
            filtered_rules_df = filtered_rules_df[filtered_rules_df["Category Type"].isin(selected_category_types)]
        if trigger_search_text.strip():
            trigger_mask = filtered_rules_df["Triggers"].astype(str).str.contains(trigger_search_text, case=False, na=False)
            filtered_rules_df = filtered_rules_df[trigger_mask]
        if account_search_text.strip():
            account_mask = filtered_rules_df["Account"].astype(str).str.contains(account_search_text, case=False, na=False)
            filtered_rules_df = filtered_rules_df[account_mask]

        st.dataframe(filtered_rules_df, use_container_width=True, hide_index=True)

        conflicts = st.session_state.get("client_rule_conflicts") or compute_rule_conflicts(rules)
        if conflicts:
            st.warning(
                f"Generation health conflict alert: {len(conflicts)} trigger phrase(s) map to more than one category."
            )
            st.dataframe(pd.DataFrame(conflicts), use_container_width=True, hide_index=True)
        else:
            st.success("No rule trigger conflicts were found.")
    else:
        st.info("Load rules to see the current rule list.")
