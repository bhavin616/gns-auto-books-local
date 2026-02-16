"""Shared CSV helpers for categorized bank statement output."""

from typing import Any, List, Dict
import json


def escape_csv_value(value: Any) -> str:
    """Escape CSV value to handle commas, quotes, and newlines."""
    if value is None:
        return ""
    value_str = str(value)
    if "," in value_str or '"' in value_str or "\n" in value_str:
        return '"' + value_str.replace('"', '""') + '"'
    return value_str


CSV_HEADER = (
    "Date,Description,Credit,Debit,Category,Source,Payee,Bank,AccountNumber,"
    "Confidence,NeedsReview,ReviewReason,RuleId,Trigger,VendorCode,MatchedViaMemo"
)


def build_categorized_csv_rows(
    transactions: List[Dict[str, Any]],
    bank: str,
    account_number: str,
    *,
    category_from_matched: bool = False,
) -> List[str]:
    """
    Build header + CSV rows for categorized transactions.
    category_from_matched: if True, use matched_categories (list) and json.dumps; else use account_name.
    """
    rows = [CSV_HEADER]
    for txn in transactions:
        if category_from_matched:
            matched = txn.get("matched_categories") or []
            if not matched:
                category = txn.get("account_name") or "Uncategorized"
            elif len(matched) == 1:
                category = (matched[0].get("account_name") or "").strip() or "Uncategorized"
            else:
                category = json.dumps([(m.get("account_name") or "").strip() for m in matched])
        else:
            category = txn.get("account_name") or "Uncategorized"
        source = txn.get("categorization_source") or ""
        payee = txn.get("payee_name") or ""
        confidence = txn.get("confidence_score")
        needs_review = bool(txn.get("needs_review_flag"))
        review_reason = txn.get("needs_review_reason") or ""
        matched_rule_id = txn.get("matched_rule_id") or ""
        matched_trigger_text = txn.get("matched_trigger_text") or ""
        vendor_code = txn.get("vendor_code") or ""
        matched_via_memo = bool(txn.get("matched_via_memo"))
        rows.append(
            f"{escape_csv_value(txn.get('date', ''))},"
            f"{escape_csv_value(txn.get('description', ''))},"
            f"{escape_csv_value(txn.get('credit', ''))},"
            f"{escape_csv_value(txn.get('debit', ''))},"
            f"{escape_csv_value(category)},"
            f"{escape_csv_value(source)},"
            f"{escape_csv_value(payee)},"
            f"{escape_csv_value(bank)},"
            f"{escape_csv_value(account_number)},"
            f"{escape_csv_value(confidence)},"
            f"{escape_csv_value(needs_review)},"
            f"{escape_csv_value(review_reason)},"
            f"{escape_csv_value(matched_rule_id)},"
            f"{escape_csv_value(matched_trigger_text)},"
            f"{escape_csv_value(vendor_code)},"
            f"{escape_csv_value(matched_via_memo)}"
        )
    return rows
