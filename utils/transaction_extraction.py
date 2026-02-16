import os
import re
from datetime import datetime
from dateutil import parser
from typing import List, Dict, Optional
from backend.utils.logger import log_message


def read_pdf_text_with_pymupdf(pdf_path: str) -> str:
    """
    Reads PDF text using PyMuPDF (most reliable for bank statements).
    pip install pymupdf
    """
    import fitz
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    return "\n".join(text)


def _parse_statement_period(text: str):
    """Extracts statement period to infer year."""
    m = re.search(
        r"([A-Za-z]+\s*\d{1,2},\s*\d{4})\s*-\s*([A-Za-z]+\s*\d{1,2},\s*\d{4})",
        text, re.IGNORECASE
    )
    if not m:
        return None, None, None
    start = parser.parse(m.group(1))
    end = parser.parse(m.group(2))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), start.year


def _ymd_from_mmdd(mmdd: str, year: Optional[int]) -> Optional[str]:
    if not mmdd or not year:
        return None
    try:
        mm, dd = map(int, mmdd.split("/"))
        return datetime(year, mm, dd).strftime("%Y-%m-%d")
    except Exception:
        return None


def _clean(s: str) -> str:
    """Normalize inner whitespace while keeping content intact."""
    return re.sub(r"\s+", " ", s.strip())


def extract_transactions_from_wf_text(text: str) -> Dict[str, object]:
    """
    Parses Wells Fargo bank statement text into structured transactions.
    Works for both Credits and Debits, even without '<' symbols.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    joined = " ".join(lines)

    # --- Bank metadata ---
    bank_name = "Wells Fargo Bank, N.A."
    acct_match = re.search(r"Account\s*number[:\s]*([0-9*]{6,})", joined, re.IGNORECASE)
    account_number = acct_match.group(1) if acct_match else None
    start_date, end_date, year = _parse_statement_period(joined)

    # --- Regex Patterns ---
    section = None  # 'credit' | 'debit' | 'checks'
    inline_row_re = re.compile(
        r"^\s*(\d{2}/\d{2})"                # posted date
        r"(?:\s+(\d{2}/\d{2}))?"            # optional effective date
        r"\s+([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})"  # amount
        r"\s*(.*)$"
    )
    date_only_re = re.compile(r"^\s*(\d{2}/\d{2})\s*$")
    amount_detail_re = re.compile(r"^\s*([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})\s+(.*)$")

    # Section Headers
    credits_header_re = re.compile(r"^\s*Credits\s*$", re.IGNORECASE)
    debits_header_re = re.compile(r"^\s*Debits\s*$", re.IGNORECASE)
    checks_header_re = re.compile(r"^\s*Checks\s+paid\s*$", re.IGNORECASE)

    skip_re = re.compile(
        r"(Electronic deposits/bank credits|Electronic debits/bank debits|Transaction detail|"
        r"Total credits|Total debits|Daily ledger balance summary|Interest summary|"
        r"Page\s+\d+\s+of\s+\d+|All rights reserved|Member FDIC|©20\d{2}|Sheet Seq)",
        re.IGNORECASE,
    )

    transactions: List[Dict[str, object]] = []
    current_txn = None
    pending_date = None

    def flush():
        nonlocal current_txn
        if current_txn:
            current_txn["description"] = _clean(current_txn["description"])
            transactions.append(current_txn)
            current_txn = None

    def section_type(line):
        if credits_header_re.match(line):
            return "credit"
        if debits_header_re.match(line):
            return "debit"
        if checks_header_re.match(line):
            return "checks"
        return None

    i = 0
    N = len(lines)

    while i < N:
        line = lines[i].strip()

        # --- Detect section ---
        s_type = section_type(line)
        if s_type:
            flush()
            section = s_type
            i += 1
            continue

        if skip_re.search(line) or not line:
            i += 1
            continue

        # --- Handle "Checks paid" section ---
        if section == "checks":
            m_chk = re.match(
                r"^\s*(\d{3,6})\s+([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})\s+(\d{2}/\d{2})\s*$", line
            )
            if m_chk:
                flush()
                chk, amt_txt, posted_mmdd = m_chk.groups()
                amt = float(amt_txt.replace(",", ""))
                posted = _ymd_from_mmdd(posted_mmdd, year)
                current_txn = {
                    "date": None,
                    "post_date": posted,
                    "credit": None,
                    "debit": amt,
                    "description": chk,  # keep literal check number
                    "bank": bank_name,
                }
                flush()
            i += 1
            continue

        # --- Inline transaction: MM/DD [MM/DD] AMOUNT Detail ---
        m_inline = inline_row_re.match(line)
        if m_inline:
            flush()
            posted_mmdd, effective_mmdd, amt_txt, detail = m_inline.groups()
            amt = float(amt_txt.replace(",", ""))
            post_date = _ymd_from_mmdd(posted_mmdd, year)
            date = _ymd_from_mmdd(effective_mmdd, year) if effective_mmdd else None
            credit, debit = (amt, None) if section == "credit" else (None, amt)

            # Collect continuation lines
            details = [detail] if detail else []
            j = i + 1
            while j < N:
                nxt = lines[j].strip()
                if (
                    inline_row_re.match(nxt)
                    or date_only_re.match(nxt)
                    or section_type(nxt)
                    or skip_re.search(nxt)
                ):
                    break
                details.append(nxt)
                j += 1

            current_txn = {
                "date": date,
                "post_date": post_date,
                "credit": credit,
                "debit": debit,
                "description": " ".join(details),
                "bank": bank_name,
            }
            flush()
            i = j
            continue

        # --- Date-only + amount line pattern ---
        m_date = date_only_re.match(line)
        if m_date and i + 1 < N:
            look = lines[i + 1].strip()
            m_amt = amount_detail_re.match(look)
            if m_amt:
                flush()
                posted_mmdd = m_date.group(1)
                amt_txt, detail = m_amt.groups()
                amt = float(amt_txt.replace(",", ""))
                post_date = _ymd_from_mmdd(posted_mmdd, year)
                credit, debit = (amt, None) if section == "credit" else (None, amt)

                details = [detail]
                j = i + 2
                while j < N:
                    nxt = lines[j].strip()
                    if (
                        inline_row_re.match(nxt)
                        or date_only_re.match(nxt)
                        or section_type(nxt)
                        or skip_re.search(nxt)
                    ):
                        break
                    details.append(nxt)
                    j += 1

                current_txn = {
                    "date": None,
                    "post_date": post_date,
                    "credit": credit,
                    "debit": debit,
                    "description": " ".join(details),
                    "bank": bank_name,
                }
                flush()
                i = j
                continue

        # --- Amount-start line in debit section (NO '<' required) ---
        if section == "debit" and re.match(r"^(?!Total)[0-9]{1,3}(?:,[0-9]{3})*\.\d{2}\b", line):
            flush()
            amt_txt = re.findall(r"([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})", line)[0]
            amt = float(amt_txt.replace(",", ""))
            detail = re.sub(r"^[0-9]{1,3}(?:,[0-9]{3})*\.\d{2}\s*", "", line)
            credit, debit = (None, amt)
            post_date = pending_date if pending_date else None
            pending_date = None

            details = [detail]
            j = i + 1
            while j < N:
                nxt = lines[j].strip()
                if (
                    inline_row_re.match(nxt)
                    or date_only_re.match(nxt)
                    or section_type(nxt)
                    or skip_re.search(nxt)
                ):
                    break
                details.append(nxt)
                j += 1

            current_txn = {
                "date": None,
                "post_date": _ymd_from_mmdd(post_date, year) if post_date else None,
                "credit": credit,
                "debit": debit,
                "description": " ".join(details),
                "bank": bank_name,
            }
            flush()
            i = j
            continue

        # --- Keep stray dates for next line ---
        if date_only_re.match(line):
            pending_date = date_only_re.match(line).group(1)
            i += 1
            continue

        i += 1

    flush()

    return {
        "bank": bank_name,
        "account_number": account_number,
        "start_date": start_date,
        "end_date": end_date,
        "transactions": transactions,
    }


# -----------------------------
# Save to Excel
# -----------------------------
def save_transactions_to_excel(response: Dict[str, object], path: str = "transactions.xlsx"):
    import pandas as pd
    txns = response.get("transactions", [])
    if not txns:
        log_message("warning", "save_transactions_to_excel: No transactions to save")
        return
    df = pd.DataFrame(txns)
    df = df[["date", "post_date", "description", "credit", "debit", "bank"]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)
    log_message("info", f"save_transactions_to_excel: Saved {len(df)} transactions to {path}")




# file_path = "Bank Statement-20251017T133837Z-1-001/Bank Statement/_123124 WellsFargo.pdf"
# file_name = os.path.basename(file_path).split("/")[-1].split(".")[0]

# text = read_pdf_text_with_pymupdf(file_path)
# For testing, you can pass the text directly if you already extracted it.
# response = extract_transactions_from_wf_text(text)
# save_transactions_to_excel(response, f"utils/output/{file_name}.xlsx")
