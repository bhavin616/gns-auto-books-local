import os
import re
import fitz
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from typing import List, Dict, Optional, Tuple


def read_pdf_text_with_pymupdf(pdf_path: str) -> str:
    """
    Reads PDF text using PyMuPDF (most reliable for bank statements).
    pip install pymupdf
    """
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text("text"))

    # Specify the file name
    filename = "data/test_output/pdf_extracted_text.txt"

    # Open the file in write mode and save the text
    with open(filename, "w", encoding="utf-8") as file:
        file.write("\n".join(text))
    return "\n".join(text)


def read_pdf_text_sorted(pdf_path: str) -> str:
    """
    Uses PyMuPDF to extract text blocks sorted by Y position.
    Fixes out-of-order multi-column issues in bank statements.
    """
    all_blocks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, ...)
            # Sort by Y coordinate, then X to preserve reading order
            blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
            for b in blocks:
                all_blocks.append(b[4].strip())

    joined_text = "\n".join(all_blocks)
    with open("data/test_output/pdf_extracted_text_sorted.txt", "w", encoding="utf-8") as f:
        f.write(joined_text)
    return joined_text


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


# --------------------------------------------------------------------------------------------

def parse_mmdd(mmdd: Optional[str], year: Optional[str]) -> Optional[str]:
    if not mmdd or not year:
        return None  # Handle the case where mmdd or year is None

    try:
        mm, dd = mmdd.split("/")
        return f"{year}-{mm}-{dd}"  # Format it as "YYYY-MM-DD"
    except ValueError:
        return None  # Handle the case where mmdd is not in "MM/DD" format


# --------------------------------------------------------------------------------------------


def extract_transactions_from_wf_text_us_roadways(text: str) -> Dict[str, object]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    joined = " ".join(lines)

    bank_name = "Wells Fargo Bank, N.A."
    acct_match = re.search(r"Account\s*number[:\s]*([0-9*]{6,})", joined, re.IGNORECASE)
    account_number = acct_match.group(1) if acct_match else None
    start_date, end_date, year = _parse_statement_period(joined)

    section = None
    inline_row_re = re.compile(
        r"^\s*(\d{2}/\d{2})(?:\s+(\d{2}/\d{2}))?\s+([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})\s*(.*)$"
    )
    date_only_re = re.compile(r"^\s*(\d{2}/\d{2})\s*$")
    amount_detail_re = re.compile(r"^\s*([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})\s+(.*)$")

    credits_header_re = re.compile(r"^\s*Credits\s*$", re.IGNORECASE)
    debits_header_re = re.compile(r"^\s*Debits\s*$", re.IGNORECASE)
    checks_header_re = re.compile(r"^\s*Checks\s+paid\s*$", re.IGNORECASE)

    # ✅ FIX: Limit skip_re only to headers / footers, not totals or ledger summaries
    skip_re = re.compile(
        r"(Electronic deposits/bank credits|Electronic debits/bank debits|Transaction detail|"
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

        s_type = section_type(line)
        if s_type:
            flush()
            section = s_type
            i += 1
            continue

        if skip_re.search(line) or not line:
            i += 1
            continue

        # ✅ FIX 1: Explicitly skip total lines (but don’t break early)
        if re.search(r"^Total\s+(credits|debits|checks)", line, re.IGNORECASE):
            i += 1
            continue

        # ✅ FIX 2: Skip known ledger/summary blocks at the very end
        if re.search(r"^Daily\s+Ledger\s+Balance\s+Summary", line, re.IGNORECASE):
            break

        # --- rest of your original working code below unchanged ---

        # Inline transaction
        m_inline = inline_row_re.match(line)
        if m_inline:
            flush()
            posted_mmdd, effective_mmdd, amt_txt, detail = m_inline.groups()
            amt = float(amt_txt.replace(",", ""))
            post_date = _ymd_from_mmdd(posted_mmdd, year)
            date = _ymd_from_mmdd(effective_mmdd, year) if effective_mmdd else None
            credit, debit = (amt, None) if section == "credit" else (None, amt)

            details = [detail] if detail else []
            j = i + 1
            while j < N:
                nxt = lines[j].strip()

                # 🩵 NEW FIX — stop if "Total" lines encountered
                if re.match(r"^Total\s+(credits|debits|checks)", nxt, re.IGNORECASE):
                    break

                # 🩵 Also stop if it’s just a numeric total line
                if re.match(r"^\$?[0-9]{1,3}(?:,[0-9]{3})*\.\d{2}\s*$", nxt):
                    break

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

        # Date-only + amount pattern
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

                    # 🩵 NEW FIX — stop if "Total" lines encountered
                    if re.match(r"^Total\s+(credits|debits|checks)", nxt, re.IGNORECASE):
                        break

                    # 🩵 Also stop if it’s just a numeric total line
                    if re.match(r"^\$?[0-9]{1,3}(?:,[0-9]{3})*\.\d{2}\s*$", nxt):
                        break

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

        # Amount-start in debit section
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

                # 🩵 NEW FIX — stop if "Total" lines encountered
                if re.match(r"^Total\s+(credits|debits|checks)", nxt, re.IGNORECASE):
                    break

                # 🩵 Also stop if it’s just a numeric total line
                if re.match(r"^\$?[0-9]{1,3}(?:,[0-9]{3})*\.\d{2}\s*$", nxt):
                    break

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

        # Keep pending date
        if date_only_re.match(line):
            pending_date = date_only_re.match(line).group(1)
            i += 1
            continue

        i += 1

    flush()

    # ✅ FIX 3: Final cleanup of stray totals/ledger descriptions
    cleaned_txns = [
        t for t in transactions
        if not re.search(r"(Total\s+(credits|debits|checks)|Daily\s+Ledger|Summary\s+of)", t["description"], re.IGNORECASE)
    ]

    return {
        "bank": bank_name,
        "account_number": account_number,
        "start_date": start_date,
        "end_date": end_date,
        "transactions": cleaned_txns,
    }

# --------------------------------------------------------------------------------------------


def extract_transactions_from_wf_text_barbar_llc(text: str) -> Dict[str, object]:
    """
    Robust parser for Wells Fargo 'Barbar LLC' combined statements.
    Handles multiple accounts, checks, and ignores summary sections.
    """
    bank_name = "Wells Fargo Bank, N.A."

    raw_lines = [ln.rstrip("\n") for ln in text.splitlines()]
    joined_for_dates = " ".join(ln.strip() for ln in raw_lines if ln.strip())

    start_date, end_date, year = _parse_statement_period(joined_for_dates)

    # ---------- Regex patterns ----------
    acct_hdr_re = re.compile(r"Account\s*number:\s*(\d{4,})", re.IGNORECASE)
    tx_hist_start_re = re.compile(r"Transaction\s+history(?:\s*\(continued\))?", re.IGNORECASE)

    # STOP sections (be aggressive: search anywhere in line)
    end_block_re = re.compile(
        r"(Summary of checks|Summary of checks written|Monthly service fee summary|"
        r"Account transaction fees summary|The Ending Daily Balance does not reflect|"
        r"\bEnding balance on\b|\bTotals\b|Fee period|How to avoid the monthly service fee)",
        re.IGNORECASE,
    )

    header_word_re = re.compile(
        r"^(Deposits/|Withdrawals/|Ending|Date|Number|Description|Credits|Debits|balance)$",
        re.IGNORECASE,
    )
    page_footer_re = re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE)
    fluff_re = re.compile(r"(All rights reserved|Member FDIC|©20\d{2}|Sheet Seq)", re.IGNORECASE)

    date_only_re = re.compile(r"^(\d{1,2}/\d{1,2})\s*$")
    amount_only_re = re.compile(r"^\$?[0-9]{1,3}(?:,[0-9]{3})*\.\d{2}\s*$")
    amount_re = re.compile(r"\$?([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})")
    check_num_re = re.compile(r"^\d{3,6}(\s*\*)?$")  # allow trailing * in summary table

    # keyword rules
    debit_kw = re.compile(
        r"(\bonline\s+transfer\s+to\b|\bcheck\b|\bpurchase\b|\bpayment\b|\bwithdrawal\b|"
        r"\bfee\b|\btax\b|\birs\b|\bdebit\b|\bchargeback\b|\bpaychex\b|\bpayroll\b)",
        re.IGNORECASE,
    )
    credit_kw = re.compile(
        r"(\brecurring\s+transfer\s+from\b|\bdeposit\b|\bcredit\b|\brefund\b|\binterest\b|"
        r"\bonline\s+transfer\s+from\b)",
        re.IGNORECASE,
    )

    def _to_ymd(mmdd: Optional[str]) -> Optional[str]:
        return _ymd_from_mmdd(mmdd, year) if (mmdd and year) else None

    # --- Find each account block ---
    def find_account_blocks(txt: str) -> List[Tuple[str, int, int]]:
        matches = list(acct_hdr_re.finditer(txt))
        blocks = []
        for i, m in enumerate(matches):
            acct = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
            blocks.append((acct, start, end))
        return blocks

    # --- Hard-truncate an account block at first end marker (prevents bleed) ---
    def truncate_at_end_marker(block_text: str) -> str:
        for ln in block_text.splitlines():
            if end_block_re.search(ln):
                # slice up to, but not including, the line that triggered end
                cut = block_text.find(ln)
                return block_text[:cut]
        return block_text

    # --- Collect transaction tables per account ---
    def collect_all_tx_tables(block_text: str) -> List[List[str]]:
        # hard truncate the whole block first
        block_text = truncate_at_end_marker(block_text)

        lines = [ln.strip() for ln in block_text.splitlines()]
        tables = []
        capturing = False
        buf = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            if tx_hist_start_re.search(ln):
                if buf:
                    tables.append(buf)
                    buf = []
                capturing = True
                i += 1
                continue

            if capturing:
                # stop capturing immediately if any end-of-section markers appear
                if end_block_re.search(ln):
                    if buf:
                        tables.append(buf)
                        buf = []
                    capturing = False
                    i += 1
                    continue

                # skip table headers/footers/fluff
                if header_word_re.match(ln) or page_footer_re.search(ln) or fluff_re.search(ln):
                    i += 1
                    continue

                # accumulate
                buf.append(ln)
                i += 1
                continue

            i += 1
        if buf:
            tables.append(buf)
        return tables

    # --- Parse tables to transactions ---
    def parse_tables_to_transactions(tables: List[List[str]], acct_number: str) -> List[Dict[str, object]]:
        out = []
        for table in tables:
            current_date = None
            group_lines = []
            seen_end = False

            def flush_group():
                nonlocal current_date, group_lines
                if not current_date or not group_lines:
                    current_date, group_lines = None, []
                    return

                glines = [g for g in group_lines if g.strip()]
                # if any line inside the group is an end marker, drop the group
                if any(end_block_re.search(g) for g in glines):
                    current_date, group_lines = None, []
                    return

                # coalesce "check number + 'Check'" into a single description token
                desc_parts = []
                i = 0
                while i < len(glines):
                    # guard against summary table: "1311", "Check" often appear in true history too,
                    # but we will never be here if we truncated correctly.
                    if i + 1 < len(glines) and check_num_re.match(glines[i]) and glines[i + 1].lower() == "check":
                        desc_parts.append(f"Check {glines[i].replace('*','').strip()}")
                        i += 2
                        continue
                    desc_parts.append(glines[i])
                    i += 1

                # find amounts in desc parts
                amounts = []
                for idx, line in enumerate(desc_parts):
                    for m in amount_re.finditer(line):
                        try:
                            amounts.append((float(m.group(1).replace(",", "")), idx))
                        except Exception:
                            pass
                if not amounts:
                    current_date, group_lines = None, []
                    return

                # choose transaction amount (not ending balance)
                txn_amount_val = txn_amount_idx = None
                if len(amounts) >= 2:
                    last_val, last_idx = amounts[-1]
                    # last pure-amount line is typically ending daily balance → use the previous one
                    if amount_only_re.match(desc_parts[last_idx]):
                        txn_amount_val, txn_amount_idx = amounts[-2]
                    else:
                        txn_amount_val, txn_amount_idx = amounts[0]
                else:
                    txn_amount_val, txn_amount_idx = amounts[0]

                # strip out pure amount lines; also remove the chosen amount text from its line
                kept_lines = []
                for idx, line in enumerate(desc_parts):
                    if amount_only_re.match(line):
                        continue
                    if idx == txn_amount_idx:
                        line_wo_amt = amount_re.sub("", line).strip()
                        if line_wo_amt:
                            kept_lines.append(line_wo_amt)
                        continue
                    kept_lines.append(amount_re.sub("", line).strip())
                description = _clean(" ".join(ln for ln in kept_lines if ln))

                # classify credit/debit
                credit = debit = None
                if txn_amount_val is not None:
                    if credit_kw.search(description) and not debit_kw.search(description):
                        credit = txn_amount_val
                    elif debit_kw.search(description) and not credit_kw.search(description):
                        debit = txn_amount_val
                    else:
                        if re.search(r"\bfrom\s+barbar\b", description, re.IGNORECASE):
                            credit = txn_amount_val
                        elif re.search(r"\bto\s+barbar\b", description, re.IGNORECASE):
                            debit = txn_amount_val
                        elif description.lower().startswith("check "):
                            debit = txn_amount_val
                        elif "deposit" in description.lower():
                            credit = txn_amount_val
                        elif "refund" in description.lower() or "interest" in description.lower():
                            credit = txn_amount_val
                        else:
                            debit = txn_amount_val

                if credit is not None or debit is not None:
                    out.append(
                        {
                            "date": None,
                            "post_date": _to_ymd(current_date),
                            "description": description,
                            "credit": credit,
                            "debit": debit,
                            "bank": bank_name,
                            "account_number": acct_number,
                        }
                    )
                current_date, group_lines = None, []

            for ln in table:
                if seen_end:
                    break
                # stop immediately on any end marker inside a table
                if end_block_re.search(ln):
                    flush_group()
                    seen_end = True
                    break

                md = date_only_re.match(ln)
                if md:
                    flush_group()
                    current_date = md.group(1)
                    group_lines = []
                    continue

                if current_date:
                    # ignore headers/footers/fluff
                    if header_word_re.match(ln) or page_footer_re.search(ln) or fluff_re.search(ln):
                        continue
                    group_lines.append(ln)

            flush_group()

        return out

    transactions = []
    for acct_number, start_idx, end_idx in find_account_blocks(text):
        block = text[start_idx:end_idx]
        tables = collect_all_tx_tables(block)
        if not tables:
            continue
        transactions.extend(parse_tables_to_transactions(tables, acct_number))

    # --- Final clean ---
    cleaned = []
    for t in transactions:
        if not (t.get("credit") or t.get("debit")):
            continue
        if not t.get("post_date"):
            continue
        if not t.get("description"):
            continue
        cleaned.append(t)

    return {
        "bank": bank_name,
        "account_number": None,  # multiple accounts are included on per-row 'account_number'
        "start_date": start_date,
        "end_date": end_date,
        "transactions": cleaned,
    }

# ----------------------------------------------------------------------------------------------------------------------


def extract_transactions_from_chase_text(text: str) -> Dict[str, object]:
    """
    Ultra-robust Chase parser for 'pdf_extracted_text_sorted.txt'-style input
    (with *start*/*end* section markers).
    Captures:
      - DEPOSITS AND ADDITIONS  (credits)
      - CHECKS PAID             (debits)
      - ATM & DEBIT CARD WITHDRAWALS (debits)
      - ELECTRONIC WITHDRAWALS  (debits)
      - OTHER WITHDRAWALS       (debits)
      - FEES                    (debits/charges)
    Ignores summaries, totals, footers, and continues seamlessly across pages.
    """

    bank_name = "JPMorgan Chase Bank, N.A."

    # ---------- Helpers ----------
    def _clean(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        # never allow a description to be just a total row
        if re.match(r"^Total\b", s, flags=re.IGNORECASE):
            return ""
        return s

    # rigid date-only line: '01/24'
    date_only_re = re.compile(r"^\s*(\d{1,2}/\d{1,2})\s*$")
    # amount-only line: '$233.54' or '233.54'
    amount_only_re = re.compile(r"^\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})\s*$")

    # Lines that are pure noise / headers / footers (to prevent cross-page bleed)
    skip_re = re.compile(
        r"^(?:"
        r"Page\b.*|^\d+\s+of\s+\d+|of\s+\d+|"
        r"DAILY\s+ENDING\s+BALANCE|ATM\s*&\s*DEBIT\s*CARD\s*SUMMARY|CHECKING SUMMARY|"
        r"IN CASE OF ERRORS|CUSTOMER SERVICE INFORMATION|"
        r"Member FDIC|JPMorgan Chase Bank|Web site:|Service Center:|"
        r"January\s+\d{1,2},?\s+\d{4}|"
        r"Account Number:|Deaf and Hard of Hearing:|International Calls:|Para Espanol:"
        r")$",
        re.IGNORECASE
    )

    # Section tags present in your sorted text file (do not change)
    SECTION_TAGS = {
        "deposits": ("*start*deposits and additions", "*end*deposits and additions"),
        "checks": ("*start*checks paid section3", "*end*checks paid section3"),
        "atm_debit": ("*start*atm debit withdrawal", "*end*atm debit withdrawal"),
        "electronic": ("*start*electronic withdrawal", "*end*electronic withdrawal"),
        "other_withdrawals": ("*start*other withdrawals", "*end*other withdrawals"),
        "fees": ("*start*fees section", "*end*fees section"),  # Added Fees section
    }

    # -------- Section I/O --------
    def extract_section_blocks(tag_start: str, tag_end: str) -> List[str]:
        """Return a list of raw blocks for this section; there can be multiple blocks (continued pages)."""
        blocks = []
        pos = 0
        while True:
            sidx = text.find(tag_start, pos)
            if sidx == -1:
                break
            eidx = text.find(tag_end, sidx)
            if eidx == -1:
                # no closing tag; take until end (defensive)
                blocks.append(text[sidx + len(tag_start):])
                break
            blocks.append(text[sidx + len(tag_start):eidx])
            pos = eidx + len(tag_end)
        return blocks

    def normalize_block(block: str) -> List[str]:
        """Strip nested tags, headers/footers, and stop cleanly at first 'Total ...' line."""
        # strip nested control tags if any
        block = re.sub(r"\*start\*.*?$", "", block, flags=re.MULTILINE)
        block = re.sub(r"\*end\*.*?$", "", block, flags=re.MULTILINE)

        raw_lines = [ln.strip() for ln in block.splitlines()]
        lines: List[str] = []
        for ln in raw_lines:
            if not ln:
                continue
            # drop obvious headers/footers (prevents continuation-page bleed)
            if skip_re.match(ln):
                continue
            # stop at totals for this block (prevents last txn picking up totals)
            if re.match(r"^\s*Total\b", ln, flags=re.IGNORECASE):
                break
            # ignore column headers and "(continued)"
            if re.match(r"^(DEPOSITS|AND|ADDITIONS|ELECTRONIC|WITHDRAWALS|DATE|DESCRIPTION|AMOUNT|CHECK NO\.?|PAID|\(continued\))$", ln, flags=re.IGNORECASE):
                continue
            lines.append(ln)
        return lines

    # ---------- Parsers ----------
    def parse_date_desc_amount_by_blocks(lines: List[str], is_credit: bool) -> List[Dict[str, object]]:
        """
        Generic parser for sections that look like:
            DATE
            description (1..N lines)
            AMOUNT (pure amount line)
        Stops at 'Total ...' (already removed in normalize_block).
        """
        txns: List[Dict[str, object]] = []
        cur_date: Optional[str] = None
        buf: List[str] = []

        def flush():
            nonlocal cur_date, buf
            if not cur_date or not buf:
                cur_date, buf = None, []
                return
            # find the LAST amount-only line (ignore any line that mentions 'Total')
            amt_val: Optional[float] = None
            amt_idx: Optional[int] = None
            for i in range(len(buf) - 1, -1, -1):
                if amount_only_re.match(buf[i]) and not re.search(r"\bTotal\b", buf[i], flags=re.IGNORECASE):
                    amt_val = float(amount_only_re.match(buf[i]).group(1).replace(",", ""))
                    amt_idx = i
                    break
            if amt_val is None:
                cur_date, buf = None, []
                return
            desc_lines = [buf[i] for i in range(len(buf)) if i != amt_idx]
            desc = _clean(" ".join(desc_lines))
            if not desc:
                cur_date, buf = None, []
                return
            txns.append({
                "date": cur_date,
                "post_date": None,
                "description": desc,
                "credit": amt_val if is_credit else None,
                "debit": None if is_credit else amt_val,
                "bank": bank_name,
                "account_number": None,
            })
            cur_date, buf = None, []

        for ln in lines:
            # if a new date appears, the prior txn is complete — flush first
            md = date_only_re.match(ln)
            if md:
                flush()
                cur_date = md.group(1)
                buf = []
                continue
            # skip any stray totals that slipped through (extra safety)
            if re.match(r"^\s*Total\b", ln, flags=re.IGNORECASE):
                flush()
                break
            buf.append(ln)

        flush()
        return txns

    def parse_checks_paid(lines: List[str]) -> List[Dict[str, object]]:
        """
        Checks Paid sequence:
          CHECK_NO
          [*] (optional on same/separate line)
          [^] (optional on same/separate line)
          DATE
          AMOUNT
        We synthesize description as "Check <#>".
        """
        txns: List[Dict[str, object]] = []
        i = 0
        check_no_re = re.compile(r"^\d{3,6}(?:\s*\*)?$")  # e.g., '2273' or '2273 *'
        marker_re = re.compile(r"^[\*\^]$")

        while i < len(lines):
            # hard stop
            if re.match(r"^\s*Total\s+Checks\s+Paid\b", lines[i], re.IGNORECASE):
                break

            if not check_no_re.match(lines[i]):
                i += 1
                continue

            check_no = re.sub(r"\D", "", lines[i])
            i += 1

            # swallow optional markers on separate lines
            while i < len(lines) and marker_re.match(lines[i]):
                i += 1

            # expect date
            if i >= len(lines) or not date_only_re.match(lines[i]):
                continue
            tx_date = date_only_re.match(lines[i]).group(1)
            i += 1

            # expect amount
            if i >= len(lines) or not amount_only_re.match(lines[i]):
                continue
            amt = float(amount_only_re.match(lines[i]).group(1).replace(",", ""))
            i += 1

            txns.append({
                "date": tx_date,
                "post_date": None,
                "description": f"Check {check_no}",
                "credit": None,
                "debit": amt,
                "bank": bank_name,
                "account_number": None,
            })

        return txns

    # ---------- Extract & parse sections ----------
    all_txns: List[Dict[str, object]] = []

    # Deposits (credits)
    dep_blocks = extract_section_blocks(*SECTION_TAGS["deposits"])
    if dep_blocks:
        dep_lines: List[str] = []
        for b in dep_blocks:
            dep_lines.extend(normalize_block(b))
        all_txns.extend(parse_date_desc_amount_by_blocks(dep_lines, is_credit=True))

    # Checks Paid (debits)
    chk_blocks = extract_section_blocks(*SECTION_TAGS["checks"])
    if chk_blocks:
        chk_lines: List[str] = []
        for b in chk_blocks:
            chk_lines.extend(normalize_block(b))
        all_txns.extend(parse_checks_paid(chk_lines))

    # ATM & Debit Card Withdrawals (debits)
    atm_blocks = extract_section_blocks(*SECTION_TAGS["atm_debit"])
    if atm_blocks:
        atm_lines: List[str] = []
        for b in atm_blocks:
            atm_lines.extend(normalize_block(b))
        all_txns.extend(parse_date_desc_amount_by_blocks(atm_lines, is_credit=False))

    # Electronic Withdrawals (debits)
    el_blocks = extract_section_blocks(*SECTION_TAGS["electronic"])
    if el_blocks:
        el_lines: List[str] = []
        for b in el_blocks:
            el_lines.extend(normalize_block(b))
        all_txns.extend(parse_date_desc_amount_by_blocks(el_lines, is_credit=False))

    # Other Withdrawals (debits)
    ow_blocks = extract_section_blocks(*SECTION_TAGS["other_withdrawals"])
    if ow_blocks:
        ow_lines: List[str] = []
        for b in ow_blocks:
            ow_lines.extend(normalize_block(b))
        all_txns.extend(parse_date_desc_amount_by_blocks(ow_lines, is_credit=False))

    # Fees (debits)
    fees_blocks = extract_section_blocks(*SECTION_TAGS["fees"])
    if fees_blocks:
        fees_lines: List[str] = []
        for b in fees_blocks:
            fees_lines.extend(normalize_block(b))
        all_txns.extend(parse_date_desc_amount_by_blocks(fees_lines, is_credit=False))

    # ---------- Deduplicate & clean ----------
    cleaned: List[Dict[str, object]] = []
    seen = set()
    for t in all_txns:
        if not t.get("description") or not (t.get("credit") or t.get("debit")):
            continue
        key = (t["date"], t["description"], t.get("credit"), t.get("debit"))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)

    # ---------- Statement period (optional) ----------
    m_period = re.search(r"([A-Za-z]+\s+\d{1,2},?\s*\d{4})\s+through\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})", text)
    start_date, end_date = (m_period.groups() if m_period else (None, None))

    return {
        "bank": bank_name,
        "account_number": None,  # keep your format
        "start_date": start_date,
        "end_date": end_date,
        "transactions": cleaned,
    }


# ------------------------------------------------------------------------------------------------------------------------


def extract_transactions_from_boa_text(text: str) -> Dict[str, object]:
    """
    Robust parser for Bank of America Business statements (extracted text).
    Captures:
      - Deposits and other credits          -> credit
      - Withdrawals and other debits        -> debit
    Ignores summaries/footers/totals and merges continued pages.
    Returns:
      {
        "bank": "Bank of America, N.A.",
        "account_number": None,
        "start_date": <str|None>,
        "end_date": <str|None>,
        "transactions": [ {date, post_date, description, credit, debit, bank, account_number}, ... ]
      }
    """

    bank_name = "Bank of America, N.A."

    # ---------- Regex helpers ----------
    hdr_credit = re.compile(r"^\s*Deposits and other credits(?:\s*-\s*continued)?\s*$", re.IGNORECASE)
    hdr_debit  = re.compile(r"^\s*Withdrawals and other debits(?:\s*-\s*continued)?\s*$", re.IGNORECASE)

    # Lines to skip entirely anywhere
    skip_line = re.compile(
        r"^(Page\s+\d+\s+of\s+\d+|This page intentionally left blank|Customer service information|"
        r"Your checking account|Account summary|IMPORTANT INFORMATION:|BANK DEPOSIT ACCOUNTS|"
        r"continued on the next page|PULL:|RAHMAN ENGINEERING SERVICES INC|Bank of America, N\.A\.|"
        r"^©|^Web site:|^Service Center:|^Member FDIC|^Deaf and Hard of Hearing:|^Para Espanol:|^International Calls:)"
        r"$",
        re.IGNORECASE
    )

    # Column headers to skip inside a section
    col_header = re.compile(r"^(Date|Description|Amount|CHECKS?|Service fees|Daily ledger balances)\s*$", re.IGNORECASE)

    # Totals/Subtotals (if we see them, we must also ignore the *next* pure amount line)
    total_or_subtotal = re.compile(
        r"^(Total\s+deposits\s+and\s+other\s+credits|Total\s+withdrawals\s+and\s+other\s+debits|"
        r"Subtotal\s+for\s+card\s+account\s+#.*?)\s*$",
        re.IGNORECASE
    )

    # Date line style (BOA uses MM/DD/YY)
    date_only = re.compile(r"^\s*(\d{2}/\d{2}/\d{2})\s*$")

    # Amount-only line: optional '$', optional leading '-', then #,###.## (or ###.##)
    amt_only = re.compile(r"^\s*-?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})\s*$")

    # Section fenceposts: encountering these ends any current section cleanly
    section_enders = re.compile(
        r"^(Service fees|Daily ledger balances|This page intentionally left blank|IMPORTANT INFORMATION:|BANK DEPOSIT ACCOUNTS)\s*$",
        re.IGNORECASE
    )

    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    # ---------- Statement period ----------
    # e.g., "for February 1, 2025 to February 28, 2025"
    m_period = re.search(r"for\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\s+to\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})", text)
    start_date, end_date = (m_period.groups() if m_period else (None, None))

    # ---------- Main scan ----------
    lines = [ln.rstrip() for ln in text.splitlines()]

    section: Optional[str] = None  # 'credit' or 'debit' or None
    skip_next_amount = False       # set after reading a Total/Subtotal line
    in_tx = False
    cur_date: Optional[str] = None
    desc_buf: List[str] = []

    out: List[Dict[str, object]] = []

    def flush_txn(amount_value: Optional[float]):
        """Flush current transaction (if coherent)."""
        nonlocal in_tx, cur_date, desc_buf
        if not (in_tx and cur_date and desc_buf and (amount_value is not None) and section in ("credit", "debit")):
            # reset buffer anyway
            in_tx = False
            cur_date = None
            desc_buf = []
            return

        desc = _clean(" ".join(desc_buf))
        if not desc:
            in_tx = False
            cur_date = None
            desc_buf = []
            return

        if section == "credit":
            credit, debit = amount_value, None
        else:
            credit, debit = None, abs(amount_value)

        out.append({
            "date": cur_date,            # keep MM/DD/YY as seen
            "post_date": None,
            "description": desc,
            "credit": credit,
            "debit": debit,
            "bank": bank_name,
            "account_number": None,
        })

        in_tx = False
        cur_date = None
        desc_buf = []

    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue

        # Global skips
        if skip_line.match(ln):
            continue

        # Section switchers
        if hdr_credit.match(ln):
            # close any open txn before switching
            flush_txn(None)
            section = "credit"
            skip_next_amount = False
            continue

        if hdr_debit.match(ln):
            flush_txn(None)
            section = "debit"
            skip_next_amount = False
            continue

        # If not in a section yet, ignore until we hit a section header
        if section is None:
            continue

        # Section enders
        if section_enders.match(ln):
            flush_txn(None)
            section = None
            skip_next_amount = False
            continue

        # Skip in-section headers
        if col_header.match(ln):
            continue

        # Totals/subtotals: flush any open txn and ignore the next pure amount line
        if total_or_subtotal.match(ln):
            flush_txn(None)
            skip_next_amount = True
            continue

        # If we are supposed to skip the next amount line and this is a pure amount line, skip it and clear flag
        if skip_next_amount and amt_only.match(ln):
            skip_next_amount = False
            continue

        # Card-account header lines inside Withdrawals—treat as noise
        if ln.lower().startswith("card account #") or ln.lower().startswith("subtotal for card account #"):
            continue

        # New transaction start (date only line)
        mdate = date_only.match(ln)
        if mdate:
            # starting a new txn: close any pending (if it never got an amount, it's dropped)
            flush_txn(None)
            in_tx = True
            cur_date = mdate.group(1)
            desc_buf = []
            continue

        # Inside a transaction, try to detect amount-only line
        mamt = amt_only.match(ln)
        if mamt and in_tx:
            # amount line for this txn
            try:
                val = float(mamt.group(1).replace(",", ""))
            except Exception:
                val = None
            flush_txn(val)
            continue

        # Otherwise, if we're inside a transaction, accumulate description
        if in_tx:
            # guard against a stray header sneaking in (e.g., "- continued" header variants)
            if hdr_credit.match(ln) or hdr_debit.match(ln):
                # a new section header mid-txn means the amount was missing; drop it
                flush_txn(None)
                # re-handle this line as a real header in next loop
                # but since we continue outer loop, header will be processed next iteration only if it reappears;
                # practically BOA repeats exact header; safe to just continue
                continue

            # ignore generic “continued” marker variants
            if "continued" in ln.lower():
                continue

            # append the line to description
            desc_buf.append(ln)
        else:
            # Not inside a tx; ignore any strays
            continue

    # Flush trailing (if any; only if amount captured)
    flush_txn(None)

    # Deduplicate in case of rare repeats
    seen = set()
    cleaned: List[Dict[str, object]] = []
    for t in out:
        key = (t["date"], t["description"], t.get("credit"), t.get("debit"))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)

    return {
        "bank": bank_name,
        "account_number": None,   # keep your format
        "start_date": start_date,
        "end_date": end_date,
        "transactions": cleaned,
    }



# ----------------------------------------------------------------------------------------------------------------------


def extract_transactions_from_bmo_text(text: str) -> Dict[str, object]:
    """
    Final BMO Bank statement parser — universal across formats.
    Works for:
      - Old 'Deposits/Withdrawals'
      - New unified 'Monthly Activity Details'
      - Spaced / unspaced OCR text
      - Checks sections (Checks / Checks Paid / Checks Processed)
    """

    bank_name = "BMO Bank N.A."

    # ---------- Normalization ----------
    raw_text = text.replace("\r", "")
    raw_text = re.sub(r"[ \t]+", " ", raw_text)
    raw_text = re.sub(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{1,2})(?!\d)(?!\s)",
        lambda m: f"{m.group(1)} {m.group(2)} ",
        raw_text,
        flags=re.IGNORECASE,
    )
    raw_text = re.sub(r"\b(\d{1,2}/\d{1,2})(?!\s)", r"\1 ", raw_text)

    # ---------- Statement Period ----------
    start_date = end_date = year_full = None
    per = re.search(
        r"Statement\s+Period\s+(\d{2})/(\d{2})/(\d{2})\s+TO\s+(\d{2})/(\d{2})/(\d{2})",
        raw_text, re.IGNORECASE
    )
    per_alt = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s+through\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",
        raw_text, re.IGNORECASE
    )
    if per:
        sm, sd, sy, em, ed, ey = per.groups()
        start_date = f"{sm}/{sd}/20{sy}"
        end_date = f"{em}/{ed}/20{ey}"
        year_full = f"20{sy}"
    elif per_alt:
        mon_map = {
            "January":"01","February":"02","March":"03","April":"04","May":"05","June":"06",
            "July":"07","August":"08","September":"09","October":"10","November":"11","December":"12"
        }
        smon, sd, sy, emon, ed, ey = per_alt.groups()
        start_date = f"{mon_map.get(smon.capitalize(),'01')}/{int(sd):02d}/{sy}"
        end_date   = f"{mon_map.get(emon.capitalize(),'12')}/{int(ed):02d}/{ey}"
        year_full = sy

    # ---------- Regex ----------
    date_mmdd = re.compile(r"\b(\d{1,2}/\d{1,2})\b")
    date_month_day = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{1,2})\b", re.IGNORECASE)
    # allow ".02" or "0.02" etc.
    amount_all = re.compile(r"\$?\(?-?(?:\d{1,3}(?:,\d{3})*|\d*)\.\d{2}\)?", re.IGNORECASE)

    # checks row detector (very tolerant):
    #  - optional "Check/CHK/CHEQUE #"
    #  - date as "mm/dd" or "Mon dd"
    #  - check number: 2-8 digits
    #  - one or more amounts; the first amount (leftmost) is the txn amount; any later "Balance" is ignored
    checks_row = re.compile(
        r"(?P<date>(?:\d{1,2}/\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{1,2}))"
        r".{0,30}?(?:(?:Check|Cheque|CHK)\s*#?\s*)?(?P<chk>\d{2,8})\b"
        r".{0,60}?(?P<amt>\$?\(?-?(?:\d{1,3}(?:,\d{3})*|\d*)\.\d{2}\)?)",
        re.IGNORECASE
    )

    footer_noise = re.compile(
        r"(PAGE\s+\d+\s+OF\s+\d+|ACCOUNT\s+NUMBER:|STATEMENT\s+PERIOD|BMO\s+BANK\s+N\.A\.|"
        r"P\.?O\.?\s*BOX|PALATINE|CUSTOMER\s+SERVICE|ACCOUNT\s+SUMMARY|Daily\s+Balance\s+Summary|"
        r"Deposit\s+Account\s+Summary|Service\s+Fees|bmo\.com|contact|888-|866-|"
        r"Important Information|IN CASE OF ERRORS|Credit Reporting Disputes|WHAT TO DO IF YOU THINK)",
        re.IGNORECASE,
    )

    # ---------- Helpers ----------
    def is_txn_header(line: str) -> Optional[str]:
        m = date_month_day.search(line)
        if m:
            mm_map = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06",
                      "Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
            mm, dd = mm_map.get(m.group(1).capitalize()), f"{int(m.group(2)):02d}"
            return f"{mm}/{dd}"
        m2 = date_mmdd.search(line)
        if m2:
            p = m2.group(1).split("/")
            return f"{int(p[0]):02d}/{int(p[1]):02d}"
        return None

    def compute_post_date(mmdd: str) -> Optional[str]:
        if not mmdd or not year_full:
            return None
        mm, dd = mmdd.split("/")
        return f"{year_full}-{mm}-{dd}"

    def clean_desc(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip())
        s = re.sub(r"-\s*-", "-", s)
        s = s.replace("--", "-")
        return s.strip()

    def strip_all_amounts(s: str) -> str:
        # remove all monetary tokens
        return re.sub(r"\$?\(?-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}\)?", "", s).strip()

    def pick_amount_from_block(buf: List[str], desc_hint: str = "") -> Optional[float]:
        """
        Prefer the leftmost amount token in the block, but if a line has 'Balance' after an amount,
        use the amount that appears BEFORE the word 'Balance'. Keeps .02 / 0.02 support.
        """
        # collect tokens with their positions to keep left-to-right order
        candidates = []
        for ln in buf:
            # if "Balance" appears, restrict to amounts BEFORE 'Balance'
            bal_idx = re.search(r"\bBalance\b", ln, re.IGNORECASE)
            if bal_idx:
                segment = ln[:bal_idx.start()]
            else:
                segment = ln
            for m in amount_all.finditer(segment):
                tok = m.group(0)
                v = tok.replace("$", "").replace(",", "")
                if v.startswith("."):  # ".02" -> "0.02"
                    v = "0" + v
                try:
                    candidates.append(abs(float(v)))
                except:
                    pass

        if not candidates:
            # fallback: any token anywhere
            for ln in buf:
                for m in amount_all.finditer(ln):
                    tok = m.group(0)
                    v = tok.replace("$", "").replace(",", "")
                    if v.startswith("."):
                        v = "0" + v
                    try:
                        candidates.append(abs(float(v)))
                        break
                    except:
                        pass
                if candidates:
                    break

        if not candidates:
            return None

        # Heuristic: choose the earliest (leftmost) we collected
        amt = round(candidates[0], 2)

        # If the description hints strongly at a credit, that's fine; polarity handled elsewhere.
        return amt

    def classify_polarity(section: Optional[str], desc: str, buf: List[str]) -> str:
        if section == "credit": return "credit"
        if section == "debit": return "debit"
        d = desc.lower()
        if any(k in d for k in ["credit","refund","reversal","returned ach","nsf","incoming wire","interest paid","interest"]):
            return "credit"
        if any(k in d for k in ["debit","fee","charge","withdrawal","payment","ach debit","check","cheque","chk"]):
            return "debit"
        for ln in buf:
            if re.search(r"\(\s*\d|\-\s*\$?", ln):
                return "debit"
        return "debit"

    def valid_description(desc: str) -> bool:
        if not desc or len(desc) < 3:
            return False
        d = desc.lower()
        short_valid = (
            "interest" in d or "nsf" in d or "fee" in d or "refund" in d or
            "return" in d or "wire" in d or "debit" in d or "credit" in d or
            "check" in d or "cheque" in d or "chk" in d
        )
        if short_valid:
            return True
        words = desc.split()
        if len(words) >= 3:
            keywords = ("ach","wire","credit","debit","fee","transfer","serv","analysis","interest",
                        "refund","return","nsf","charge","check","cheque","chk")
            return any(k in d for k in keywords)
        return False

    # ---------- Core Parser ----------
    def parse_block(block_text: str, section_label: Optional[str]) -> List[Dict[str, object]]:
        txns, cur_date, buf = [], None, []
        cleaned = []
        for ln in block_text.splitlines():
            if not ln.strip():
                continue
            if footer_noise.search(ln):
                continue
            cleaned.append(ln.strip())

        # detect if this block is a Checks section
        is_checks_block = bool(re.search(r"\b(Checks?|Checks\s+Paid|Checks\s+Processed|Check\s+Activity)\b",
                                         block_text, re.IGNORECASE))

        def flush_generic():
            nonlocal cur_date, buf
            if not cur_date or not buf:
                cur_date, buf = None, []
                return
            desc_raw = " ".join(buf)
            amt = pick_amount_from_block(buf, desc_raw)
            desc = clean_desc(strip_all_amounts(desc_raw))

            # target footer junk only when it's truly footer-ish
            footer_markers = r"P\.?O\.?\s*BOX|ACCOUNT\s*NUMBER|IL\s*\d{5,}|[A-Z]{2}\s*\d{5,}"
            if (
                re.search(footer_markers, desc, re.IGNORECASE)
                and (
                    "interest" in desc.lower() or len(desc.split()) <= 6
                )
                and not any(k in desc.lower() for k in ["ach", "debit", "lease", "services", "billpay", "wire", "check"])
            ):
                desc = re.split(footer_markers, desc, 1, flags=re.IGNORECASE)[0].strip()

            # Normalize broken interest
            desc = re.sub(r"\bINTEREST\s*P[A-Z]?\b", "INTEREST PAID", desc, flags=re.IGNORECASE)

            if not amt or not valid_description(desc):
                cur_date, buf = None, []
                return

            pol = classify_polarity(section_label, desc, buf)
            txns.append({
                "date": cur_date,
                "post_date": compute_post_date(cur_date),
                "description": desc.strip(),
                "credit": amt if pol == "credit" else None,
                "debit": amt if pol == "debit" else None,
                "bank": bank_name,
                "account_number": None,
            })
            cur_date, buf = None, []

        def flush():
            # For checks block we still rely on generic flush when building multi-line descs
            flush_generic()

        i = 0
        while i < len(cleaned):
            ln = cleaned[i]

            # Stop early at footers/disclaimers but flush any pending txn first
            if re.search(r"Ending\s+Balance|Important\s+Information", ln, re.IGNORECASE):
                if buf and cur_date:
                    flush()
                break
            if i + 1 < len(cleaned) and re.search(r"Ending\s+Balance", cleaned[i + 1], re.IGNORECASE):
                if buf and cur_date:
                    flush()
                break

            # Totals
            if re.match(r"^\s*Total\b", ln, re.IGNORECASE):
                if buf and cur_date:
                    flush()
                i += 1
                continue

            if is_checks_block:
                # Try to parse a checks row directly
                m = checks_row.search(ln)
                if m:
                    mmdd = m.group("date")
                    hdr_date = is_txn_header(mmdd)  # normalize
                    chk_no = m.group("chk")
                    amt_tok = m.group("amt")
                    v = amt_tok.replace("$", "").replace(",", "")
                    if v.startswith("."):
                        v = "0" + v
                    try:
                        amt_val = abs(float(v))
                    except:
                        amt_val = None

                    if hdr_date and amt_val is not None:
                        # Build description "CHECK #NNNN"
                        desc = f"CHECK #{chk_no}"
                        txns.append({
                            "date": hdr_date,
                            "post_date": compute_post_date(hdr_date),
                            "description": desc,
                            "credit": None,          # checks are debits
                            "debit": amt_val,
                            "bank": bank_name,
                            "account_number": None,
                        })
                        # Reset any in-progress generic buffer since checks rows are row-wise
                        cur_date, buf = None, []
                        i += 1
                        continue
                # If not a checks row, fall through to generic accumulation (some PDFs wrap)
                # and we’ll let flush() classify by keywords ("check") as debit.

            # Generic header?
            hdr = is_txn_header(ln)
            if hdr:
                # If header line itself has stuff, keep remainder
                ln_remainder = re.sub(r"^(?:[A-Za-z]{3}\s*\d{1,2}|\d{1,2}/\d{1,2})", "", ln).strip()
                if buf and cur_date:
                    flush()
                cur_date = hdr
                buf = [ln_remainder] if ln_remainder else []
            elif cur_date:
                buf.append(ln)

            i += 1

        # Flush at block end
        if buf and cur_date:
            flush()

        return txns

    # ---------- Section Detection ----------
    def find_sections(txt: str):
        credits_hdr = re.compile(r"(Deposits?\s*(and|&)\s*Other\s*Credits?)", re.IGNORECASE)
        debits_hdr = re.compile(r"(Withdrawals?\s*(and|&)\s*Other\s*Debits?)", re.IGNORECASE)
        unified_hdr = re.compile(r"(Monthly\s+Activity\s+Details|Account\s+Activity)", re.IGNORECASE)
        checks_hdr = re.compile(r"\b(Checks?|Checks\s+Paid|Checks\s+Processed|Check\s+Activity)\b", re.IGNORECASE)
        stop_hdr = re.compile(
            r"(Daily\s+Balance\s+Summary|Statement\s+Period|Account\s+Summary|Interest\s+Summary|"
            r"Service\s+Fees|Messages\s+and\s+Notices|Balance\s+Summary)",
            re.IGNORECASE,
        )
        spans = []
        for m in credits_hdr.finditer(txt): spans.append(("credit", m.start(), m.end()))
        for m in debits_hdr.finditer(txt): spans.append(("debit", m.start(), m.end()))
        for m in unified_hdr.finditer(txt): spans.append((None, m.start(), m.end()))
        for m in checks_hdr.finditer(txt): spans.append(("checks", m.start(), m.end()))
        spans.sort(key=lambda x: x[1])
        blocks = []
        for i, (lbl, s, _) in enumerate(spans):
            e = spans[i + 1][1] if i + 1 < len(spans) else len(txt)
            stop = stop_hdr.search(txt, s, e)
            if stop: e = min(e, stop.start())
            blocks.append((lbl, s, e))
        return blocks

    # ---------- Parse ----------
    sections = find_sections(raw_text)
    all_txns = []
    if sections:
        for label, s, e in sections:
            all_txns += parse_block(raw_text[s:e], section_label=label)
    else:
        all_txns += parse_block(raw_text, section_label=None)

    # ---------- Deduplicate ----------
    seen, deduped = set(), []
    for t in all_txns:
        if not (t.get("credit") or t.get("debit")):
            continue
        key = (t["date"], t["description"].lower(), f"{t.get('credit') or t.get('debit'):.2f}")
        if key in seen: 
            continue
        seen.add(key)
        deduped.append(t)

    return {
        "bank": bank_name,
        "account_number": None,
        "start_date": start_date,
        "end_date": end_date,
        "transactions": deduped,
    }


# ----------------------------------------------------------------------------------------------------------------------


def extract_transactions_from_bmo_text_without_spacing(text: str) -> Dict[str, object]:
    """
    Final BMO Bank statement parser for the format with no spacing.
    ✅ Handles:
       - Multi-line descriptions
       - Footer handling
       - Amount association across lines
       - Proper transaction extraction
    """

    bank_name = "BMO Bank N.A."

    # ---------- Normalization ----------
    raw_text = text.replace("\r", "")
    raw_text = re.sub(r"[ \t]+", " ", raw_text)
    raw_text = re.sub(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{1,2})(?!\d)(?!\s)",
        lambda m: f"{m.group(1)} {m.group(2)} ",
        raw_text,
        flags=re.IGNORECASE,
    )
    raw_text = re.sub(r"\b(\d{1,2}/\d{1,2})(?!\s)", r"\1 ", raw_text)

    # ---------- Statement Period ----------
    start_date = end_date = year_full = None
    per = re.search(
        r"Statement\s+Period\s+(\d{2})/(\d{2})/(\d{2})\s+TO\s+(\d{2})/(\d{2})/(\d{2})",
        raw_text, re.IGNORECASE
    )
    per_alt = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s+through\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",
        raw_text, re.IGNORECASE
    )
    if per:
        sm, sd, sy, em, ed, ey = per.groups()
        start_date = f"{sm}/{sd}/20{sy}"
        end_date = f"{em}/{ed}/20{ey}"
        year_full = f"20{sy}"
    elif per_alt:
        mon_map = {
            "January":"01","February":"02","March":"03","April":"04","May":"05","June":"06",
            "July":"07","August":"08","September":"09","October":"10","November":"11","December":"12"
        }
        smon, sd, sy, emon, ed, ey = per_alt.groups()
        start_date = f"{mon_map.get(smon.capitalize(),'01')}/{int(sd):02d}/{sy}"
        end_date   = f"{mon_map.get(emon.capitalize(),'12')}/{int(ed):02d}/{ey}"
        year_full = sy

    # ---------- Regex ----------
    date_mmdd = re.compile(r"\b(\d{1,2}/\d{1,2})\b")
    date_month_day = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{1,2})\b", re.IGNORECASE)
    amount_all = re.compile(
        r"\$?\(?-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}\)?",  # ✅ allow ".02" or "0.02"
        re.IGNORECASE,
    )
    footer_noise = re.compile(
        r"(PAGE\s+\d+\s+OF\s+\d+|ACCOUNT\s+NUMBER:|STATEMENT\s+PERIOD|BMO\s+BANK\s+N\.A\.|"
        r"P\.?O\.?\s*BOX|PALATINE|CUSTOMER\s+SERVICE|ACCOUNT\s+SUMMARY|Daily\s+Balance\s+Summary|"
        r"Deposit\s+Account\s+Summary|Service\s+Fees|bmo\.com|contact|888-|866-|"
        r"Important Information|IN CASE OF ERRORS|Credit Reporting Disputes|WHAT TO DO IF YOU THINK)",
        re.IGNORECASE,
    )

    # ---------- Helpers ----------
    def is_txn_header(line: str) -> Optional[str]:
        m = date_month_day.search(line)
        if m:
            mm_map = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06",
                      "Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
            mm, dd = mm_map.get(m.group(1).capitalize()), f"{int(m.group(2)):02d}"
            return f"{mm}/{dd}"
        m2 = date_mmdd.search(line)
        if m2:
            p = m2.group(1).split("/")
            return f"{int(p[0]):02d}/{int(p[1]):02d}"
        return None

    def compute_post_date(mmdd: str) -> Optional[str]:
        if not mmdd or not year_full:
            return None
        mm, dd = mmdd.split("/")
        return f"{year_full}-{mm}-{dd}"

    def clean_desc(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip())
        s = re.sub(r"-\s*-", "-", s)
        s = s.replace("--", "-")
        return s.strip()

    def strip_all_amounts(s: str) -> str:
        return re.sub(r"\$?\(?-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}\)?", "", s).strip()

    def pick_amount_from_block(buf: List[str], desc_hint: str = "") -> Optional[float]:
        tokens = []
        for ln in buf:
            tokens += amount_all.findall(ln)
        if not tokens:
            return None
        vals = []
        for t in tokens:
            v = t.replace("$", "").replace(",", "")
            if v.startswith("."):  # handle cases like ".02" -> "0.02"
                v = "0" + v
            try: vals.append(abs(float(v)))
            except: pass
        if not vals:
            return None
        desc_l = desc_hint.lower()
        if any(k in desc_l for k in ["credit", "wire", "nsf", "returned"]):
            return round(vals[0], 2)
        if any("balance" in ln.lower() for ln in buf):
            return round(min(vals), 2)
        return round(vals[0], 2)

    def classify_polarity(section: Optional[str], desc: str, buf: List[str]) -> str:
        if section == "credit": return "credit"
        if section == "debit": return "debit"
        d = desc.lower()
        if any(k in d for k in ["credit","refund","reversal","returned ach","nsf","incoming wire","interest paid"]):
            return "credit"
        if any(k in d for k in ["debit","fee","charge","withdrawal","payment","ach debit"]):
            return "debit"
        for ln in buf:
            if re.search(r"\(\s*\d|\-\s*\$?", ln):
                return "debit"
        return "debit"

    def valid_description(desc: str) -> bool:
        """
        Returns True if description looks like a legitimate transaction.
        Fix: allow short but valid phrases like 'INTEREST PAID', 'NSF RETURN', etc.
        """
        if not desc or len(desc) < 4:
            return False
        d = desc.lower()
        # Allow short but meaningful transaction descriptors
        short_valid = (
            "interest" in d
            or "nsf" in d
            or "fee" in d
            or "refund" in d
            or "return" in d
            or "wire" in d
            or "debit" in d
            or "credit" in d
        )
        if short_valid:
            return True

        # Default: at least 3 words OR contains typical transaction keywords
        words = desc.split()
        if len(words) >= 3:
            keywords = (
                "ach","wire","credit","debit","fee","transfer","serv","analysis","interest",
                "refund","return","nsf","charge"
            )
            return any(k in d for k in keywords)
        return False

    # ---------- Core Parser ----------
    def parse_block(block_text: str, section_label: Optional[str]) -> List[Dict[str, object]]:
        txns, cur_date, buf = [], None, []
        cleaned = []
        for ln in block_text.splitlines():
            if not ln.strip():
                continue
            if footer_noise.search(ln):
                continue
            cleaned.append(ln.strip())

        def flush():
            nonlocal cur_date, buf
            if not cur_date or not buf:
                cur_date, buf = None, []
                return

            desc_raw = " ".join(buf)
            amt = pick_amount_from_block(buf, desc_raw)
            desc = clean_desc(strip_all_amounts(desc_raw))

            # --- Footer cleanup: only trim when clearly footer-style garbage ---
            footer_markers = r"P\.?O\.?\s*BOX|ACCOUNT\s*NUMBER|IL\s*\d{5,}|[A-Z]{2}\s*\d{5,}"
            if (
                re.search(footer_markers, desc, re.IGNORECASE)
                and (
                    "interest" in desc.lower() or len(desc.split()) <= 6
                )
                and not any(k in desc.lower() for k in ["ach", "debit", "lease", "services", "billpay", "wire"])
            ):
                desc = re.split(footer_markers, desc, 1, flags=re.IGNORECASE)[0].strip()

            # --- Normalize common broken descriptions ---
            desc = re.sub(r"\bINTEREST\s*P[A-Z]?\b", "INTEREST PAID", desc, flags=re.IGNORECASE)

            # --- Skip invalid or empty transactions ---
            if not amt or not valid_description(desc):
                cur_date, buf = None, []
                return

            pol = classify_polarity(section_label, desc, buf)
            txns.append({
                "date": cur_date,
                "post_date": compute_post_date(cur_date),
                "description": desc.strip(),
                "credit": amt if pol == "credit" else None,
                "debit": amt if pol == "debit" else None,
                "bank": bank_name,
                "account_number": None,
            })
            cur_date, buf = None, []

        for i, ln in enumerate(cleaned):
            # If nearing the footer or disclaimer section
            if re.search(r"Ending\s+Balance|Important\s+Information", ln, re.IGNORECASE):
                # Flush whatever is pending before breaking
                if buf and cur_date:
                    flush()
                break

            # If the current line looks like a "balance only" value immediately before footer
            if i + 1 < len(cleaned) and re.search(r"Ending\s+Balance", cleaned[i + 1], re.IGNORECASE):
                if buf and cur_date:
                    flush()
                break

            # Handle total summary lines
            if re.match(r"^\s*Total\b", ln, re.IGNORECASE):
                if buf and cur_date:
                    flush()
                continue

            hdr = is_txn_header(ln)
            if hdr:
                # new transaction header detected
                ln_remainder = re.sub(r"^(?:[A-Za-z]{3}\s*\d{1,2}|\d{1,2}/\d{1,2})", "", ln).strip()
                if buf and cur_date:
                    flush()
                cur_date = hdr
                buf = [ln_remainder] if ln_remainder else []
            elif cur_date:
                buf.append(ln)

        # 🩵 NEW SAFETY: Flush uncommitted transaction if EOF reached normally
        if buf and cur_date:
            flush()
        return txns

    # ---------- Section Detection ----------
    def find_sections(txt: str):
        credits_hdr = re.compile(r"(Deposits?\s*(and|&)\s*Other\s*Credits?)", re.IGNORECASE)
        debits_hdr = re.compile(r"(Withdrawals?\s*(and|&)\s*Other\s*Debits?)", re.IGNORECASE)
        unified_hdr = re.compile(r"(Monthly\s+Activity\s+Details|Account\s+Activity)", re.IGNORECASE)
        stop_hdr = re.compile(
            r"(Daily\s+Balance\s+Summary|Statement\s+Period|Account\s+Summary|Interest\s+Summary|"
            r"Service\s+Fees|Messages\s+and\s+Notices|Balance\s+Summary)",
            re.IGNORECASE,
        )
        spans = []
        for m in credits_hdr.finditer(txt): spans.append(("credit", m.start(), m.end()))
        for m in debits_hdr.finditer(txt): spans.append(("debit", m.start(), m.end()))
        for m in unified_hdr.finditer(txt): spans.append((None, m.start(), m.end()))
        spans.sort(key=lambda x: x[1])
        blocks = []
        for i, (lbl, s, _) in enumerate(spans):
            e = spans[i + 1][1] if i + 1 < len(spans) else len(txt)
            stop = stop_hdr.search(txt, s, e)
            if stop: e = min(e, stop.start())
            blocks.append((lbl, s, e))
        return blocks

    # ---------- Parse ----------
    sections = find_sections(raw_text)
    all_txns = []
    if sections:
        for label, s, e in sections:
            all_txns += parse_block(raw_text[s:e], section_label=label)
    else:
        all_txns += parse_block(raw_text, section_label=None)

    # ---------- Deduplicate ----------
    seen, deduped = set(), []
    for t in all_txns:
        if not (t.get("credit") or t.get("debit")):
            continue
        key = (t["date"], t["description"].lower(), f"{t.get('credit') or t.get('debit'):.2f}")
        if key in seen: continue
        seen.add(key)
        deduped.append(t)

    return {
        "bank": bank_name,
        "account_number": None,
        "start_date": start_date,
        "end_date": end_date,
        "transactions": deduped,
    }


# ----------------------------------------------------------------------------------------------------------------------

def extract_transactions_wf_credit_card(text: str) -> Dict[str, object]:
    """
    Extracts credit card transactions from Wells Fargo statements, focusing on the 'Transaction Details' section.
    This function captures the correct transaction amounts and handles additional text correctly.
    """

    # Initialize bank details
    bank_name = "Wells Fargo"
    transactions = []

    # Regular expression patterns for detecting dates and amounts
    date_re = re.compile(r"(\d{1,2}/\d{1,2})")  # Match dates in MM/DD format
    amount_re = re.compile(r"\$?([0-9]{1,3}(?:,\d{3})*\.\d{2})")  # Match amounts (e.g., $1,000.00)
    
    # Process the extracted text and split into lines
    lines = text.splitlines()

    # Flags and variables for processing
    current_date = None
    description_lines = []
    amount = None
    capturing_transaction = False  # Flag to indicate we are in the transaction section
    capture_amount = False  # Flag to stop further amount capture after the first valid amount

    for line in lines:
        line = line.strip()

        # Skip lines that are empty or footer lines
        if not line or re.match(r"^(Page|Account)", line):
            continue
        
        # Identify the start of the 'Transaction Details' section
        if "Transaction Details" in line:
            capturing_transaction = True
            continue  # Skip the "Transaction Details" heading line

        if capturing_transaction:
            # Extract the date from the line
            date_match = date_re.match(line)
            if date_match:
                # If we have a valid date and there is an accumulated transaction, store it first
                if current_date and description_lines and amount is not None:
                    description = " ".join(description_lines).strip()
                    transactions.append({
                        "date": current_date,
                        "post_date": current_date,  # Assuming the post date is same as transaction date
                        "description": description,
                        "credit": amount if amount > 0 else None,
                        "debit": amount if amount < 0 else None,
                        "bank": bank_name,
                    })

                # Start a new transaction
                current_date = date_match.group(1)
                description_lines = []
                amount = None
                capture_amount = False  # Reset amount capture flag for the new transaction
                continue

            # Extract any amounts mentioned in the line
            amount_match = amount_re.search(line)
            if amount_match and not capture_amount:
                # Capture the correct amount only when the line matches a valid description or transaction amount
                amount = float(amount_match.group(1).replace(",", "").replace("$", ""))
                capture_amount = True  # Stop capturing amount for this transaction once it's set
            
            # If the line contains any transaction data (reference number or description)
            if amount is None and len(line.split()) > 1:  # Skip lines with no amount
                description_lines.append(line)

            # If we reached the end of the transaction section, stop capturing
            if "Summary" in line or "Total" in line:
                break

    # Handle the last transaction if any
    if current_date and description_lines and amount is not None:
        description = " ".join(description_lines).strip()
        transactions.append({
            "date": current_date,
            "post_date": current_date,
            "description": description,
            "credit": amount if amount > 0 else None,
            "debit": amount if amount < 0 else None,
            "bank": bank_name,
        })

    # Return the results in the specified format
    return {
        "bank": bank_name,
        "account_number": None,  # Account number is not being extracted in this case
        "start_date": None,  # Could be set based on the first transaction's date if needed
        "end_date": None,  # Could be set based on the last transaction's date if needed
        "transactions": transactions,
    }


# ----------------------------------------------------------------------------------------------------------------------

def extract_transactions_from_chase_credit_card(text: str) -> Dict[str, object]:
    bank_name = "JPMorgan Chase Bank, N.A."
    transactions = []

    date_re = re.compile(r"\d{2}/\d{2}")
    amount_re = re.compile(r"\$?([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})")
    reference_re = re.compile(r"([A-Za-z0-9]+(?:\s?\*)?)")

    skip_re = re.compile(
        r"^(Page\b.*|Date of Transaction|Account Summary|Customer Service|"
        r"Member FDIC|Your Balance|Available Credit|Late Payment|"
        r"Minimum Payment Due)", re.IGNORECASE)

    ignore_amount_line = re.compile(
        r"(TRANSACTIONS THIS CYCLE|PREVIOUS BALANCE|NEW BALANCE|PAYMENT|INTEREST|FEES)",
        re.IGNORECASE
    )

    # 🔥 REAL hard-stop markers that ALWAYS appear after last transaction
    end_of_transactions_re = re.compile(
        r"(Total fees charged|Total interest charged|Year-to-date totals|INTEREST CHARGES|Annual Percentage Rate)",
        re.IGNORECASE
    )

    current_date = None
    post_date = None
    description_lines = []
    amount = None

    capturing_transaction = False
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line or skip_re.match(line):
            continue

        # Start capturing after ACCOUNT ACTIVITY
        if "ACCOUNT ACTIVITY" in line:
            capturing_transaction = True
            continue

        if capturing_transaction:

            # 🔥 HARD STOP — end of ALL transactions
            if end_of_transactions_re.search(line):
                capturing_transaction = False
                break

            # DATE
            date_match = date_re.match(line)
            if date_match:
                if current_date and description_lines and amount is not None:
                    description = " ".join(description_lines).strip()
                    reference_match = reference_re.search(description)
                    reference_number = reference_match.group(1) if reference_match else None

                    transactions.append({
                        "date": current_date,
                        "post_date": post_date or current_date,
                        "description": description + (f" Reference: {reference_number}" if reference_number else ""),
                        "credit": amount if amount > 0 else None,
                        "debit": amount if amount < 0 else None,
                        "bank": bank_name,
                    })

                current_date = date_match.group(0)
                post_date = None
                description_lines = []
                amount = None
                continue

            # AMOUNT
            amount_match = amount_re.search(line)
            if amount_match:
                if not ignore_amount_line.search(line):
                    amount = float(amount_match.group(1).replace(",", "").replace("$", ""))
                continue

            # DESCRIPTION
            description_lines.append(line)

    # FLUSH LAST TRANSACTION
    if current_date and description_lines and amount is not None:
        description = " ".join(description_lines).strip()
        reference_match = reference_re.search(description)
        reference_number = reference_match.group(1) if reference_match else None

        transactions.append({
            "date": current_date,
            "post_date": post_date or current_date,
            "description": description + (f" Reference: {reference_number}" if reference_number else ""),
            "credit": amount if amount > 0 else None,
            "debit": amount if amount < 0 else None,
            "bank": bank_name,
        })

    return {
        "bank": bank_name,
        "account_number": None,
        "start_date": None,
        "end_date": None,
        "transactions": transactions,
    }

# ----------------------------------------------------------------------------------------------------------------------

def extract_transactions_from_boa_credit_card(text: str) -> Dict[str, object]:

    bank_name = "Bank of America"

    # ============================
    # REGEX DEFINITIONS
    # ============================
    date_re = re.compile(r"^\s*(\d{2}/\d{2})\s*$")
    amount_re = re.compile(r"\$?([0-9]{1,3}(?:,[0-9]{3})*\.\d{2})")
    account_num_re = re.compile(r"Account Number:\s*([0-9X]+)")

    # Section markers used in all BoA layouts
    SECTION_HEADERS = {
        "payments": "PAYMENTS AND OTHER CREDITS",
        "purchases": "PURCHASES AND OTHER CHARGES",
        "fees": "FEES AND ADJUSTMENTS",
        "cash": "CASH ADVANCES",
        "returns": "CREDITS AND RETURNS",
    }

    # For mapping section → debit/credit logic
    SECTION_TYPE = {
        "payments": "credit",
        "returns": "credit",
        "purchases": "debit",
        "fees": "debit",
        "cash": "debit",
    }

    # ============================
    # EXTRACT ACCOUNT NUMBER
    # ============================
    account_number = None
    m = account_num_re.search(text)
    if m:
        account_number = m.group(1)

    # ============================
    # LOCATE STATEMENT PERIOD
    # ============================
    period_re = re.compile(r"(\w+\s+\d{1,2},\s*\d{4})\s*-\s*(\w+\s+\d{1,2},\s*\d{4})")
    start_date, end_date = None, None
    m = period_re.search(text)
    if m:
        start_date, end_date = m.group(1), m.group(2)

    # ============================
    # PROCESS LINES
    # ============================
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    transactions: List[Dict] = []

    current_section = None
    current_date = None
    post_date = None
    desc_buffer = []
    amount = None

    def flush_transaction():
        nonlocal current_date, post_date, desc_buffer, amount, current_section

        if not current_date or amount is None or not desc_buffer:
            # Not a complete transaction
            current_date = None
            desc_buffer = []
            amount = None
            post_date = None
            return

        description = " ".join(desc_buffer).strip()

        txn_type = SECTION_TYPE.get(current_section)
        credit = amount if txn_type == "credit" else None
        debit = amount if txn_type == "debit" else None

        transactions.append({
            "date": current_date,
            "post_date": post_date if post_date else current_date,
            "description": description,
            "credit": credit,
            "debit": debit,
            "bank": bank_name,
        })

        # reset for next
        current_date = None
        post_date = None
        desc_buffer = []
        amount = None

    # ============================
    # MAIN PARSE LOOP
    # ============================
    i = 0
    while i < len(lines):
        line = lines[i]

        # ---- Detect section header ----
        for key, hdr in SECTION_HEADERS.items():
            if hdr.lower() in line.lower():
                flush_transaction()
                current_section = key
                current_date = None
                desc_buffer = []
                amount = None
                break

        # Ignore lines until a section is detected
        if current_section is None:
            i += 1
            continue

        # ---- Date detection ----
        m = date_re.match(line)
        if m:
            flush_transaction()
            current_date = m.group(1)       # new transaction begins
            desc_buffer = []
            amount = None
            i += 1
            continue

        # ---- Amount detection ----
        m = amount_re.search(line)
        if m:
            amount = float(m.group(1).replace(",", "").replace("$", ""))

            # description is already buffered
            i += 1
            continue

        # ---- Post date (BoA sometimes shows 2 dates) ----
        if current_date and not post_date and date_re.match(line):
            post_date = date_re.match(line).group(1)
            i += 1
            continue

        # ---- Description line ----
        if current_date:
            desc_buffer.append(line)

        i += 1

    # Flush last transaction
    flush_transaction()

    # ============================
    # FINAL RESPONSE FORMAT
    # ============================
    return {
        "bank": bank_name,
        "account_number": account_number,
        "start_date": start_date,
        "end_date": end_date,
        "transactions": transactions,
    }

# ----------------------------------------------------------------------------------------------------------------------

# pattern for all illegal / non-printable ASCII control characters
_illegal_char_re = re.compile(r"[\000-\010\013\014\016-\037]")

def remove_illegal_chars(s):
    """Remove characters Excel/openpyxl cannot handle."""
    if isinstance(s, str):
        return _illegal_char_re.sub("", s)
    return s

# -----------------------------
# Save to Excel
# -----------------------------
def save_transactions_to_excel(response: Dict[str, object], path: str = "transactions.xlsx"):
    txns = response.get("transactions", [])
    if not txns:
        print("No transactions to save.")
        return
    df = pd.DataFrame(txns)
    df = df[["date", "post_date", "description", "credit", "debit", "bank"]]

    # --- Clean every cell to prevent IllegalCharacterError ---
    # for col in df.columns:
    #     df[col] = df[col].map(remove_illegal_chars)

    df.to_excel(path, index=False)
    print(f"✅ Saved {len(df)} transactions to {path}")

