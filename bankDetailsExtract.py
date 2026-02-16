import os
import re
import pdfplumber
import pandas as pd
from decimal import Decimal
from dateutil import parser as dateparser
from pathlib import Path
from datetime import datetime
from backend.utils.logger import log_message


# ==================================================
# Helpers
# ==================================================
def clean_amount(s):
    """Convert string to Decimal, handle negatives, commas, $."""
    if not s:
        return None
    s = str(s).replace("$", "").replace(",", "").replace("−", "-").strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    try:
        val = Decimal(s)
    except Exception:
        return None
    if negative or val < 0:
        return -abs(val)
    return val

def try_parse_date(s):
    """Try to convert string into YYYY-MM-DD date."""
    if not s:
        return None
    try:
        return dateparser.parse(s, fuzzy=True, dayfirst=False).date().isoformat()
    except Exception:
        return None

def detect_bank(text):
    if not text:
        return "Unknown"
    text_upper = text.upper().replace("\u00A0", " ")

    if "WELLS FARGO" in text_upper:
        return "Wells Fargo"
    if "BMO" in text_upper:   # ✅ catch BMO Harris/Bank
        return "BMO"
    if "JPMORGAN CHASE BANK" in text_upper or "CHASE BANK" in text_upper:
        return "Chase Bank"
    if "BANK OF AMERICA" in text_upper: #Bank of amrica check
        return "Bank of America"
    if "CHASE" in text_upper and "CREDIT CARD STATEMENT" in text_upper:
        return "Chase Credit Card"
    if "CREDIT CARD STATEMENT" in text_upper:
        return "Credit Card"
    return "Unknown"


# ==================================================
# Specialized Wells Fargo Parser
# ==================================================

#Parse Wells Fargo business statements Start

# ----------------------------
# Wells Fargo (A) Optimize Business Checking — inline lines
# ----------------------------

def parse_wellsfargo_optimize(pdf_path):
    """
    Wells Fargo — Optimize Business Checking.
    Parses the two ledger sections:
      - Electronic deposits/bank credits  -> credit column
      - Electronic debits/bank debits     -> debit column
    Rows may begin with 1 or 2 dates (Effective, Posted). We use Posted if present.
    """
    rows = []

    # Dates like MM/DD or MM/DD/YY or MM/DD/YYYY
    date_token = r"\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    # Money amounts like 1,234.56 (may include $ or leading -)
    money_re = re.compile(r"-?\$?\d[\d,]*\.\d{2}")

    # Helpers
    def clean_amount(s):
        s = s.replace("$", "").replace(",", "").strip()
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        return float(s)

    def try_parse_date(s):
        from datetime import datetime, date
        s = s.strip()
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m/%d"):
            try:
                if fmt == "%m/%d" and s.count("/") == 1:
                    # assume current year if missing
                    s = f"{s}/{datetime.today().year}"
                    return datetime.strptime(s, "%m/%d/%Y").date()
                return datetime.strptime(s, fmt).date()
            except Exception:
                pass
        return s  # leave as-is if unparsable

    # Section detection
    credits_hdr = re.compile(r"electronic deposits/?bank credits", re.IGNORECASE)
    debits_hdr  = re.compile(r"electronic debits/?bank debits", re.IGNORECASE)

    # Lines to skip outright (column headers & boilerplate)
    skip_prefixes = tuple(x.lower() for x in [
        "account number", "account summary", "credits", "debits",
        "checks paid", "daily ledger balance summary", "notice:",
        "effective", "posted", "amount", "transaction detail",
        "questions?", "page", "sheet seq", "sheet", "©2010",
        "all rights reserved", "member fdic"
    ])

    current_section = None   # 'credit' or 'debit'

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.split("\n"):
                line = raw.strip()
                if not line:
                    continue
                low = line.lower()

                # Detect section boundaries
                if credits_hdr.search(low):
                    current_section = "credit"
                    continue
                if debits_hdr.search(low):
                    current_section = "debit"
                    continue
                # Break out when a new major block starts
                if low.startswith(("checks paid", "daily ledger balance summary")):
                    current_section = None

                # Ignore table/boilerplate lines
                if any(low.startswith(p) for p in skip_prefixes):
                    continue
                if current_section not in ("credit", "debit"):
                    continue  # outside a txn section

                # ---- Parse a transaction or a continuation line ----
                # Row can start with 1 or 2 dates:
                #   "07/01 229,600.50  Description..."         (credits)
                #   "06/28  07/01 24,150.84  Description..."  (debits)
                m = re.match(rf"^\s*({date_token})(?:\s+({date_token}))?\s+(.*)$", line)
                if not m:
                    # Continuation: append to last description
                    if rows:
                        rows[-1]["description"] = (rows[-1]["description"] + " " + line).strip()
                    continue

                eff_date, posted_date, tail = m.group(1), m.group(2), m.group(3)
                date_s = posted_date or eff_date  # prefer Posted date if present

                # Pull the first money token on the line as the txn amount
                amounts = money_re.findall(tail)
                credit = debit = balance = None  # balance not shown in these sections
                if amounts:
                    amt = clean_amount(amounts[0])
                    # Remove all money tokens from description
                    desc = money_re.sub("", tail).replace("<", "").strip()
                    if current_section == "credit":
                        credit = amt
                    else:
                        debit = amt
                else:
                    # Rare: line with dates but no amount → treat as continuation
                    if rows:
                        rows[-1]["description"] = (rows[-1]["description"] + " " + tail).strip()
                    continue

                rows.append({
                    "date": try_parse_date(date_s),
                    "description": desc,
                    "debit": debit,
                    "credit": credit,
                    "balance": balance  # remains None in these sections
                })

    return rows

# ----------------------------
# Wells Fargo (B) Combined Statement / Navigate Business Checking — tabular "Transaction history"

def parse_wellsfargo_combined_navbiz(pdf_path, account_name_hint="Navigate Business Checking"):
    """
    Wells Fargo Combined Statement (Navigate Business Checking).
    Aligns Deposits/Credits, Withdrawals/Debits, Balance columns using
    a PDF-specific keyword classifier for 2-number rows.
    """
    rows = []
    date_re  = re.compile(r"^\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
    money_re = re.compile(r"-?\$?\d[\d,]*\.\d{2}")

    # Headers/sections
    start_section_re = re.compile(rf"{re.escape(account_name_hint)}", re.IGNORECASE)
    txn_header_re    = re.compile(r"transaction history", re.IGNORECASE)
    end_markers = [
        "Ending balance on",
        "Summary of checks written",
        "Business Market Rate Savings",
        "Monthly service fee summary",
        "Account transaction fees summary",
        "Important Information You Should Know",
        "Totals $",  # end of table block
    ]

    # --- Classifier tuned to your PDF text ---
    CREDIT_HINTS = [
        # table header bucket: Deposits/Credits
        "deposit", "edeposit", "e deposit", "wells fargo rewards", "rewards",
        "interest payment", "interest",
        # Heartland merchant settlement text in your PDF
        "hrtland", "heartland", "pmt sys", "txns/fees", "slam dunk sports bar",
        # other typical incoming types
        "online transfer from", "zelle payment from", "wt fed", "wire in", "incoming wire",
    ]
    DEBIT_HINTS = [
        # table header bucket: Withdrawals/Debits
        "ach debit", "business to business ach debit",
        "purchase authorized", "recurring payment", "online transfer to",
        "check", "deposited or cashed check",
        "currency ordered fee", "coin ordered fee", "fee",
        "lottery lotto invoices", "ins prem", "payroll",
        "transfer to", "bill pay", "dtv*directv", "amazon",
    ]

    def looks_credit(desc: str) -> bool:
        d = desc.lower()
        return any(k in d for k in CREDIT_HINTS)

    def looks_debit(desc: str) -> bool:
        d = desc.lower()
        return any(k in d for k in DEBIT_HINTS)

    in_nav = False
    in_txn = False

    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

            for line in lines:
                # enter/exit the transaction table
                if not in_nav and start_section_re.search(line):
                    in_nav = True
                    continue
                if in_nav and not in_txn and txn_header_re.search(line):
                    in_txn = True
                    continue
                if in_txn and any(line.startswith(m) for m in end_markers):
                    in_txn = False
                    in_nav = False
                    continue
                if not in_txn:
                    continue

                # skip table headings
                low = line.lower()
                if ("page" in low and "transaction history" in low) or \
                   low.startswith(("date ", "check ", "deposits/credits", "withdrawals/debits", "ending daily")):
                    continue

                if not date_re.match(line):
                    # continuation: append to previous description
                    if rows:
                        rows[-1]["description"] = (rows[-1]["description"] + " " + line).strip()
                    continue

                # parse a transaction row
                parts = line.split()
                date_s = parts[0]
                rest   = " ".join(parts[1:])
                nums   = [float(a.replace("$","").replace(",","")) for a in money_re.findall(rest)]
                desc   = money_re.sub("", rest).strip()

                credit = debit = balance = None

                if len(nums) == 3:
                    # PDF order is Deposits/Credits, Withdrawals/Debits, Ending daily balance
                    credit, debit, balance = nums
                elif len(nums) == 2:
                    # Usually amount + ending balance. Decide which side via description.
                    amt, bal = nums
                    balance = bal
                    if looks_credit(desc) and not looks_debit(desc):
                        credit = amt
                    elif looks_debit(desc) and not looks_credit(desc):
                        debit = amt
                    else:
                        # tie-breakers:
                        # Heartland rows (merchant settlements) are credits in your PDF
                        if "hrtland" in desc.lower() or "heartland" in desc.lower():
                            credit = amt
                        # lines that literally contain 'deposit' or 'interest' are credits
                        elif any(k in desc.lower() for k in ("deposit", "interest", "rewards")):
                            credit = amt
                        else:
                            # conservative default: treat as debit only if strong debit signals; else credit
                            debit_keywords = ("ach debit", "purchase authorized", "recurring payment",
                                              "online transfer to", "check", "fee", "lottery")
                            if any(k in desc.lower() for k in debit_keywords):
                                debit = amt
                            else:
                                credit = amt
                elif len(nums) == 1:
                    # If only one number appears, most lines in this layout use it as a txn amount
                    amt = nums[0]
                    if looks_credit(desc) and not looks_debit(desc):
                        credit = amt
                    elif looks_debit(desc) and not looks_credit(desc):
                        debit = amt
                    else:
                        # strong fallbacks by explicit nouns
                        if any(k in desc.lower() for k in ("deposit", "interest", "rewards", "hrtland", "heartland", "txns/fees")):
                            credit = amt
                        else:
                            debit = amt

                rows.append({
                    "date": try_parse_date(date_s),
                    "description": desc,
                    "credit": credit,
                    "debit": debit,
                    "balance": balance
                })

    return rows

#3rd type of Wells Fargo statement functions if any

def parse_wellsfargo_business_card(pdf_path: str):
    """
    Parse Wells Fargo Business Credit Card statement.
    Returns list of dicts: {trans_date, post_date, description, credit, debit}
    """
    rows = []
    date_re = re.compile(r"^\d{2}/\d{2}")  # MM/DD
    money_re = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.split("\n"):
                line = raw.strip()
                if not line:
                    continue

                # Skip headers
                if line.startswith(("Trans", "Page", "See reverse side", "WELLS FARGO")):
                    continue

                if date_re.match(line):
                    parts = line.split()
                    trans_date = parts[0]
                    post_date = parts[1] if re.match(r"\d{2}/\d{2}", parts[1]) else None
                    rest = " ".join(parts[2:]) if post_date else " ".join(parts[1:])

                    # Find money values
                    amounts = money_re.findall(rest)
                    credit = debit = None
                    if amounts:
                        if "PAYMENT" in rest.upper():
                            credit = clean_amount(amounts[-1])
                        else:
                            debit = clean_amount(amounts[-1])

                    desc = money_re.sub("", rest).strip()

                    rows.append({
                        "date": try_parse_date(trans_date),
                        "post_date": try_parse_date(post_date) if post_date else None,
                        "description": desc,
                        "credit": credit,
                        "debit": debit,
                        "balance": None
                    })

    return rows


# ----------------------------
# Dispatcher helper for Wells Fargo, auto-detect the layout
# ----------------------------
def parse_wellsfargo(pdf_path):
    """
    Auto-detect which Wells Fargo format the PDF is:
      - Optimize Business Checking (inline): use parse_wellsfargo_optimize
      - Combined Statement / Navigate Business Checking (tabular): use parse_wellsfargo_combined_navbiz
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([p.extract_text() or "" for p in pdf.pages])

    text_l = text.lower()

    # Strong hints per your two samples:
    #   A) Optimize Business Checking (U.S. Roadways): has "Optimize Business Checking" and "Electronic deposits/bank credits"
    #   B) Combined Statement (Barbar LLC): has "Combined Statement of Accounts" and "Navigate Business Checking"

    if "business card" in text_l or "prepared for" in text_l:
        return parse_wellsfargo_business_card(pdf_path)
    if ("combined statement of accounts" in text_l) or ("navigate business checking" in text_l):
        return parse_wellsfargo_combined_navbiz(pdf_path)
    if "optimize business checking" in text_l:
        return parse_wellsfargo_optimize(pdf_path)

    # Fallback: decide by presence of "Transaction history" (tabular) vs "Electronic deposits/bank credits" (inline)
    if "transaction history" in text_l:
        return parse_wellsfargo_combined_navbiz(pdf_path)
    else:
        return parse_wellsfargo_optimize(pdf_path)

#Parse Wells Fargo business statements End

#For chase bank credit card statements

def parse_chase_credit(pdf_path):
    """Parse Chase credit card statement transactions."""
    rows = []
    txn_pattern = re.compile(r"^(\d{2}/\d{2})\s+(.+?)\s+(-?[\d,]+\.\d{2})$")

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                m = txn_pattern.match(line)
                if m:
                    date, desc, amount = m.groups()
                    amt = clean_amount(amount)
                    debit, credit = (abs(amt), None) if amt < 0 else (None, amt)

                    rows.append({
                        "date": try_parse_date(date),
                        "description": desc.strip(),
                        "debit": debit,
                        "credit": credit,
                        "balance": None
                    })
                else:
                    # Append continuation lines to last transaction description
                    if rows:
                        rows[-1]["description"] += " " + line.strip()
    return rows

#BMO bank is correct handle both old and new categryries
#Old style like Sample9

def parse_bmo_old(pdf_path):
    """
    Parse BMO Business Checking (Old Style like sample9).
    Sections:
      - Deposits and Other Credits  (Date Amount Description)
      - Withdrawals and Other Debits (Date Amount Description)
      - Daily Balance Summary (two columns of Date Balance pairs)
    Output rows: {date, description, debit, credit, balance}
    """
    rows = []
    section = None
    statement_year = None

    # Helpers
    MONTH = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

    def date_with_year(mon, day):
        # Use year from "Statement Period ... TO ..." (first page)
        if statement_year:
            return f"{mon} {int(day)} {statement_year}"
        return f"{mon} {int(day)}"  # fallback

    with pdfplumber.open(pdf_path) as pdf:
        # ---- Get the statement year from page 1, e.g. "Statement Period 04/01/25 TO 04/30/25"
        first_text = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
        m_yr = re.search(r"Statement\s+Period\s+\d{2}/\d{2}/(\d{2,4})\s+TO\s+\d{2}/\d{2}/(\d{2,4})", first_text, re.I)
        if m_yr:
            y = m_yr.group(2)
            y = f"20{y}" if len(y) == 2 else y
            try:
                statement_year = int(y)
            except:
                statement_year = None

        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue

                # ---- Section switches
                if re.search(r"Deposits\s+and\s+other\s+credits", line, re.I):
                    section = "deposits"
                    continue
                if re.search(r"Withdrawals\s+and\s+other\s+debits", line, re.I):
                    section = "withdrawals"
                    continue
                if re.search(r"(Daily\s+Balance\s+Summary|Daily\s+ledger\s+balances)", line, re.I):
                    section = "balances"
                    continue

                # ---- Skip obvious headers / totals / boilerplate
                if re.search(r"^Date\s+Amount\s+Description", line, re.I):
                    continue
                if re.search(r"(Total deposits|Total withdrawals|Service fee|Statement Period Rates|DEPOSIT ACCOUNT SUMMARY)", line, re.I):
                    continue
                if re.fullmatch(r"Interest Paid(?:.*)?", line, re.I):
                    # handled by the Deposits section row itself when it appears with a date
                    continue

                # ---- Deposits / Withdrawals: "Mon DD  Amount  Description"
                if section in ("deposits", "withdrawals"):
                    m = re.match(
                        rf"^{MONTH}\s+(\d{{1,2}})\s+(\(?-?\$?[\d,]*\.\d{{2}}\)?|\.\d{{2}})\s+(.*)$",
                        line
                    )
                    if m:
                        mon, day, amt_s, desc = m.group(1), m.group(2), m.group(3), m.group(4)
                        amt = clean_amount(amt_s)  # preserves sign/parentheses exactly
                        rows.append({
                            "date": try_parse_date(date_with_year(mon, day)),
                            "description": desc.strip(),
                            "debit": amt if section == "withdrawals" else None,
                            "credit": amt if section == "deposits" else None,
                            "balance": None
                        })
                    else:
                        # wrapped description line → append to last transaction
                        if rows:
                            rows[-1]["description"] = (rows[-1]["description"] + " " + line).strip()
                    continue

                # ---- Daily Balance Summary: each line has up to TWO "Mon DD Amount" pairs
                if section == "balances":
                    # find all "Mon DD Amount" triples on the line
                    triples = list(re.finditer(
                        rf"{MONTH}\s+(\d{{1,2}})\s+(\(?-?\$?[\d,]*\.\d{{2}}\)?|\.\d{{2}})",
                        line
                    ))
                    for t in triples:
                        mon = t.group(1)
                        day = t.group(2)
                        bal_s = t.group(3)
                        rows.append({
                            "date": try_parse_date(date_with_year(mon, day)),
                            "description": "Daily Balance",
                            "debit": None,
                            "credit": None,
                            "balance": clean_amount(bal_s)
                        })
                    continue

                # Fallback: continuation line for any previous row
                if rows:
                    rows[-1]["description"] = (rows[-1]["description"] + " " + line).strip()

    return rows

#New style like Sample10
#Helper functions inside
def _bmo_clean_amount_exact(s: str):
    """Parse amount exactly as shown; keep sign/parentheses."""
    if not s:
        return None
    s = str(s).strip().replace(",", "").replace("$", "")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("+", "")
    try:
        val = Decimal(s)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d{1,2})?", s)
        if not m:
            return None
        try:
            val = Decimal(m.group(0))
        except Exception:
            return None
    return -val if neg else val
def _bmo_find_header(words):
    """Find the y of the header line and the header words on that line."""
    rows = {}
    for w in words:
        y = round(w["top"], 1)
        rows.setdefault(y, []).append(w)

    for y in sorted(rows.keys()):
        line = " ".join(w["text"] for w in sorted(rows[y], key=lambda ww: ww["x0"])).lower()
        if ("date" in line and "description" in line and "withdraw" in line
            and "deposit" in line and "balance" in line):
            return y, sorted(rows[y], key=lambda ww: ww["x0"])
    return None, []
def _bmo_header_centers(header_words):
    """Map canonical columns to their x-centers, using the header words."""
    centers = {}
    for w in header_words:
        t = w["text"].strip().lower()
        cx = (w["x0"] + w["x1"]) / 2.0
        if t == "date" and "date" not in centers:
            centers["date"] = cx
        elif "withdraw" in t and "withdrawal" not in centers:
            centers["withdrawal"] = cx
        elif "deposit" in t and "deposit" not in centers:
            centers["deposit"] = cx
        elif "balance" in t and "balance" not in centers:
            centers["balance"] = cx
        elif "description" in t and "description" not in centers:
            centers["description"] = cx
        elif t == "transaction" and "description" not in centers:
            centers["description"] = cx  # header may read "Transaction description"

    # If description is missing, estimate it mid-way between date and withdrawal
    if "description" not in centers and "date" in centers and "withdrawal" in centers:
        centers["description"] = (centers["date"] + centers["withdrawal"]) / 2.0
    return centers
def _bmo_edges_from_centers(centers, xmin, xmax):
    """Build left/right spans for each column from header centers."""
    items = sorted(centers.items(), key=lambda kv: kv[1])
    spans = []
    for i, (name, cx) in enumerate(items):
        left = xmin if i == 0 else (items[i-1][1] + cx) / 2.0
        right = xmax if i == len(items)-1 else (cx + items[i+1][1]) / 2.0
        spans.append((name, left, right))
    return spans
def _bmo_pick_col(xc, spans):
    for name, left, right in spans:
        if left <= xc <= right:
            return name
    return None
#Helper End For BMO
def parse_bmo_new(pdf_path):
    """
    STRICT parser for BMO Business Checking (Sample10-style).
    Columns are taken *exactly* as in the PDF:
      Withdrawal -> debit,  Deposit -> credit,  Balance -> balance.
    Signs are preserved exactly (no abs(), no remapping).
    Wrapped description lines get appended to the previous row.
    """
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=1.5, y_tolerance=2.0,
                keep_blank_chars=False, use_text_flow=True
            )
            if not words:
                continue

            header_top, header_words = _bmo_find_header(words)
            if header_top is None:
                continue

            centers = _bmo_header_centers(header_words)
            # Must have at least these; description is inferred if missing
            if not {"date", "withdrawal", "deposit", "balance"}.issubset(centers.keys()):
                continue

            xmin = min(w["x0"] for w in words)
            xmax = max(w["x1"] for w in words)
            spans = _bmo_edges_from_centers(centers, xmin, xmax)

            # Group words by row below the header
            line_map = {}
            for w in words:
                if w["top"] <= header_top:
                    continue
                y = round(w["top"], 1)
                line_map.setdefault(y, []).append(w)

            for y in sorted(line_map):
                rwords = sorted(line_map[y], key=lambda ww: ww["x0"])
                cols = {"date": [], "description": [], "withdrawal": [], "deposit": [], "balance": []}
                for w in rwords:
                    col = _bmo_pick_col((w["x0"] + w["x1"]) / 2.0, spans)
                    if not col:
                        continue
                    t = w["text"]
                    if t.lower().startswith("page "):
                        continue
                    cols.setdefault(col, []).append(t)

                date_txt = " ".join(cols.get("date", [])).strip()
                desc_txt = " ".join(cols.get("description", [])).strip()
                w_txt    = " ".join(cols.get("withdrawal", [])).strip()
                d_txt    = " ".join(cols.get("deposit", [])).strip()
                b_txt    = " ".join(cols.get("balance", [])).strip()

                # Skip empty lines
                if not any([date_txt, desc_txt, w_txt, d_txt, b_txt]):
                    continue

                # Wrapped description line (no date/amounts): append to previous
                if (not date_txt) and (not w_txt) and (not d_txt) and (not b_txt) and desc_txt:
                    if rows:
                        rows[-1]["description"] = (rows[-1]["description"] + " " + desc_txt).strip()
                    continue

                debit   = _bmo_clean_amount_exact(w_txt) if w_txt else None
                credit  = _bmo_clean_amount_exact(d_txt) if d_txt else None
                balance = _bmo_clean_amount_exact(b_txt) if b_txt else None

                rows.append({
                    "date": date_txt or None,  # keep as seen; you can run try_parse_date() later if you want ISO
                    "description": desc_txt,
                    "debit": float(debit) if debit is not None else None,
                    "credit": float(credit) if credit is not None else None,
                    "balance": float(balance) if balance is not None else None,
                })
    return rows


#BMO credit card parser start



def clean_amount(value: str) -> Decimal:
    """Convert amount string into Decimal"""
    value = value.replace("$", "").replace(",", "").strip()
    try:
        return Decimal(value)
    except:
        return Decimal(0)

def try_parse_date(value: str):
    """Try parsing dates like 'Aug 14 2025' or 'Aug 14, 2025'"""
    for fmt in ["%b %d %Y", "%b %d, %Y", "%b %d %y"]:
        try:
            return datetime.strptime(value.replace(",", ""), fmt).date()
        except ValueError:
            continue
    return None


def parse_bmo_creditcard(pdf_path: str) -> pd.DataFrame:
    """
    Parse BMO Business Platinum Credit Card statement into a DataFrame.
    Returns columns: date, description, debit, credit, balance.
    """
    rows = []
    money_re = re.compile(r"-?\$?\d[\d,]*\.\d{2}")
    date_re = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}",
        re.I,
    )

    with pdfplumber.open(pdf_path) as pdf:
        pending_desc = None
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.split("\n"):
                line = " ".join(raw.strip().split())
                if not line:
                    continue

                # Case 1: line contains amount → store description+amount
                m_amt = money_re.search(line)
                if m_amt:
                    amt = clean_amount(m_amt.group(0))
                    desc = money_re.sub("", line).strip()
                    pending_desc = (desc, amt)
                    continue

                # Case 2: line contains date → pair with stored desc+amount
                if date_re.match(line) and pending_desc:
                    desc, amt = pending_desc
                    debit, credit = None, None
                    if amt < 0:
                        credit = abs(amt)
                    else:
                        debit = amt

                    parsed_date = try_parse_date(line)
                    row = {
                        "date": parsed_date,
                        "description": desc,
                        "debit": debit,
                        "credit": credit,
                        "balance": None,
                    }
                    log_message("debug", f"parse_bmo_credit: APPEND: {row}")
                    rows.append(row)
                    pending_desc = None

    return pd.DataFrame(rows, columns=["date", "description", "debit", "credit", "balance"])


#BMO credit card parser end



#BMO end 2 categery BMO business also available

#Bank of Amrica parser start
def parse_bofa(pdf_path):
    """
    Parse Bank of America Business Advantage Fundamentals Banking statements.
    Extracts:
      - Deposits and other credits
      - Withdrawals and other debits
      - Daily ledger balances
    """
    rows = []
    section = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.split("\n"):
                line = raw.strip()
                if not line:
                    continue

                # Section headers
                if re.search(r"Deposits and other credits", line, re.I):
                    section = "credits"
                    continue
                if re.search(r"Withdrawals and other debits", line, re.I):
                    section = "debits"
                    continue
                if re.search(r"Daily ledger balances", line, re.I):
                    section = "balances"
                    continue

                # Skip headers/totals
                if re.match(r"^Date\s+Description\s+Amount", line, re.I):
                    continue
                if re.match(r"Total deposits", line, re.I) or \
                   re.match(r"Total withdrawals", line, re.I) or \
                   re.match(r"Service fees", line, re.I):
                    continue

                # ---- Deposits / Withdrawals ----
                if section in ("credits", "debits"):
                    m = re.match(r"^(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(-?\$?[\d,]+\.\d{2})$", line)
                    if m:
                        date_s, desc, amt_s = m.groups()
                        amt = clean_amount(amt_s)
                        rows.append({
                            "date": try_parse_date(date_s),
                            "description": desc.strip(),
                            "credit": amt if section == "credits" else None,
                            "debit": amt if section == "debits" else None,
                            "balance": None
                        })
                    else:
                        # Continuation line → append to previous description
                        if rows:
                            rows[-1]["description"] += " " + line
                    continue

                # ---- Daily Ledger Balances ----
                if section == "balances":
                    m = re.findall(r"(\d{2}/\d{2})\s+([\d,]+\.\d{2})", line)
                    for date_s, bal_s in m:
                        rows.append({
                            "date": try_parse_date(date_s),
                            "description": "Daily Balance",
                            "credit": None,
                            "debit": None,
                            "balance": clean_amount(bal_s)
                        })
                    continue

    return rows

#Bank of Amrica parser end

#Parse JPMorgan Chase bank statements (sample6/7/8) start
#Helper Functions inside

#Parse JPMorgan Chase bank statements (sample6/7/8) end 


#Bank of Amrica parser start


#Bank of Amrica parser end


# ==================================================
# Universal extractor (for unknown formats)
# ==================================================
def extract_transactions(pdf_path):
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if not words:
                continue
            rows = {}
            for w in words:
                y = round(w["top"], 1)
                rows.setdefault(y, []).append(w)
            for y in sorted(rows.keys()):
                row = sorted(rows[y], key=lambda x: x["x0"])
                line = " ".join(w["text"] for w in row)
                m = re.match(r"^(\d{2}/\d{2}|[A-Za-z]{3}\s+\d{1,2})\s+(.+?)\s+(-?\$?[\d,]+\.\d{2})$", line)
                if m:
                    date_s, desc, amt = m.groups()
                    amt_val = clean_amount(amt)
                    debit, credit = (abs(amt_val), None) if amt_val and amt_val < 0 else (None, amt_val)
                    results.append({
                        "date": try_parse_date(date_s),
                        "description": desc.strip(),
                        "debit": debit,
                        "credit": credit,
                        "balance": None,
                        "raw": line
                    })
                else:
                    if results:
                        results[-1]["description"] += " " + line
                        results[-1]["raw"] += " | " + line
    return results

# ==================================================
# Unified parser
# ==================================================
def parse_statement(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([p.extract_text() or "" for p in pdf.pages])

    bank = detect_bank(text)
    log_message("info", f"parse_statement: Detected bank: {bank}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
    except Exception as e:
        log_message("error", f"parse_statement: Failed to open {pdf_path}: {e}")
        return pd.DataFrame(columns=["date", "description", "debit", "credit", "balance", "bank"])

    rows = []  # default fallback

    if bank == "Wells Fargo":
        rows = parse_wellsfargo(pdf_path)
    elif bank == "Chase Credit Card":
        rows = parse_chase_credit(pdf_path)
    # elif bank == "Chase Bank":
    #     rows = parse_chase_JPMorgan(pdf_path)
    elif bank == "Bank of America":
        rows = parse_bofa(pdf_path)
    elif "BMO" in first_page_text or "Business Platinum Credit Card" in first_page_text:
        rows = parse_bmo_creditcard(pdf_path)
        bank = "BMO"
    elif bank == "BMO":
        if "Monthly Activity Details" in text:
            rows = parse_bmo_new(pdf_path)
        else:
            rows = parse_bmo_old(pdf_path)
    else:
        rows = extract_transactions(pdf_path)

    # --- Normalize to DataFrame and add bank column ---
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bank"] = bank

    return df

# ==================================================
# Batch Processor
# ==================================================
BASE_DIR = Path(__file__).parent.resolve()
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_pdfs():
    pdf_files = list(INPUT_DIR.glob("*.pdf")) + list(INPUT_DIR.glob("*.PDF"))
    if not pdf_files:
        log_message("warning", f"process_pdfs: No PDF files found in {INPUT_DIR}")
        return

    for pdf_file in pdf_files:
        log_message("info", f"process_pdfs: Processing: {pdf_file.name}")
        df = parse_statement(pdf_file)
        if df.empty:
            log_message("warning", f"process_pdfs: No transactions found in {pdf_file.name}")
            output_file = OUTPUT_DIR / f"_FAILED{pdf_file.stem}.xlsx"
            df.to_excel(output_file, index=False)
            continue
        output_file = OUTPUT_DIR / f"{pdf_file.stem}.xlsx"
        df.to_excel(output_file, index=False)
        log_message("info", f"process_pdfs: Saved: {output_file}")


if __name__ == "__main__":
    process_pdfs()
   
