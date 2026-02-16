"""
Bank Statement Categorizer Service

Categorizes bank statement transactions using client-specific rules.
Uses Gemini to extract transactions and applies rules for categorization.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from backend.utils.logger import log_message
from backend.services.gemini_pdf_extractor import GeminiPDFExtractor, _is_quota_error
from newversion.services.rule_base_mongodb import get_client_rule_base
from newversion.utils import build_categorized_csv_rows
import re
from io import BytesIO
import pandas as pd
from datetime import datetime
import asyncio
import os
import json
import hashlib
from dateutil import parser as date_parser
from newversion.services.amortization import find_loan_by_transaction

try:
    import google.genai as genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai_types = None
    log_message("warning", "google-genai not installed. Gemini fallback will not be available.")

class BankStatementCategorizer:
    """Categorize bank statement transactions using client rules."""

    AMA_CATEGORY = "Ask My Accountant"
    MIN_AI_CONFIDENCE = 0.70
    AI_RETRY_BACKOFF_SECONDS = [10, 20, 40]
    GENERIC_TRIGGER_BLOCKLIST = {
        "DIRECT PAY",
        "DIRECTPAY",
        "ACH PAYMENT",
        "ACH TRANSFER",
        "WIRE TRANSFER",
        "WIRE",
        "CASH CHECK",
        "PAYMENT",
        "ACH DEBIT",
        "ACH CREDIT",
    }
    
    def __init__(self):
        """Initialize the categorizer with Gemini PDF extractor."""
        self.pdf_extractor = GeminiPDFExtractor()
        
        # Initialize Gemini client for fallback categorization
        if GENAI_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self.gemini_client = genai.Client(api_key=api_key)
                self.gemini_model = "gemini-3-flash-preview"  # Use Gemini 3 Flash for categorization
                log_message("info", "Gemini 3 Flash fallback categorization enabled")
            else:
                self.gemini_client = None
                log_message("warning", "GOOGLE_API_KEY not set, Gemini fallback disabled")
        else:
            self.gemini_client = None
        self._rate_limit_retries_used = 0

    def _reset_run_counters(self) -> None:
        """Reset counters that are tracked per categorization run."""
        self._rate_limit_retries_used = 0

    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        """Return unique values while preserving first occurrence order."""
        out: List[str] = []
        seen = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Convert numeric-like values to float safely."""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                clean = value.replace(",", "").replace("$", "").strip()
                if clean == "":
                    return None
                return float(clean)
            return float(value)
        except Exception:
            return None

    def _build_transaction_fingerprint(self, transaction: Dict[str, Any]) -> str:
        """
        Build stable fingerprint for a transaction row so one-time edits can be targeted safely.
        """
        date = str(transaction.get("date") or "").strip()
        description = str(transaction.get("description") or "").strip().upper()
        credit = self._to_float(transaction.get("credit"))
        debit = self._to_float(transaction.get("debit"))
        amount = credit if credit not in (None, 0) else (-debit if debit not in (None, 0) else 0.0)
        bank_name = str(transaction.get("bank_name") or transaction.get("bank") or "").strip().upper()
        account_number = str(transaction.get("account_number") or "").strip().upper()
        raw = f"{date}|{description}|{amount:.2f}|{bank_name}|{account_number}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _is_ach_vendor_payment(description: str) -> bool:
        """
        Restrict vendor-code logic to ACH vendor payment style transactions only.
        """
        if not description:
            return False
        d = description.upper()
        has_ach_or_direct_pay = ("ACH" in d) or ("DIRECT PAY" in d) or ("DIRECTPAY" in d)
        has_vendor_payment_hint = any(
            token in d
            for token in ["VENDOR", "PAYMENT", "PAY ", "TRNSFR", "TRANSFER", "CCD", "PPD", "WEB"]
        )
        return has_ach_or_direct_pay and has_vendor_payment_hint

    @staticmethod
    def _extract_ach_vendor_code(description: str) -> Optional[str]:
        """
        Extract 4 digit vendor code from ACH vendor payment description.
        Only used after _is_ach_vendor_payment returns True.
        """
        if not description:
            return None
        d = description.upper()

        # Common forms: "DIRECT PAY 2013", "ACH ... CODE 3004", "VENDOR 4021"
        explicit_patterns = [
            r"\bDIRECT\s*PAY\s+(\d{4})\b",
            r"\bACH\b.*?\b(?:CODE|VENDOR|PAYEE|ID)\s*[:#-]?\s*(\d{4})\b",
            r"\b(?:VENDOR|PAYEE)\s*[:#-]?\s*(\d{4})\b",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, d)
            if match:
                return match.group(1)

        # Safe fallback: find standalone 4-digit token near ACH/DIRECT PAY
        tokens = re.findall(r"\b\d{4}\b", d)
        if not tokens:
            return None
        if "ACH" in d or "DIRECT PAY" in d or "DIRECTPAY" in d:
            return tokens[0]
        return None

    def _match_transaction_by_vendor_code(
        self,
        transaction: Dict[str, Any],
        rules: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Vendor-code matching for ACH vendor payment transactions.
        Returns: (rule, vendor_code, reason_if_not_found)
        """
        description = transaction.get("description", "")
        if not self._is_ach_vendor_payment(description):
            return None, None, None

        vendor_code = self._extract_ach_vendor_code(description)
        if not vendor_code:
            return None, None, "ACH vendor payment detected, but no valid 4 digit vendor code found."

        # Preferred source: explicit vendor_code field in rule
        for rule in rules:
            rule_code = str(rule.get("vendor_code") or "").strip()
            if rule_code and rule_code == vendor_code:
                return rule, vendor_code, None

        # Backward-compatible fallback: trigger contains same 4 digit code
        for rule in rules:
            triggers = rule.get("trigger_payee_contains", []) or []
            for trigger in triggers:
                if re.search(rf"\b{re.escape(vendor_code)}\b", str(trigger).upper()):
                    return rule, vendor_code, None

        return None, vendor_code, f"No rule mapping found for ACH vendor code {vendor_code}."

    def _compute_needs_review_reason(self, txn: Dict[str, Any]) -> Optional[str]:
        """
        Return a short review reason when row should be checked by staff.
        """
        source = (txn.get("categorization_source") or "").strip()
        account_name = (txn.get("account_name") or "").strip()
        confidence = self._to_float(txn.get("confidence_score")) or 0.0

        if account_name == self.AMA_CATEGORY:
            return txn.get("needs_review_reason") or "No matching client rule was found."
        if source == "AI" and confidence < 0.80:
            return "Matched by artificial intelligence with lower confidence."
        trigger = (txn.get("matched_trigger_text") or "").strip()
        if source == "Rule" and trigger and len(trigger) <= 3:
            return "Matched with a very short trigger keyword."
        return txn.get("needs_review_reason")

    def _is_generic_trigger_text(self, trigger: str) -> bool:
        """Check whether a trigger is too generic for coded vendor rows."""
        if not trigger:
            return False
        t = str(trigger).upper().strip()
        # If a 4 digit code is present, it is not generic.
        if re.search(r"\b\d{4}\b", t):
            return False
        compact = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", t)).strip()
        if compact in self.GENERIC_TRIGGER_BLOCKLIST:
            return True
        return any(compact.startswith(block + " ") for block in self.GENERIC_TRIGGER_BLOCKLIST)

    def _rule_is_generic_payment_only(self, rule: Dict[str, Any]) -> bool:
        """Return True when every trigger in a rule is generic payment wording."""
        triggers = rule.get("trigger_payee_contains", []) or []
        if not triggers:
            return False
        generic_count = sum(1 for trigger in triggers if self._is_generic_trigger_text(str(trigger)))
        return generic_count == len(triggers)

    def _filter_rules_for_unknown_vendor_code(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove generic payment-only rules when a vendor code is present but mapping is missing.
        This avoids incorrectly forcing coded rows into broad buckets.
        """
        filtered = [rule for rule in rules if not self._rule_is_generic_payment_only(rule)]
        return filtered if filtered else rules

    @staticmethod
    def _build_unknown_vendor_code_list(
        unknown_vendor_code_stats: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert unknown vendor code stats into stable response list."""
        out = []
        for code in sorted(unknown_vendor_code_stats.keys()):
            data = unknown_vendor_code_stats[code]
            out.append(
                {
                    "vendor_code": code,
                    "count": int(data.get("count", 0)),
                    "sample_descriptions": data.get("sample_descriptions") or [],
                }
            )
        return out

    def _build_run_warnings(
        self,
        ama_rate: float,
        unknown_vendor_codes: List[Dict[str, Any]],
        rate_limit_retries_used: int,
    ) -> List[str]:
        """Build plain-language warnings for run summary."""
        warnings: List[str] = []
        if ama_rate > 40:
            warnings.append(
                f"{ama_rate:.1f}% of transactions are Ask My Accountant. Consider refreshing rules with newer general ledger data."
            )
        if unknown_vendor_codes:
            total_rows = sum(int(item.get("count", 0)) for item in unknown_vendor_codes)
            warnings.append(
                f"Found {len(unknown_vendor_codes)} unknown vendor code pattern(s) across {total_rows} transaction row(s)."
            )
        if rate_limit_retries_used > 0:
            warnings.append(
                f"Rate limit retries were used {rate_limit_retries_used} time(s) during processing."
            )
        return warnings

    def _parse_check_register(self, file) -> Dict[str, Dict[str, str]]:
        """
        Parse check register CSV/Excel file to extract check information including payee name, category, and memo.
        
        Args:
            file: UploadFile object containing check register
            
        Returns:
            Dict mapping check numbers to dict with payee_name, category, and memo
            Example: {"1836": {"payee_name": "Victor Patri", "category": "Labor Expense", "memo": "4 Days Wor Driver Pay"}}
        """
        try:
            filename = (file.filename or "").lower()
            ext = filename.split(".")[-1] if "." in filename else ""
            
            file.file.seek(0)
            raw = file.file.read()
            file.file.seek(0)
            
            if not raw:
                return {}
            
            # Load dataframe
            if ext == "csv":
                df = pd.read_csv(BytesIO(raw))
            else:
                engine = "openpyxl" if ext in {"xlsx", "xlsm"} else None
                df = pd.read_excel(BytesIO(raw), engine=engine)
            
            if df is None or df.empty:
                return {}
            
            # Find columns (case-insensitive)
            norm_cols = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns}
            
            # Find check number column
            check_col = None
            for orig, norm in norm_cols.items():
                if norm in ["check number", "check_number", "check #", "check#", "chk", "chk number", "check no", "num"]:
                    check_col = orig
                    break
            
            # Find payee name column
            payee_col = None
            for orig, norm in norm_cols.items():
                if norm in ["payee", "payee name", "payee_name", "pay to", "vendor", "vendor name", "name"]:
                    payee_col = orig
                    break
            
            # Find category column (optional)
            category_col = None
            for orig, norm in norm_cols.items():
                if norm in ["category", "account", "account name", "account_name", "classification", "split"]:
                    category_col = orig
                    break

            # Find type column (optional)
            type_col = None
            for orig, norm in norm_cols.items():
                if norm in ["type", "transaction type", "txn type", "entry type"]:
                    type_col = orig
                    break

            # Find split amount column (optional)
            split_amount_col = None
            for orig, norm in norm_cols.items():
                if norm in ["split_amount", "split amount", "split amt", "amount split"]:
                    split_amount_col = orig
                    break
            
            # Find memo/description column (optional)
            memo_col = None
            for orig, norm in norm_cols.items():
                if norm in ["memo", "description", "notes", "details", "memo/description"]:
                    memo_col = orig
                    break
            
            if not check_col:
                log_message("warning", f"Check register missing check number column. Found: {list(df.columns)}")
                return {}
            
            if not payee_col:
                log_message("warning", f"Check register missing payee name column. Found: {list(df.columns)}")
                return {}
            
            # Build mapping with all available information
            check_mapping = {}
            for _, row in df.iterrows():
                check_num = row.get(check_col)
                payee = row.get(payee_col)
                
                if pd.isna(check_num):
                    continue
                
                # Normalize check number (remove spaces, leading zeros)
                check_num_str = str(check_num).strip().replace(" ", "").lstrip("0") or "0"
                
                # Build check info dict
                check_info = {}
                
                # Add payee name
                if not pd.isna(payee):
                    check_info["payee_name"] = str(payee).strip()
                
                # Add category if available
                if category_col and not pd.isna(row.get(category_col)):
                    check_info["category"] = str(row.get(category_col)).strip()
                
                # Add memo if available
                if memo_col and not pd.isna(row.get(memo_col)):
                    check_info["memo"] = str(row.get(memo_col)).strip()

                # Add type if available
                if type_col and not pd.isna(row.get(type_col)):
                    check_info["type"] = str(row.get(type_col)).strip()

                # Add split amount if available
                if split_amount_col and not pd.isna(row.get(split_amount_col)):
                    check_info["split_amount"] = str(row.get(split_amount_col)).strip()
                
                if check_info:  # Only add if we have at least some info
                    check_mapping[check_num_str] = check_info
            
            log_message(
                "info",
                f"Loaded {len(check_mapping)} check mappings from register "
                f"(columns: check_num, payee_name"
                f"{', category' if category_col else ''}"
                f"{', memo' if memo_col else ''}"
                f"{', type' if type_col else ''}"
                f"{', split_amount' if split_amount_col else ''})"
            )
            return check_mapping
            
        except Exception as e:
            log_message("warning", f"Failed to parse check register: {e}")
            import traceback
            log_message("warning", f"Traceback: {traceback.format_exc()}")
            return {}
    
    def _extract_check_number(self, description: str) -> Optional[str]:
        """
        Extract check number from transaction description.
        
        Args:
            description: Transaction description
            
        Returns:
            Check number or None
        """
        if not description:
            return None
        
        # Common patterns (number after or before "Check"/"CHK"):
        # "Check 1234", "CHK 1234", "2550 Check", "2550 CHK", "Deposited OR Cashed Check 1234"
        patterns = [
            r'(\d+)\s*[Cc]heck\b',           # "2550 Check" (number first - e.g. Wells Fargo)
            r'(\d+)\s*[Cc][Hh][Kk]\b',      # "2550 CHK" (number first)
            r'[Cc]heck\s*#?\s*(\d+)',       # "Check 1234", "Check # 1234"
            r'[Cc][Hh][Kk]\s*#?\s*(\d+)',   # "CHK 1234"
            r'[Cc]hk\s+(\d+)',              # "Chk 1234"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                check_num = match.group(1).lstrip("0") or "0"
                return check_num
        
        return None

    def _is_check_transaction(self, description: str) -> bool:
        """
        Hardcoded: return True if this transaction looks like a check (for Ask My Accountant when no CSV).
        Matches: check number in description, or description contains 'check' / 'chk' as word.
        """
        if not description:
            return False
        if self._extract_check_number(description) is not None:
            return True
        d = description.strip().lower()
        if not d:
            return False
        return "check" in d or d.startswith("chk ") or " chk " in d or d.endswith(" chk")
    
    async def _match_check_with_register(
        self,
        transaction: Dict[str, Any],
        check_mapping: Dict[str, Dict[str, str]],
        rules: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Match a check transaction using check register data. Returns all rules that match
        the payee (e.g. same payee in "Cost of Goods Sold" and "Wages" → both categories returned).
        
        Args:
            transaction: Transaction dict
            check_mapping: Dict of check_number -> dict with payee_name, category, memo
            rules: List of rule dictionaries
            
        Returns:
            Tuple of (matching rule list, matched_via_memo flag)
        """
        description = transaction.get("description", "")
        
        # Extract check number from description
        check_num = self._extract_check_number(description)
        if not check_num:
            return [], False
        
        # Look up check info in check register
        check_info = check_mapping.get(check_num)
        if not check_info:
            log_message("debug", f"Check #{check_num} not found in check register")
            return [], False
        
        payee_name = str(check_info.get("payee_name") or "").strip()
        memo_text = str(check_info.get("memo") or "").strip()

        if not payee_name and not memo_text:
            log_message("debug", f"Check #{check_num} has no payee or memo in check register")
            return [], False

        if payee_name:
            log_message("info", f"Found check #{check_num} -> payee: {payee_name}")
            temp_transaction = {
                "description": payee_name,
                "payee_name": payee_name
            }
            matched_rules = self._match_transaction_to_all_rules(temp_transaction, rules)
            if matched_rules:
                log_message(
                    "info",
                    f"Matched check #{check_num} (payee: {payee_name}) to {len(matched_rules)} rule(s): "
                    f"{[r.get('rule_id') for r in matched_rules]}"
                )
                return matched_rules, False

            if self.gemini_client:
                matched_rule, ai_confidence, _retries, _reason = await self._match_transaction_with_gemini(
                    temp_transaction,
                    rules
                )
                if matched_rule:
                    log_message(
                        "info",
                        f"Gemini matched check #{check_num} (payee: {payee_name}) to rule '{matched_rule.get('rule_id')}' "
                        f"with confidence {ai_confidence:.2f}"
                    )
                    return [matched_rule], False

        # If no category and no payee match, try memo-based matching
        if memo_text:
            memo_transaction = {
                "description": memo_text,
                "payee_name": payee_name,
            }
            matched_rules = self._match_transaction_to_all_rules(memo_transaction, rules)
            if matched_rules:
                log_message(
                    "info",
                    f"Matched check #{check_num} via memo text to {len(matched_rules)} rule(s): "
                    f"{[r.get('rule_id') for r in matched_rules]}"
                )
                return matched_rules, True

            if self.gemini_client:
                matched_rule, ai_confidence, _retries, _reason = await self._match_transaction_with_gemini(
                    memo_transaction,
                    rules
                )
                if matched_rule:
                    log_message(
                        "info",
                        f"Gemini matched check #{check_num} via memo to rule '{matched_rule.get('rule_id')}' "
                        f"with confidence {ai_confidence:.2f}"
                    )
                    return [matched_rule], True

        log_message("debug", f"No rule match for check #{check_num} (payee: {payee_name}, memo present: {bool(memo_text)})")
        return [], False
    
    @staticmethod
    def _normalize_date(value: Any) -> Optional[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return str(value).strip() or None
            # Convert to ISO date string
            if isinstance(dt, pd.Timestamp):
                return dt.date().isoformat()
            if isinstance(dt, datetime):
                return dt.date().isoformat()
            return str(dt).split(" ")[0]
        except Exception:
            return str(value).strip() or None

    def _extract_transactions_from_tabular(self, file) -> Dict[str, Any]:
        """
        Extract transactions from CSV/Excel bank statement exports.
        Supports common column variants:
        - Date / Transaction Date / Posted Date
        - Description / Memo / Payee / Details
        - Amount (signed) OR Credit+Debit columns
        """
        filename = (file.filename or "").lower()
        ext = filename.split(".")[-1] if "." in filename else ""

        file.file.seek(0)
        raw = file.file.read()
        file.file.seek(0)

        if not raw:
            return {"bank": "Unknown", "account_number": None, "transactions": []}

        # Load dataframe
        if ext == "csv":
            df = pd.read_csv(BytesIO(raw))
        else:
            engine = "openpyxl" if ext in {"xlsx", "xlsm"} else None
            df = pd.read_excel(BytesIO(raw), engine=engine)

        if df is None or df.empty:
            return {"bank": "Unknown", "account_number": None, "transactions": []}

        # Normalize columns for matching
        orig_cols = list(df.columns)
        norm_cols = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in orig_cols}

        def pick_col(candidates: List[str]) -> Optional[str]:
            for cand in candidates:
                for orig, norm in norm_cols.items():
                    if cand == norm:
                        return orig
            # fallback contains-match
            for cand in candidates:
                for orig, norm in norm_cols.items():
                    if cand in norm:
                        return orig
            return None

        date_col = pick_col(["date", "transaction date", "posted date", "post date"])
        desc_col = pick_col(["description", "memo", "details", "payee", "name"])
        amount_col = pick_col(["amount", "amt"])
        credit_col = pick_col(["credit", "deposit"])
        debit_col = pick_col(["debit", "withdrawal", "payment"])
        acct_col = pick_col(["account number", "acct", "account"])
        bank_col = pick_col(["bank", "bank name"])

        transactions: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            date_val = row.get(date_col) if date_col else None
            desc_val = row.get(desc_col) if desc_col else None

            description = "" if pd.isna(desc_val) else str(desc_val).strip()
            if not description:
                # Skip empty lines
                continue

            credit = None
            debit = None

            # Prefer explicit credit/debit columns if available
            if credit_col or debit_col:
                c = row.get(credit_col) if credit_col else None
                d = row.get(debit_col) if debit_col else None
                try:
                    if c is not None and not pd.isna(c) and str(c).strip() != "":
                        credit = float(str(c).replace(",", "").replace("$", "").strip())
                except Exception:
                    credit = None
                try:
                    if d is not None and not pd.isna(d) and str(d).strip() != "":
                        debit = float(str(d).replace(",", "").replace("$", "").strip())
                except Exception:
                    debit = None

            # Fallback to signed amount
            if credit is None and debit is None and amount_col:
                a = row.get(amount_col)
                try:
                    if a is not None and not pd.isna(a) and str(a).strip() != "":
                        amt = float(str(a).replace(",", "").replace("$", "").strip())
                        if amt >= 0:
                            credit = amt
                        else:
                            debit = abs(amt)
                except Exception:
                    pass

            # Keep only rows with amount
            if not credit and not debit:
                continue

            txn = {
                "date": self._normalize_date(date_val),
                "post_date": None,
                "description": description,
                "credit": credit,
                "debit": debit,
                "bank": (str(row.get(bank_col)).strip() if bank_col and not pd.isna(row.get(bank_col)) else "Unknown"),
                "account_number": (str(row.get(acct_col)).strip() if acct_col and not pd.isna(row.get(acct_col)) else None),
                "payee_name": None,
            }
            transactions.append(txn)

        return {
            "bank": "Unknown",
            "account_number": None,
            "start_date": None,
            "end_date": None,
            "transactions": transactions,
        }
    
    @staticmethod
    def _strip_bank_noise(text: str) -> str:
        """
        Strip common bank prefixes and symbols so only core payee name is used for matching.
        Does not hardcode any merchant names - only generic patterns (prefixes, *, phone-like, etc.).
        """
        if not text or not text.strip():
            return ""
        s = text.strip()
        # Remove common leading prefixes (case-insensitive)
        prefixes = [
            "Recurring Payment authorized on",
            "Recurring payment authorized on",
            "Recurring Payment authorized",
            "Purchase authorized on",
            "Purchase authorized",
            "Payment authorized on",
            "Payment authorized",
        ]
        for p in prefixes:
            if s.upper().startswith(p.upper()):
                s = s[len(p):].strip()
                break
        # Remove leading date (e.g. "07/01 ", "07/01/2025 ")
        s = re.sub(r"^\s*\d{1,2}/\d{1,2}(/\d{2,4})?\s*", "", s)
        # Replace asterisk with space so "Fsp*American" -> "Fsp American", "PY *Homeshield" -> "PY  Homeshield"
        s = re.sub(r"\*+", " ", s)
        # Remove common processor prefixes at start (Fsp, Sq, PY, etc.) - 2-4 letters + space
        s = re.sub(r"^\s*[A-Za-z]{2,4}\s+", "", s)
        s = s.strip()
        # Remove phone number patterns (e.g. 714-7021058, 949-476-5018)
        s = re.sub(r"\d{3}[-.]?\d{3}[-.]?\d{4}", " ", s)
        # Remove trailing " CA " and ref/card suffixes (e.g. "CA S585182400985379 Card 2979")
        s = re.sub(r"\s+CA\s+S?\d+\s+Card\s+\d+.*$", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+Card\s+\d+.*$", " ", s, flags=re.IGNORECASE)
        return s.strip()
    
    def _match_transaction_to_rule_with_trace(
        self,
        transaction: Dict[str, Any],
        rules: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Match a transaction to a rule and return both the rule and the trigger text used.
        """
        description = transaction.get("description", "")
        payee_name = transaction.get("payee_name", "")
        normalized_desc = self._strip_bank_noise(description)
        search_text = f"{normalized_desc} {description} {payee_name}".strip()
        if not search_text:
            return None, None

        search_text_upper = search_text.upper()

        for rule in rules:
            if rule.get("ach_vendor_only") and not self._is_ach_vendor_payment(description):
                continue
            triggers = rule.get("trigger_payee_contains", [])
            if not triggers:
                continue

            for trigger in triggers:
                trigger_upper = str(trigger).upper().strip()
                if not trigger_upper:
                    continue

                # Method 1: Exact substring match
                if trigger_upper in search_text_upper:
                    log_message(
                        "info",
                        f"Matched transaction '{description[:50]}' to rule '{rule.get('rule_id')}' "
                        f"using trigger '{trigger}'"
                    )
                    return rule, str(trigger).strip()

                # Method 2: Space-normalized and truncation matching
                search_no_space = search_text_upper.replace(" ", "")
                trigger_no_space = trigger_upper.replace(" ", "")
                if len(trigger_no_space) >= 3:
                    if trigger_no_space in search_no_space:
                        return rule, str(trigger).strip()
                    if trigger_no_space[:-1] in search_no_space and len(trigger_no_space) >= 5:
                        return rule, str(trigger).strip()
                    if len(search_no_space) >= 5 and search_no_space in trigger_no_space:
                        return rule, str(trigger).strip()
                    if search_no_space in trigger_no_space and len(search_no_space) >= 4:
                        overlap_ratio = len(search_no_space) / len(trigger_no_space)
                        if overlap_ratio >= 0.6:
                            return rule, str(trigger).strip()

                # Method 3: Word-based matching
                trigger_words = [w.strip() for w in trigger_upper.split() if len(w.strip()) > 2]
                if len(trigger_words) >= 2:
                    words_found = sum(1 for word in trigger_words if word in search_text_upper)
                    if words_found >= len(trigger_words) * 0.8:
                        return rule, str(trigger).strip()
                    if trigger_words[0] in search_text_upper and len(trigger_words[0]) >= 5:
                        if search_no_space.startswith(trigger_words[0].replace(" ", "")):
                            return rule, str(trigger).strip()
                    if len(trigger_words) == 2 and trigger_words[0] in search_text_upper:
                        second = trigger_words[1]
                        if second in search_no_space or (second[:-1] in search_no_space and len(second) >= 4):
                            return rule, str(trigger).strip()

        return None, None

    def _match_transaction_to_rule(
        self,
        transaction: Dict[str, Any],
        rules: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Backward compatible wrapper that returns only the matched rule.
        """
        rule, _trigger = self._match_transaction_to_rule_with_trace(transaction, rules)
        return rule
    
    def _match_transaction_to_all_rules(
        self,
        transaction: Dict[str, Any],
        rules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match a transaction to all rules whose trigger matches (same logic as _match_transaction_to_rule).
        Used for check transactions when a payee may map to multiple categories (e.g. Cost of Goods Sold and Wages).
        
        Returns:
            List of matching rule dicts (no duplicates by rule_id).
        """
        description = transaction.get("description", "")
        payee_name = transaction.get("payee_name", "")
        normalized_desc = self._strip_bank_noise(description)
        search_text = f"{normalized_desc} {description} {payee_name}".strip()
        if not search_text:
            return []
        
        search_text_upper = search_text.upper()
        search_no_space = search_text_upper.replace(" ", "")
        matched: List[Dict[str, Any]] = []
        seen_rule_ids: set = set()
        
        for rule in rules:
            rule_id = rule.get("rule_id")
            if rule_id in seen_rule_ids:
                continue
            if rule.get("ach_vendor_only") and not self._is_ach_vendor_payment(description):
                continue
            triggers = rule.get("trigger_payee_contains", [])
            if not triggers:
                continue
            for trigger in triggers:
                trigger_upper = str(trigger).upper().strip()
                if not trigger_upper:
                    continue
                if trigger_upper in search_text_upper:
                    seen_rule_ids.add(rule_id)
                    matched.append(rule)
                    log_message("info", f"Matched transaction '{description[:50]}' to rule '{rule_id}' using trigger '{trigger}'")
                    break
                trigger_no_space = trigger_upper.replace(" ", "")
                if len(trigger_no_space) >= 3:
                    if trigger_no_space in search_no_space:
                        seen_rule_ids.add(rule_id)
                        matched.append(rule)
                        log_message("info", f"Matched transaction '{description[:50]}' to rule '{rule_id}' (space-normalized)")
                        break
                    if trigger_no_space[:-1] in search_no_space and len(trigger_no_space) >= 5:
                        seen_rule_ids.add(rule_id)
                        matched.append(rule)
                        break
                    if len(search_no_space) >= 5 and search_no_space in trigger_no_space:
                        seen_rule_ids.add(rule_id)
                        matched.append(rule)
                        break
                    if search_no_space in trigger_no_space and len(search_no_space) >= 4:
                        if len(search_no_space) / len(trigger_no_space) >= 0.6:
                            seen_rule_ids.add(rule_id)
                            matched.append(rule)
                            break
                trigger_words = [w.strip() for w in trigger_upper.split() if len(w.strip()) > 2]
                if len(trigger_words) >= 2:
                    words_found = sum(1 for word in trigger_words if word in search_text_upper)
                    if words_found >= len(trigger_words) * 0.8:
                        seen_rule_ids.add(rule_id)
                        matched.append(rule)
                        break
                    if trigger_words[0] in search_text_upper and len(trigger_words[0]) >= 5:
                        if search_no_space.startswith(trigger_words[0].replace(" ", "")):
                            seen_rule_ids.add(rule_id)
                            matched.append(rule)
                            break
                    if len(trigger_words) == 2 and trigger_words[0] in search_text_upper:
                        second = trigger_words[1]
                        if second in search_no_space or (second[:-1] in search_no_space and len(second) >= 4):
                            seen_rule_ids.add(rule_id)
                            matched.append(rule)
                            break
        return matched
    
    async def _match_transaction_with_gemini(
        self,
        transaction: Dict[str, Any],
        rules: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], float, int, Optional[str]]:
        """
        Use Gemini to match a transaction to rules when manual matching fails.
        Gemini analyzes the description and finds which trigger_payee_contains matches.
        
        Args:
            transaction: Transaction dict with description/payee_name
            rules: List of rule dictionaries
            
        Returns:
            Tuple: (matching rule or None, confidence, retries_used, rejection_reason)
        """
        if not self.gemini_client:
            return None, 0.0, 0, "Artificial intelligence matching is disabled."
        
        retries_used = 0
        try:
            description = transaction.get("description", "")
            payee_name = transaction.get("payee_name", "")
            
            if not description and not payee_name:
                return None, 0.0, retries_used, None
            
            # Build complete rules list with ALL triggers for Gemini
            rules_list = []
            for rule in rules:
                triggers = rule.get("trigger_payee_contains", [])
                if not triggers:
                    continue
                
                rules_list.append({
                    "rule_id": rule.get("rule_id"),
                    "account_id": rule.get("account_id"),
                    "account_name": rule.get("account_name"),
                    "trigger_payee_contains": triggers
                })
            
            # Format rules for prompt
            rules_json = json.dumps(rules_list, indent=2)
            
            prompt = f"""### UNIVERSAL KEYWORD RECONCILIATION
Task: Map the bank description to the best category from the provided rules.

1. Ignore noise: Do NOT use the raw string as-is. Strip:
   - Any leading prefix (e.g. "Recurring Payment authorized on", "Fsp*", "PY *", "Sq*"). The asterisk (*) and text before it are NOT part of the payee name.
   - Phone numbers, " CA ", "Card 1234" and similar suffixes.
   - Extract ONLY the core merchant/payee name (e.g. from "PY *Homeshield Inl 714-7021058 CA..." use "Homeshield Inl"; from "Fsp*American Mini 949-476-5018 CA..." use "American Mini").
2. Keyword match: Compare this extracted core name against EVERY string in trigger_payee_contains for ALL rules. If the core name is a subset of a trigger, or the trigger is a subset of the core name, or they share the same leading words, it is a MANDATORY MATCH.
3. Do not require exact or full-string match. If the merchant identity is clearly the same (same first word, truncation, or partial name), map it.

Data:
Bank Description: {description}
Available Rules: {rules_json}

Return only: {{"account_name": "...", "account_id": "...", "confidence": 0.0}} or {{"account_name": null, "account_id": null, "confidence": 0.0}} if no match.
Confidence must be between 0 and 1."""

            # Call Gemini with retry and exponential backoff on rate-limit errors
            response = None
            last_error: Optional[Exception] = None
            max_attempts = 1 + len(self.AI_RETRY_BACKOFF_SECONDS)
            for attempt in range(max_attempts):
                try:
                    response = await asyncio.to_thread(
                        self.gemini_client.models.generate_content,
                        model=self.gemini_model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=0,
                            max_output_tokens=220,
                            response_mime_type="application/json"
                        )
                    )
                    break
                except Exception as call_error:
                    last_error = call_error
                    if _is_quota_error(call_error) and attempt < len(self.AI_RETRY_BACKOFF_SECONDS):
                        wait_seconds = self.AI_RETRY_BACKOFF_SECONDS[attempt]
                        retries_used += 1
                        self._rate_limit_retries_used += 1
                        log_message(
                            "warning",
                            f"Artificial intelligence match rate limit hit. Retrying in {wait_seconds} seconds (attempt {attempt + 1}/{max_attempts - 1})."
                        )
                        await asyncio.sleep(wait_seconds)
                        continue
                    raise

            if response is None:
                if last_error:
                    raise last_error
                return None, 0.0, retries_used, None

            response_text = response.text if hasattr(response, 'text') else str(response)
            response_text = response_text.strip()

            if not response_text:
                return None, 0.0, retries_used, None

            # Parse JSON - accept either account_name/account_id or rule_id format
            json_match = re.search(r'\{[^}]*(?:"account_name"|"rule_id")[^}]*\}', response_text)
            if json_match:
                response_text = json_match.group(0)

            result = json.loads(response_text)

            # Parse confidence value safely
            ai_confidence = self._to_float(result.get("confidence"))
            if ai_confidence is None:
                ai_confidence = 0.0
            ai_confidence = max(0.0, min(1.0, ai_confidence))

            # New format: account_name + account_id
            gemini_account_name = result.get("account_name")
            gemini_account_id = result.get("account_id")

            if gemini_account_name or gemini_account_id:
                # Find the rule that matches this account_name or account_id
                for rule in rules:
                    rule_name = (rule.get("account_name") or "").strip()
                    rule_id = (rule.get("account_id") or "").strip()
                    if not rule_name and not rule_id:
                        continue
                    name_match = gemini_account_name and rule_name and (
                        rule_name.lower() == gemini_account_name.lower() or
                        gemini_account_name.lower() in rule_name.lower()
                    )
                    id_match = gemini_account_id and rule_id and (
                        rule_id.lower() == gemini_account_id.lower() or
                        gemini_account_id.lower() in rule_id.lower()
                    )
                    if name_match or id_match:
                        if ai_confidence < self.MIN_AI_CONFIDENCE:
                            reason = (
                                f"Artificial intelligence confidence {ai_confidence:.2f} is below "
                                f"the minimum threshold {self.MIN_AI_CONFIDENCE:.2f}."
                            )
                            return None, ai_confidence, retries_used, reason
                        log_message(
                            "info",
                            f"Gemini matched '{description[:50]}' -> '{rule.get('account_name')}' "
                            f"(account_id: {rule.get('account_id')}, confidence: {ai_confidence:.2f})"
                        )
                        return rule, ai_confidence, retries_used, None

            # Fallback: old format rule_id
            matched_rule_id = result.get("rule_id")
            if matched_rule_id:
                for rule in rules:
                    if rule.get("rule_id") == matched_rule_id:
                        if ai_confidence < self.MIN_AI_CONFIDENCE:
                            reason = (
                                f"Artificial intelligence confidence {ai_confidence:.2f} is below "
                                f"the minimum threshold {self.MIN_AI_CONFIDENCE:.2f}."
                            )
                            return None, ai_confidence, retries_used, reason
                        log_message(
                            "info",
                            f"Gemini matched '{description[:50]}' to rule '{matched_rule_id}' "
                            f"-> '{rule.get('account_name')}' (confidence: {ai_confidence:.2f})"
                        )
                        return rule, ai_confidence, retries_used, None

            return None, ai_confidence, retries_used, None
            
        except json.JSONDecodeError as e:
            log_message("debug", f"Gemini matching JSON parse failed for '{description[:50]}': {e}")
            return None, 0.0, retries_used, None
        except Exception as e:
            log_message("debug", f"Gemini matching failed for '{description[:50]}': {e}")
            if _is_quota_error(e):
                return None, 0.0, retries_used, "Artificial intelligence matching failed due to rate limit."
            return None, 0.0, retries_used, None
    
    async def categorize_bank_statement(
        self,
        client_id: str,
        bank_statement_file,
        check_register_file=None
    ) -> Dict[str, Any]:
        """
        Categorize bank statement transactions using client rules.
        
        Args:
            client_id: Client ID to lookup rules
            bank_statement_file: UploadFile object containing bank statement PDF
            
        Returns:
            Dict with categorized transactions:
            {
                "client_id": str,
                "client_name": str,
                "bank": str,
                "account_number": str,
                "start_date": str,
                "end_date": str,
                "transactions": [
                    {
                        "date": str,
                        "description": str,
                        "credit": float | None,
                        "debit": float | None,
                        "account_number": str,
                        "bank_name": str,
                        "account_name": str | None,  # Only present if rule matched
                        "category_type": str | None  # Only present if rule matched
                    }
                ]
            }
        """
        try:
            self._reset_run_counters()
            # Step 1: Get client rules from MongoDB using client_id
            log_message("info", f"Looking up rules for client_id: {client_id}")
            client_rules_doc = await get_client_rule_base(client_id)
            
            if not client_rules_doc:
                raise ValueError(f"No rules found for client_id: {client_id}")
            
            rules = client_rules_doc.get("rules", [])
            client_name = client_rules_doc.get("client_name", "Unknown")
            log_message("info", f"Found {len(rules)} rules for client_id: {client_id} (client_name: {client_name})")
            
            if not rules:
                log_message("warning", f"Client {client_id} has no rules defined")
            
            # Step 2: Extract transactions from bank statement (PDF via Gemini, CSV/Excel via pandas)
            filename_lower = (getattr(bank_statement_file, "filename", "") or "").lower()
            if filename_lower.endswith(".pdf"):
                log_message("info", "Extracting transactions from PDF using Gemini")
                extraction_result = await self.pdf_extractor.extract_transactions(
                    bank_statement_file,
                    is_credit_card=False
                )
            else:
                log_message("info", "Extracting transactions from CSV/Excel bank export")
                extraction_result = self._extract_transactions_from_tabular(bank_statement_file)
            
            if not extraction_result:
                raise ValueError("Failed to extract transactions from bank statement")

            extraction_retry_count = int(extraction_result.get("rate_limit_retries_used") or 0)
            if extraction_retry_count > 0:
                self._rate_limit_retries_used += extraction_retry_count
            
            transactions = extraction_result.get("transactions", [])
            log_message("info", f"Extracted {len(transactions)} transactions from bank statement")
            
            # Step 2.5: Parse check register if provided
            check_mapping = {}
            if check_register_file:
                log_message("info", "Parsing check register file")
                check_mapping = self._parse_check_register(check_register_file)
            
            # Step 3: Categorize each transaction using rules
            categorized_transactions = []
            gemini_fallback_count = 0
            check_matched_count = 0
            unknown_vendor_code_stats: Dict[str, Dict[str, Any]] = {}
            
            for transaction in transactions:
                # Create output transaction and always include explainability fields
                categorized_txn = {
                    "date": transaction.get("date"),
                    "description": transaction.get("description", ""),
                    "credit": transaction.get("credit"),
                    "debit": transaction.get("debit"),
                    "account_number": transaction.get("account_number"),
                    "bank_name": transaction.get("bank") or extraction_result.get("bank", "Unknown"),
                    "account_name": self.AMA_CATEGORY,
                    "category_type": None,
                    "categorization_source": self.AMA_CATEGORY,
                    "matched_rule_id": None,
                    "matched_trigger_text": None,
                    "confidence_score": 0.0,
                    "needs_review_reason": None,
                }
                categorized_txn["transaction_fingerprint"] = self._build_transaction_fingerprint(categorized_txn)
                
                # Detect check transactions (hardcoded: check number or description contains check/chk)
                description = transaction.get("description", "")
                check_num = self._extract_check_number(description)
                ach_vendor_unmatched_reason: Optional[str] = None
                
                # If check register CSV provided and has info for this check number, use it directly
                check_has_category_from_csv = False
                if check_num and check_mapping and check_num in check_mapping:
                    check_info = check_mapping.get(check_num)
                    
                    # Add payee name if available
                    if check_info.get("payee_name"):
                        categorized_txn["payee_name"] = check_info.get("payee_name")
                    
                    # If category provided in check CSV, use it directly (don't check rules)
                    if check_info.get("category"):
                        categorized_txn["account_name"] = check_info.get("category")
                        categorized_txn["category_type"] = check_info.get("type") or "Expense"
                        if check_info.get("split_amount"):
                            categorized_txn["split_amount"] = check_info.get("split_amount")
                        categorized_txn["categorization_source"] = "Check Register"
                        categorized_txn["matched_trigger_text"] = f"Check {check_num} category from check register"
                        categorized_txn["confidence_score"] = 1.0
                        check_has_category_from_csv = True
                        log_message("info", f"Check #{check_num}: Using category from CSV: {check_info.get('category')}")
                    
                    # If category was provided in CSV, skip rule matching
                    if check_has_category_from_csv:
                        categorized_txn["needs_review_reason"] = self._compute_needs_review_reason(categorized_txn)
                        categorized_txn["needs_review_flag"] = bool(categorized_txn["needs_review_reason"])
                        categorized_transactions.append(categorized_txn)
                        continue
                
                # If no check register CSV uploaded but this is a check -> keep Ask My Accountant and mark for review
                if not check_mapping and self._is_check_transaction(description):
                    categorized_txn["needs_review_reason"] = "Check transaction has no check register mapping."
                    categorized_txn["needs_review_flag"] = True
                    categorized_transactions.append(categorized_txn)
                    continue
                
                # Try to match transaction to rule(s)
                matched_rules: List[Dict[str, Any]] = []
                matched_via_check = False  # multiple categories only for check transactions
                matched_via_vendor_code = False
                matched_via_ai = False
                matched_via_memo = False
                matched_trigger_text: Optional[str] = None
                vendor_code_used: Optional[str] = None

                # First, if ACH vendor payment has a mapped 4 digit code, use it before generic text matching
                vendor_rule, vendor_code, vendor_reason = self._match_transaction_by_vendor_code(transaction, rules)
                if vendor_reason:
                    ach_vendor_unmatched_reason = vendor_reason
                if vendor_rule:
                    matched_rules = [vendor_rule]
                    matched_via_vendor_code = True
                    vendor_code_used = vendor_code
                    matched_trigger_text = f"ACH vendor code {vendor_code}"

                # Track unknown coded vendor rows for review summary
                if vendor_code and (not vendor_rule):
                    stat = unknown_vendor_code_stats.setdefault(
                        vendor_code,
                        {"count": 0, "sample_descriptions": []},
                    )
                    stat["count"] += 1
                    desc_sample = (transaction.get("description") or "").strip()
                    if desc_sample and len(stat["sample_descriptions"]) < 3:
                        stat["sample_descriptions"].append(desc_sample)
                    stat["sample_descriptions"] = self._dedupe_preserve_order(stat["sample_descriptions"])

                candidate_rules = rules
                if vendor_code and (not vendor_rule):
                    candidate_rules = self._filter_rules_for_unknown_vendor_code(rules)
                
                # Then, check if this is a check transaction and we have check register (can match multiple rules)
                if not matched_rules and check_mapping and check_num:
                    matched_rules, matched_via_memo = await self._match_check_with_register(transaction, check_mapping, candidate_rules)
                    if matched_rules:
                        check_matched_count += 1
                        matched_via_check = True
                        if matched_via_memo:
                            matched_trigger_text = f"Check {check_num} memo from check register"
                        else:
                            matched_trigger_text = f"Check {check_num} payee from check register"
                
                # If not matched via check register, try single manual matching
                if not matched_rules:
                    matched_rule, matched_trigger = self._match_transaction_to_rule_with_trace(transaction, candidate_rules)
                    if matched_rule:
                        matched_rules = [matched_rule]
                        matched_trigger_text = matched_trigger
                
                # If manual matching failed, try Gemini fallback
                if not matched_rules and self.gemini_client:
                    matched_rule, ai_confidence, _retries_used, ai_rejection_reason = await self._match_transaction_with_gemini(
                        transaction,
                        candidate_rules
                    )
                    if matched_rule:
                        matched_rules = [matched_rule]
                        matched_via_ai = True
                        gemini_fallback_count += 1
                        triggers = matched_rule.get("trigger_payee_contains", []) or []
                        matched_trigger_text = ", ".join(str(t) for t in triggers[:3]) or "Gemini semantic match"
                        categorized_txn["confidence_score"] = ai_confidence
                    elif ai_rejection_reason:
                        categorized_txn["needs_review_reason"] = ai_rejection_reason
                
                if matched_rules:
                    if matched_via_check:
                        # Check transaction: allow multiple categories (comma-separated + matched_categories)
                        categorized_txn["account_name"] = ", ".join(r.get("account_name") or "" for r in matched_rules)
                        categorized_txn["category_type"] = ", ".join(r.get("category_type") or "" for r in matched_rules)
                        categorized_txn["account_id"] = ", ".join(r.get("account_id") or "" for r in matched_rules)
                        categorized_txn["matched_categories"] = [
                            {"account_name": r.get("account_name"), "category_type": r.get("category_type")}
                            for r in matched_rules
                        ]
                    else:
                        # Non-check: single category only (no matched_categories)
                        categorized_txn["account_name"] = matched_rules[0].get("account_name")
                        categorized_txn["category_type"] = matched_rules[0].get("category_type")
                        categorized_txn["account_id"] = matched_rules[0].get("account_id")
                    categorized_txn["matched_rule_id"] = matched_rules[0].get("rule_id")
                    categorized_txn["matched_trigger_text"] = matched_trigger_text
                    if matched_via_vendor_code:
                        categorized_txn["categorization_source"] = "Vendor Code"
                        categorized_txn["confidence_score"] = 0.98
                        categorized_txn["vendor_code"] = vendor_code_used
                    elif matched_via_check:
                        categorized_txn["categorization_source"] = "Check Register"
                        categorized_txn["confidence_score"] = 0.90
                        if matched_via_memo:
                            categorized_txn["matched_via_memo"] = True
                    elif matched_via_ai:
                        categorized_txn["categorization_source"] = "AI"
                        if categorized_txn.get("confidence_score") in (None, 0, 0.0):
                            categorized_txn["confidence_score"] = 0.74
                    else:
                        categorized_txn["categorization_source"] = "Rule"
                        categorized_txn["confidence_score"] = 0.95
                else:
                    log_message(
                        "debug",
                        f"No rule match for transaction: {transaction.get('description', '')[:50]}"
                    )
                    if ach_vendor_unmatched_reason:
                        categorized_txn["needs_review_reason"] = ach_vendor_unmatched_reason
                
                # Check if category contains "Loan" and client_id is available
                category = categorized_txn.get("account_name", "")
                should_split_loan = client_id and category and "loan" in category.lower()
                log_message("info", f"[TEST] Check register file provided: {category}, should_split_loan={should_split_loan}")
                
                if should_split_loan:
                    try:
                        # Parse transaction date
                        transaction_date = date_parser.parse(categorized_txn.get("date"))
                        
                        # Find loan details
                        loan_result = await find_loan_by_transaction(
                            transaction_date=transaction_date,
                            memo=categorized_txn.get("description", ""),
                            amount=abs(categorized_txn.get("debit") or categorized_txn.get("credit") or 0),
                            client_id=client_id
                        )
                        
                        if loan_result and loan_result.get("schedule"):
                            schedule = loan_result.get("schedule")
                            interest_expense = schedule.get("interest_expense", 0)
                            principal_liability = schedule.get("principal_liability", 0)
                            
                            # Only split if we have valid amounts
                            if interest_expense > 0 and principal_liability > 0:
                                # First transaction: Principal with original category
                                principal_txn = categorized_txn.copy()
                                if categorized_txn.get("credit"):
                                    principal_txn["credit"] = principal_liability
                                    principal_txn["debit"] = None
                                else:
                                    principal_txn["credit"] = None
                                    principal_txn["debit"] = principal_liability
                                principal_txn["transaction_fingerprint"] = self._build_transaction_fingerprint(principal_txn)
                                principal_txn["needs_review_reason"] = self._compute_needs_review_reason(principal_txn)
                                principal_txn["needs_review_flag"] = bool(principal_txn["needs_review_reason"])
                                
                                categorized_transactions.append(principal_txn)
                                
                                # Second transaction: Interest with "Interest" category
                                interest_txn = categorized_txn.copy()
                                interest_txn["account_name"] = "Interest"
                                if categorized_txn.get("credit"):
                                    interest_txn["credit"] = interest_expense
                                    interest_txn["debit"] = None
                                else:
                                    interest_txn["credit"] = None
                                    interest_txn["debit"] = interest_expense
                                interest_txn["transaction_fingerprint"] = self._build_transaction_fingerprint(interest_txn)
                                interest_txn["needs_review_reason"] = self._compute_needs_review_reason(interest_txn)
                                interest_txn["needs_review_flag"] = bool(interest_txn["needs_review_reason"])
                                
                                categorized_transactions.append(interest_txn)
                                
                                log_message("info", f"Split loan transaction: {categorized_txn.get('description', '')[:50]} - Principal: {principal_liability}, Interest: {interest_expense}")
                                continue  # Skip adding the original transaction
                            else:
                                log_message("warning", f"Loan schedule found but invalid amounts: interest={interest_expense}, principal={principal_liability}")
                        else:
                            log_message("warning", f"Loan transaction found but no schedule available: {categorized_txn.get('description', '')[:50]}")
                    except Exception as e:
                        log_message("error", f"Error splitting loan transaction {categorized_txn.get('description', '')[:50]}: {e}")
                        # Fall through to add original transaction

                categorized_txn["needs_review_reason"] = self._compute_needs_review_reason(categorized_txn)
                categorized_txn["needs_review_flag"] = bool(categorized_txn["needs_review_reason"])
                categorized_transactions.append(categorized_txn)

            source_counts: Dict[str, int] = {}
            for txn in categorized_transactions:
                src = txn.get("categorization_source") or self.AMA_CATEGORY
                source_counts[src] = source_counts.get(src, 0) + 1

            matched_count = sum(
                1 for txn in categorized_transactions
                if (txn.get("account_name") or "").strip() and (txn.get("account_name") or "").strip() != self.AMA_CATEGORY
            )
            unmatched_count = len(categorized_transactions) - matched_count
            ama_rate = (unmatched_count / len(categorized_transactions) * 100.0) if categorized_transactions else 0.0
            unknown_vendor_codes = self._build_unknown_vendor_code_list(unknown_vendor_code_stats)
            warnings = self._build_run_warnings(
                ama_rate=ama_rate,
                unknown_vendor_codes=unknown_vendor_codes,
                rate_limit_retries_used=self._rate_limit_retries_used,
            )
            
            if check_matched_count > 0:
                log_message(
                    "info",
                    f"Check register matched {check_matched_count} check transactions"
                )
            
            if gemini_fallback_count > 0:
                log_message(
                    "info",
                    f"Gemini fallback matched {gemini_fallback_count} additional transactions"
                )
            
            log_message(
                "info",
                f"Matched {matched_count} out of {len(categorized_transactions)} transactions"
            )
            
            # Step 4: Build final result
            result = {
                "client_id": client_id,
                "client_name": client_name,
                "bank": extraction_result.get("bank", "Unknown"),
                "account_number": extraction_result.get("account_number"),
                "start_date": extraction_result.get("start_date"),
                "end_date": extraction_result.get("end_date"),
                "total_transactions": len(categorized_transactions),
                "matched_transactions": matched_count,
                "unmatched_transactions": unmatched_count,
                "source_counts": source_counts,
                "warnings": warnings,
                "unknown_vendor_codes": unknown_vendor_codes,
                "ama_rate": round(ama_rate, 2),
                "rate_limit_retries_used": self._rate_limit_retries_used,
                "transactions": categorized_transactions
            }
            
            return result
            
        except Exception as e:
            log_message("error", f"Failed to categorize bank statement: {e}")
            raise Exception(f"Bank statement categorization failed: {e}")

    async def recategorize_existing_transactions(
        self,
        client_id: str,
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Re-apply current client rules to existing transactions without re-uploading statement files.
        Used by the review screen after rule edits.
        """
        self._reset_run_counters()
        client_rules_doc = await get_client_rule_base(client_id)
        if not client_rules_doc:
            raise ValueError(f"No rules found for client_id: {client_id}")

        rules = client_rules_doc.get("rules", [])
        recategorized: List[Dict[str, Any]] = []
        unknown_vendor_code_stats: Dict[str, Dict[str, Any]] = {}

        for transaction in transactions or []:
            categorized_txn = {
                "date": transaction.get("date"),
                "description": transaction.get("description", ""),
                "credit": transaction.get("credit"),
                "debit": transaction.get("debit"),
                "account_number": transaction.get("account_number"),
                "bank_name": transaction.get("bank_name") or transaction.get("bank") or "Unknown",
                "payee_name": transaction.get("payee_name"),
                "account_name": self.AMA_CATEGORY,
                "category_type": None,
                "categorization_source": self.AMA_CATEGORY,
                "matched_rule_id": None,
                "matched_trigger_text": None,
                "confidence_score": 0.0,
                "needs_review_reason": None,
            }
            categorized_txn["transaction_fingerprint"] = transaction.get("transaction_fingerprint") or self._build_transaction_fingerprint(categorized_txn)

            matched_rule = None
            matched_trigger = None

            vendor_rule, vendor_code, vendor_reason = self._match_transaction_by_vendor_code(transaction, rules)
            if vendor_rule:
                matched_rule = vendor_rule
                categorized_txn["categorization_source"] = "Vendor Code"
                categorized_txn["confidence_score"] = 0.98
                categorized_txn["matched_trigger_text"] = f"ACH vendor code {vendor_code}"
                categorized_txn["vendor_code"] = vendor_code
            else:
                candidate_rules = rules
                if vendor_code and (not vendor_rule):
                    stat = unknown_vendor_code_stats.setdefault(
                        vendor_code,
                        {"count": 0, "sample_descriptions": []},
                    )
                    stat["count"] += 1
                    desc_sample = (transaction.get("description") or "").strip()
                    if desc_sample and len(stat["sample_descriptions"]) < 3:
                        stat["sample_descriptions"].append(desc_sample)
                    stat["sample_descriptions"] = self._dedupe_preserve_order(stat["sample_descriptions"])
                    candidate_rules = self._filter_rules_for_unknown_vendor_code(rules)

                rule_match, trigger_match = self._match_transaction_to_rule_with_trace(transaction, candidate_rules)
                if rule_match:
                    matched_rule = rule_match
                    matched_trigger = trigger_match
                    categorized_txn["categorization_source"] = "Rule"
                    categorized_txn["confidence_score"] = 0.95
                    categorized_txn["matched_trigger_text"] = trigger_match
                elif self.gemini_client:
                    ai_rule, ai_confidence, _retries_used, ai_rejection_reason = await self._match_transaction_with_gemini(
                        transaction,
                        candidate_rules
                    )
                    if ai_rule:
                        matched_rule = ai_rule
                        categorized_txn["categorization_source"] = "AI"
                        categorized_txn["confidence_score"] = ai_confidence
                        triggers = ai_rule.get("trigger_payee_contains", []) or []
                        categorized_txn["matched_trigger_text"] = ", ".join(str(t) for t in triggers[:3]) or "Gemini semantic match"
                    elif ai_rejection_reason:
                        categorized_txn["needs_review_reason"] = ai_rejection_reason
                    elif vendor_reason:
                        categorized_txn["needs_review_reason"] = vendor_reason
                elif vendor_reason:
                    categorized_txn["needs_review_reason"] = vendor_reason

            if matched_rule:
                categorized_txn["account_name"] = matched_rule.get("account_name")
                categorized_txn["category_type"] = matched_rule.get("category_type")
                categorized_txn["account_id"] = matched_rule.get("account_id")
                categorized_txn["matched_rule_id"] = matched_rule.get("rule_id")
                if not categorized_txn.get("matched_trigger_text"):
                    categorized_txn["matched_trigger_text"] = matched_trigger

            categorized_txn["needs_review_reason"] = self._compute_needs_review_reason(categorized_txn)
            categorized_txn["needs_review_flag"] = bool(categorized_txn["needs_review_reason"])
            recategorized.append(categorized_txn)

        matched_count = sum(
            1 for txn in recategorized
            if (txn.get("account_name") or "").strip() and (txn.get("account_name") or "").strip() != self.AMA_CATEGORY
        )
        source_counts: Dict[str, int] = {}
        for txn in recategorized:
            src = txn.get("categorization_source") or self.AMA_CATEGORY
            source_counts[src] = source_counts.get(src, 0) + 1

        unmatched_count = len(recategorized) - matched_count
        ama_rate = (unmatched_count / len(recategorized) * 100.0) if recategorized else 0.0
        unknown_vendor_codes = self._build_unknown_vendor_code_list(unknown_vendor_code_stats)
        warnings = self._build_run_warnings(
            ama_rate=ama_rate,
            unknown_vendor_codes=unknown_vendor_codes,
            rate_limit_retries_used=self._rate_limit_retries_used,
        )

        return {
            "transactions": recategorized,
            "matched_transactions": matched_count,
            "unmatched_transactions": unmatched_count,
            "source_counts": source_counts,
            "warnings": warnings,
            "unknown_vendor_codes": unknown_vendor_codes,
            "ama_rate": round(ama_rate, 2),
            "rate_limit_retries_used": self._rate_limit_retries_used,
        }

    async def categorize_bank_statement_streaming(
        self,
        client_id: str,
        bank_statement_file,
        batch_size: int = 10,
        check_register_file=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Categorize bank statement in batches and yield each batch for streaming.
        Now includes fake progress events during PDF extraction to keep the connection alive
        and give better UX feedback.
        """
        try:
            # ──────────────────────────────────────────────────────────────
            # 1. Immediate connection feedback
            # ──────────────────────────────────────────────────────────────
            yield {
                "batch_num": 0,
                "total_batches": "starting",
                "batch_csv": "",
                "batch_transactions": [],
                "meta": {
                    "status": "connected",
                    "message": "Connected – preparing to process bank statement"
                }
            }

            client_rules_doc = await get_client_rule_base(client_id)
            if not client_rules_doc:
                raise ValueError(f"No rules found for client_id: {client_id}")

            rules = client_rules_doc.get("rules", [])
            client_name = client_rules_doc.get("client_name", "Unknown")
            log_message("info", f"Streaming categorize: {len(rules)} rules for client_id={client_id}")

            filename_lower = (getattr(bank_statement_file, "filename", "") or "").lower()
            is_pdf = filename_lower.endswith(".pdf")

            # ──────────────────────────────────────────────────────────────
            # 2. Fake progress for PDFs (the long blocking part)
            # Adjust sleep time based on your average Gemini call duration (~20–40 seconds)
            # ──────────────────────────────────────────────────────────────
            if is_pdf:
                log_message("info", "PDF detected – sending simulated progress during extraction")
                for step in range(1, 11):  # 10 steps → 10% to 100%
                    yield {
                        "batch_num": 0,
                        "total_batches": "extracting",
                        "batch_csv": "",
                        "batch_transactions": [],
                        "meta": {
                            "status": "extracting_pdf",
                            "progress": step * 10,
                            "message": f"AI analyzing PDF pages ({step*10}%) — usually takes 20–60 seconds"
                        }
                    }
                    await asyncio.sleep(2.5)  # ≈ 25 seconds total — tune this value

            # ──────────────────────────────────────────────────────────────
            # 3. Actual transaction extraction (this is the slow part for PDFs)
            # ──────────────────────────────────────────────────────────────
            if is_pdf:
                extraction_result = await self.pdf_extractor.extract_transactions(
                    bank_statement_file,
                    is_credit_card=False,
                )
            else:
                extraction_result = self._extract_transactions_from_tabular(bank_statement_file)

            if not extraction_result:
                raise ValueError("Failed to extract transactions from bank statement")

            # Extraction finished – notify frontend
            yield {
                "batch_num": 0,
                "total_batches": "extraction_done",
                "batch_csv": "",
                "batch_transactions": [],
                "meta": {
                    "status": "categorizing",
                    "message": "Extraction complete – now categorizing transactions"
                }
            }

            transactions = extraction_result.get("transactions", [])
            bank = extraction_result.get("bank", "Unknown")
            account_number = extraction_result.get("account_number") or "NA"
            log_message("info", f"Streaming categorize: {len(transactions)} transactions, batch_size={batch_size}")

            # Parse check register if provided
            check_mapping = {}
            if check_register_file:
                log_message("info", "Parsing check register file for streaming")
                check_mapping = self._parse_check_register(check_register_file)

            total_batches = max(1, (len(transactions) + batch_size - 1) // batch_size)
            matched_count = 0

            # ──────────────────────────────────────────────────────────────
            # 4. Real batch processing – this part is fast
            # ──────────────────────────────────────────────────────────────
            gemini_fallback_count = 0
            check_matched_count = 0
            for batch_start in range(0, len(transactions), batch_size):
                batch = transactions[batch_start : batch_start + batch_size]
                categorized_batch: List[Dict[str, Any]] = []

                for transaction in batch:
                    categorized_txn = {
                        "date": transaction.get("date"),
                        "description": transaction.get("description", ""),
                        "credit": transaction.get("credit"),
                        "debit": transaction.get("debit"),
                        "account_number": transaction.get("account_number"),
                        "bank_name": transaction.get("bank") or bank,
                    }
                    
                    # Detect check transactions (hardcoded: check number or description contains check/chk)
                    description = transaction.get("description", "")
                    check_num = self._extract_check_number(description)
                    
                    # If check register CSV provided and has info for this check number, use it directly
                    check_has_category_from_csv = False
                    if check_num and check_mapping and check_num in check_mapping:
                        check_info = check_mapping.get(check_num)
                        
                        # Add payee name if available
                        if check_info.get("payee_name"):
                            categorized_txn["payee_name"] = check_info.get("payee_name")
                        
                        # If category provided in check CSV, use it directly (don't check rules)
                        if check_info.get("category"):
                            categorized_txn["account_name"] = check_info.get("category")
                            categorized_txn["category_type"] = check_info.get("type") or "Expense"
                            if check_info.get("split_amount"):
                                categorized_txn["split_amount"] = check_info.get("split_amount")
                            check_has_category_from_csv = True
                            matched_count += 1
                            log_message("info", f"[STREAMING] Check #{check_num}: Using category from CSV: {check_info.get('category')}")
                        
                        # If category was provided in CSV, skip rule matching
                        if check_has_category_from_csv:
                            categorized_batch.append(categorized_txn)
                            continue
                    
                    # If no check register CSV uploaded but this is a check -> assign Ask My Accountant
                    if not check_mapping and self._is_check_transaction(description):
                        categorized_txn["account_name"] = self.AMA_CATEGORY
                        matched_count += 1
                        categorized_batch.append(categorized_txn)
                        continue
                    
                    # Try to match transaction to rule(s)
                    matched_rules_list: List[Dict[str, Any]] = []
                    matched_via_check = False  # multiple categories only for check transactions
                    matched_via_memo = False
                    
                    # First, check if this is a check transaction and we have check register (can match multiple rules)
                    if check_mapping and check_num:
                        matched_rules_list, matched_via_memo = await self._match_check_with_register(transaction, check_mapping, rules)
                        if matched_rules_list:
                            check_matched_count += 1
                            matched_via_check = True
                            if matched_via_memo:
                                categorized_txn["matched_via_memo"] = True
                    
                    # If not matched via check register, try single manual matching
                    if not matched_rules_list:
                        matched_rule = self._match_transaction_to_rule(transaction, rules)
                        if matched_rule:
                            matched_rules_list = [matched_rule]
                    
                    # If manual matching failed, try Gemini fallback
                    if not matched_rules_list and self.gemini_client:
                        matched_rule, ai_confidence, _retries_used, ai_rejection_reason = await self._match_transaction_with_gemini(
                            transaction,
                            rules
                        )
                        if matched_rule:
                            matched_rules_list = [matched_rule]
                            gemini_fallback_count += 1
                            categorized_txn["confidence_score"] = ai_confidence
                        elif ai_rejection_reason:
                            categorized_txn["needs_review_reason"] = ai_rejection_reason
                    
                    if matched_rules_list:
                        if matched_via_check:
                            # Check transaction: allow multiple categories (comma-separated + matched_categories)
                            categorized_txn["account_name"] = ", ".join(r.get("account_name") or "" for r in matched_rules_list)
                            categorized_txn["category_type"] = ", ".join(r.get("category_type") or "" for r in matched_rules_list)
                            categorized_txn["matched_categories"] = [
                                {"account_name": r.get("account_name"), "category_type": r.get("category_type")}
                                for r in matched_rules_list
                            ]
                        else:
                            # Non-check: single category only (no matched_categories)
                            categorized_txn["account_name"] = matched_rules_list[0].get("account_name")
                            categorized_txn["category_type"] = matched_rules_list[0].get("category_type")
                        matched_count += 1
                    
                    # Loan splitting logic (same as non-streaming)
                    category = categorized_txn.get("account_name", "")
                    should_split_loan = client_id and category and "loan" in category.lower()
                    log_message("info", f"[TEST] Check register file provided: {category}, should_split_loan={should_split_loan}")
                    if should_split_loan:
                        try:
                            transaction_date = date_parser.parse(categorized_txn.get("date"))
                            loan_result = await find_loan_by_transaction(
                                transaction_date=transaction_date,
                                memo=categorized_txn.get("description", ""),
                                amount=abs(categorized_txn.get("debit") or categorized_txn.get("credit") or 0),
                                client_id=client_id
                            )
                            
                            if loan_result and loan_result.get("schedule"):
                                schedule = loan_result.get("schedule")
                                interest_expense = schedule.get("interest_expense", 0)
                                principal_liability = schedule.get("principal_liability", 0)
                                
                                if interest_expense > 0 and principal_liability > 0:
                                    # Principal transaction
                                    principal_txn = categorized_txn.copy()
                                    if categorized_txn.get("credit"):
                                        principal_txn["credit"] = principal_liability
                                        principal_txn["debit"] = None
                                    else:
                                        principal_txn["credit"] = None
                                        principal_txn["debit"] = principal_liability
                                    
                                    categorized_batch.append(principal_txn)
                                    
                                    # Interest transaction
                                    interest_txn = categorized_txn.copy()
                                    interest_txn["account_name"] = "Interest"
                                    if categorized_txn.get("credit"):
                                        interest_txn["credit"] = interest_expense
                                        interest_txn["debit"] = None
                                    else:
                                        interest_txn["credit"] = None
                                        interest_txn["debit"] = interest_expense
                                    
                                    categorized_batch.append(interest_txn)
                                    continue
                        except Exception as e:
                            log_message("error", f"Error splitting loan transaction: {e}")
                    
                    categorized_batch.append(categorized_txn)

                batch_csv_rows = build_categorized_csv_rows(
                    categorized_batch, bank, account_number, category_from_matched=False
                )
                batch_csv = "\n".join(batch_csv_rows)

                batch_num = batch_start // batch_size + 1
                meta: Optional[Dict[str, Any]] = None
                if batch_num == 1:
                    meta = {
                        "bank": bank,
                        "account_number": account_number,
                        "client_id": client_id,
                        "client_name": client_name,
                        "start_date": extraction_result.get("start_date"),
                        "end_date": extraction_result.get("end_date"),
                        "total_transactions": len(transactions),
                    }

                yield {
                    "batch_num": batch_num,
                    "total_batches": total_batches,
                    "batch_csv": batch_csv,
                    "batch_transactions": categorized_batch,
                    "meta": meta,
                }

                # Small sleep helps event loop + flushing in some environments
                await asyncio.sleep(0.01)

            log_message(
                "info",
                f"Streaming categorize done: matched {matched_count}/{len(transactions)} for client_id={client_id}"
                + (f" (Check register: {check_matched_count})" if check_matched_count > 0 else "")
                + (f" (Gemini fallback: {gemini_fallback_count})" if gemini_fallback_count > 0 else ""),
            )

        except Exception as e:
            log_message("error", f"Streaming categorize failed: {e}")
            yield {
                "batch_num": 0,
                "total_batches": "error",
                "batch_csv": "",
                "batch_transactions": [],
                "meta": {
                    "status": "error",
                    "message": str(e)
                }
            }
