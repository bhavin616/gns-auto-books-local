"""
Rule Base Routes

FastAPI endpoints for rule generation and bank statement categorization.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Path, Body, Query, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Any, Dict
import json
import traceback
import uuid
import asyncio
from io import BytesIO
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd
from backend.utils.logger import log_message
from newversion.utils import is_quota_error, build_categorized_csv_rows
from newversion.services.gemini_rule_generator import GeminiRuleGenerator
from newversion.services.rule_base_mongodb import (
    create_or_update_client_rule_base,
    create_client_rule,
    get_client_rule_base,
    get_client_rule_by_id,
    get_all_client_rule_bases,
    update_client_rule,
    delete_client_rule,
    store_regenerated_rules,
    add_old_rule_to_current_rules,
    store_client_bank_statement_data,
    get_client_bank_statements,
    get_statement_run_by_client_and_run_id,
    get_statement_run_overrides,
    upsert_statement_run_override,
    update_statement_run_transactions,
)
from newversion.services.bank_statement_categorizer import BankStatementCategorizer


router = APIRouter(prefix="/api/v2/rule-base", tags=["Rule Base"])


QUOTA_ERROR_MESSAGE = "Quota error. Please retry after some time."

ALLOWED_GL_COA_EXTENSIONS = {".csv", ".txt", ".json", ".xlsx", ".xls", ".xlsm"}
ALLOWED_BANK_STATEMENT_EXTENSIONS = (".pdf", ".csv", ".xlsx", ".xls", ".xlsm")
ALLOWED_CHECK_REGISTER_EXTENSIONS = (".csv", ".xlsx", ".xls", ".xlsm")


def _serialize_for_json(obj: Any) -> Any:
    """Recursively convert datetime and ObjectId to JSON-serializable types (same as backend get_client_bank_extract_data)."""
    from bson import ObjectId
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(i) for i in obj]
    return obj


# Require at least this many matches to consider the PDF a bank statement.
_BANK_STATEMENT_KEYWORDS = [
    "bank statement",
    "account statement",
    "statement period",
    "account number",
    "account summary",
    "transactions",
    "deposits",
    "withdrawals",
    "debit",
    "credit",
    "balance",
    "opening balance",
    "closing balance",
    "statement date",
    "transaction history",
]
_BANK_STATEMENT_MIN_KEYWORDS = 2


def _is_bank_statement_pdf(pdf_bytes: bytes) -> bool:
    """
    Text-based check: extract text from first 2 pages and look for bank-statement keywords.
    Returns True if the PDF appears to be a bank statement, False otherwise.
    """
    if not pdf_bytes or len(pdf_bytes) < 100:
        return False
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= 2:
                    break
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        combined = "\n".join(text_parts).lower()
        matches = sum(1 for k in _BANK_STATEMENT_KEYWORDS if k in combined)
        return matches >= _BANK_STATEMENT_MIN_KEYWORDS
    except Exception:
        return False


def _build_categorized_csv_response(result: dict) -> dict:
    """Build CSV-style response for categorized bank statement (single result)."""
    bank = result.get("bank", "Unknown")
    account_number = result.get("account_number") or "NA"
    transactions = result.get("transactions", []) or []
    csv_rows = build_categorized_csv_rows(
        transactions, bank, account_number, category_from_matched=True
    )
    return {
        "data": {
            "banks": [
                {
                    "bankName": bank,
                    "accounts": [
                        {
                            "accountNo": account_number,
                            "csv_content": "\n".join(csv_rows),
                        }
                    ],
                }
            ]
        }
    }


def _summarize_transactions(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build matched and source summary for categorized transactions."""
    source_counts: Dict[str, int] = {}
    for txn in transactions:
        source = txn.get("categorization_source") or BankStatementCategorizer.AMA_CATEGORY
        source_counts[source] = source_counts.get(source, 0) + 1

    matched = sum(
        1
        for txn in transactions
        if (txn.get("account_name") or "").strip()
        and (txn.get("account_name") or "").strip() != BankStatementCategorizer.AMA_CATEGORY
    )
    total = len(transactions)
    return {
        "matched_transactions": matched,
        "unmatched_transactions": total - matched,
        "total_transactions": total,
        "ama_rate": round(((total - matched) / total * 100.0), 2) if total else 0.0,
        "source_counts": source_counts,
    }


def _merge_unknown_vendor_codes(items: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge unknown vendor code summaries from multiple result objects."""
    merged: Dict[str, Dict[str, Any]] = {}
    for group in items:
        for item in group or []:
            code = str(item.get("vendor_code") or "").strip()
            if not code:
                continue
            if code not in merged:
                merged[code] = {"vendor_code": code, "count": 0, "sample_descriptions": []}
            merged[code]["count"] += int(item.get("count") or 0)
            samples = item.get("sample_descriptions") or []
            for sample in samples:
                sample_text = str(sample).strip()
                if sample_text and sample_text not in merged[code]["sample_descriptions"] and len(merged[code]["sample_descriptions"]) < 3:
                    merged[code]["sample_descriptions"].append(sample_text)
    return [merged[key] for key in sorted(merged.keys())]


def _merge_warning_lists(items: List[List[str]]) -> List[str]:
    """Merge warning message lists while preserving first-seen order."""
    merged: List[str] = []
    for group in items:
        for warning in group or []:
            text = str(warning).strip()
            if text and text not in merged:
                merged.append(text)
    return merged

class RuleUpdateRequest(BaseModel):
    trigger_payee_contains: Optional[List[str]] = Field(
        default=None,
        description="Updated trigger keywords (stored uppercased)"
    )
    account_id: Optional[str] = Field(default=None, description="Updated account_id")
    account_name: Optional[str] = Field(default=None, description="Updated account_name")
    category_type: Optional[str] = Field(default=None, description="Updated category_type")
    logic: Optional[str] = Field(default=None, description="Updated rule logic/description")
    vendor_code: Optional[str] = Field(default=None, description="Optional 4 digit vendor code for ACH vendor payments")
    ach_vendor_only: Optional[bool] = Field(default=None, description="If true, apply this rule only to ACH vendor payment style rows")


class RuleCreateRequest(BaseModel):
    trigger_payee_contains: List[str] = Field(..., description="Trigger keywords for the rule")
    account_id: str = Field(..., description="Account identifier")
    account_name: str = Field(..., description="Account name")
    category_type: Optional[str] = Field(default="Expense", description="Category type")
    logic: Optional[str] = Field(default="", description="Plain language reason for this rule")
    vendor_code: Optional[str] = Field(default=None, description="Optional 4 digit vendor code for ACH vendor payments")
    ach_vendor_only: Optional[bool] = Field(default=False, description="If true, this rule is restricted to ACH vendor payment style rows")


class StatementRunOverrideRequest(BaseModel):
    transaction_fingerprint: str = Field(..., description="Stable transaction fingerprint from categorization result")
    account_name: str = Field(..., description="Category name for one-time override")
    category_type: Optional[str] = Field(default=None, description="Optional category type for one-time override")
    reason: Optional[str] = Field(default=None, description="Optional review note")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")

class TestCategorizeRequest(BaseModel):
    """Request body for test categorize by description."""
    client_id: str = Field(..., description="Client ID to lookup rules")
    description: str = Field(..., description="Transaction description to categorize")


def _get_gl_date_range(gl_data: str) -> dict:
    """
    Parse GL text (CSV or Excel-derived CSV) and return transaction date range.
    Looks for a date column (date, transaction date, posting date, etc.) and
    returns start_date, end_date, and year(s) for the response.

    Returns:
        dict with gl_start_date, gl_end_date, gl_year (or gl_years if spanning multiple years)
    """
    result: dict[str, Any] = {
        "gl_start_date": None,
        "gl_end_date": None,
        "gl_year": None,
    }
    if not (gl_data and gl_data.strip()):
        return result
    try:
        df = pd.read_csv(BytesIO(gl_data.encode("utf-8")), nrows=50000)
        if df.empty:
            return result
        # Find date column (case-insensitive, common GL column names)
        date_col = None
        for col in df.columns:
            c = (col or "").strip().lower()
            if c in ("date", "transaction date", "posting date", "post date", "trans date", "doc date"):
                date_col = col
                break
        if not date_col:
            # Try any column name containing "date"
            for col in df.columns:
                if "date" in (col or "").lower():
                    date_col = col
                    break
        if not date_col:
            return result
        # Parse dates (handles 1/1/24, 01/01/2024, 2024-01-01, etc.)
        ser = pd.to_datetime(df[date_col], errors="coerce")
        ser = ser.dropna()
        if ser.empty:
            return result
        min_dt = ser.min()
        max_dt = ser.max()
        # Format as m/d/yy (e.g. 1/1/24, 7/8/24)
        if hasattr(min_dt, "month"):
            result["gl_start_date"] = f"{min_dt.month}/{min_dt.day}/{str(min_dt.year)[2:]}"
            result["gl_end_date"] = f"{max_dt.month}/{max_dt.day}/{str(max_dt.year)[2:]}"
        else:
            result["gl_start_date"] = str(min_dt)
            result["gl_end_date"] = str(max_dt)
        years = sorted(ser.dt.year.unique().tolist())
        result["gl_year"] = years[0] if len(years) == 1 else years
        return result
    except Exception as e:
        log_message("warning", f"Could not extract GL date range: {e}")
        return result


def _read_file_content(file: UploadFile) -> str:
    """
    Read file content and handle encoding issues.
    
    Args:
        file: UploadFile object
        
    Returns:
        File content as string
    """
    try:
        filename = (file.filename or "").lower()
        ext = filename.split(".")[-1] if "." in filename else ""

        # Excel support (.xlsx/.xls/.xlsm): convert to CSV-like text for the prompt
        if ext in {"xlsx", "xls", "xlsm"}:
            file.file.seek(0)
            raw = file.file.read()
            file.file.seek(0)
            if not raw:
                return ""

            try:
                engine = "openpyxl" if ext in {"xlsx", "xlsm"} else None
                # Note: .xls requires xlrd installed; engine=None will try defaults
                df = pd.read_excel(BytesIO(raw), engine=engine)
            except ImportError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Excel parsing dependency missing: {e}. Please upload CSV or install required Excel engine.",
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {e}")

            # Normalize to CSV text (keeps structure readable for LLM)
            text = df.to_csv(index=False)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            log_message("info", f"Successfully read Excel file '{file.filename}' ({len(df)} rows)")
            return text

        # Read file content
        file.file.seek(0)
        content = file.file.read()
        file.file.seek(0)
        
        # Remove NUL bytes that cause issues
        content = content.replace(b'\x00', b'')
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                # Normalize line endings
                text = text.replace('\r\n', '\n').replace('\r', '\n')
                log_message("info", f"Successfully read file '{file.filename}' using {encoding} encoding")
                return text
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, use utf-8 with errors='replace'
        text = content.decode('utf-8', errors='replace')
        log_message("warning", f"Read file '{file.filename}' with encoding errors replaced")
        return text
        
    except Exception as e:
        log_message("error", f"Failed to read file '{file.filename}': {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")


@router.post("/generate")
async def generate_rules(
    client_id: str = Form(..., description="Client ID"),
    client_name: str = Form(..., description="Client name"),
    nature_of_business: str = Form(..., description="Nature of business"),
    gl_file: UploadFile = File(..., description="General Ledger file"),
    coa_file: UploadFile = File(..., description="Chart of Accounts file")):
    """
    Generate categorization rules based on GL, COA, and nature of business.
    
    Args:
        client_id: Client ID (provided by frontend)
        client_name: Client name
        nature_of_business: Nature of business
        gl_file: General Ledger file (CSV, TXT, or JSON)
        coa_file: Chart of Accounts file (CSV, TXT, or JSON)
        
    Returns:
        JSON response with generated rules
    """
    try:
        log_message("info", f"Rule generation request received for client_id: {client_id}, client_name: {client_name}")
        
        existing_rule_base = await get_client_rule_base(client_id)
        if existing_rule_base and existing_rule_base.get("rules"):
            rules = existing_rule_base.get("rules", [])
            # Include GL date range from uploaded file for response
            try:
                gl_data_exists = _read_file_content(gl_file)
                gl_date_range = _get_gl_date_range(gl_data_exists)
            except Exception:
                gl_date_range = {"gl_start_date": None, "gl_end_date": None, "gl_year": None}
            try:
                coa_data_exists = _read_file_content(coa_file)
                generation_health = GeminiRuleGenerator.evaluate_generation_health(rules=rules, coa_data=coa_data_exists)
            except Exception:
                generation_health = {
                    "rule_count": len(rules),
                    "warnings": [],
                    "warning_count": 0,
                    "low_rule_count": len(rules) < 30,
                    "missing_account_mappings": [],
                    "generic_trigger_rules": [],
                    "overlapping_triggers": [],
                }
            log_message("info", f"Existing rule base found for client_id: {client_id}, returning {len(rules)} existing rules")
            return _serialize_for_json({
                "status": "exists",
                "message": "Rule base already exists for this client. Returning existing rules.",
                "client_id": existing_rule_base.get("client_id", client_id),
                "client_name": existing_rule_base.get("client_name", client_name),
                "nature_of_business": existing_rule_base.get("nature_of_business", nature_of_business),
                "rule_count": existing_rule_base.get("rule_count", len(rules)),
                "rules": rules,
                "generation_health": generation_health,
                "warnings": generation_health.get("warnings", []),
                "gl_start_date": gl_date_range.get("gl_start_date"),
                "gl_end_date": gl_date_range.get("gl_end_date"),
                "gl_year": gl_date_range.get("gl_year"),
            })
        gl_name = (gl_file.filename or "").lower()
        coa_name = (coa_file.filename or "").lower()
        if not any(gl_name.endswith(ext) for ext in ALLOWED_GL_COA_EXTENSIONS):
            raise HTTPException(status_code=400, detail="GL file must be CSV, Excel (xlsx/xls), TXT, or JSON")
        if not any(coa_name.endswith(ext) for ext in ALLOWED_GL_COA_EXTENSIONS):
            raise HTTPException(status_code=400, detail="COA file must be CSV, Excel (xlsx/xls), TXT, or JSON")

        # Read file contents
        log_message("info", f"Reading GL file: {gl_file.filename}")
        gl_data = _read_file_content(gl_file)
        
        log_message("info", f"Reading COA file: {coa_file.filename}")
        coa_data = _read_file_content(coa_file)
        
        # Extract GL transaction date range for response (which year, start/end dates)
        gl_date_range = _get_gl_date_range(gl_data)
        
        # Initialize rule generator
        rule_generator = GeminiRuleGenerator()
        
        # Generate rules using Gemini
        log_message("info", "Generating rules using Gemini API")
        rules = await rule_generator.generate_rules(
            client_name=client_name,
            nature_of_business=nature_of_business,
            coa_data=coa_data,
            gl_data=gl_data
        )
        generation_health = GeminiRuleGenerator.evaluate_generation_health(rules=rules, coa_data=coa_data)
        
        if not rules:
            raise HTTPException(status_code=500, detail="No rules were generated")
        
        # Store rules in MongoDB (including GL date range)
        log_message("info", f"Storing {len(rules)} rules in MongoDB for client_id: {client_id}")
        success = await create_or_update_client_rule_base(
            client_id=client_id,
            client_name=client_name,
            nature_of_business=nature_of_business,
            rules=rules,
            gl_start_date=gl_date_range.get("gl_start_date"),
            gl_end_date=gl_date_range.get("gl_end_date"),
            gl_year=gl_date_range.get("gl_year"),
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store rules in MongoDB")
        
        log_message("info", f"Successfully generated and stored {len(rules)} rules for client_id: {client_id}")
        
        return {
            "status": "success",
            "message": f"Successfully generated and stored {len(rules)} rules",
            "client_id": client_id,
            "client_name": client_name,
            "nature_of_business": nature_of_business,
            "rule_count": len(rules),
            "rules": rules,
            "generation_health": generation_health,
            "warnings": generation_health.get("warnings", []),
            "gl_start_date": gl_date_range.get("gl_start_date"),
            "gl_end_date": gl_date_range.get("gl_end_date"),
            "gl_year": gl_date_range.get("gl_year"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Rule generation failed: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        if is_quota_error(e):
            return JSONResponse(
                status_code=429,
                content={
                    "succeeded": False,
                    "message": QUOTA_ERROR_MESSAGE,
                    "status_code": 429,
                },
            )
        raise HTTPException(status_code=500, detail=f"Rule generation failed: {e}")


@router.post("/regenerate")
async def regenerate_rules(
    client_id: str = Form(..., description="Client ID"),
    client_name: str = Form(..., description="Client name"),
    nature_of_business: str = Form(..., description="Nature of business"),
    gl_file: UploadFile = File(..., description="General Ledger file"),
    coa_file: UploadFile = File(..., description="Chart of Accounts file")):
    """
    Regenerate rules for a client that already has rules in MongoDB.
    Accepts client_id, client_name, nature_of_business, GL and COA files;
    generates new rules and returns both the old rules (from MongoDB) and the new rules.
    Regenerate is allowed only when an old rule base exists for the client.
    """
    try:
        log_message("info", f"Regenerate rules request received for client_id: {client_id}")

        # Regenerate only when old rule exists
        existing_rule_base = await get_client_rule_base(client_id)
        if not existing_rule_base or not existing_rule_base.get("rules"):
            raise HTTPException(
                status_code=400,
                detail=f"No existing rule base found for client_id: {client_id}. Regenerate is only allowed when rules already exist. Use /generate to create rules first."
            )

        # Old rules = slot at index 0 (for response). rules_slots: [index0=older, index1=newer].
        slots = existing_rule_base.get("rules_slots") or []
        if len(slots) >= 2:
            old_rules = slots[0]  # previous older slot
        elif len(slots) == 1:
            old_rules = slots[0]  # first regenerate: index 0 was first generate
        else:
            old_rules = existing_rule_base.get("rules", [])  # backward compat: no rules_slots yet
        log_message("info", f"Found {len(old_rules)} existing rules for client_id: {client_id}, regenerating")

        # Validate file types (CSV/TXT/JSON/Excel)
        gl_name = (gl_file.filename or "").lower()
        coa_name = (coa_file.filename or "").lower()
        if not any(gl_name.endswith(ext) for ext in ALLOWED_GL_COA_EXTENSIONS):
            raise HTTPException(status_code=400, detail="GL file must be CSV, Excel (xlsx/xls), TXT, or JSON")
        if not any(coa_name.endswith(ext) for ext in ALLOWED_GL_COA_EXTENSIONS):
            raise HTTPException(status_code=400, detail="COA file must be CSV, Excel (xlsx/xls), TXT, or JSON")

        # Read file contents
        log_message("info", f"Reading GL file: {gl_file.filename}")
        gl_data = _read_file_content(gl_file)
        log_message("info", f"Reading COA file: {coa_file.filename}")
        coa_data = _read_file_content(coa_file)

        # Extract GL transaction date range for response
        gl_date_range = _get_gl_date_range(gl_data)

        # Generate new rules using Gemini
        rule_generator = GeminiRuleGenerator()
        log_message("info", "Generating new rules using Gemini API")
        new_rules = await rule_generator.generate_rules(
            client_name=client_name,
            nature_of_business=nature_of_business,
            coa_data=coa_data,
            gl_data=gl_data
        )
        generation_health = GeminiRuleGenerator.evaluate_generation_health(rules=new_rules, coa_data=coa_data)

        if not new_rules:
            raise HTTPException(status_code=500, detail="No rules were generated")

        # Store new rules in MongoDB: index 0 = older, index 1 = newer (shift on 3rd+ regenerate)
        log_message("info", f"Storing {len(new_rules)} new rules in MongoDB for client_id: {client_id} (two-slot history)")
        success = await store_regenerated_rules(
            client_id=client_id,
            client_name=client_name,
            nature_of_business=nature_of_business,
            new_rules=new_rules,
            gl_start_date=gl_date_range.get("gl_start_date"),
            gl_end_date=gl_date_range.get("gl_end_date"),
            gl_year=gl_date_range.get("gl_year"),
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store new rules in MongoDB")

        log_message("info", f"Successfully regenerated rules for client_id: {client_id}, old_count={len(old_rules)}, new_count={len(new_rules)}")

        # GL date range for the uploaded GL (applies to new_rules)
        new_rules_gl = {
            "gl_start_date": gl_date_range.get("gl_start_date"),
            "gl_end_date": gl_date_range.get("gl_end_date"),
            "gl_year": gl_date_range.get("gl_year"),
        }
        return {
            "status": "success",
            "message": "Rules regenerated successfully. Old rules and new rules are returned.",
            "client_id": client_id,
            "client_name": client_name,
            "nature_of_business": nature_of_business,
            "old_rules": old_rules,
            "new_rules": new_rules,
            "new_rules_gl": new_rules_gl,
            "old_rule_count": len(old_rules),
            "new_rule_count": len(new_rules),
            "generation_health": generation_health,
            "warnings": generation_health.get("warnings", []),
            "gl_start_date": gl_date_range.get("gl_start_date"),
            "gl_end_date": gl_date_range.get("gl_end_date"),
            "gl_year": gl_date_range.get("gl_year"),
        }

    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Rule regenerate failed: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        if is_quota_error(e):
            return JSONResponse(
                status_code=429,
                content={
                    "succeeded": False,
                    "message": QUOTA_ERROR_MESSAGE,
                    "status_code": 429,
                },
            )
        raise HTTPException(status_code=500, detail=f"Rule regenerate failed: {e}")

class AddFromOldRequest(BaseModel):
    rule_ids: str = Field(..., description="Comma separated rule IDs")


@router.put("/{client_id}/rules/add-from-old")
async def add_old_rule_to_new(
    client_id: str = Path(..., description="Client ID"),
    body: AddFromOldRequest = Body(...),):
    try:
        # ── 1. Split comma separated string into list ─────────────────────────
        rule_ids = [rid.strip() for rid in body.rule_ids.split(",")]
        # "COGS_DRIVER_PAY_001,COGS_MAINTENANCE_001"
        # becomes → ["COGS_DRIVER_PAY_001", "COGS_MAINTENANCE_001"]

        # ── 2. Validate client exists ─────────────────────────────────────────
        existing_rule_base = await get_client_rule_base(client_id)
        if not existing_rule_base:
            raise HTTPException(
                status_code=404,
                detail=f"No rule base found for client_id: {client_id}",
            )

        # ── 3. Validate old/new slots exist ───────────────────────────────────
        slots = existing_rule_base.get("rules_slots") or []
        if len(slots) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Client {client_id} has no old/new slot history. Run regenerate first.",
            )

        # ── 4. Loop and add each rule ─────────────────────────────────────────
        succeeded = []
        failed = []

        for rule_id in rule_ids:
            try:
                added_rule = await add_old_rule_to_current_rules(
                    client_id=client_id,
                    rule_id=rule_id,
                )
                if added_rule is None:
                    failed.append({"rule_id": rule_id, "reason": "not found in old rules"})
                else:
                    succeeded.append({"rule_id": rule_id, "added_rule": added_rule})
            except Exception as e:
                failed.append({"rule_id": rule_id, "reason": str(e)})

        # ── 5. Response ───────────────────────────────────────────────────────
        # Single rule → simple response
        if len(rule_ids) == 1:
            if failed:
                raise HTTPException(
                    status_code=404,
                    detail=f"Rule rule_id={rule_ids[0]} not found in old rules for client_id: {client_id}",
                )
            return {
                "status": "success",
                "message": "Old rule added to current (new) rules. Old rule unchanged in old slot.",
                "client_id": client_id,
                "rule_id": succeeded[0]["rule_id"],
                "added_rule": succeeded[0]["added_rule"],
            }

        # Multiple rules → detailed response
        return JSONResponse(
            status_code=200 if not failed else 207,
            content={
                "status": "success" if not failed else "partial",
                "client_id": client_id,
                "total_requested": len(rule_ids),
                "total_succeeded": len(succeeded),
                "total_failed": len(failed),
                "succeeded": succeeded,
                "failed": failed,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Add old rule to new failed: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Add old rule to new failed: {e}")


@router.get("/{client_id}/rules/{rule_id}")
async def get_rule_by_id(
    client_id: str = Path(..., description="Client ID"),
    rule_id: str = Path(..., description="Rule ID to fetch"),
):
    """Get one client rule by rule_id."""
    try:
        rule = await get_client_rule_by_id(client_id=client_id, rule_id=rule_id)
        if not rule:
            raise HTTPException(
                status_code=404,
                detail=f"No matching rule found for client_id={client_id}, rule_id={rule_id}",
            )

        return {
            "status": "success",
            "client_id": client_id,
            "rule_id": rule_id,
            "rule": rule,
        }
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to get rule for client_id={client_id}, rule_id={rule_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get rule: {e}")


@router.post("/{client_id}/rules")
async def create_rule(
    client_id: str = Path(..., description="Client ID"),
    payload: RuleCreateRequest = Body(...),
):
    """Create one new rule in the client's active rule set."""
    try:
        rule_data = payload.model_dump()
        created = await create_client_rule(client_id=client_id, rule_data=rule_data)
        if not created:
            raise HTTPException(
                status_code=400,
                detail="Unable to create rule. Check required fields and client rule base.",
            )

        return {
            "status": "success",
            "message": "Rule created successfully",
            "client_id": client_id,
            "rule": created,
        }
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to create rule for client_id={client_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {e}")


@router.put("/{client_id}/rules/{rule_id}")
async def update_rule(
    client_id: str = Path(..., description="Client ID"),
    rule_id: str = Path(..., description="Rule ID to update"),
    payload: RuleUpdateRequest = Body(...),):
    """
    Update a single rule for a client by rule_id.
    
    Updates allowed fields:
    - trigger_payee_contains
    - account_id
    - account_name
    - category_type
    - logic
    - vendor_code
    - ach_vendor_only
    """
    try:
        raw_data = payload.model_dump(exclude_unset=True)

        # Build update_data while honoring "keep as-is" semantics:
        # - Ignore fields that are null
        # - Ignore string fields that are empty/whitespace ("")
        # - For trigger_payee_contains: if null or empty list, ignore (no change)
        update_data = {}
        for key, value in raw_data.items():
            # Skip explicit nulls
            if value is None:
                continue

            # Special handling for trigger list
            if key == "trigger_payee_contains":
                # If frontend sends empty list, treat as "no change"
                if not value:
                    continue

                triggers = value or []
                normalized = []
                for t in triggers:
                    t_str = str(t).upper().strip()
                    if t_str:
                        normalized.append(t_str)

                if not normalized:
                    # All triggers were empty → don't update this field
                    continue

                # de-dup + stable sort for consistent storage
                update_data["trigger_payee_contains"] = sorted(set(normalized))
                continue

            # For other string fields, ignore empty/whitespace-only values
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    # blank string means "keep existing value"
                    continue
                update_data[key] = cleaned
            else:
                update_data[key] = value

        if not update_data:
            raise HTTPException(status_code=400, detail="No valid update fields provided")
        
        updated_rule = await update_client_rule(
            client_id=client_id,
            rule_id=rule_id,
            updates=update_data,
        )
        
        if not updated_rule:
            raise HTTPException(
                status_code=404,
                detail=f"No matching rule found for client_id={client_id}, rule_id={rule_id}",
            )
        
        return {
            "status": "success",
            "message": "Rule updated successfully",
            "client_id": client_id,
            "rule_id": rule_id,
            "rule": updated_rule,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to update rule for client_id={client_id}, rule_id={rule_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update rule: {e}")


@router.delete("/{client_id}/rules/{rule_id}")
async def delete_rule(
    client_id: str = Path(..., description="Client ID"),
    rule_id: str = Path(..., description="Rule ID to delete"),):
    """
    Delete a single rule from a client's rule base by rule_id.
    """
    try:
        success = await delete_client_rule(client_id=client_id, rule_id=rule_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No matching rule found for client_id={client_id}, rule_id={rule_id}",
            )

        return {
            "status": "success",
            "message": "Rule deleted successfully",
            "client_id": client_id,
            "rule_id": rule_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to delete rule for client_id={client_id}, rule_id={rule_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete rule: {e}")


ALLOWED_CATEGORIZE_BANK_STATEMENT_FORM_KEYS = {"client_id", "user_id", "bank_statement", "bank_statements", "check_register"}


@router.post("/categorize-bank-statement")
async def categorize_bank_statement(
    request: Request,
    client_id: str = Form(..., description="Client ID (for rule lookup)"),
    user_id: str = Form(..., description="User ID (for storing result; use this in GET /client-bank-statement-data)"),
    bank_statements: List[UploadFile] = File(..., description="Bank statement files (PDF/CSV/Excel) - can upload multiple"),
    check_register: Optional[UploadFile] = File(None, description="Optional: CSV/Excel with check numbers and payee names (same logic as test endpoint)")):
    """
    Categorize bank statement transactions using client-specific rules.

    Uses the same logic as /categorize-bank-statement-test: manual rule matching first,
    then check register (if provided) for check transactions, then Gemini fallback for
    unmatched transactions.

    Now supports multiple bank statement files in a single request. Files are processed
    sequentially and grouped by bank/account (similar to /api/process endpoint).

    Stored data is keyed by user_id; use GET /client-bank-statement-data?user_id=... to retrieve.

    Args:
        client_id: Client ID to lookup rules
        user_id: User ID to store result under (retrieve later with GET ?user_id=...)
        bank_statements: List of bank statement files (PDF/CSV/Excel) - can upload multiple
        check_register: Optional CSV/Excel with check number and payee name columns

    Returns:
        CSV-style JSON with data.banks[].accounts[].csv_content
        - Grouped by bank and account number
        - Each file processed sequentially, results returned after all processing
    """
    try:
        from collections import defaultdict
        
        form = await request.form()
        received_keys = set(form.keys())
        # Update validation to allow both 'bank_statement' (single) and 'bank_statements' (multiple)
        allowed_keys = {"client_id", "user_id", "bank_statement", "bank_statements", "check_register"}
        invalid_keys = received_keys - allowed_keys
        if invalid_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid form field name(s): {', '.join(sorted(invalid_keys))}. Expected: client_id, user_id, bank_statements, and optionally check_register."
            )

        if not bank_statements:
            raise HTTPException(
                status_code=400,
                detail="At least one bank statement file is required"
            )

        log_message("info", f"Bank statement categorization request received for client_id: {client_id}, user_id: {user_id}, files: {len(bank_statements)}")

        # Validate all files
        for idx, bank_statement in enumerate(bank_statements):
            filename_lower = (bank_statement.filename or "").lower()
            if not filename_lower.endswith(ALLOWED_BANK_STATEMENT_EXTENSIONS):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {idx+1} ({bank_statement.filename}): Bank statement must be PDF, CSV, or Excel (xlsx/xls)"
                )
            if filename_lower.endswith(".pdf"):
                pdf_bytes = await bank_statement.read()
                bank_statement.file.seek(0)
                if not _is_bank_statement_pdf(pdf_bytes):
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {idx+1} ({bank_statement.filename}): Invalid pdf upload valid bank statement"
                    )
        
        if check_register:
            check_filename_lower = (check_register.filename or "").lower()
            if not check_filename_lower.endswith(ALLOWED_CHECK_REGISTER_EXTENSIONS):
                raise HTTPException(status_code=400, detail="Check register must be CSV or Excel (xlsx/xls)")

        categorizer = BankStatementCategorizer()
        
        # Group results by (bank, account_number) similar to backend /process endpoint
        # Key: (bank, account_number), Value: list of results
        statements_by_account = defaultdict(list)
        inter_file_delay_seconds = 2
        delay_applied_count = 0
        
        # Process each bank statement file
        for idx, bank_statement in enumerate(bank_statements):
            log_message("info", f"Categorizing bank statement {idx+1}/{len(bank_statements)} ({bank_statement.filename}) for client_id: {client_id}")
            
            result = await categorizer.categorize_bank_statement(
                client_id=client_id,
                bank_statement_file=bank_statement,
                check_register_file=check_register
            )
            
            # Get bank and account number from result
            bank = result.get("bank", "Unknown")
            account_number = result.get("account_number") or "NA"
            
            # Group by (bank, account_number)
            key = (bank, account_number)
            statements_by_account[key].append({
                "filename": bank_statement.filename,
                "result": result,
                "matched": result.get("matched_transactions", 0),
                "total": result.get("total_transactions", 0)
            })
            
            log_message(
                "info",
                f"File {idx+1}/{len(bank_statements)}: Categorized {result.get('matched_transactions', 0)}/{result.get('total_transactions', 0)} transactions from {bank}/{account_number}"
            )
            if idx < len(bank_statements) - 1:
                delay_applied_count += 1
                log_message(
                    "info",
                    f"Applying {inter_file_delay_seconds} second spacing before next file to reduce rate limit risk."
                )
                await asyncio.sleep(inter_file_delay_seconds)
        
        log_message("info", f"Grouped {len(bank_statements)} file(s) into {len(statements_by_account)} account group(s)")
        
        # Now merge results for each account group
        account_group_results = {}
        
        for (bank, account_number), statements in statements_by_account.items():
            account_key = f"{bank}_{account_number}" if account_number != "NA" else f"{bank}_NA"
            
            # Merge all transactions from statements in this group
            all_transactions = []
            total_matched = 0
            total_transactions = 0
            total_rate_limit_retries_used = 0
            combined_result = None
            grouped_warnings: List[List[str]] = []
            grouped_unknown_vendor_codes: List[List[Dict[str, Any]]] = []
            
            for stmt_data in statements:
                result = stmt_data["result"]
                
                # Store metadata from first file
                if combined_result is None:
                    combined_result = result.copy()
                    combined_result["transactions"] = []
                
                # Merge transactions
                all_transactions.extend(result.get("transactions", []))
                total_matched += stmt_data["matched"]
                total_transactions += stmt_data["total"]
                total_rate_limit_retries_used += int(result.get("rate_limit_retries_used") or 0)
                grouped_warnings.append(result.get("warnings") or [])
                grouped_unknown_vendor_codes.append(result.get("unknown_vendor_codes") or [])
            
            # Update combined result
            combined_result["transactions"] = all_transactions
            combined_result["matched_transactions"] = total_matched
            combined_result["total_transactions"] = total_transactions
            combined_result["unmatched_transactions"] = max(total_transactions - total_matched, 0)
            combined_result["ama_rate"] = round(
                ((combined_result["unmatched_transactions"] / total_transactions) * 100.0), 2
            ) if total_transactions else 0.0
            combined_result["rate_limit_retries_used"] = total_rate_limit_retries_used
            combined_result["unknown_vendor_codes"] = _merge_unknown_vendor_codes(grouped_unknown_vendor_codes)
            combined_result["warnings"] = _merge_warning_lists(grouped_warnings)
            source_counts: Dict[str, int] = {}
            for txn in all_transactions:
                src = txn.get("categorization_source") or "Ask My Accountant"
                source_counts[src] = source_counts.get(src, 0) + 1
            combined_result["source_counts"] = source_counts
            
            account_group_results[account_key] = combined_result
            
            log_message(
                "info",
                f"Account group {account_key}: Combined {total_matched}/{total_transactions} transactions from {len(statements)} file(s)"
            )
        
        # Build final response grouped by bank and account (similar to backend /process)
        banks_dict = defaultdict(lambda: defaultdict(list))
        
        for account_key, result in account_group_results.items():
            bank = result.get("bank", "Unknown")
            account_number = result.get("account_number") or "NA"
            
            banks_dict[bank][account_number].append(result)
        
        # Format response using same structure as _build_categorized_csv_response
        response_banks = []
        
        for bank_name, accounts_dict in banks_dict.items():
            bank_accounts = []
            
            for account_no, results in accounts_dict.items():
                # Merge all transactions for this account
                all_transactions = []
                for res in results:
                    all_transactions.extend(res.get("transactions", []))
                
                # Build CSV for this account (with category_from_matched=True for account_name field)
                csv_rows = build_categorized_csv_rows(
                    transactions=all_transactions,
                    bank=bank_name,
                    account_number=account_no,
                    category_from_matched=True
                )
                csv_content = "\n".join(csv_rows)
                
                bank_accounts.append({
                    "accountNo": account_no,
                    "csv_content": csv_content
                })
            
            response_banks.append({
                "bankName": bank_name,
                "accounts": bank_accounts
            })
        
        # Return in same format as _build_categorized_csv_response - simple data structure
        total_transactions_all = sum((r.get("total_transactions") or 0) for r in account_group_results.values())
        total_matched_all = sum((r.get("matched_transactions") or 0) for r in account_group_results.values())
        total_unmatched_all = sum((r.get("unmatched_transactions") or 0) for r in account_group_results.values())
        ama_rate_all = round((total_unmatched_all / total_transactions_all * 100.0), 2) if total_transactions_all else 0.0
        unknown_vendor_codes_all = _merge_unknown_vendor_codes(
            [r.get("unknown_vendor_codes") or [] for r in account_group_results.values()]
        )
        warnings_all = _merge_warning_lists(
            [r.get("warnings") or [] for r in account_group_results.values()]
        )
        if delay_applied_count > 0:
            warnings_all.append(
                f"Applied a {inter_file_delay_seconds} second spacing between files {delay_applied_count} time(s) to reduce rate limit risk."
            )
        rate_limit_retries_used_all = sum(
            int(r.get("rate_limit_retries_used") or 0) for r in account_group_results.values()
        )
        final_response = {
            "data": {
                "banks": response_banks
            },
            "summary": {
                "total_transactions": total_transactions_all,
                "matched_transactions": total_matched_all,
                "unmatched_transactions": total_unmatched_all,
                "ama_rate": ama_rate_all,
            },
            "warnings": warnings_all,
            "unknown_vendor_codes": unknown_vendor_codes_all,
            "ama_rate": ama_rate_all,
            "rate_limit_retries_used": rate_limit_retries_used_all,
        }
        
        # Store each account group separately in MongoDB
        stored_runs = []
        delay_warning_message = None
        if delay_applied_count > 0:
            delay_warning_message = (
                f"Applied a {inter_file_delay_seconds} second spacing between files {delay_applied_count} time(s) to reduce rate limit risk."
            )
        for account_key, result in account_group_results.items():
            result_to_store = dict(result)
            warnings_for_store = result_to_store.get("warnings") or []
            if delay_warning_message and delay_warning_message not in warnings_for_store:
                warnings_for_store = warnings_for_store + [delay_warning_message]
            result_to_store["warnings"] = warnings_for_store
            store_result = await store_client_bank_statement_data(
                user_id=user_id, result=result_to_store, client_id=client_id,
                client_name=result_to_store.get("client_name"),
            )
            stored_runs.append(
                {
                    "account_key": account_key,
                    "bank": result_to_store.get("bank", "Unknown"),
                    "account_number": result_to_store.get("account_number") or "NA",
                    "run_id": store_result.get("run_id"),
                    "stored": bool(store_result.get("success")),
                }
            )

        final_response["statement_runs"] = stored_runs
        
        log_message(
            "info",
            f"Successfully processed and stored {len(account_group_results)} account group(s) for user_id: {user_id}"
        )
        
        return final_response

    except HTTPException:
        raise
    except ValueError as e:
        log_message("error", f"Bank statement categorization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_message("error", f"Bank statement categorization failed: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        if is_quota_error(e):
            return JSONResponse(
                status_code=429,
                content={
                    "succeeded": False,
                    "message": QUOTA_ERROR_MESSAGE,
                    "status_code": 429,
                },
            )
        raise HTTPException(status_code=500, detail=f"Bank statement categorization failed: {e}")


@router.get("/client-bank-statement-data")
async def get_client_bank_statement_data(
    user_id: str = Query(..., description="User ID to retrieve categorized bank statement data (stored by user_id)")):
    """
    Get all categorized bank statement data for a user (by user_id).

    Returns the same response format as backend GET /client_bank_extract_data:
    succeeded, message, data: { records, count }, status_code. Each record has
    id, user_id, client_name, data (banks with accounts and csv_content), created_at, updated_at.

    Args:
        user_id: User ID (query parameter)

    Returns:
        JSON: { succeeded, message, data: { records: [...], count }, status_code }
    """
    try:
        log_message("info", f"Get client bank statement data request for user_id: {user_id}")

        docs = await get_client_bank_statements(user_id)

        if not docs:
            return JSONResponse(
                status_code=200,
                content=_serialize_for_json({
                    "succeeded": True,
                    "message": "No bank statement data found for user_id",
                    "data": {"records": [], "count": 0},
                    "status_code": 200,
                }),
            )

        # Build records in same shape as backend get_client_bank_extract_data
        records = []
        for doc in docs:
            result_for_csv = {
                "bank": doc.get("bank", "Unknown"),
                "account_number": doc.get("account_number") or "NA",
                "transactions": doc.get("transactions") or [],
            }
            data_banks = _build_categorized_csv_response(result_for_csv)["data"]
            record = {
                "id": str(doc["_id"]) if doc.get("_id") else None,
                "user_id": doc.get("user_id", ""),
                "client_id": doc.get("client_id", ""),
                "client_name": doc.get("client_name") or "",
                "run_id": doc.get("run_id"),
                "matched_transactions": doc.get("matched_transactions", 0),
                "unmatched_transactions": doc.get("unmatched_transactions", 0),
                "total_transactions": doc.get("total_transactions", 0),
                "source_counts": doc.get("source_counts") or {},
                "warnings": doc.get("warnings") or [],
                "unknown_vendor_codes": doc.get("unknown_vendor_codes") or [],
                "ama_rate": float(doc.get("ama_rate") or 0.0),
                "rate_limit_retries_used": int(doc.get("rate_limit_retries_used") or 0),
                "data": data_banks,
                "transactions": doc.get("transactions") or [],
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at") or doc.get("created_at"),
            }
            records.append(record)

        serialized = _serialize_for_json({
            "succeeded": True,
            "message": f"Retrieved {len(records)} bank statement data record(s)",
            "data": {"records": records, "count": len(records)},
            "status_code": 200,
        })
        log_message("info", f"Returning {len(docs)} stored statement(s) for user_id: {user_id}")
        return JSONResponse(status_code=200, content=serialized)

    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to get client bank statement data: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get client bank statement data: {e}"
        )


@router.post("/categorize-bank-statement-test")
async def categorize_bank_statement_test(
    client_id: str = Form(..., description="Client ID"),
    bank_statement: UploadFile = File(..., description="Bank statement file (PDF/CSV/Excel)"),
    check_register: Optional[UploadFile] = File(None, description="Optional: CSV file with check numbers and payee names")):
    """
    Test endpoint: Categorize bank statement and return the raw JSON structure
    from the categorizer (same as original behavior of /categorize-bank-statement).
    
    Optional check_register CSV file format:
    - Must have columns: check_number (or Check Number, Check#, etc.) and payee_name (or Payee, Payee Name, etc.)
    - Used to match check transactions to categories based on payee names
    """
    try:
        log_message("info", f"[TEST] Bank statement categorization request received for client_id: {client_id}")

        # Validate file type
        filename_lower = (bank_statement.filename or "").lower()
        if not filename_lower.endswith(ALLOWED_BANK_STATEMENT_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Bank statement must be PDF, CSV, or Excel (xlsx/xls)")
        if check_register:
            check_filename_lower = (check_register.filename or "").lower()
            if not check_filename_lower.endswith(ALLOWED_CHECK_REGISTER_EXTENSIONS):
                raise HTTPException(status_code=400, detail="Check register must be CSV or Excel (xlsx/xls)")
            log_message("info", f"[TEST] Check register file provided: {check_register.filename}")

        categorizer = BankStatementCategorizer()
        log_message("info", f"[TEST] Categorizing bank statement for client_id: {client_id}")
        result = await categorizer.categorize_bank_statement(
            client_id=client_id,
            bank_statement_file=bank_statement,
            check_register_file=check_register
        )

        log_message(
            "info",
            f"[TEST] Successfully categorized {result.get('matched_transactions', 0)}/{result.get('total_transactions', 0)} "
            f"transactions for client_id: {client_id}"
        )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        log_message("error", f"[TEST] Bank statement categorization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_message("error", f"[TEST] Bank statement categorization failed: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        if is_quota_error(e):
            return JSONResponse(
                status_code=429,
                content={
                    "succeeded": False,
                    "message": QUOTA_ERROR_MESSAGE,
                    "status_code": 429,
                },
            )
        raise HTTPException(status_code=500, detail=f"Bank statement categorization failed: {e}")


@router.post("/{client_id}/statement-runs/{run_id}/overrides")
async def save_statement_run_override(
    client_id: str = Path(..., description="Client ID"),
    run_id: str = Path(..., description="Statement run identifier"),
    payload: StatementRunOverrideRequest = Body(...),
):
    """
    Save a one-time category override for one statement run row.
    This does not change future runs unless user also updates or creates a rule.
    """
    try:
        run_doc = await get_statement_run_by_client_and_run_id(client_id=client_id, run_id=run_id)
        if not run_doc:
            raise HTTPException(status_code=404, detail=f"Statement run not found for client_id={client_id}, run_id={run_id}")

        transactions = run_doc.get("transactions") or []
        fingerprint = payload.transaction_fingerprint.strip()
        updated_transactions: List[Dict[str, Any]] = []
        target_found = False
        updated_row: Optional[Dict[str, Any]] = None

        for txn in transactions:
            txn_copy = dict(txn)
            if (txn_copy.get("transaction_fingerprint") or "").strip() == fingerprint:
                target_found = True
                txn_copy["account_name"] = payload.account_name.strip()
                if payload.category_type is not None:
                    txn_copy["category_type"] = payload.category_type
                txn_copy["categorization_source"] = "Override"
                txn_copy["matched_rule_id"] = None
                txn_copy["matched_trigger_text"] = "One-time override from review screen"
                txn_copy["confidence_score"] = 1.0
                txn_copy["needs_review_reason"] = None
                txn_copy["needs_review_flag"] = False
                updated_row = txn_copy
            updated_transactions.append(txn_copy)

        if not target_found:
            raise HTTPException(status_code=404, detail="Transaction fingerprint not found in this statement run")

        save_ok = await upsert_statement_run_override(
            client_id=client_id,
            run_id=run_id,
            transaction_fingerprint=fingerprint,
            account_name=payload.account_name.strip(),
            category_type=payload.category_type,
            reason=payload.reason,
            user_id=payload.user_id,
            action_type="one_time_override",
        )
        if not save_ok:
            raise HTTPException(status_code=500, detail="Failed to store one-time override")

        summary = _summarize_transactions(updated_transactions)
        override_warnings = _merge_warning_lists([run_doc.get("warnings") or []])
        if summary["ama_rate"] > 40:
            ama_warning = (
                f"{summary['ama_rate']:.1f}% of transactions are Ask My Accountant. "
                "Consider refreshing rules with newer general ledger data."
            )
            if ama_warning not in override_warnings:
                override_warnings.append(ama_warning)
        existing_unknown_vendor_codes = run_doc.get("unknown_vendor_codes") or []
        existing_rate_limit_retries_used = int(run_doc.get("rate_limit_retries_used") or 0)
        update_ok = await update_statement_run_transactions(
            client_id=client_id,
            run_id=run_id,
            transactions=updated_transactions,
            matched_transactions=summary["matched_transactions"],
            unmatched_transactions=summary["unmatched_transactions"],
            source_counts=summary["source_counts"],
            warnings=override_warnings,
            unknown_vendor_codes=existing_unknown_vendor_codes,
            ama_rate=summary["ama_rate"],
            rate_limit_retries_used=existing_rate_limit_retries_used,
        )
        if not update_ok:
            raise HTTPException(status_code=500, detail="Failed to update statement run with override")

        return {
            "status": "success",
            "message": "One-time override saved",
            "client_id": client_id,
            "run_id": run_id,
            "updated_transaction": updated_row,
            "summary": summary,
            "warnings": override_warnings,
            "unknown_vendor_codes": existing_unknown_vendor_codes,
            "ama_rate": summary["ama_rate"],
            "rate_limit_retries_used": existing_rate_limit_retries_used,
        }
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to save statement override for client_id={client_id}, run_id={run_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to save statement override: {e}")


@router.post("/{client_id}/statement-runs/{run_id}/reapply")
async def reapply_statement_run(
    client_id: str = Path(..., description="Client ID"),
    run_id: str = Path(..., description="Statement run identifier"),
):
    """
    Reapply latest client rules to the same statement run transactions.
    Re-applies one-time overrides after rule-based recategorization.
    """
    try:
        run_doc = await get_statement_run_by_client_and_run_id(client_id=client_id, run_id=run_id)
        if not run_doc:
            raise HTTPException(status_code=404, detail=f"Statement run not found for client_id={client_id}, run_id={run_id}")

        existing_transactions = run_doc.get("transactions") or []
        categorizer = BankStatementCategorizer()
        recategorized_result = await categorizer.recategorize_existing_transactions(
            client_id=client_id,
            transactions=existing_transactions,
        )
        recategorized_transactions = recategorized_result.get("transactions") or []

        # Re-apply one-time overrides as final step for this run
        overrides = await get_statement_run_overrides(client_id=client_id, run_id=run_id)
        override_map = {
            (ov.get("transaction_fingerprint") or "").strip(): ov
            for ov in overrides
            if (ov.get("transaction_fingerprint") or "").strip()
        }
        applied_override_count = 0
        if override_map:
            for txn in recategorized_transactions:
                fingerprint = (txn.get("transaction_fingerprint") or "").strip()
                override = override_map.get(fingerprint)
                if not override:
                    continue
                txn["account_name"] = override.get("account_name") or txn.get("account_name")
                if override.get("category_type") is not None:
                    txn["category_type"] = override.get("category_type")
                txn["categorization_source"] = "Override"
                txn["matched_rule_id"] = None
                txn["matched_trigger_text"] = "One-time override from review screen"
                txn["confidence_score"] = 1.0
                txn["needs_review_reason"] = None
                txn["needs_review_flag"] = False
                applied_override_count += 1

        summary = _summarize_transactions(recategorized_transactions)
        recategorized_warnings = recategorized_result.get("warnings") or []
        recategorized_unknown_vendor_codes = recategorized_result.get("unknown_vendor_codes") or []
        recategorized_ama_rate = float(recategorized_result.get("ama_rate") or summary["ama_rate"])
        recategorized_rate_limit_retries_used = int(recategorized_result.get("rate_limit_retries_used") or 0)
        updated = await update_statement_run_transactions(
            client_id=client_id,
            run_id=run_id,
            transactions=recategorized_transactions,
            matched_transactions=summary["matched_transactions"],
            unmatched_transactions=summary["unmatched_transactions"],
            source_counts=summary["source_counts"],
            warnings=recategorized_warnings,
            unknown_vendor_codes=recategorized_unknown_vendor_codes,
            ama_rate=recategorized_ama_rate,
            rate_limit_retries_used=recategorized_rate_limit_retries_used,
        )
        if not updated:
            raise HTTPException(status_code=500, detail="Failed to update statement run after reapply")

        bank = run_doc.get("bank", "Unknown")
        account_number = run_doc.get("account_number") or "NA"
        csv_rows = build_categorized_csv_rows(
            transactions=recategorized_transactions,
            bank=bank,
            account_number=account_number,
            category_from_matched=True,
        )

        response = {
            "status": "success",
            "message": "Reapplied latest rules to stored statement run",
            "client_id": client_id,
            "run_id": run_id,
            "summary": summary,
            "applied_override_count": applied_override_count,
            "warnings": recategorized_warnings,
            "unknown_vendor_codes": recategorized_unknown_vendor_codes,
            "ama_rate": recategorized_ama_rate,
            "rate_limit_retries_used": recategorized_rate_limit_retries_used,
            "data": {
                "banks": [
                    {
                        "bankName": bank,
                        "accounts": [
                            {
                                "accountNo": account_number,
                                "csv_content": "\n".join(csv_rows),
                            }
                        ],
                    }
                ]
            },
            "transactions": recategorized_transactions,
        }
        return _serialize_for_json(response)
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to reapply statement run for client_id={client_id}, run_id={run_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to reapply statement run: {e}")


@router.post("/test-categorize")
async def test_categorize_by_description(
    body: TestCategorizeRequest = Body(..., description="Client ID and transaction description")):
    """
    Test API: Categorize a single description using the client's rule base.
    Accepts client_id and description; returns the category and the rule that was applied.

    Request body:
        - client_id: Client ID to lookup rules
        - description: Transaction description to categorize

    Response:
        - category: Matched category (account_name) or null if no rule matched
        - rule: The rule that was applied (rule_id, trigger_payee_contains, account_name, category_type, logic, etc.) or null
    """
    try:
        client_id = body.client_id
        description = body.description
        log_message("info", f"Test categorize request for client_id: {client_id}, description: {description[:80]}...")

        # Get client rules from MongoDB
        client_rules_doc = await get_client_rule_base(client_id)
        if not client_rules_doc:
            raise HTTPException(
                status_code=404,
                detail=f"No rule base found for client_id: {client_id}"
            )

        rules = client_rules_doc.get("rules", [])
        if not rules:
            return {
                "status": "success",
                "client_id": client_id,
                "description": description,
                "category": None,
                "rule": None,
                "message": "No rules defined for this client.",
            }

        # Build minimal transaction for matching
        transaction = {"description": description, "payee_name": ""}
        categorizer = BankStatementCategorizer()
        matched_rule = categorizer._match_transaction_to_rule(transaction, rules)

        if not matched_rule:
            return {
                "status": "success",
                "client_id": client_id,
                "description": description,
                "category": None,
                "rule": None,
                "message": "No rule matched this description.",
            }

        # Return category and the full rule that was applied
        category = matched_rule.get("account_name")
        return {
            "status": "success",
            "client_id": client_id,
            "description": description,
            "category": category,
            "rule": matched_rule,
            "message": f"Matched to category: {category}",
        }

    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Test categorize failed: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Test categorize failed: {e}")


@router.get("/clients")
async def get_all_clients_with_rules():
    """
    Get list of all clients that have rule bases.
    
    Returns:
        JSON response with list of clients (client_id and client_name only)
    """
    try:
        log_message("info", "Get all clients with rules request received")
        
        # Get all rule bases from MongoDB
        all_rule_bases = await get_all_client_rule_bases()
        
        # Extract only client_id and client_name
        clients = []
        for rule_base in all_rule_bases:
            clients.append({
                "client_id": rule_base.get("client_id"),
                "client_name": rule_base.get("client_name"),
                "rule_count": rule_base.get("rule_count", len(rule_base.get("rules") or [])),
                "gl_start_date": rule_base.get("gl_start_date"),
                "gl_end_date": rule_base.get("gl_end_date"),
                "gl_year": rule_base.get("gl_year"),
            })
        
        log_message("info", f"Found {len(clients)} clients with rule bases")
        
        return {
            "status": "success",
            "total_clients": len(clients),
            "clients": clients
        }
        
    except Exception as e:
        log_message("error", f"Failed to get all clients with rules: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get clients: {e}")


async def _get_serialized_rule_base_or_404(client_id: str) -> dict:
    """Fetch rule base by client_id, return JSON-serializable dict or raise HTTPException 404."""
    rule_base = await get_client_rule_base(client_id)
    if not rule_base:
        raise HTTPException(status_code=404, detail=f"No rule base found for client_id: {client_id}")
    return _serialize_for_json(rule_base)


@router.get("/rules")
async def get_rules_by_client_id(
    client_id: str = Query(..., description="Client ID to lookup rules in MongoDB")):
    """Get rule base for a client by client_id (query param). Returns 404 if not found."""
    try:
        log_message("info", f"Get rules by client_id request received for client_id: {client_id}")
        data = await _get_serialized_rule_base_or_404(client_id)
        log_message("info", f"Successfully retrieved rule base for client_id: {client_id}")
        return {"status": "success", "data": data}
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to get rules for client_id {client_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {e}")


@router.get("/{client_id}")
async def get_rule_base_by_id(
    client_id: str = Path(..., description="Client ID to retrieve rules")):
    """Get rule base for a client by client_id (path param). Returns 404 if not found."""
    try:
        log_message("info", f"Get rule base request received for client_id: {client_id}")
        data = await _get_serialized_rule_base_or_404(client_id)
        log_message("info", f"Successfully retrieved rule base for client_id: {client_id}")
        return {"status": "success", "data": data}
    except HTTPException:
        raise
    except Exception as e:
        log_message("error", f"Failed to get rule base for client_id {client_id}: {e}")
        log_message("error", f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get rule base: {e}")
