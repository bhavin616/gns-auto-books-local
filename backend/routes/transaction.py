"""
Transaction Routes - FastAPI Router

This module exposes endpoints for:
1. Uploading client documents into VectorDB
2. Categorizing transactions using AI
3. Extracting transactions from PDF statements
4. Full automated bank statement pipeline: PDF -> CSV -> Categorized CSV

Design Principles:
- Google-style docstrings and naming conventions
- Consistent JSON response schema
- Centralized error handling
- Efficient file IO
- Logging hooks (logger provided separately)
"""

from fastapi import APIRouter, UploadFile, File, status, Body, Form, BackgroundTasks, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from backend.interactors import transaction as transaction_interactors
from backend.utils.logger import log_message
from backend.utils.analytics import log_event
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from datetime import datetime
from dateutil import parser as date_parser
import csv
import io
import os
import json
import re
import pandas as pd
from io import BytesIO
from collections import defaultdict

router = APIRouter(prefix="/api")


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------

def serialize_for_json(obj):
    """
    Recursively convert datetime and ObjectId objects to strings for JSON serialization.
    
    Args:
        obj: Object to serialize (can be dict, list, datetime, ObjectId, or other types)
    
    Returns:
        JSON-serializable object
    """
    from bson import ObjectId
    
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


def normalize_description_for_comparison(description: str) -> str:
    """
    Normalize description/memo for comparison to handle variations in:
    - Case (uppercase/lowercase)
    - Special characters (#, -, etc.)
    - Whitespace (multiple spaces, tabs, etc.)
    
    Examples:
    - "CHECK # 3360" -> "check 3360"
    - "Check 3360" -> "check 3360"
    - "CHECK  3360" -> "check 3360" (multiple spaces normalized)
    
    Args:
        description: Description string to normalize
    
    Returns:
        Normalized description string
    """
    if not description:
        return ""
    
    # Convert to lowercase
    normalized = description.strip().lower()
    
    # Remove or normalize special characters that cause mismatches
    normalized = normalized.replace("#", " ")
    
    # Normalize other common separators to spaces
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("_", " ")
    
    # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
    import re
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Strip again after normalization
    normalized = normalized.strip()
    
    return normalized


def create_response(success: bool, message: str, data=None, code=status.HTTP_200_OK):
    """Factory to create consistent JSON API responses."""
    # Serialize data to handle datetime and ObjectId objects
    serialized_data = serialize_for_json(data) if data is not None else None
    
    return JSONResponse(
        content={
            "succeeded": success,
            "message": message,
            "data": serialized_data,
            "status_code": code,
        },
        status_code=code,
    )


def generate_csv_content_from_transaction(txn_obj: Dict[str, Any]) -> str:
    """
    Generate CSV content string (header + row) from a transaction object.
    
    Args:
        txn_obj: Transaction object with date, description, credit, debit, category, bank, payee, accountNumber
        
    Returns:
        CSV content string with header row and transaction row
    """
    def escape_csv(value):
        if value is None:
            return ""
        value_str = str(value)
        if ',' in value_str or '"' in value_str or '\n' in value_str:
            return '"' + value_str.replace('"', '""') + '"'
        return value_str
    
    # CSV header
    header = "Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber"
    
    # Transaction row
    row = (
        f"{escape_csv(txn_obj.get('date', ''))},"
        f"{escape_csv(txn_obj.get('description', ''))},"
        f"{escape_csv(txn_obj.get('credit', ''))},"
        f"{escape_csv(txn_obj.get('debit', ''))},"
        f"{escape_csv(txn_obj.get('category', ''))},"
        f"{escape_csv(txn_obj.get('bank', ''))},"
        f"{escape_csv(txn_obj.get('payee', ''))},"
        f"{escape_csv(txn_obj.get('accountNumber', ''))}"
    )
    
    return f"{header}\n{row}"


def parse_csv_to_transactions(csv_content: str) -> List[Dict[str, Any]]:
    """
    Parse CSV content to extract transaction objects.
    
    Args:
        csv_content: CSV string with headers: Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber
        
    Returns:
        List of transaction dictionaries
    """
    if not csv_content or not csv_content.strip():
        return []
    
    transactions = []
    try:
        # Use Python's csv module for proper parsing
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        for row in csv_reader:
            # Skip empty rows
            if not any(row.values()):
                continue
            transactions.append(dict(row))
    except Exception as e:
        log_message("warning", f"Error parsing CSV content: {e}")
        # Fallback to simple parsing if csv module fails
        lines = csv_content.strip().split('\n')
        if len(lines) < 2:
            return []
        
        header = [col.strip() for col in lines[0].split(',')]
        for line in lines[1:]:
            if not line.strip():
                continue
            values = [v.strip().strip('"') for v in line.split(',')]
            if len(values) >= len(header):
                txn = {header[i]: values[i] if i < len(values) else "" for i in range(len(header))}
                transactions.append(txn)
    
    return transactions


def restructure_response_by_bank_and_account(account_group_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restructure account group results into the format:
    {
        "data": {
            "banks": [
                {
                    "bankName": "BOA",
                    "accounts": [
                        {
                            "accountNo": "123",
                            "transactions": [transaction objects],
                            "csv_content": "CSV string"
                        }
                    ]
                }
            ]
        }
    }
    
    Also handles deduplication - if same bank/account exists, merges transactions
    and avoids duplicates based on (date, memo, amount).
    
    Args:
        account_group_results: Dict mapping account_key to result dict with csv_content, bank, account_number, etc.
        
    Returns:
        Restructured response dict
    """
    from collections import defaultdict
    
    # Structure: bank_name -> account_number -> set of (date, memo, amount) -> transaction
    banks_dict = defaultdict(lambda: defaultdict(lambda: {}))  # bank -> account -> {(date, memo, amount): txn}
    
    # Process each account group
    for account_key, result in account_group_results.items():
        bank = result.get("bank", "Unknown")
        account_number = result.get("account_number") or "NA"
        csv_content = result.get("csv_content", "")
        
        # Parse CSV to get transactions
        # Log first few lines of CSV for debugging (to verify category is in CSV)
        if csv_content:
            csv_lines = csv_content.strip().split('\n')
            if len(csv_lines) > 1:
                log_message("debug", f"CSV header: {csv_lines[0]}")
                if len(csv_lines) > 1:
                    log_message("debug", f"CSV first row: {csv_lines[1][:200]}...")
        
        transactions = parse_csv_to_transactions(csv_content)
        
        # Also check subaccounts
        subaccounts = result.get("subaccounts", {})
        for subaccount_num, subaccount_data in subaccounts.items():
            subaccount_csv = subaccount_data.get("csv_content", "")
            if subaccount_csv:
                subaccount_txns = parse_csv_to_transactions(subaccount_csv)
                transactions.extend(subaccount_txns)
        
        # Deduplicate and add transactions
        for txn in transactions:
            date = txn.get("Date", "").strip()
            memo = txn.get("Description", txn.get("bank_memo", "")).strip()
            credit = txn.get("Credit", "").strip()
            debit = txn.get("Debit", "").strip()
            
            # Get category from CSV - check multiple possible column names (case-insensitive)
            category_from_csv = txn.get("Category") or txn.get("category") or "Uncategorized"
            # Also check all keys case-insensitively
            if category_from_csv == "Uncategorized":
                for key in txn.keys():
                    if key.lower() == "category":
                        category_from_csv = txn.get(key, "Uncategorized")
                        break
            
            # Log category from CSV for debugging
            if category_from_csv and category_from_csv != "Ask My Accountant" and category_from_csv != "Uncategorized":
                log_message("debug", f"Category from CSV for '{memo[:50]}': '{category_from_csv}'")
            
            # Determine amount (debits in CSV are already negative)
            if credit:
                try:
                    amount = float(credit)
                except:
                    amount = 0.0
            elif debit:
                try:
                    # Debits in CSV are already stored as negative values
                    amount = float(debit)
                except:
                    amount = 0.0
            else:
                amount = 0.0
            
            # Use account number from transaction if available, otherwise use group account
            # If no account number, use "NA"
            txn_account = txn.get("AccountNumber", "").strip() or account_number or "NA"
            
            # Create deduplication key
            dedup_key = (date, memo.lower(), f"{amount:.2f}")
            
            # Only add if not already exists (deduplication)
            if dedup_key not in banks_dict[bank][txn_account]:
                # Store transaction object with all fields for generating CSV later
                # Use "NA" for empty account numbers
                account_num_display = txn_account if txn_account and txn_account != "default" else "NA"
                txn_obj = {
                    "date": date,
                    "description": memo,
                    "credit": credit if credit else "",
                    "debit": debit if debit else "",
                    "category": category_from_csv,  # Use category from CSV (preserves LLM predictions)
                    "bank": txn.get("Bank", bank),
                    "payee": txn.get("Payee") or "",  # Handle None values
                    "accountNumber": account_num_display
                }
                
                # Log final category in txn_obj for debugging
                if txn_obj.get("category") and txn_obj.get("category") != "Ask My Accountant" and txn_obj.get("category") != "Uncategorized":
                    log_message("debug", f"Final category in txn_obj for '{memo[:50]}': '{txn_obj.get('category')}'")
                
                banks_dict[bank][txn_account][dedup_key] = txn_obj
    
    # Convert to final structure
    banks_list = []
    for bank_name in sorted(banks_dict.keys()):
        accounts_list = []
        for account_no in sorted(banks_dict[bank_name].keys()):
            # Get all transactions for this account
            transactions_list = list(banks_dict[bank_name][account_no].values())
            
            # Generate ONE csv_content for this account with all transactions
            # Use proper CSV escaping to handle commas in dates and other fields
            def escape_csv_value(value):
                """Escape CSV value to handle commas, quotes, and newlines."""
                if value is None:
                    return ""
                value_str = str(value)
                if ',' in value_str or '"' in value_str or '\n' in value_str:
                    return '"' + value_str.replace('"', '""') + '"'
                return value_str
            
            csv_header = "Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber"
            csv_rows = [csv_header]
            
            for txn_obj in transactions_list:
                # Show "NA" for empty account numbers
                account_num = txn_obj.get('accountNumber', '') or 'NA'
                csv_row = (
                    f"{escape_csv_value(txn_obj.get('date', ''))},"
                    f"{escape_csv_value(txn_obj.get('description', ''))},"
                    f"{escape_csv_value(txn_obj.get('credit', ''))},"
                    f"{escape_csv_value(txn_obj.get('debit', ''))},"
                    f"{escape_csv_value(txn_obj.get('category', ''))},"
                    f"{escape_csv_value(txn_obj.get('bank', ''))},"
                    f"{escape_csv_value(txn_obj.get('payee', ''))},"
                    f"{escape_csv_value(account_num)}"
                )
                csv_rows.append(csv_row)
            
            account_csv_content = "\n".join(csv_rows)
            
            accounts_list.append({
                "accountNo": account_no,
                "csv_content": account_csv_content
            })
        
        banks_list.append({
            "bankName": bank_name,
            "accounts": accounts_list
        })
    
    return {
        "data": {
            "banks": banks_list
        }
    }


def detect_file_type(filename: str) -> str:
    """
    Detect file type from filename extension.
    
    Returns: 'pdf', 'csv', 'excel', or 'unknown'
    """
    if not filename:
        return "unknown"
    
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return 'pdf'
    elif filename_lower.endswith('.csv'):
        return 'csv'
    elif filename_lower.endswith(('.xlsx', '.xls', '.xlsm')):
        return 'excel'
    else:
        return 'unknown'


def _load_bankname_prompt() -> str:
    """
    Load bank name detection prompt from file.
    
    Returns:
        Prompt text as string
    """
    try:
        from pathlib import Path
        current_dir = Path(__file__).parent.parent
        prompt_file_path = current_dir / "services" / "prompts" / "bankname_extract.txt"
        
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        log_message("info", f"Loaded bank name detection prompt from: {prompt_file_path}")
        return prompt_text
        
    except FileNotFoundError:
        log_message("error", f"Bank name prompt file not found: bankname_extract.txt")
        raise FileNotFoundError(f"Bank name prompt file not found: bankname_extract.txt")
    except Exception as e:
        log_message("error", f"Failed to load bank name prompt file: {e}")
        raise Exception(f"Failed to load bank name prompt file: {e}")


async def contains_bank_name_llm(text: str, prompt: str, llm) -> bool:
    """
    Check if a text string contains a bank name using LLM based on prompt.
    
    Args:
        text: Text string to check (e.g., from CSV Split column)
        prompt: Bank name detection prompt
        llm: LLM instance (ChatGoogleGenerativeAI)
    
    Returns:
        True if the text contains a bank name pattern, False otherwise
    """
    if not text or not isinstance(text, str) or not text.strip():
        return False
    
    try:
        # Create classification prompt
        classification_prompt = f"""{prompt}

Based on the above rules and examples, analyze the following text and determine if it contains a bank name.

Text to analyze: "{text}"

Respond with ONLY "YES" if the text contains a bank name, or "NO" if it does not. Do not include any other text or explanation."""
        
        # Call LLM
        response = await asyncio.to_thread(llm.invoke, classification_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip().upper()
        
        # Check if response indicates bank name
        return "YES" in response_text or response_text.startswith("YES")
        
    except Exception as e:
        log_message("error", f"LLM bank name detection failed for text '{text}': {e}")
        # Fallback to False on error
        return False


async def batch_detect_bank_names(texts: List[str], prompt: str, client, model_name: str) -> Dict[str, bool]:
    """
    Batch detect bank names for multiple texts using google.genai package.
    
    Args:
        texts: List of text strings to check
        prompt: Bank name detection prompt
        client: google.genai.Client instance
        model_name: Model name (e.g., "gemini-3-flash-preview")
    
    Returns:
        Dictionary mapping text -> bool (True if contains bank name)
        Returns empty dict if quota exhausted (to signal caller to stop)
    """
    if not texts:
        return {}
    
    try:
        # Create batch classification prompt
        texts_list = "\n".join([f"- \"{text}\"" for text in texts])
        batch_prompt = f"""{prompt}

Based on the above rules and examples, analyze the following texts and determine which ones contain bank names.

Texts to analyze:
{texts_list}

Respond with a JSON object where each key is the text (exactly as provided) and the value is true if it contains a bank name, false otherwise.

Example format:
{{
  "Wells Fargo Chk XXXX8615": true,
  "Office Supplies": false,
  "Wells Fargo #3468": true
}}

Return ONLY the JSON object, no other text."""
        
        # Call LLM with exponential backoff for rate limits
        max_attempts = 3
        base_delay = 2.0
        
        for attempt in range(max_attempts):
            try:
                # Use google.genai package (new SDK)
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=[batch_prompt]
                )
                
                # Extract response text
                response_text = response.text if hasattr(response, 'text') else str(response)
                response_text = response_text.strip()
                
                # Try to extract JSON from response
                # Remove markdown code blocks if present
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
                
                # Parse JSON
                try:
                    result = json.loads(response_text)
                    # Ensure all input texts are in the result
                    for text in texts:
                        if text not in result:
                            result[text] = False
                    return result
                except json.JSONDecodeError:
                    log_message("error", f"Failed to parse LLM response as JSON: {response_text}")
                    # Fallback: return all False
                    return {text: False for text in texts}
                    
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a quota exhaustion error (ResourceExhausted: 429)
                is_quota_error = (
                    "ResourceExhausted" in error_str or
                    "429" in error_str or
                    "quota" in error_str.lower() or
                    "exceeded your current quota" in error_str.lower()
                )
                
                if is_quota_error:
                    log_message("error", f"Gemini API quota exhausted: {error_str}")
                    log_message("info", "Stopping batch processing due to quota exhaustion. Returning empty dict to signal caller.")
                    # Return empty dict to signal caller that quota is exhausted
                    return {}
                
                # For other errors, retry with exponential backoff
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                    log_message("warning", f"LLM call failed (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    log_message("error", f"Batch LLM bank name detection failed after {max_attempts} attempts: {e}")
                    # Fallback: return all False
                    return {text: False for text in texts}
        
        # Should not reach here, but fallback
        return {text: False for text in texts}
        
    except Exception as e:
        error_str = str(e)
        # Check if it's a quota exhaustion error
        is_quota_error = (
            "ResourceExhausted" in error_str or
            "429" in error_str or
            "quota" in error_str.lower() or
            "exceeded your current quota" in error_str.lower()
        )
        
        if is_quota_error:
            log_message("error", f"Gemini API quota exhausted: {error_str}")
            return {}  # Return empty dict to signal quota exhaustion
        
        log_message("error", f"Batch LLM bank name detection failed: {e}")
        # Fallback: return all False
        return {text: False for text in texts}


async def filter_gl_file_for_bank_names(raw: bytes, filename: str) -> bytes:
    """
    Filter CSV/Excel file to remove rows where Split column contains bank names.
    
    This function applies the same bank name filtering logic as /filter_csv_columns endpoint,
    but returns the filtered file as bytes for further processing.
    
    Args:
        raw: Raw file bytes (CSV or Excel)
        filename: Original filename (for type detection)
    
    Returns:
        Filtered file bytes (same format as input)
    """
    try:
        filename_lower = filename.lower() if filename else ""
        is_excel = filename_lower.endswith(('.xlsx', '.xls', '.xlsm'))
        is_csv = filename_lower.endswith('.csv')
        
        if not (is_csv or is_excel):
            log_message("warning", f"Unsupported file type for filtering: {filename}. Returning original file.")
            return raw
        
        # Read file into DataFrame
        if is_excel:
            df = pd.read_excel(BytesIO(raw), engine='openpyxl')
            input_fieldnames = list(df.columns)
        else:
            # Try multiple encodings for CSV
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    text_data = raw.decode(encoding)
                    input_io = io.StringIO(text_data)
                    df = pd.read_csv(input_io)
                    input_fieldnames = list(df.columns)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                log_message("warning", f"Failed to decode CSV file. Returning original file.")
                return raw
        
        # Check if Split column exists
        if "Split" not in input_fieldnames:
            # Try case-insensitive search
            split_col = None
            for col in input_fieldnames:
                if col.lower() == "split":
                    split_col = col
                    break
            
            if not split_col:
                log_message("info", f"No 'Split' column found in file. Skipping bank name filtering.")
                # Return original file as bytes
                if is_excel:
                    output_io = BytesIO()
                    df.to_excel(output_io, index=False, engine='openpyxl')
                    return output_io.getvalue()
                else:
                    output_io = io.StringIO()
                    df.to_csv(output_io, index=False)
                    return output_io.getvalue().encode('utf-8')
        else:
            split_col = "Split"
        
        # Apply bank name filtering
        initial_row_count = len(df)
        df = df.fillna("")
        
        try:
            # Load bank name detection prompt
            bankname_prompt = _load_bankname_prompt()
            
            # Initialize google.genai package (new SDK, not deprecated)
            try:
                import google.genai as genai
            except ImportError:
                log_message("error", "google-genai package not installed. Install with: pip install google-genai")
                raise ImportError("google-genai package is required. Install with: pip install google-genai")
            
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
            # Create client (same pattern as gemini_pdf_extractor.py)
            client = genai.Client(api_key=api_key)
            model_name = "gemini-3-flash-preview"
            
            log_message("info", f"Initialized google.genai client for bank name detection with model: {model_name}")
            
            # Get unique Split values for batch processing
            unique_splits = df[split_col].astype(str).unique().tolist()
            unique_splits = [s for s in unique_splits if s and s.strip() and s != "nan"]
            
            if not unique_splits:
                log_message("info", "No Split values to process. Returning original file.")
                if is_excel:
                    output_io = BytesIO()
                    df.to_excel(output_io, index=False, engine='openpyxl')
                    return output_io.getvalue()
                else:
                    output_io = io.StringIO()
                    df.to_csv(output_io, index=False)
                    return output_io.getvalue().encode('utf-8')
            
            log_message("info", f"Processing {len(unique_splits)} unique Split values for bank name detection in GL file")
            
            # Batch process unique Split values
            batch_size = 50
            bank_name_map = {}
            quota_exhausted = False
            
            total_batches = (len(unique_splits) + batch_size - 1) // batch_size
            
            for i in range(0, len(unique_splits), batch_size):
                batch = unique_splits[i:i + batch_size]
                batch_num = i//batch_size + 1
                log_message("info", f"Processing batch {batch_num}/{total_batches} ({len(batch)} items) for GL file filtering")
                
                # Add delay between batches to respect rate limits (2 seconds between batches)
                if i > 0:
                    await asyncio.sleep(2.0)
                
                batch_results = await batch_detect_bank_names(batch, bankname_prompt, client, model_name)
                
                # Check if quota is exhausted (empty dict returned)
                if batch_results == {} and len(batch) > 0:
                    quota_exhausted = True
                    log_message("warning", f"Quota exhausted after processing {batch_num - 1} batches. Remaining batches will be skipped.")
                    break
                
                bank_name_map.update(batch_results)
            
            if quota_exhausted:
                log_message("warning", f"Quota exhausted during GL file processing. Processed {len(bank_name_map)} unique Split values. Proceeding with partial results.")
            
            # Create a mask: keep rows where Split does NOT contain bank names
            def is_bank_name(split_value: str) -> bool:
                split_str = str(split_value) if pd.notna(split_value) else ""
                return bank_name_map.get(split_str, False)
            
            mask = ~df[split_col].astype(str).apply(is_bank_name)
            filtered_df = df[mask].copy()
            
            filtered_row_count = len(filtered_df)
            removed_count = initial_row_count - filtered_row_count
            
            log_message(
                "info",
                f"GL file filtered: {initial_row_count} rows -> {filtered_row_count} rows "
                f"(removed {removed_count} bank-related rows)"
            )
            
        except Exception as e:
            log_message("error", f"Bank name filtering failed for GL file: {e}. Using original file.")
            import traceback
            log_message("error", f"Traceback: {traceback.format_exc()}")
            # Return original file if filtering fails
            filtered_df = df
        
        # Convert filtered DataFrame back to bytes
        if is_excel:
            output_io = BytesIO()
            filtered_df.to_excel(output_io, index=False, engine='openpyxl')
            return output_io.getvalue()
        else:
            output_io = io.StringIO()
            filtered_df.to_csv(output_io, index=False)
            return output_io.getvalue().encode('utf-8')
            
    except Exception as e:
        log_message("error", f"Error filtering GL file: {e}. Returning original file.")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return raw


def _run_add_gl_background(client_id: str, client_name: str, index_name: str, raw: bytes, filename: str) -> None:
    """
    Synchronous wrapper to run the async GL ingestion in a background task.

    This is needed because FastAPI's BackgroundTasks expects a regular
    callable, not a coroutine or asyncio.create_task.
    
    This function:
    1. Sets status to "processing" at start
    2. Detects duplicates (same memo + same split) after filtering bank rows
    3. Stores duplicates in client_uniquedata collection
    4. Sets status to "success" at end or "failed" on error
    """
    # Create a new event loop for this background task
    # This is necessary because FastAPI's BackgroundTasks runs in a thread pool
    # and we need a fresh event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        async def _async_background_task():
            """Inner async function that handles all async operations."""
            from backend.services.mongodb import create_or_update_client_status, insert_client_unique_data
            
            # Use nonlocal to modify the outer function's raw variable
            nonlocal raw
            
            try:
                log_message("info", f"Starting background GL ingestion for client_id={client_id}, client={client_name}, index_name={index_name}, filename={filename}")
                
                # Create a new MongoDB client for this event loop (don't touch the global one)
                from motor.motor_asyncio import AsyncIOMotorClient
                import os
                from dotenv import load_dotenv
                load_dotenv()
                
                mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
                mongodb_db_name = os.getenv("MONGODB_DB_NAME", "cpa_backend_migration")
                
                # Create a local MongoDB client for this background task's event loop
                local_client = AsyncIOMotorClient(mongodb_url)
                local_db = local_client[mongodb_db_name]
                
                try:
                    # Set status to "processing" using local database
                    status_collection = local_db["client_status"]
                    from datetime import datetime as dt
                    
                    existing_status = await status_collection.find_one({"client_id": client_id})
                    if existing_status:
                        await status_collection.update_one(
                            {"client_id": client_id},
                            {"$set": {"client_name": client_name, "status": "processing", "updated_at": dt.utcnow()}}
                        )
                    else:
                        await status_collection.insert_one({
                            "client_id": client_id,
                            "client_name": client_name,
                            "status": "processing",
                            "created_at": dt.utcnow(),
                            "updated_at": dt.utcnow()
                        })
                    log_message("info", f"Set status to 'processing' for client_id={client_id}")
                    
                    # Step 1: Filter out bank-related rows from Split column
                    log_message("info", f"Filtering GL file to remove bank-related rows from Split column: {filename}")
                    try:
                        filtered_raw = await filter_gl_file_for_bank_names(raw, filename)
                        log_message("info", f"GL file filtered successfully. Using filtered version for processing.")
                        raw = filtered_raw  # Use filtered file
                    except Exception as filter_error:
                        log_message("warning", f"Failed to filter GL file: {filter_error}. Using original file.")
                        # Continue with original file if filtering fails
                    
                    # Step 2: Parse the filtered GL file to detect duplicates
                    # First, read the file into a DataFrame to check for duplicates
                    filename_lower = filename.lower() if filename else ""
                    is_excel = filename_lower.endswith(('.xlsx', '.xls', '.xlsm'))
                    
                    duplicate_data = []
                    if is_excel:
                        df = pd.read_excel(BytesIO(raw), engine='openpyxl')
                    else:
                        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
                        df = None
                        for encoding in encodings:
                            try:
                                text_data = raw.decode(encoding)
                                input_io = io.StringIO(text_data)
                                df = pd.read_csv(input_io)
                                break
                            except (UnicodeDecodeError, UnicodeError):
                                continue
                        
                        if df is None:
                            raise ValueError(f"Failed to decode CSV file with any encoding")
                    
                    # Normalize column names (case-insensitive, strip whitespace)
                    df.columns = df.columns.str.strip().str.lower()
                    
                    # Find memo and split columns
                    memo_col = None
                    split_col = None
                    date_col = None
                    credit_col = None
                    debit_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if memo_col is None and 'memo' in col_lower:
                            memo_col = col
                        if split_col is None and ('split' in col_lower or 'category' in col_lower):
                            split_col = col
                        if date_col is None and 'date' in col_lower:
                            date_col = col
                        if credit_col is None and 'credit' in col_lower:
                            credit_col = col
                        if debit_col is None and 'debit' in col_lower:
                            debit_col = col
                    
                    # Detect duplicates: same memo + same split
                    if memo_col and split_col:
                        df = df.fillna("")
                        # Filter out rows with empty memo or split (these are not valid duplicates)
                        df_filtered = df[
                            (df[memo_col].astype(str).str.strip() != "") & 
                            (df[split_col].astype(str).str.strip() != "")
                        ].copy()
                        
                        if len(df_filtered) == 0:
                            log_message("info", "No valid rows with memo and split found for duplicate detection")
                        else:
                            # Create a key for duplicate detection: memo + split (case-insensitive)
                            df_filtered['_dup_key'] = df_filtered[memo_col].astype(str).str.strip().str.lower() + "|||" + df_filtered[split_col].astype(str).str.strip().str.lower()
                            
                            # Find duplicates (keep=False means keep all occurrences, not just the first)
                            duplicates = df_filtered[df_filtered.duplicated(subset=['_dup_key'], keep=False)].copy()
                            
                            if len(duplicates) > 0:
                                log_message("info", f"Found {len(duplicates)} duplicate rows (same memo + same split) in GL file")
                                
                                # Group duplicates by memo+split
                                for dup_key, group in duplicates.groupby('_dup_key'):
                                    # Get the first occurrence values
                                    first_row = group.iloc[0]
                                    
                                    # Build duplicate record
                                    dup_record = {
                                        "memo": str(first_row[memo_col]).strip() if memo_col else "",
                                        "split": str(first_row[split_col]).strip() if split_col else "",
                                        "occurrence_count": len(group)
                                    }
                                    
                                    # Add date, credit, debit if available
                                    if date_col:
                                        dup_record["date"] = str(first_row[date_col]).strip() if pd.notna(first_row[date_col]) else ""
                                    if credit_col:
                                        dup_record["credit"] = str(first_row[credit_col]).strip() if pd.notna(first_row[credit_col]) else ""
                                    if debit_col:
                                        dup_record["debit"] = str(first_row[debit_col]).strip() if pd.notna(first_row[debit_col]) else ""
                                    
                                    duplicate_data.append(dup_record)
                                
                                # Store duplicates in client_uniquedata collection using local database
                                if duplicate_data:
                                    uniquedata_collection = local_db["client_uniquedata"]
                                    unique_doc = {
                                        "client_id": client_id,
                                        "client_name": client_name,
                                        "pinecone_index_name": index_name,
                                        "duplicate_data": duplicate_data,
                                        "duplicate_count": len(duplicate_data),
                                        "created_at": dt.utcnow()
                                    }
                                    await uniquedata_collection.insert_one(unique_doc)
                                    log_message("info", f"Stored {len(duplicate_data)} duplicate groups in client_uniquedata collection")
                            else:
                                log_message("info", "No duplicates found in GL file (same memo + same split)")
                    
                    # Now run the actual GL ingestion
                    result = await transaction_interactors.add_doc_call_from_bytes(
                        client_name=client_name,
                        index_name=index_name,
                        raw=raw,
                        filename=filename,
                    )
                    
                    # Check if ingestion was successful
                    if result.get("status") == "success":
                        # Set status to "success" using local database
                        await status_collection.update_one(
                            {"client_id": client_id},
                            {"$set": {"status": "success", "updated_at": dt.utcnow()}}
                        )
                        log_message("info", f"Background GL ingestion completed successfully for client_id={client_id}, client={client_name}, filename={filename}")
                    else:
                        # Set status to "failed"
                        error_msg = result.get("error", "Unknown error")
                        await status_collection.update_one(
                            {"client_id": client_id},
                            {"$set": {"status": "failed", "updated_at": dt.utcnow()}}
                        )
                        log_message("error", f"Background GL ingestion failed for client_id={client_id}, client={client_name}, filename={filename}: {error_msg}")
                        raise Exception(error_msg)
                finally:
                    # Close the local client
                    local_client.close()
            
            except Exception as e:
                # Set status to "failed" on exception using local database
                try:
                    status_collection_err = local_db["client_status"]
                    await status_collection_err.update_one(
                        {"client_id": client_id},
                        {"$set": {"status": "failed", "updated_at": dt.utcnow()}}
                    )
                except Exception as status_error:
                    log_message("error", f"Failed to update status to failed: {status_error}")
                
                log_message("error", f"Background GL ingestion failed for client_id={client_id}, client={client_name}, filename={filename}: {e}")
                import traceback
                log_message("error", f"Traceback: {traceback.format_exc()}")
                raise
        
        # Run the async function in the new event loop
        loop.run_until_complete(_async_background_task())
    finally:
        # Clean up the event loop
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Wait for tasks to be cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


# -------------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------------

@router.post("/add_gl")
async def add_gl_to_vectordb(
    request: Request,
    background_tasks: BackgroundTasks,
    client_name: str = Form(...),
    client_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Form(None),
):
    """
    Upload a client GL file and store it inside the VectorDB.
    
    This endpoint automatically filters out bank-related rows from the Split column before uploading:
    1. Detects bank names in Split column using LLM (based on bankname_extract.txt prompt)
    2. Removes rows where Split contains bank names (e.g., "Wells Fargo Chk XXXX8615")
    3. Uploads only the filtered data to Pinecone

    This endpoint automatically handles Pinecone index resolution:
    1. First checks if the client has a Pinecone index in the client_pinecone_index collection
    2. If not found, retrieves the client from the clients collection
    3. Automatically creates a Pinecone index based on the client's name from the clients collection
    4. Stores the mapping in client_pinecone_index collection for future use
    
    """
    import time
    start_time = time.time()
    
    try:
        if not client_name or not client_name.strip():
            return create_response(
                False,
                "client_name is required and cannot be empty.",
                code=status.HTTP_400_BAD_REQUEST,
            )

        if not client_id or not client_id.strip():
            return create_response(
                False,
                "client_id is required and cannot be empty.",
                code=status.HTTP_400_BAD_REQUEST,
            )

        raw = await file.read()
        filename = file.filename or ""
        
        # Note: Bank name filtering will be done in the background task
        # This allows the API to return immediately

        # Get index name from client_pinecone_index collection
        from backend.services.mongodb import get_client_pinecone_index, get_client_by_id
        from backend.services.pinecone import create_index_for_client
        
        client_index = await get_client_pinecone_index(client_id)
        resolved_index_name = None
        db_client_name = None
        
        # If client_index exists and has pinecone_index_name, use it
        if client_index and client_index.get("pinecone_index_name"):
            resolved_index_name = client_index.get("pinecone_index_name")
            db_client_name = client_index.get("name")
            log_message(
                "info",
                f"Found stored pinecone_index_name for client_id={client_id}: {resolved_index_name}"
            )
        
        # If no pinecone_index_name found, we need to create it
        if not resolved_index_name:
            # Case 1: Client exists in client_pinecone_index but no pinecone_index_name
            # Use the name from client_pinecone_index to create the index
            if client_index and client_index.get("name"):
                db_client_name = client_index.get("name")
                log_message(
                    "info",
                    f"Client found in client_pinecone_index but no pinecone_index_name for client_id={client_id}. "
                    f"Using name '{db_client_name}' from client_pinecone_index to create index..."
                )
            else:
                # Case 2: Client NOT found in client_pinecone_index
                # Fetch from clients collection and add to client_pinecone_index
                log_message(
                    "info",
                    f"Client not found in client_pinecone_index for client_id={client_id}. "
                    f"Fetching from clients collection..."
                )
                
                # Get client from clients collection
                client = await get_client_by_id(client_id)
                if not client:
                    return create_response(
                        False,
                        f"Client with ID '{client_id}' not found in clients collection. "
                        f"Please ensure the client exists before uploading GL files.",
                        code=status.HTTP_400_BAD_REQUEST,
                    )
                
                # Get client name from clients collection
                # First try to get 'company', if null or empty then use 'name'
                db_client_name = client.get("company")
                if not db_client_name or not db_client_name.strip():
                    db_client_name = client.get("name")
                
                if not db_client_name or not db_client_name.strip():
                    return create_response(
                        False,
                        f"Client with ID '{client_id}' does not have a company or name in the clients collection. "
                        f"Please ensure the client has a company or name before uploading GL files.",
                        code=status.HTTP_400_BAD_REQUEST,
                    )
                
                log_message(
                    "info",
                    f"Found client in clients collection: client_id={client_id}, name={db_client_name}. "
                    f"Adding to client_pinecone_index table..."
                )
                
                # Add client info to client_pinecone_index table (without pinecone_index_name initially)
                from backend.services.mongodb import create_or_update_client_pinecone_index
                
                # Store client info in client_pinecone_index (without pinecone_index_name)
                store_success = await create_or_update_client_pinecone_index(
                    client_id=client_id,
                    client_name=db_client_name,
                    pinecone_index_name=None  # Store without index name first
                )
                
                if not store_success:
                    log_message(
                        "warning",
                        f"Failed to store client info in client_pinecone_index for client_id={client_id}. "
                        f"Continuing with index creation..."
                    )
                else:
                    log_message(
                        "info",
                        f"Stored client info in client_pinecone_index: client_id={client_id}, name={db_client_name}"
                    )
            
            # Now create Pinecone index based on client name (from either client_pinecone_index or clients collection)
            log_message(
                "info",
                f"Creating Pinecone index for client '{db_client_name}'..."
            )
            
            # Create Pinecone index based on client name
            index_result = create_index_for_client(db_client_name, None)
            
            if index_result.get("status") != "success":
                return create_response(
                    False,
                    f"Failed to create Pinecone index for client '{db_client_name}': {index_result.get('message', 'Unknown error')}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            
            resolved_index_name = index_result.get("index_name")
            log_message(
                "info",
                f"Created Pinecone index '{resolved_index_name}' for client '{db_client_name}'"
            )
            
            # Update client_pinecone_index with the pinecone_index_name
            from backend.services.mongodb import create_or_update_client_pinecone_index
            
            update_success = await create_or_update_client_pinecone_index(
                client_id=client_id,
                client_name=db_client_name,
                pinecone_index_name=resolved_index_name
            )
            
            if update_success:
                log_message(
                    "info",
                    f"Updated client_pinecone_index with pinecone_index_name: "
                    f"client_id={client_id}, name={db_client_name}, index_name={resolved_index_name}"
                )
            else:
                log_message(
                    "warning",
                    f"Created Pinecone index but failed to update mapping in client_pinecone_index. "
                    f"The index '{resolved_index_name}' is available but may need to be recreated on next upload."
                )

        # Use client name from DB if available for consistency
        if db_client_name:
            client_name = db_client_name

        # Ensure resolved_index_name is set (should be set by now)
        if not resolved_index_name:
            return create_response(
                False,
                f"Failed to resolve Pinecone index name for client_id='{client_id}'. "
                f"This should not happen. Please contact support.",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Verify Pinecone index exists (should already exist from above logic)
        try:
            from pinecone import Pinecone
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
            if PINECONE_API_KEY:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                if not pc.has_index(resolved_index_name):
                    log_message(
                        "warning",
                        f"Pinecone index '{resolved_index_name}' does not exist in Pinecone. "
                        f"This may indicate an issue with index creation. Attempting to create..."
                    )
                    # Safety fallback: Create the Pinecone index if somehow it doesn't exist
                    from pinecone import ServerlessSpec
                    pc.create_index(
                        name=resolved_index_name,
                        dimension=768,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    log_message("info", f"Pinecone index '{resolved_index_name}' created successfully (fallback).")
                else:
                    log_message("info", f"Pinecone index '{resolved_index_name}' verified in Pinecone.")
            else:
                log_message("warning", "PINECONE_API_KEY not found. Cannot verify Pinecone index.")
        except Exception as e:
            log_message("error", f"Error verifying Pinecone index: {e}")
            return create_response(
                False,
                f"Failed to verify Pinecone index '{resolved_index_name}': {e}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        log_message(
            "info",
            f"Received GL upload (queued) for client={client_name}, client_id={client_id}, "
            f"index_name={resolved_index_name}, filename={filename}",
        )

        # Queue background ingestion (all processing runs in background)
        # This includes: bank name filtering, duplicate detection, and Pinecone upload
        background_tasks.add_task(
            _run_add_gl_background,
            client_id,
            db_client_name or client_name,
            resolved_index_name,
            raw,
            filename,
        )

        # Log analytics event for successful GL upload queue
        # Use user_id if available, otherwise use client_id
        analytics_id = user_id if user_id else client_id
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            file_size = len(raw) if raw else 0
            
            await log_event(
                user_id_or_client_id=analytics_id,
                event_type='tool_usage',
                category='GLUpload',
                action='gl_queued',
                metadata={
                    'toolName': 'GLUpload',
                    'toolMode': 'upload',
                    'fileName': filename,
                    'fileSize': file_size,
                    'fileSizeKB': round(file_size / 1024, 2) if file_size else 0,
                    'clientName': db_client_name or client_name,
                    'clientId': client_id,
                    'indexName': resolved_index_name,
                    'status': 'queued'
                },
                request=request,
                options={
                    'duration': duration_ms,
                    'status': 'success',
                    'clientId': client_id  # Pass client_id in options
                }
            )
            log_message('info', f"Analytics event logged: GLUpload/gl_queued for {'user_id' if user_id else 'client_id'}={analytics_id}")
        except Exception as analytics_error:
            log_message('error', f"Error logging GL upload analytics: {analytics_error}")
            import traceback
            log_message('error', f"Analytics traceback: {traceback.format_exc()}")
            # Don't fail the main request if analytics logging fails

        # Return immediately - all processing happens in background
        return create_response(
            True,
            "GL upload queued for processing. All operations (filtering, duplicate detection, upload) will run in background.",
            {
                "queued": True,
                "client_name": db_client_name or client_name,
                "client_id": client_id,
                "index_name": resolved_index_name,
                "filename": filename,
                "status": "queued",
                "message": "Processing started in background. Use /api/gl_status to check progress."
            },
        )

    except Exception as e:
        log_message("error", f"VectorDB upload failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        
        # Log analytics event for failed GL upload
        # Use user_id if available, otherwise use client_id
        analytics_id = user_id if user_id else client_id
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            file_size = len(raw) if 'raw' in locals() and raw else 0
            
            await log_event(
                user_id_or_client_id=analytics_id,
                event_type='tool_usage',
                category='GLUpload',
                action='gl_upload_failed',
                metadata={
                    'toolName': 'GLUpload',
                    'toolMode': 'upload',
                    'fileName': filename if 'filename' in locals() else 'unknown',
                    'fileSize': file_size,
                    'fileSizeKB': round(file_size / 1024, 2) if file_size else 0,
                    'clientName': client_name,
                    'clientId': client_id,
                    'error': str(e)
                },
                request=request,
                options={
                    'duration': duration_ms,
                    'status': 'error',
                    'errorMessage': str(e),
                    'clientId': client_id  # Pass client_id in options
                }
            )
            log_message('info', f"Analytics error event logged for {'user_id' if user_id else 'client_id'}={analytics_id}")
        except Exception as analytics_error:
            log_message('error', f"Error logging GL upload error analytics: {analytics_error}")
            import traceback as tb
            log_message('error', f"Analytics traceback: {tb.format_exc()}")
            # Don't fail the main request if analytics logging fails
        
        return create_response(False, f"Error saving file to VectorDB: {e}", code=status.HTTP_400_BAD_REQUEST)


@router.post("/get_category")
async def categorize_file(file: UploadFile, client_name: str | None = None):
    """
    Categorize uploaded CSV/Excel file using AI.

    Args:
        file: UploadFile containing extracted transaction CSV.
        client_name: Optional client name for context.

    Returns:
        Categorization results.
    """
    try:
        log_message("info", f"Categorization requested (client={client_name})")
        content = await file.read()
        file_name = file.filename.rsplit(".", 1)[0]

        response = await transaction_interactors.call(content, file_name, client_name)
        return create_response(True, "Successfully processed categorization", response)

    except Exception as e:
        log_message("error", f"Categorization failed: {e}")
        return create_response(False, f"Categorization error: {e}", code=status.HTTP_400_BAD_REQUEST)


@router.post("/extract")
async def extract_transactions(file: UploadFile):
    """
    Extract transactions from uploaded PDF and generate CSV.

    Args:
        file: PDF bank statement.

    Returns:
        JSON with extraction status and CSV path.
    """
    try:
        log_message("info", f"Extract request for file={file.filename}")
        response = await transaction_interactors.extract_transactions(file)

        if response.get("status") == "failed":
            return create_response(False, response.get("message"), code=status.HTTP_501_NOT_IMPLEMENTED)

        return create_response(True, "Successfully extracted transactions", response.get("data"))

    except Exception as e:
        log_message("error", f"Extraction failed: {e}")
        return create_response(False, f"Error extracting transactions: {e}", code=status.HTTP_400_BAD_REQUEST)


@router.delete("/clear_pinecone")
async def clear_pinecone_data():
    """
    Delete all data from Pinecone index.
    
    This endpoint clears all vectors from the Pinecone index.
    Use this to remove empty or unwanted data before re-uploading clean data.

    Returns:
        JSON with deletion status and count.
    """
    try:
        from backend.services.pinecone import delete_all_records
        
        log_message("info", "Received request to clear all Pinecone data")
        result = delete_all_records()
        
        if result.get("status") == "success":
            return create_response(True, result.get("message", "Pinecone data cleared successfully"), result)
        else:
            return create_response(False, result.get("error", "Failed to clear Pinecone data"), result, code=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        log_message("error", f"Clear Pinecone failed: {e}")
        return create_response(False, f"Error clearing Pinecone: {e}", code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# -------------------------------------------------------------------------
# Pinecone Index Management
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Client Pinecone Index CRUD Endpoints
# -------------------------------------------------------------------------

class CreateClientPineconeIndexRequest(BaseModel):
    """Request model to create a client Pinecone index record."""
    client_id: str
    name: str
    pinecone_index_name: Optional[str] = ""


class UpdateClientPineconeIndexRequest(BaseModel):
    """Request model to update a client Pinecone index record."""
    name: Optional[str] = None
    pinecone_index_name: Optional[str] = None


@router.post("/client_pinecone_index")
async def create_client_pinecone_index(request: CreateClientPineconeIndexRequest):
    """
    Create a new client Pinecone index record in the database.
    
    This endpoint allows you to manually create a client_pinecone_index record
    without creating the actual Pinecone index. Use this if you want to manage
    the mapping separately.
    
    Args:
        request: Request body with:
            - client_id: Client identifier (required)
            - name: Client name (required)
            - pinecone_index_name: Pinecone index name (required)
    
    Returns:
        JSON response with created record including _id.
    """
    try:
        from backend.services.mongodb import create_or_update_client_pinecone_index
        
        if not request.client_id or not request.client_id.strip():
            return create_response(
                False,
                "client_id is required",
                code=status.HTTP_400_BAD_REQUEST,
            )
        
        if not request.name or not request.name.strip():
            return create_response(
                False,
                "name is required",
                code=status.HTTP_400_BAD_REQUEST,
            )
        
        # pinecone_index_name is optional - if empty string or None, store only client_id and name
        pinecone_index_name = request.pinecone_index_name.strip() if request.pinecone_index_name and request.pinecone_index_name.strip() else None
        
        log_message(
            "info",
            f"Creating client_pinecone_index record: client_id={request.client_id}, "
            f"name={request.name}, index_name={pinecone_index_name or '(not provided)'}"
        )
        
        # Create or update the record
        success = await create_or_update_client_pinecone_index(
            client_id=request.client_id,
            client_name=request.name,
            pinecone_index_name=pinecone_index_name
        )
        
        if not success:
            return create_response(
                False,
                "Failed to create client_pinecone_index record",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
        # Retrieve the created record to return with _id
        from backend.services.mongodb import get_client_pinecone_index
        created_record = await get_client_pinecone_index(request.client_id)
        
        if created_record:
            # Add id field for convenience (serialization handled by create_response)
            if "_id" in created_record:
                created_record["id"] = str(created_record["_id"])
            
            return create_response(
                True,
                "Client Pinecone index record created successfully",
                data=created_record,
            )
        else:
            return create_response(
                False,
                "Record created but could not be retrieved",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
    except Exception as e:
        log_message("error", f"Create client_pinecone_index endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error creating client_pinecone_index record: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/client_pinecone_index")
async def get_client_pinecone_index_endpoint(client_id: Optional[str] = None):
    """
    Get client Pinecone index record(s) from the database.
    
    If client_id is provided, returns the specific record for that client.
    If client_id is not provided, returns all records.
    
    Args:
        client_id: Client identifier (optional, if not provided returns all records)
    
    Returns:
        JSON response with record(s) including _id and id fields.
    """
    try:
        from backend.services.mongodb import get_client_pinecone_index, get_database
        from bson import ObjectId
        
        if client_id:
            # Get specific record by client_id
            record = await get_client_pinecone_index(client_id)
            
            if not record:
                return create_response(
                    False,
                    f"Client Pinecone index record not found for client_id: {client_id}",
                    code=status.HTTP_404_NOT_FOUND,
                )
            
            # Add id field for convenience (serialization handled by create_response)
            if "_id" in record:
                record["id"] = str(record["_id"])
            
            return create_response(
                True,
                "Client Pinecone index record retrieved successfully",
                data=record,
            )
        else:
            # Get all records
            db = get_database()
            collection = db["client_pinecone_index"]
            
            cursor = collection.find({})
            records = await cursor.to_list(length=None)
            
            # Add id field for convenience (serialization handled by create_response)
            for record in records:
                if "_id" in record:
                    record["id"] = str(record["_id"])
            
            return create_response(
                True,
                f"Retrieved {len(records)} client Pinecone index record(s)",
                data={
                    "records": records,
                    "count": len(records)
                },
            )
        
    except Exception as e:
        log_message("error", f"Get client_pinecone_index endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error retrieving client_pinecone_index record(s): {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/client_pinecone_index/all")
async def get_all_client_pinecone_index_endpoint():
    """
    Get all client Pinecone index records from the database.
    
    This endpoint explicitly returns all records from the client_pinecone_index collection.
    Use this when you want to retrieve all client Pinecone index mappings.
    
    Returns:
        JSON response with:
            - records: List of all client Pinecone index records
            - count: Total number of records
            Each record includes:
                - _id: MongoDB ObjectId
                - id: String representation of _id (for convenience)
                - client_id: Client identifier
                - name: Client name
                - pinecone_index_name: Pinecone index name (if set)
    """
    try:
        from backend.services.mongodb import get_database
        
        log_message("info", "Retrieving all client_pinecone_index records")
        
        db = get_database()
        collection = db["client_pinecone_index"]
        
        # Get all records from the collection
        cursor = collection.find({})
        records = await cursor.to_list(length=None)
        
        # Add id field for convenience and serialize for JSON
        for record in records:
            if "_id" in record:
                record["id"] = str(record["_id"])
        
        # Serialize all records to handle datetime and ObjectId objects
        serialized_records = [serialize_for_json(record) for record in records]
        
        log_message("info", f"Retrieved {len(serialized_records)} client_pinecone_index record(s)")
        
        return create_response(
            True,
            f"Successfully retrieved all {len(serialized_records)} client Pinecone index record(s)",
            data={
                "records": serialized_records,
                "count": len(serialized_records)
            },
        )
        
    except Exception as e:
        log_message("error", f"Get all client_pinecone_index endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error retrieving all client_pinecone_index records: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.put("/client_pinecone_index/{client_id}")
async def update_client_pinecone_index(
    client_id: str,
    request: UpdateClientPineconeIndexRequest
):
    """
    Update an existing client Pinecone index record.
    
    Args:
        client_id: Client identifier (path parameter)
        request: Request body with fields to update:
            - name: Client name (optional)
            - pinecone_index_name: Pinecone index name (optional)
    
    Returns:
        JSON response with updated record including _id.
    """
    try:
        from backend.services.mongodb import get_client_pinecone_index, create_or_update_client_pinecone_index
        
        # Check if record exists
        existing = await get_client_pinecone_index(client_id)
        if not existing:
            return create_response(
                False,
                f"Client Pinecone index record not found for client_id: {client_id}",
                code=status.HTTP_404_NOT_FOUND,
            )
        
        # Use existing values if not provided in update
        updated_name = request.name if request.name else existing.get("name")
        updated_index_name = request.pinecone_index_name if request.pinecone_index_name else existing.get("pinecone_index_name")
        
        if not updated_name or not updated_index_name:
            return create_response(
                False,
                "Both name and pinecone_index_name are required",
                code=status.HTTP_400_BAD_REQUEST,
            )
        
        log_message(
            "info",
            f"Updating client_pinecone_index record: client_id={client_id}, "
            f"name={updated_name}, index_name={updated_index_name}"
        )
        
        # Update the record
        success = await create_or_update_client_pinecone_index(
            client_id=client_id,
            client_name=updated_name,
            pinecone_index_name=updated_index_name
        )
        
        if not success:
            return create_response(
                False,
                "Failed to update client_pinecone_index record",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
        # Retrieve the updated record
        updated_record = await get_client_pinecone_index(client_id)
        
        if updated_record:
            # Add id field for convenience (serialization handled by create_response)
            if "_id" in updated_record:
                updated_record["id"] = str(updated_record["_id"])
            
            return create_response(
                True,
                "Client Pinecone index record updated successfully",
                data=updated_record,
            )
        else:
            return create_response(
                False,
                "Record updated but could not be retrieved",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
    except Exception as e:
        log_message("error", f"Update client_pinecone_index endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error updating client_pinecone_index record: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.delete("/client_pinecone_index/{client_id}")
async def delete_client_pinecone_index(client_id: str):
    """
    Delete a client Pinecone index record from the database.
    
    Args:
        client_id: Client identifier (path parameter)
    
    Returns:
        JSON response with deletion status.
    """
    try:
        from backend.services.mongodb import get_database
        
        db = get_database()
        collection = db["client_pinecone_index"]
        
        # Check if record exists
        existing = await collection.find_one({"client_id": client_id})
        if not existing:
            return create_response(
                False,
                f"Client Pinecone index record not found for client_id: {client_id}",
                code=status.HTTP_404_NOT_FOUND,
            )
        
        # Delete the record
        result = await collection.delete_one({"client_id": client_id})
        
        if result.deleted_count > 0:
            log_message("info", f"Deleted client_pinecone_index record for client_id: {client_id}")
            return create_response(
                True,
                f"Client Pinecone index record deleted successfully for client_id: {client_id}",
                data={"client_id": client_id, "deleted": True},
            )
        else:
            return create_response(
                False,
                "Failed to delete client_pinecone_index record",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
    except Exception as e:
        log_message("error", f"Delete client_pinecone_index endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error deleting client_pinecone_index record: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class CreatePineconeIndexRequest(BaseModel):
    """Request model to create a Pinecone index for a client."""
    client_name: str
    client_id: Optional[str] = None


@router.post("/create_pinecone_index")
async def create_pinecone_index(request: CreatePineconeIndexRequest):
    """
    Create (or ensure) a Pinecone index for a given client.

    The client only needs to provide `client_name`. The index name will be
    automatically generated based on `client_name` using the format:
    client_name.lower().replace(" ", "-") + "-data"

    The index will be created with static parameters:
        dimension = 768
        metric    = "cosine"
        spec      = ServerlessSpec(cloud="aws", region="us-east-1")

    If `client_id` is provided, the created index name will be stored in MongoDB
    for future reference. This allows the /add_gl endpoint to automatically
    find and use the index.

    If the index already exists, the endpoint will return success with
    `created = false`.

    Args:
        request: Request body with:
            - client_name: Name of the client (required)
            - client_id: Client ID from MongoDB (optional, used to store index name)

    Returns:
        JSON response with:
            - status: "success" | "failed"
            - index_name: The generated Pinecone index name
            - created: bool (True if new index was created, False if already existed)
            - message: Human-readable description
            - mongodb_updated: bool (True if index name was stored in MongoDB)
    """
    try:
        from backend.services.pinecone import create_index_for_client

        if not request.client_name or not request.client_name.strip():
            return create_response(
                False,
                "client_name is required",
                code=status.HTTP_400_BAD_REQUEST,
            )

        log_message(
            "info",
            f"Received request to create Pinecone index for client='{request.client_name}', client_id='{request.client_id}'",
        )
        
        # Create index (or ensure it exists)
        # create_index_for_client will automatically generate index_name from client_name
        result = create_index_for_client(request.client_name, None)

        if result.get("status") == "success":
            index_name = result.get("index_name")
            
            # Store in client_pinecone_index collection if client_id is provided
            if request.client_id and index_name:
                from backend.services.mongodb import create_or_update_client_pinecone_index
                
                update_success = await create_or_update_client_pinecone_index(
                    client_id=request.client_id,
                    client_name=request.client_name,
                    pinecone_index_name=index_name
                )
                if update_success:
                    log_message(
                        "info",
                        f"Stored in client_pinecone_index collection: client_id='{request.client_id}', "
                        f"index_name='{index_name}'"
                    )
                    result["mongodb_updated"] = True
                else:
                    log_message(
                        "warning",
                        f"Failed to store in client_pinecone_index collection for client_id='{request.client_id}'"
                    )
                    result["mongodb_updated"] = False
            else:
                result["mongodb_updated"] = None  # client_id not provided
            
            return create_response(
                True,
                result.get("message", "Pinecone index operation completed successfully."),
                data=result,
            )

        return create_response(
            False,
            result.get("message", "Failed to create Pinecone index."),
            data=result,
            code=status.HTTP_400_BAD_REQUEST,
        )

    except Exception as e:
        log_message("error", f"Create Pinecone index endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error creating Pinecone index: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/pinecone_index_name")
async def get_pinecone_index_name(client_name: str | None = None):
    """
    Resolve the Pinecone index name for a given client without creating it.

    This uses the same deterministic convention as the backend:
        - lowercases the name
        - replaces spaces with '-'
        - ensures suffix '-data'

    If `client_name` is not provided, the default index name from environment
    will be returned.
    """
    try:
        from backend.services.pinecone import get_index_name_from_client

        index_name = get_index_name_from_client(client_name, log_result=False)

        return create_response(
            True,
            "Resolved Pinecone index name successfully.",
            data={"client_name": client_name, "index_name": index_name},
        )

    except Exception as e:
        log_message("error", f"Get Pinecone index name endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error resolving Pinecone index name: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/pinecone_indexes")
async def list_pinecone_indexes():
    """
    List all Pinecone indexes for the current API key.

    This endpoint allows the frontend to discover newly created indexes
    automatically, without any hard-coded mapping in the backend.
    """
    try:
        from backend.services.pinecone import list_all_indexes

        result = list_all_indexes()

        if result.get("status") == "success":
            return create_response(
                True,
                result.get("message", "Successfully retrieved Pinecone indexes."),
                data=result,
            )

        return create_response(
            False,
            result.get("error", "Failed to list Pinecone indexes."),
            data=result,
            code=status.HTTP_400_BAD_REQUEST,
        )

    except Exception as e:
        log_message("error", f"List Pinecone indexes endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error listing Pinecone indexes: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/process")
async def process_pipeline(
    request: Request,
    bank_files: List[UploadFile] = File(...),
    coa_file: UploadFile = File(...),
    check_mapping_file: UploadFile = File(None),
    client_name: str = None,
    client_id: str = None,
    user_id: str = None,
    stream: bool = Query(False, description="Enable streaming mode to receive batches as they complete"),
    llm: bool = Query(True, description="Use LLM to predict category from COA when Ask My Accountant (default: True)")
):
    """
    Process bank transactions with COA and categorize using RAG (Pinecone + AI).
    
    Bank transactions: Multiple PDF, CSV, or Excel (XLSX) files supported
    COA: PDF, CSV, or Excel
    Check Mapping: CSV or Excel file with Check Number and Payee_name columns (optional)
    
    If multiple bank statements are from the same account (same bank + account_number),
    they will be combined into a single response. If they are from different accounts,
    separate responses will be returned for each.

    Pinecone Index Resolution:
    - If client_id is provided: Looks up client_pinecone_index collection in MongoDB
      to get the stored pinecone_index_name (dynamic approach)
    - If client_id not found or not provided: Falls back to generating index name
      from client_name using convention (backward compatible)
    - If neither provided: Uses default index (may result in uncategorized transactions)

    Args:
        bank_files: List of bank transaction files - PDF, CSV, or Excel (XLSX) format (at least one required)
            Excel format expected columns: Date, Description, Amount, Cheque No (optional)
        coa_file: Chart of Accounts file (PDF, CSV, or Excel)
        check_mapping_file: CSV or Excel file with check number to payee mapping (optional)
            Required Columns: Check Number, Payee_name (or Payee Name)
            Used to lookup payee names for check transactions
            Supports: .csv, .xlsx, .xls formats
        client_name: Client name for Pinecone filtering (optional, used as fallback if client_id not found)
        client_id: Client ID for:
            - Pinecone index lookup in MongoDB (primary method)
            - Loan transaction splitting (optional)
        user_id: User ID (from user table) for storing bank extract data in MongoDB (optional)
        stream: Enable streaming mode to receive batches as they complete (default: False)
        llm: Use LLM to predict category from COA when transaction gets "Ask My Accountant" (default: True)
            If True: Uses Gemini LLM to suggest a category from COA for "Ask My Accountant" transactions
            If False: Uses current working logic (no LLM fallback)

    Returns:
        JSON response with categorized transactions:
        - If all statements are from the same account: single combined response
        - If statements are from different accounts: separate responses for each account
    """
    import time
    start_time = time.time()
    
    try:
        # Validate at least one bank file is provided
        if not bank_files or len(bank_files) == 0:
            return create_response(
                False,
                "At least one bank statement PDF file is required.",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message("info", f"Received {len(bank_files)} bank file(s), coa_file={coa_file.filename}, check_mapping_file={check_mapping_file.filename if check_mapping_file else None}, client_name={client_name}, client_id={client_id}")

        # Parse check mapping file if provided (supports CSV and Excel)
        check_mapping = {}
        if check_mapping_file:
            try:
                filename_lower = check_mapping_file.filename.lower() if check_mapping_file.filename else ""
                is_excel = filename_lower.endswith(('.xlsx', '.xls', '.xlsm'))
                is_csv = filename_lower.endswith('.csv')
                
                log_message("info", f"Parsing check mapping file: {check_mapping_file.filename} (Excel: {is_excel}, CSV: {is_csv})")
                
                check_file_content = await check_mapping_file.read()
                
                # Try Excel parsing first if it's an Excel file OR if CSV parsing might fail
                # This handles cases where .xls files might be detected as something else
                parsed_successfully = False
                
                # Parse based on file type - try Excel first if detected as Excel
                if is_excel:
                    try:
                        # Parse Excel file - use appropriate engine based on file extension
                        # .xlsx -> openpyxl, .xls -> xlrd (older format)
                        engine = None
                        if filename_lower.endswith('.xlsx'):
                            engine = 'openpyxl'
                        elif filename_lower.endswith('.xls'):
                            engine = 'xlrd'  # For older Excel format
                        
                        df = pd.read_excel(BytesIO(check_file_content), engine=engine)
                        total_rows = len(df)
                        valid_mappings = 0
                        
                        # Find check number and payee columns (case-insensitive)
                        check_col = None
                        payee_col = None
                        
                        for col in df.columns:
                            col_lower = str(col).lower()
                            if not check_col and 'check' in col_lower and 'number' in col_lower:
                                check_col = col
                            if not payee_col and 'payee' in col_lower and 'name' in col_lower:
                                payee_col = col
                        
                        if not check_col or not payee_col:
                            log_message("warning", f"Could not find Check Number or Payee Name columns in Excel. Columns found: {list(df.columns)}")
                        else:
                            # Iterate through rows
                            for _, row in df.iterrows():
                                check_num = row.get(check_col)
                                payee_name = row.get(payee_col)
                                
                                if pd.notna(check_num) and pd.notna(payee_name):
                                    check_num_str = str(check_num).strip()
                                    payee_name_str = str(payee_name).strip()
                                    
                                    if check_num_str and payee_name_str:
                                        # Normalize check number (remove leading zeros)
                                        check_num_normalized = check_num_str.lstrip('0') or '0'
                                        check_mapping[check_num_normalized] = payee_name_str
                                        valid_mappings += 1
                            
                            log_message("info", f"Parsed check mapping Excel: {total_rows} rows, {valid_mappings} valid check numbers mapped")
                            log_message("info", f"Loaded {len(check_mapping)} check number mappings from Excel")
                        
                        parsed_successfully = True
                    except Exception as excel_error:
                        log_message("error", f"Failed to parse Excel file: {excel_error}. Will try CSV parsing as fallback.")
                        parsed_successfully = False
                
                # Try CSV parsing only if:
                # 1. File is detected as CSV, OR
                # 2. Excel parsing failed (fallback)
                # But skip CSV if we successfully parsed as Excel
                if not parsed_successfully:
                    try:
                        # Parse CSV file
                        # Try different encodings
                        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
                        check_csv_text = None
                        for encoding in encodings:
                            try:
                                check_csv_text = check_file_content.decode(encoding)
                                break
                            except (UnicodeDecodeError, UnicodeError):
                                continue
                        
                        if not check_csv_text:
                            log_message("warning", "Failed to decode check mapping file. Skipping check mapping.")
                        else:
                            # Parse CSV
                            csv_reader = csv.DictReader(io.StringIO(check_csv_text))
                            total_rows = 0
                            valid_mappings = 0
                            
                            for row in csv_reader:
                                total_rows += 1
                                # Look for Check Number column (case-insensitive)
                                check_num = None
                                payee_name = None
                                
                                # Find check number column
                                for key in row.keys():
                                    if key and 'check' in key.lower() and 'number' in key.lower():
                                        check_num = row[key]
                                    elif key and 'payee' in key.lower() and 'name' in key.lower():
                                        payee_name = row[key]
                                
                                # Also try exact column names
                                if not check_num:
                                    check_num = row.get('Check Number') or row.get('check number') or row.get('Check_Number')
                                if not payee_name:
                                    payee_name = row.get('Payee_name') or row.get('payee_name') or row.get('Payee Name') or row.get('payee name')
                                
                                # Store mapping if both check number and payee name exist
                                if check_num and check_num.strip() and payee_name and payee_name.strip():
                                    # Normalize check number (remove leading zeros, whitespace)
                                    check_num_normalized = check_num.strip().lstrip('0') or '0'
                                    check_mapping[check_num_normalized] = payee_name.strip()
                                    valid_mappings += 1
                        
                            log_message("info", f"Parsed check mapping CSV: {total_rows} rows, {valid_mappings} valid check numbers mapped")
                            log_message("info", f"Loaded {len(check_mapping)} check number mappings from CSV")
                            parsed_successfully = True
                    except Exception as csv_error:
                        log_message("error", f"Failed to parse CSV file: {csv_error}. The file might be corrupted or in an unsupported format.")
                        if is_excel:
                            log_message("info", "Note: This file has an Excel extension (.xls/.xlsx) but couldn't be parsed as Excel or CSV. Please ensure the file is not corrupted.")
                        parsed_successfully = False
                
                # If neither Excel nor CSV parsing succeeded, log warning
                if not parsed_successfully:
                    log_message("warning", f"Could not parse check mapping file: {check_mapping_file.filename}. Only CSV and Excel files are supported.")
                    
            except Exception as e:
                log_message("error", f"Failed to parse check mapping file: {e}. Continuing without check mapping.")
                import traceback
                log_message("error", f"Traceback: {traceback.format_exc()}")
                check_mapping = {}

        # Resolve Pinecone index name from client_id (if provided) or fallback to client_name
        resolved_index_name = None
        if client_id:
            try:
                from backend.services.mongodb import get_client_pinecone_index
                client_index = await get_client_pinecone_index(client_id)
                if client_index and client_index.get("pinecone_index_name"):
                    resolved_index_name = client_index.get("pinecone_index_name")
                    log_message("info", f"Resolved Pinecone index from MongoDB: client_id={client_id}, index_name={resolved_index_name}")
                else:
                    log_message("warning", f"Client ID '{client_id}' not found in client_pinecone_index collection. Falling back to client_name-based index resolution.")
            except Exception as e:
                log_message("error", f"Error looking up client_pinecone_index for client_id={client_id}: {e}. Falling back to client_name-based index resolution.")
        
        # Fallback: Generate index name from client_name if not resolved from MongoDB
        if not resolved_index_name and client_name:
            from backend.services.pinecone import get_index_name_from_client
            resolved_index_name = get_index_name_from_client(client_name, log_result=True)
            log_message("info", f"Generated Pinecone index name from client_name: {client_name} -> {resolved_index_name}")
        
        # If still no index name, log warning but continue (will use default index)
        if not resolved_index_name:
            log_message("warning", "No Pinecone index name resolved. Will use default index or may result in uncategorized transactions.")

        # Validate that GL data exists in Pinecone index before processing
        if resolved_index_name:
            from backend.services.pinecone import check_index_has_data
            index_check = check_index_has_data(resolved_index_name)
            
            if index_check.get("status") == "failed":
                return create_response(
                    False,
                    f"Failed to check Pinecone index '{resolved_index_name}': {index_check.get('error', 'Unknown error')}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            if not index_check.get("has_data", False):
                # Index exists but has no data, or index doesn't exist
                if not index_check.get("index_exists", False):
                    return create_response(
                        False,
                        f"Pinecone index '{resolved_index_name}' does not exist. Please upload GL data first using the /api/add_gl endpoint.",
                        {
                            "index_name": resolved_index_name,
                            "client_id": client_id,
                            "client_name": client_name
                        },
                        code=status.HTTP_400_BAD_REQUEST
                    )
                else:
                    return create_response(
                        False,
                        f"Pinecone index '{resolved_index_name}' exists but has no GL data. Please upload GL data first using the /api/add_gl endpoint.",
                        {
                            "index_name": resolved_index_name,
                            "total_vectors": index_check.get("total_vectors", 0),
                            "client_id": client_id,
                            "client_name": client_name
                        },
                        code=status.HTTP_400_BAD_REQUEST
                    )
            
            # GL data exists, log the count
            log_message(
                "info",
                f"Verified GL data exists in Pinecone index '{resolved_index_name}': {index_check.get('total_vectors', 0)} vectors found."
            )
        else:
            # No index name resolved - check default index
            from backend.services.pinecone import check_index_has_data, get_index_name_from_client
            default_index_name = get_index_name_from_client(None)  # None returns default INDEX_NAME
            index_check = check_index_has_data(default_index_name)
            
            if not index_check.get("has_data", False):
                return create_response(
                    False,
                    f"No GL data found in default Pinecone index '{default_index_name}'. Please upload GL data first using the /api/add_gl endpoint, or provide client_id/client_name to use a client-specific index.",
                    {
                        "default_index": default_index_name,
                        "client_id": client_id,
                        "client_name": client_name
                    },
                    code=status.HTTP_400_BAD_REQUEST
                )
            
            log_message(
                "info",
                f"Using default Pinecone index '{default_index_name}' with {index_check.get('total_vectors', 0)} vectors."
            )

        # Validate all bank files are PDFs, CSVs, or Excel files
        for idx, bank_file in enumerate(bank_files):
            bank_file_type = detect_file_type(bank_file.filename)
            if bank_file_type not in ['pdf', 'csv', 'excel']:
                return create_response(
                    False,
                    f"Bank file #{idx + 1} ({bank_file.filename}) must be a PDF, CSV, or Excel document.",
                    code=status.HTTP_400_BAD_REQUEST
                )

        # Validate COA file type
        coa_file_type = detect_file_type(coa_file.filename)
        if coa_file_type not in ['pdf', 'csv', 'excel']:
            return create_response(
                False,
                "COA file must be a PDF, CSV, or Excel document.",
                code=status.HTTP_400_BAD_REQUEST
            )

        # Parse COA file once (shared across all bank statements)
        coa_result = None
        if coa_file_type == 'pdf':
            coa_result = await transaction_interactors.parse_coa_pdf(coa_file)
        elif coa_file_type == 'csv':
            coa_result = await transaction_interactors.parse_coa_csv(coa_file)
        elif coa_file_type == 'excel':
            coa_result = await transaction_interactors.parse_coa_excel(coa_file)
        
        if coa_result.get("status") == "failed":
            return create_response(
                False,
                coa_result.get("error", "Unable to process COA file. Please try again."),
                code=status.HTTP_400_BAD_REQUEST
            )
        
        coa_list = coa_result.get("coa_list", [])
        
        # Extract unique categories from COA
        try:
            coa_categories = transaction_interactors.extract_coa_categories(coa_list)
            log_message("info", f"Extracted {len(coa_categories)} unique categories from COA: {coa_categories[:10]}...")
        except Exception as e:
            log_message("error", f"COA category extraction failed: {e}")
            coa_categories = []

        # Process each bank file separately
        bank_statements = []  # List of (metadata, transactions, filename)
        
        for bank_file in bank_files:
            log_message("info", f"Processing bank file: {bank_file.filename}")
            
            # Detect file type (PDF or CSV)
            bank_file_type = detect_file_type(bank_file.filename)
            
            # Parse bank file based on type
            if bank_file_type == 'pdf':
                bank_result = await transaction_interactors.parse_bank_pdf(bank_file)
            elif bank_file_type == 'csv':
                bank_result = await transaction_interactors.parse_bank_csv(bank_file)
            elif bank_file_type == 'excel':
                bank_result = await transaction_interactors.parse_bank_excel(bank_file)
            else:
                return create_response(
                    False,
                    f"Unsupported bank file type for '{bank_file.filename}'. Only PDF, CSV, and Excel are supported.",
                    code=status.HTTP_400_BAD_REQUEST
                )
            
            if bank_result.get("status") == "failed":
                log_message("error", f"Failed to parse bank file {bank_file.filename}: {bank_result.get('error')}")
                return create_response(
                    False,
                    f"Failed to process bank file '{bank_file.filename}': {bank_result.get('error')}",
                    code=status.HTTP_400_BAD_REQUEST
                )
            
            metadata = bank_result.get("metadata", {})
            transactions = bank_result.get("transactions", [])
            
            if not transactions:
                log_message("warning", f"No transactions found in bank file {bank_file.filename}")
                continue
            
            bank_statements.append({
                "filename": bank_file.filename,
                "metadata": metadata,
                "transactions": transactions
            })
            
            log_message("info", f"Extracted {len(transactions)} transactions from {bank_file.filename} (Bank: {metadata.get('bank')}, Account: {metadata.get('account_number')})")
        
        if not bank_statements:
            return create_response(
                False,
                "No transactions found in any of the bank statements. Please check the files and try again.",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        # Group bank statements by bank + account_number
        # Key: (bank, account_number), Value: list of statements
        statements_by_account = defaultdict(list)
        
        for stmt in bank_statements:
            bank = stmt["metadata"].get("bank", "Unknown")
            account_number = stmt["metadata"].get("account_number") or "NA"
            key = (bank, account_number)
            statements_by_account[key].append(stmt)
        
        log_message("info", f"Grouped {len(bank_statements)} statement(s) into {len(statements_by_account)} account group(s)")
        for (bank, account), stmts in statements_by_account.items():
            log_message("info", f"  Account group: Bank={bank}, Account={account}, Statements={len(stmts)}")
        
        # If streaming mode, handle all account groups in streaming fashion
        if stream:
            # Streaming mode: process batches incrementally and yield results
            async def stream_all_account_groups():
                """Generator function to stream batches via SSE for all account groups"""
                try:
                    # Send initial message with COA categories
                    yield f"data: {json.dumps({'type': 'start', 'message': 'Processing started', 'total_account_groups': len(statements_by_account), 'coa_categories': coa_categories})}\n\n"
                    
                    account_group_results = {}
                    
                    for (bank, account_number), statements in statements_by_account.items():
                        account_key = f"{bank}_{account_number}" if account_number != "NA" else f"{bank}_NA"
                        
                        # Send account group start message
                        yield f"data: {json.dumps({'type': 'account_group_start', 'account_key': account_key, 'bank': bank, 'account_number': account_number})}\n\n"
                        
                        # Combine all transactions from statements in this group
                        all_raw_transactions = []
                        for stmt in statements:
                            all_raw_transactions.extend(stmt["transactions"])
                        
                        if not all_raw_transactions:
                            log_message("warning", f"No transactions to process for account group {account_key}")
                            continue
                        
                        log_message("info", f"Processing {len(all_raw_transactions)} transactions for account group: {account_key}")
                        
                        # Standardize transactions
                        standardized_transactions = transaction_interactors.standardize_bank_transactions(all_raw_transactions)
                        
                        if not standardized_transactions:
                            log_message("warning", f"No valid transactions after standardization for account group {account_key}")
                            continue
                        
                        # Apply check number mapping if available (same logic as non-streaming)
                        if check_mapping:
                            log_message("info", f"Applying check number mapping to {len(standardized_transactions)} transactions")
                            matched_count = 0
                            for idx, txn in enumerate(standardized_transactions):
                                original_memo = txn.get("Memo", "") or txn.get("memo", "")
                                check_num = None
                                if idx < len(all_raw_transactions):
                                    raw_txn = all_raw_transactions[idx]
                                    check_num = raw_txn.get("check_number") or raw_txn.get("check_num") or raw_txn.get("checkNumber")
                                if not check_num:
                                    check_num = txn.get("check_number") or txn.get("check_num") or txn.get("checkNumber")
                                if not check_num and original_memo:
                                    check_patterns = [
                                        r'check\s*#?\s*(\d+)',
                                        r'chk\s*#?\s*(\d+)',
                                        r'ck\s*#?\s*(\d+)'
                                    ]
                                    for pattern in check_patterns:
                                        match = re.search(pattern, original_memo, re.IGNORECASE)
                                        if match:
                                            check_num = match.group(1)
                                            break
                                if check_num:
                                    check_num_normalized = str(check_num).strip().lstrip('0') or '0'
                                    if check_num_normalized in check_mapping:
                                        payee_name = check_mapping[check_num_normalized]
                                        txn["payee"] = payee_name
                                        txn["Payee"] = payee_name
                                        matched_count += 1
                            if matched_count > 0:
                                log_message("info", f"Check mapping summary: {matched_count} check(s) matched with payee names from CSV")
                        
                        # Mark check transactions
                        check_transaction_count = 0
                        for idx, txn in enumerate(standardized_transactions):
                            original_memo = txn.get("Memo", "") or txn.get("memo", "")
                            check_num = None
                            if idx < len(all_raw_transactions):
                                raw_txn = all_raw_transactions[idx]
                                check_num = raw_txn.get("check_number") or raw_txn.get("check_num") or raw_txn.get("checkNumber")
                            if not check_num:
                                check_num = txn.get("check_number") or txn.get("check_num") or txn.get("checkNumber")
                            if not check_num and original_memo:
                                check_patterns = [
                                    r'check\s*#?\s*(\d+)',
                                    r'chk\s*#?\s*(\d+)',
                                    r'ck\s*#?\s*(\d+)'
                                ]
                                for pattern in check_patterns:
                                    match = re.search(pattern, original_memo, re.IGNORECASE)
                                    if match:
                                        check_num = match.group(1)
                                        break
                            if check_num:
                                txn["is_check_transaction"] = True
                                check_transaction_count += 1
                        
                        if check_transaction_count > 0:
                            log_message("info", f"Identified {check_transaction_count} check transaction(s) - will be categorized as 'Ask My Accountant'")
                        
                        # Group standardized transactions by account_number
                        transactions_by_subaccount = defaultdict(list)
                        account_number_map = {}
                        for idx, raw_txn in enumerate(all_raw_transactions):
                            txn_account_number = raw_txn.get("account_number")
                            if not txn_account_number:
                                for stmt in statements:
                                    txn_account_number = stmt["metadata"].get("account_number")
                                    if txn_account_number:
                                        break
                            txn_account_number = txn_account_number or account_number or "NA"
                            account_number_map[idx] = txn_account_number
                        
                        for idx, txn in enumerate(standardized_transactions):
                            txn_account_number = account_number_map.get(idx, account_number or "NA")
                            transactions_by_subaccount[txn_account_number].append((idx, txn))
                        
                        # Process each sub-account separately
                        subaccount_results = {}
                        all_csv_lines = []
                        header_written = False
                        csv_header = "Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber"
                        
                        for subaccount_number, subaccount_transactions_with_idx in transactions_by_subaccount.items():
                            subaccount_transactions = [txn for _, txn in subaccount_transactions_with_idx]
                            
                            working_list_result = transaction_interactors.create_working_list_for_pinecone(
                                subaccount_transactions
                            )
                            
                            # Define callback to store batches in MongoDB
                            async def batch_callback(batch_num, total_batches, batch_csv, batch_transactions):
                                if user_id:
                                    try:
                                        from backend.services.mongodb import insert_client_bank_extract_batch
                                        await insert_client_bank_extract_batch(
                                            user_id=user_id,
                                            client_name=client_name or "Unknown",
                                            batch_num=batch_num,
                                            total_batches=total_batches,
                                            batch_csv=batch_csv,
                                            batch_transactions=batch_transactions,
                                            account_key=account_key
                                        )
                                    except Exception as e:
                                        log_message("error", f"Failed to store batch in MongoDB: {e}")
                            
                            # Process batches with streaming
                            async for batch_num, total_batches, batch_csv, batch_transactions in transaction_interactors.process_all_batches_with_pinecone_streaming(
                                subaccount_transactions,
                                working_list_result,
                                client_name=client_name,
                                client_id=client_id,
                                account_number=subaccount_number if subaccount_number != "NA" else "",
                                index_name=resolved_index_name,
                                user_id=user_id,
                                stream_callback=batch_callback,
                                llm=llm,
                                coa_categories=coa_categories
                            ):
                                # Send batch via SSE (include COA categories in first batch for each account group)
                                batch_data = {
                                    'type': 'batch',
                                    'batch_num': batch_num,
                                    'total_batches': total_batches,
                                    'account_key': account_key,
                                    'subaccount_number': subaccount_number if subaccount_number != "NA" else None,
                                    'batch_csv': batch_csv,
                                    'batch_transactions': batch_transactions,
                                    'transaction_count': len(batch_transactions)
                                }
                                # Include COA categories in the first batch of the first account group
                                if batch_num == 1:
                                    batch_data['coa_categories'] = coa_categories
                                yield f"data: {json.dumps(batch_data)}\n\n"
                                
                                # Also accumulate for final response
                                csv_lines = batch_csv.strip().split('\n')
                                if not header_written and csv_lines:
                                    all_csv_lines.append(csv_lines[0])  # Header
                                    header_written = True
                                    for line in csv_lines[1:]:
                                        if line.strip():
                                            all_csv_lines.append(line)
                                else:
                                    for line in csv_lines[1:]:
                                        if line.strip():
                                            all_csv_lines.append(line)
                            
                            # Store subaccount result
                            subaccount_csv = "\n".join([csv_header] + [line for line in all_csv_lines if line != csv_header and line.strip()])
                            subaccount_results[subaccount_number] = subaccount_csv if subaccount_csv else None
                        
                        # Combine all sub-account CSVs
                        combined_csv = "\n".join(all_csv_lines) if all_csv_lines else ""
                        
                        account_group_results[account_key] = {
                            "bank": bank,
                            "account_number": account_number if account_number != "NA" else None,
                            "statements": [stmt["filename"] for stmt in statements],
                            "transaction_count": len(all_raw_transactions),
                            "csv_content": combined_csv,
                            "subaccounts": {
                                subaccount_num: {
                                    "transaction_count": len(txns),
                                    "csv_content": subaccount_results.get(subaccount_num, "")
                                }
                                for subaccount_num, txns in transactions_by_subaccount.items()
                            }
                        }
                        
                        # Send account group completion message
                        yield f"data: {json.dumps({'type': 'account_group_complete', 'account_key': account_key})}\n\n"
                    
                    # Restructure response to group by bank, then by account
                    restructured_data = restructure_response_by_bank_and_account(account_group_results)
                    restructured_data["data"]["coa_categories"] = coa_categories
                    
                    # Store final data in MongoDB if user_id is provided
                    stored_record_id = None
                    if user_id:
                        try:
                            from backend.services.mongodb import insert_client_bank_extract_data
                            stored_record_id = await insert_client_bank_extract_data(
                                user_id=user_id,
                                client_name=client_name or "Unknown",
                                bank_extract_data=restructured_data["data"]
                            )
                        except Exception as e:
                            log_message("error", f"Error storing final bank extract data: {e}")
                    
                    # Add stored_record_id to response if available (same as non-streaming)
                    response_data = restructured_data["data"].copy()
                    if stored_record_id:
                        response_data["stored_record_id"] = stored_record_id
                    
                    # Log analytics event for successful bank statement processing (streaming mode)
                    if user_id:
                        try:
                            duration_ms = int((time.time() - start_time) * 1000)
                            
                            # Calculate total file sizes
                            total_bank_file_size = sum(bank_file.size for bank_file in bank_files if hasattr(bank_file, 'size'))
                            coa_file_size = coa_file.size if hasattr(coa_file, 'size') else 0
                            check_mapping_file_size = check_mapping_file.size if check_mapping_file and hasattr(check_mapping_file, 'size') else 0
                            
                            # Get transaction count from response
                            total_transactions = 0
                            if "banks" in response_data:
                                for bank in response_data["banks"]:
                                    if "accounts" in bank:
                                        for account in bank["accounts"]:
                                            if "transactions" in account:
                                                total_transactions += len(account["transactions"])
                            
                            await log_event(
                                user_id_or_client_id=user_id,
                                event_type='tool_usage',
                                category='Bank Reconciliation',
                                action='file_processed',
                                metadata={
                                    'toolName': 'Bank Reconciliation',
                                    'toolMode': 'process',
                                    'bankFileCount': len(bank_files),
                                    'bankFileNames': [f.filename for f in bank_files],
                                    'coaFileName': coa_file.filename,
                                    'checkMappingFileName': check_mapping_file.filename if check_mapping_file else None,
                                    'totalBankFileSize': total_bank_file_size,
                                    'coaFileSize': coa_file_size,
                                    'checkMappingFileSize': check_mapping_file_size,
                                    'totalFileSize': total_bank_file_size + coa_file_size + check_mapping_file_size,
                                    'clientName': client_name,
                                    'clientId': client_id,
                                    'accountGroupsCount': len(account_group_results),
                                    'totalTransactions': total_transactions,
                                    'streamMode': stream,
                                    'llmEnabled': llm
                                },
                                request=request,
                                options={
                                    'duration': duration_ms,
                                    'status': 'success'
                                }
                            )
                            log_message('info', f"Analytics event logged: Bank Reconciliation/file_processed for user_id={user_id}")
                        except Exception as analytics_error:
                            log_message('error', f"Error logging bank statement processing analytics (streaming): {analytics_error}")
                            import traceback
                            log_message('error', f"Analytics traceback: {traceback.format_exc()}")
                            # Don't fail the main request if analytics logging fails
                    
                    # Send final completion message with SAME format as non-streaming response
                    # This matches the create_response() format exactly
                    completion_data = {
                        "succeeded": True,
                        "message": f"Successfully processed transactions from {len(account_group_results)} account group(s)",
                        "data": serialize_for_json(response_data),
                        "status_code": 200
                    }
                    yield f"data: {json.dumps(completion_data)}\n\n"
                    
                except Exception as e:
                    # Log analytics event for failed bank statement processing (streaming mode)
                    if user_id:
                        try:
                            duration_ms = int((time.time() - start_time) * 1000)
                            
                            # Calculate file sizes even on failure
                            total_bank_file_size = sum(bank_file.size for bank_file in bank_files if hasattr(bank_file, 'size'))
                            coa_file_size = coa_file.size if hasattr(coa_file, 'size') else 0
                            check_mapping_file_size = check_mapping_file.size if check_mapping_file and hasattr(check_mapping_file, 'size') else 0
                            
                            await log_event(
                                user_id_or_client_id=user_id,
                                event_type='tool_usage',
                                category='Bank Reconciliation',
                                action='file_processed',
                                metadata={
                                    'toolName': 'Bank Reconciliation',
                                    'toolMode': 'process',
                                    'bankFileCount': len(bank_files),
                                    'bankFileNames': [f.filename for f in bank_files],
                                    'coaFileName': coa_file.filename,
                                    'checkMappingFileName': check_mapping_file.filename if check_mapping_file else None,
                                    'totalBankFileSize': total_bank_file_size,
                                    'coaFileSize': coa_file_size,
                                    'checkMappingFileSize': check_mapping_file_size,
                                    'totalFileSize': total_bank_file_size + coa_file_size + check_mapping_file_size,
                                    'clientName': client_name,
                                    'clientId': client_id,
                                    'error': str(e),
                                    'streamMode': stream,
                                    'llmEnabled': llm
                                },
                                request=request,
                                options={
                                    'duration': duration_ms,
                                    'status': 'error',
                                    'errorMessage': str(e)
                                }
                            )
                            log_message('info', f"Analytics error event logged for user_id={user_id}")
                        except Exception as analytics_error:
                            log_message('error', f"Error logging bank statement processing error analytics (streaming): {analytics_error}")
                            # Don't fail the main request if analytics logging fails
                    
                    error_data = {
                        'type': 'error',
                        'error': str(e),
                        'message': 'Error processing transactions'
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    log_message("error", f"Streaming error: {e}")
                    import traceback
                    log_message("error", f"Traceback: {traceback.format_exc()}")
            
            # Return streaming response
            return StreamingResponse(
                stream_all_account_groups(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable buffering in nginx
                }
            )
        
        # Non-streaming mode (existing behavior - backward compatible)
        # Process each account group
        account_group_results = {}
        
        for (bank, account_number), statements in statements_by_account.items():
            account_key = f"{bank}_{account_number}" if account_number != "NA" else f"{bank}_NA"
            
            # Combine all transactions from statements in this group
            all_raw_transactions = []
            for stmt in statements:
                all_raw_transactions.extend(stmt["transactions"])
            
            if not all_raw_transactions:
                log_message("warning", f"No transactions to process for account group {account_key}")
                continue
            
            log_message("info", f"Processing {len(all_raw_transactions)} transactions for account group: {account_key}")
            
            # Standardize transactions
            standardized_transactions = transaction_interactors.standardize_bank_transactions(all_raw_transactions)
            
            if not standardized_transactions:
                log_message("warning", f"No valid transactions after standardization for account group {account_key}")
                continue
            
            # Apply check number mapping if available
            if check_mapping:
                log_message("info", f"Applying check number mapping to {len(standardized_transactions)} transactions")
                matched_count = 0
                for idx, txn in enumerate(standardized_transactions):
                    # Get memo from standardized transaction (field name is "Memo" with capital M)
                    original_memo = txn.get("Memo", "") or txn.get("memo", "")
                    
                    # Check if raw transaction has a check number field
                    check_num = None
                    if idx < len(all_raw_transactions):
                        raw_txn = all_raw_transactions[idx]
                        check_num = raw_txn.get("check_number") or raw_txn.get("check_num") or raw_txn.get("checkNumber")
                    
                    # Also check standardized transaction
                    if not check_num:
                        check_num = txn.get("check_number") or txn.get("check_num") or txn.get("checkNumber")
                    
                    # Extract check number from memo/description if not found
                    if not check_num and original_memo:
                        # Look for patterns like "Check #1234", "Chk 1234", "Check 1234", "Check 3343"
                        check_patterns = [
                            r'check\s*#?\s*(\d+)',
                            r'chk\s*#?\s*(\d+)',
                            r'ck\s*#?\s*(\d+)'
                        ]
                        for pattern in check_patterns:
                            match = re.search(pattern, original_memo, re.IGNORECASE)
                            if match:
                                check_num = match.group(1)
                                break
                    
                    if check_num:
                        # Normalize check number (remove leading zeros)
                        check_num_normalized = str(check_num).strip().lstrip('0') or '0'
                        
                        # Lookup payee name from check mapping
                        if check_num_normalized in check_mapping:
                            payee_name = check_mapping[check_num_normalized]
                            # Set both lowercase and uppercase versions to ensure it's preserved
                            txn["payee"] = payee_name
                            txn["Payee"] = payee_name
                            matched_count += 1
                            # Only log when check number is FOUND in mapping
                            log_message("info", f"CHECK #{check_num} (normalized: {check_num_normalized}) -> PAYEE: '{payee_name}' | Memo: {original_memo[:60]}...")
                
                # Only show summary if matches were found
                if matched_count > 0:
                    log_message("info", f"Check mapping summary: {matched_count} check(s) matched with payee names from CSV")
            
            # Mark check transactions for "Ask My Accountant" category
            # Detect check transactions (with or without check mapping file)
            check_transaction_count = 0
            for idx, txn in enumerate(standardized_transactions):
                original_memo = txn.get("Memo", "") or txn.get("memo", "")
                
                # Check if raw transaction has a check number field
                check_num = None
                if idx < len(all_raw_transactions):
                    raw_txn = all_raw_transactions[idx]
                    check_num = raw_txn.get("check_number") or raw_txn.get("check_num") or raw_txn.get("checkNumber")
                
                # Also check standardized transaction
                if not check_num:
                    check_num = txn.get("check_number") or txn.get("check_num") or txn.get("checkNumber")
                
                # Extract check number from memo/description if not found
                if not check_num and original_memo:
                    # Look for patterns like "Check #1234", "Chk 1234", "Check 1234", "Check 3343"
                    check_patterns = [
                        r'check\s*#?\s*(\d+)',
                        r'chk\s*#?\s*(\d+)',
                        r'ck\s*#?\s*(\d+)'
                    ]
                    for pattern in check_patterns:
                        match = re.search(pattern, original_memo, re.IGNORECASE)
                        if match:
                            check_num = match.group(1)
                            break
                
                # If check number found, mark this transaction as a check transaction
                if check_num:
                    txn["is_check_transaction"] = True
                    check_transaction_count += 1
            
            if check_transaction_count > 0:
                log_message("info", f"Identified {check_transaction_count} check transaction(s) - will be categorized as 'Ask My Accountant'")
            
            # Group standardized transactions by account_number (in case there are sub-accounts)
            transactions_by_subaccount = defaultdict(list)
            account_number_map = {}
            
            # Create mapping from raw transactions to account_number
            for idx, raw_txn in enumerate(all_raw_transactions):
                txn_account_number = raw_txn.get("account_number")
                if not txn_account_number:
                    # Try to get from metadata of the statement this transaction came from
                    for stmt in statements:
                        txn_account_number = stmt["metadata"].get("account_number")
                        if txn_account_number:
                            break
                txn_account_number = txn_account_number or account_number or "NA"
                account_number_map[idx] = txn_account_number
            
            # Group standardized transactions by account_number
            for idx, txn in enumerate(standardized_transactions):
                txn_account_number = account_number_map.get(idx, account_number or "NA")
                transactions_by_subaccount[txn_account_number].append((idx, txn))
            
            # Process each sub-account separately
            subaccount_results = {}
            all_csv_lines = []
            header_written = False
            
            for subaccount_number, subaccount_transactions_with_idx in transactions_by_subaccount.items():
                subaccount_transactions = [txn for _, txn in subaccount_transactions_with_idx]
                
                working_list_result = transaction_interactors.create_working_list_for_pinecone(
                    subaccount_transactions
                )

                csv_result = await transaction_interactors.process_all_batches_with_pinecone(
                    subaccount_transactions,
                    working_list_result,
                    client_name=client_name,
                    client_id=client_id,
                    account_number=subaccount_number if subaccount_number != "NA" else "",
                    index_name=resolved_index_name,
                    llm=llm,
                    coa_categories=coa_categories
                )

                if csv_result and len(csv_result.strip()) >= 20:
                    csv_lines = csv_result.strip().split('\n')
                    
                    if not header_written and csv_lines:
                        all_csv_lines.append(csv_lines[0])  # Header
                        header_written = True
                        for line in csv_lines[1:]:
                            if line.strip():
                                all_csv_lines.append(line)
                    else:
                        for line in csv_lines[1:]:
                            if line.strip():
                                all_csv_lines.append(line)
                    
                    subaccount_results[subaccount_number] = csv_result
                else:
                    log_message("warning", f"No valid CSV generated for sub-account {subaccount_number}")
                    subaccount_results[subaccount_number] = None
            
            # Combine all sub-account CSVs
            combined_csv = "\n".join(all_csv_lines) if all_csv_lines else ""
            
            account_group_results[account_key] = {
                "bank": bank,
                "account_number": account_number if account_number != "NA" else None,
                "statements": [stmt["filename"] for stmt in statements],
                "transaction_count": len(all_raw_transactions),
                "csv_content": combined_csv,
                "subaccounts": {
                    subaccount_num: {
                        "transaction_count": len(txns),
                        "csv_content": subaccount_results.get(subaccount_num, "")
                    }
                    for subaccount_num, txns in transactions_by_subaccount.items()
                }
            }
        
        # Restructure response to group by bank, then by account
        # This handles merging same bank/account from multiple PDFs and deduplication
        restructured_data = restructure_response_by_bank_and_account(account_group_results)
        
        # Add COA categories to the response
        restructured_data["data"]["coa_categories"] = coa_categories
        
        # Store the processed data in MongoDB if user_id is provided
        stored_record_id = None
        if user_id:
            try:
                from backend.services.mongodb import insert_client_bank_extract_data
                stored_record_id = await insert_client_bank_extract_data(
                    user_id=user_id,
                    client_name=client_name or "Unknown",
                    bank_extract_data=restructured_data["data"]
                )
                if stored_record_id:
                    log_message("info", f"Stored bank extract data in MongoDB: user_id={user_id}, record_id={stored_record_id}")
                else:
                    log_message("warning", f"Failed to store bank extract data in MongoDB for user_id={user_id}")
            except Exception as e:
                log_message("error", f"Error storing bank extract data in MongoDB: {e}")
                # Continue with response even if storage fails
                import traceback
                log_message("error", f"Traceback: {traceback.format_exc()}")
        
        # Add stored_record_id to response if available
        response_data = restructured_data["data"].copy()
        if stored_record_id:
            response_data["stored_record_id"] = stored_record_id
        
        # Log analytics event for successful bank statement processing
        if user_id:
            try:
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Calculate total file sizes
                total_bank_file_size = sum(bank_file.size for bank_file in bank_files if hasattr(bank_file, 'size'))
                coa_file_size = coa_file.size if hasattr(coa_file, 'size') else 0
                check_mapping_file_size = check_mapping_file.size if check_mapping_file and hasattr(check_mapping_file, 'size') else 0
                
                # Get transaction count from response
                total_transactions = 0
                if "banks" in response_data:
                    for bank in response_data["banks"]:
                        if "accounts" in bank:
                            for account in bank["accounts"]:
                                if "transactions" in account:
                                    total_transactions += len(account["transactions"])
                
                await log_event(
                    user_id_or_client_id=user_id,
                    event_type='tool_usage',
                    category='Bank Reconciliation',
                    action='file_processed',
                    metadata={
                        'toolName': 'Bank Reconciliation',
                        'toolMode': 'process',
                        'bankFileCount': len(bank_files),
                        'bankFileNames': [f.filename for f in bank_files],
                        'coaFileName': coa_file.filename,
                        'checkMappingFileName': check_mapping_file.filename if check_mapping_file else None,
                        'totalBankFileSize': total_bank_file_size,
                        'coaFileSize': coa_file_size,
                        'checkMappingFileSize': check_mapping_file_size,
                        'totalFileSize': total_bank_file_size + coa_file_size + check_mapping_file_size,
                        'clientName': client_name,
                        'clientId': client_id,
                        'accountGroupsCount': len(account_group_results),
                        'totalTransactions': total_transactions,
                        'streamMode': stream,
                        'llmEnabled': llm
                    },
                    request=request,
                    options={
                        'duration': duration_ms,
                        'status': 'success'
                    }
                )
                log_message('info', f"Analytics event logged: Bank Reconciliation/file_processed for user_id={user_id}")
            except Exception as analytics_error:
                log_message('error', f"Error logging bank statement processing analytics: {analytics_error}")
                import traceback
                log_message('error', f"Analytics traceback: {traceback.format_exc()}")
                # Don't fail the main request if analytics logging fails
        
        return create_response(
            True,
            f"Successfully processed transactions from {len(account_group_results)} account group(s)",
            response_data
        )

    except Exception as e:
        log_message("critical", f"Pipeline crashed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        
        # Log analytics event for failed bank statement processing
        if user_id:
            try:
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Calculate file sizes even on failure
                total_bank_file_size = sum(bank_file.size for bank_file in bank_files if hasattr(bank_file, 'size'))
                coa_file_size = coa_file.size if hasattr(coa_file, 'size') else 0
                check_mapping_file_size = check_mapping_file.size if check_mapping_file and hasattr(check_mapping_file, 'size') else 0
                
                await log_event(
                    user_id_or_client_id=user_id,
                    event_type='tool_usage',
                    category='Bank Reconciliation',
                    action='file_processed',
                    metadata={
                        'toolName': 'Bank Reconciliation',
                        'toolMode': 'process',
                        'bankFileCount': len(bank_files),
                        'bankFileNames': [f.filename for f in bank_files],
                        'coaFileName': coa_file.filename,
                        'checkMappingFileName': check_mapping_file.filename if check_mapping_file else None,
                        'totalBankFileSize': total_bank_file_size,
                        'coaFileSize': coa_file_size,
                        'checkMappingFileSize': check_mapping_file_size,
                        'totalFileSize': total_bank_file_size + coa_file_size + check_mapping_file_size,
                        'clientName': client_name,
                        'clientId': client_id,
                        'error': str(e),
                        'streamMode': stream,
                        'llmEnabled': llm
                    },
                    request=request,
                    options={
                        'duration': duration_ms,
                        'status': 'error',
                        'errorMessage': str(e)
                    }
                )
                log_message('info', f"Analytics error event logged for user_id={user_id}")
            except Exception as analytics_error:
                log_message('error', f"Error logging bank statement processing error analytics: {analytics_error}")
                import traceback
                log_message('error', f"Analytics traceback: {traceback.format_exc()}")
                # Don't fail the main request if analytics logging fails
        
        return create_response(
            False, 
            "Processing failed. Please try again or contact support if the issue persists.", 
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/extract_loan_details")
async def extract_loan_details(
    request: Request,
    file: UploadFile = File(...),
    client_id: str = Form(...),
    user_id: str = Form(None)
):
    """
    Extract loan details from a PDF document and store in MongoDB.
    
    Extracts the following loan information:
    - Legal Name
    - Funding Provided
    - Annual Percentage Rate (APR)
    - Finance Charge
    - Total Payment Amount
    - Payment
    - Account No.
    - Start Date
    
    Args:
        file: PDF file containing loan document.
        client_id: Client identifier (required for MongoDB storage).
        user_id: Optional user identifier for analytics tracking.
    
    Returns:
        JSON response with extracted loan details and MongoDB record ID.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate client_id
        if not client_id:
            return create_response(
                False,
                "client_id is required",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return create_response(
                False,
                "File must be a PDF document",
                {"file_type": detect_file_type(file.filename)},
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message("info", f"Received loan extraction request for file={file.filename}, client_id={client_id}")
        
        # Validate client_id exists (foreign key check)
        from backend.services.mongodb import validate_client_exists
        client_exists = await validate_client_exists(client_id)
        if not client_exists:
            return create_response(
                False,
                f"Client with ID '{client_id}' does not exist. Please create the client first.",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        # Extract loan details
        result = await transaction_interactors.extract_loan_details(file, client_id)
        
        if result.get("status") == "failed":
            return create_response(
                False,
                result.get("error", "Failed to extract loan details"),
                code=status.HTTP_400_BAD_REQUEST
            )
        
        # Log analytics event for successful loan extraction
        # Use user_id if available, otherwise use client_id
        analytics_id = user_id if user_id else client_id
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            file_size = file.size if hasattr(file, 'size') else 0
            
            # Extract loan data for metadata
            loan_data = result.get("data", {})
            loan_details = loan_data.get("loan_details", {})
            
            await log_event(
                user_id_or_client_id=analytics_id,
                event_type='tool_usage',
                category='Loan Extraction',
                action='loan_extracted',
                metadata={
                    'toolName': 'Loan Extraction',
                    'toolMode': 'extract',
                    'fileName': file.filename,
                    'fileSize': file_size,
                    'fileSizeKB': round(file_size / 1024, 2) if file_size else 0,
                    'clientId': client_id,
                    'legalName': loan_details.get('legal_name'),
                    'fundingProvided': loan_details.get('funding_provided'),
                    'apr': loan_details.get('apr'),
                    'accountNo': loan_details.get('account_no'),
                    'recordId': loan_data.get('record_id')
                },
                request=request,
                options={
                    'duration': duration_ms,
                    'status': 'success',
                    'clientId': client_id  # Pass client_id in options
                }
            )
            log_message('info', f"Analytics event logged: Loan Extraction/loan_extracted for {'user_id' if user_id else 'client_id'}={analytics_id}")
        except Exception as analytics_error:
            log_message('error', f"Error logging loan extraction analytics: {analytics_error}")
            import traceback
            log_message('error', f"Analytics traceback: {traceback.format_exc()}")
            # Don't fail the main request if analytics logging fails
        
        return create_response(
            True,
            "Successfully extracted loan details and saved to MongoDB",
            result.get("data", {})
        )
        
    except Exception as e:
        log_message("error", f"Loan extraction endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        
        # Log analytics event for failed loan extraction
        # Use user_id if available, otherwise use client_id
        analytics_id = user_id if user_id else client_id
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            file_size = file.size if hasattr(file, 'size') else 0
            
            await log_event(
                user_id_or_client_id=analytics_id,
                event_type='tool_usage',
                category='Loan Extraction',
                action='loan_extraction_failed',
                metadata={
                    'toolName': 'Loan Extraction',
                    'toolMode': 'extract',
                    'fileName': file.filename if hasattr(file, 'filename') else 'unknown',
                    'fileSize': file_size,
                    'fileSizeKB': round(file_size / 1024, 2) if file_size else 0,
                    'clientId': client_id,
                    'error': str(e)
                },
                request=request,
                options={
                    'duration': duration_ms,
                    'status': 'error',
                    'errorMessage': str(e),
                    'clientId': client_id  # Pass client_id in options
                }
            )
            log_message('info', f"Analytics error event logged for {'user_id' if user_id else 'client_id'}={analytics_id}")
        except Exception as analytics_error:
            log_message('error', f"Error logging loan extraction error analytics: {analytics_error}")
            import traceback as tb
            log_message('error', f"Analytics traceback: {tb.format_exc()}")
            # Don't fail the main request if analytics logging fails
        
        return create_response(
            False,
            f"Error extracting loan details: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# -------------------------------------------------------------------------
# Testing Endpoint
# -------------------------------------------------------------------------

class TestTransactionRequest(BaseModel):
    memo: str
    client_id: str


class TestNameSearchRequest(BaseModel):
    name: str
    client_id: str


@router.post("/test_transaction")
async def test_transaction(request: TestTransactionRequest):
    """
    Test endpoint to query Pinecone for a single transaction and get top 25 results.
    
    This is for testing/debugging purposes to see what Pinecone returns for a transaction.
    Uses dynamic index selection based on client_id (same as /process endpoint).
    
    Logic (same as /process endpoint):
    1. Query top 25 similar records from Pinecone
    2. If exact memo match found, use that (newest by date)
    3. If no exact match: filter by score >= 0.800, sort by date (newest first), use first
    4. If no high-score results (>= 0.800), return "Ask My Accountant"
    5. Return all 25 results with the selected category
    
    Pinecone Index Resolution:
    - Looks up client_pinecone_index collection in MongoDB using client_id
      to get the stored pinecone_index_name (same as /process endpoint)
    
    Args:
        request: JSON body with:
            - memo: The transaction memo/description to test (required)
            - client_id: Client ID to determine which Pinecone index to use (required)
                         Looks up client_pinecone_index table to get pinecone_index_name
    
    Returns:
        JSON with:
            - transaction_memo: The input transaction memo
            - client_id: The client ID used
            - client_name: The client name from client_pinecone_index
            - pinecone_index: The Pinecone index name that was queried
            - selected_category: The category selected based on the logic
            - all_25_results: List of all 25 similar records with details
            - exact_match_found: Whether exact memo match was found
            - high_score_used: Whether high score (>= 0.800) was used
            - ask_accountant: Whether "Ask My Accountant" category was assigned
            - total_results: Number of results found
    """
    try:
        # Validate required fields
        if not request.memo or not request.memo.strip():
            return create_response(
                False,
                "memo is required and cannot be empty",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        if not request.client_id or not request.client_id.strip():
            return create_response(
                False,
                "client_id is required and cannot be empty",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message("info", f"Test transaction request: memo='{request.memo}', client_id='{request.client_id}'")
        
        # Resolve Pinecone index name from client_id (same logic as /process endpoint)
        resolved_index_name = None
        client_name = None
        
        try:
            from backend.services.mongodb import get_client_pinecone_index
            client_index = await get_client_pinecone_index(request.client_id)
            if client_index and client_index.get("pinecone_index_name"):
                resolved_index_name = client_index.get("pinecone_index_name")
                client_name = client_index.get("name", "")
                log_message("info", f"Resolved Pinecone index from MongoDB: client_id={request.client_id}, index_name={resolved_index_name}")
            else:
                log_message("warning", f"Client ID '{request.client_id}' not found in client_pinecone_index collection. Cannot proceed without index.")
                return create_response(
                    False,
                    f"Client ID '{request.client_id}' not found in client_pinecone_index collection. Please ensure the client has a Pinecone index configured.",
                    code=status.HTTP_404_NOT_FOUND
                )
        except Exception as e:
            log_message("error", f"Error looking up client_pinecone_index for client_id={request.client_id}: {e}")
            return create_response(
                False,
                f"Error looking up client: {e}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Validate that GL data exists in Pinecone index before processing
        if resolved_index_name:
            from backend.services.pinecone import check_index_has_data
            index_check = check_index_has_data(resolved_index_name)
            
            if index_check.get("status") == "failed":
                return create_response(
                    False,
                    f"Failed to check Pinecone index '{resolved_index_name}': {index_check.get('error', 'Unknown error')}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            if not index_check.get("has_data", False):
                if not index_check.get("index_exists", False):
                    return create_response(
                        False,
                        f"Pinecone index '{resolved_index_name}' does not exist. Please upload GL data first using the /api/add_gl endpoint.",
                        {
                            "index_name": resolved_index_name,
                            "client_id": request.client_id
                        },
                        code=status.HTTP_400_BAD_REQUEST
                    )
                else:
                    return create_response(
                        False,
                        f"Pinecone index '{resolved_index_name}' exists but has no GL data. Please upload GL data first using the /api/add_gl endpoint.",
                        {
                            "index_name": resolved_index_name,
                            "total_vectors": index_check.get("total_vectors", 0),
                            "client_id": request.client_id
                        },
                        code=status.HTTP_400_BAD_REQUEST
                    )
            
            log_message(
                "info",
                f"Verified GL data exists in Pinecone index '{resolved_index_name}': {index_check.get('total_vectors', 0)} vectors found."
            )
        
        # Call the interactor function with client_id and resolved index_name
        result = await transaction_interactors.test_single_transaction(
            transaction_memo=request.memo,
            client_id=request.client_id,
            client_name=client_name,
            index_name=resolved_index_name
        )
        
        if result.get("status") == "failed":
            return create_response(
                False,
                result.get("error", "Test failed"),
                code=status.HTTP_400_BAD_REQUEST
            )
        
        return create_response(
            True,
            "Test transaction query completed",
            data=result.get("data", {})
        )
        
    except Exception as e:
        log_message("error", f"Test transaction endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Test transaction error: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/test_name_search")
async def test_name_search(request: TestNameSearchRequest):
    """
    Test endpoint to query Pinecone by name and get top 50 results.
    
    This is for testing/debugging purposes to see what Pinecone returns when searching by name.
    Uses dynamic index selection based on client_id (same as /process endpoint).
    
    Logic (same as name-based fallback in /process endpoint):
    1. Generate embedding for the name
    2. Query top 50 similar records from Pinecone
    3. First check for exact name matches (case-insensitive)
    4. If exact matches found, use newest by date
    5. If no exact matches, use similar matches with score >= 0.8, newest by date
    6. Return all 50 results with the selected category
    
    Pinecone Index Resolution:
    - Looks up client_pinecone_index collection in MongoDB using client_id
      to get the stored pinecone_index_name (same as /process endpoint)
    
    Args:
        request: JSON body with:
            - name: The name to test (required)
            - client_id: Client ID to determine which Pinecone index to use (required)
                         Looks up client_pinecone_index table to get pinecone_index_name
    
    Returns:
        JSON with:
            - input_name: The input name
            - client_id: The client ID used
            - client_name: The client name from client_pinecone_index
            - pinecone_index: The Pinecone index name that was queried
            - selected_category: The category selected based on the logic
            - match_type: "exact" or "similar" (or None if no match)
            - all_50_results: List of all 50 similar records with details
            - ask_accountant: Whether "Ask My Accountant" category was assigned
            - total_results: Number of results found
    """
    try:
        # Validate required fields
        if not request.name or not request.name.strip():
            return create_response(
                False,
                "name is required and cannot be empty",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        if not request.client_id or not request.client_id.strip():
            return create_response(
                False,
                "client_id is required and cannot be empty",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message("info", f"Test name search request: name='{request.name}', client_id='{request.client_id}'")
        
        # Resolve Pinecone index name from client_id (same logic as /process endpoint)
        resolved_index_name = None
        client_name = None
        
        try:
            from backend.services.mongodb import get_client_pinecone_index
            client_index = await get_client_pinecone_index(request.client_id)
            if client_index and client_index.get("pinecone_index_name"):
                resolved_index_name = client_index.get("pinecone_index_name")
                client_name = client_index.get("name", "")
                log_message("info", f"Resolved Pinecone index from MongoDB: client_id={request.client_id}, index_name={resolved_index_name}")
            else:
                log_message("warning", f"Client ID '{request.client_id}' not found in client_pinecone_index collection. Cannot proceed without index.")
                return create_response(
                    False,
                    f"Client ID '{request.client_id}' not found in client_pinecone_index collection. Please ensure the client has a Pinecone index configured.",
                    code=status.HTTP_404_NOT_FOUND
                )
        except Exception as e:
            log_message("error", f"Error looking up client_pinecone_index for client_id={request.client_id}: {e}")
            return create_response(
                False,
                f"Error looking up client: {e}",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Validate that GL data exists in Pinecone index before processing
        if resolved_index_name:
            from backend.services.pinecone import check_index_has_data
            index_check = check_index_has_data(resolved_index_name)
            
            if index_check.get("status") == "failed":
                return create_response(
                    False,
                    f"Failed to check Pinecone index '{resolved_index_name}': {index_check.get('error', 'Unknown error')}",
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            if not index_check.get("has_data", False):
                if not index_check.get("index_exists", False):
                    return create_response(
                        False,
                        f"Pinecone index '{resolved_index_name}' does not exist. Please upload GL data first using the /api/add_gl endpoint.",
                        {
                            "index_name": resolved_index_name,
                            "client_id": request.client_id
                        },
                        code=status.HTTP_400_BAD_REQUEST
                    )
                else:
                    return create_response(
                        False,
                        f"Pinecone index '{resolved_index_name}' exists but has no GL data. Please upload GL data first using the /api/add_gl endpoint.",
                        {
                            "index_name": resolved_index_name,
                            "total_vectors": index_check.get("total_vectors", 0),
                            "client_id": request.client_id
                        },
                        code=status.HTTP_400_BAD_REQUEST
                    )
            
            log_message(
                "info",
                f"Verified GL data exists in Pinecone index '{resolved_index_name}': {index_check.get('total_vectors', 0)} vectors found."
            )
        
        # Call the interactor function with client_id and resolved index_name
        result = await transaction_interactors.test_name_search(
            name=request.name,
            client_id=request.client_id,
            client_name=client_name,
            index_name=resolved_index_name
        )
        
        if result.get("status") == "failed":
            return create_response(
                False,
                result.get("error", "Test failed"),
                code=status.HTTP_400_BAD_REQUEST
            )
        
        return create_response(
            True,
            "Test name search query completed",
            data=result.get("data", {})
        )
        
    except Exception as e:
        log_message("error", f"Test name search endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Test name search error: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# -------------------------------------------------------------------------
# Loan Amortization Endpoint
# -------------------------------------------------------------------------

class UpdateTransactionCategoryRequest(BaseModel):
    """Request model for updating transaction category in Pinecone."""
    client_id: str
    description: str
    category: str
    force_update: bool = False  # If True, updates even if exact match exists


class FindLoanByTransactionRequest(BaseModel):
    """Request model for finding loan by transaction details."""
    transaction_date: str  # ISO format date string (e.g., "2024-01-15" or "2024-01-15T10:30:00")
    memo: str
    amount: float
    client_id: str


@router.post("/find_loan_by_transaction")
async def find_loan_by_transaction_endpoint(request: FindLoanByTransactionRequest):
    """
    Find a loan from MongoDB based on transaction details.
    
    This endpoint searches for loans by matching:
    - Client ID (required)
    - Account ID extracted from memo
    - Transaction amount with loan payment amount (exact match)
    
    Args:
        request: JSON body with:
            - transaction_date: Date of the transaction (ISO format string, e.g., "2024-01-15")
            - memo: Transaction memo/description (must contain account ID/number)
            - amount: Transaction amount (float)
            - client_id: Client ID to filter loans
    
    Returns:
        JSON response with:
            - loan: The matched loan document from MongoDB
            - schedule: Amortization schedule if loan details are sufficient
            - match_reason: Explanation of why this loan was matched
    """
    try:
        # Parse transaction date
        try:
            transaction_date = date_parser.parse(request.transaction_date)
        except Exception as e:
            return create_response(
                False,
                f"Invalid transaction_date format: {request.transaction_date}. Use ISO format (e.g., '2024-01-15' or '2024-01-15T10:30:00')",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate required fields
        if not request.memo:
            return create_response(
                False,
                "memo is required and must contain account ID",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        if not request.amount or request.amount <= 0:
            return create_response(
                False,
                "amount must be a positive number",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        if not request.client_id:
            return create_response(
                False,
                "client_id is required",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message(
            "info",
            f"Finding loan by transaction: date={transaction_date}, amount={request.amount}, "
            f"client_id={request.client_id}, memo={request.memo[:50]}..."
        )
        
        # Import the amortization service function
        from backend.services.amortization import find_loan_by_transaction
        
        # Call the function
        result = await find_loan_by_transaction(
            transaction_date=transaction_date,
            memo=request.memo,
            amount=request.amount,
            client_id=request.client_id
        )
        
        if result is None:
            return create_response(
                False,
                "No matching loan found for the given transaction details",
                code=status.HTTP_404_NOT_FOUND
            )
        
        # Convert datetime and ObjectId objects to strings for JSON serialization
        def serialize_for_json(obj):
            """Recursively convert datetime and ObjectId objects to strings."""
            from bson import ObjectId
            
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, ObjectId):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: serialize_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [serialize_for_json(item) for item in obj]
            return obj
        
        # Serialize the result to make it JSON-compatible
        serialized_result = serialize_for_json(result)
        
        return create_response(
            True,
            "Successfully found matching loan",
            serialized_result
        )
        
    except Exception as e:
        log_message("error", f"Find loan by transaction endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error finding loan by transaction: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/update_transaction_category")
async def update_transaction_category_endpoint(request: UpdateTransactionCategoryRequest):
    """
    Update or add a transaction category in Pinecone based on description.
    
    This endpoint:
    1. Looks up the Pinecone index name from MongoDB using client_id
    2. Searches Pinecone for an exact match on BOTH description AND category
    3. If exact match found and force_update=False: Returns "already_exists" (frontend shows popup)
    4. If exact match found and force_update=True: Updates the existing vector
    5. If description exists but category is different: Updates the category
    6. If not found: Adds a new vector with the description and category
    
    Args:
        request: JSON body with:
            - client_id: Client identifier (required)
            - description: Transaction description/memo to search for (required)
            - category: Category to update or assign (required)
            - force_update: If True, updates even if exact match exists (default: False)
    
    Returns:
        JSON response with:
            - action: "updated" | "created" | "already_exists"
            - vector_id: The Pinecone vector ID that was updated or created
            - index_name: The Pinecone index name used
            - message: Human-readable description
    """
    try:
        # Validate required fields
        if not request.client_id or not request.client_id.strip():
            return create_response(
                False,
                "client_id is required",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        if not request.description or not request.description.strip():
            return create_response(
                False,
                "description is required",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        if not request.category or not request.category.strip():
            return create_response(
                False,
                "category is required",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message(
            "info",
            f"Update transaction category request: client_id={request.client_id}, "
            f"description='{request.description[:50]}...', category='{request.category}', "
            f"force_update={request.force_update}"
        )
        
        # Get Pinecone index name from MongoDB
        from backend.services.mongodb import get_client_pinecone_index
        client_index = await get_client_pinecone_index(request.client_id)
        
        if not client_index or not client_index.get("pinecone_index_name"):
            return create_response(
                False,
                f"Client with ID '{request.client_id}' not found in client_pinecone_index collection, "
                f"or no Pinecone index name is associated. Please ensure the client has a Pinecone index.",
                code=status.HTTP_404_NOT_FOUND
            )
        
        index_name = client_index.get("pinecone_index_name")
        client_name = client_index.get("name", "")
        
        log_message("info", f"Resolved Pinecone index: {index_name} for client_id={request.client_id}")
        
        # Verify index exists
        from pinecone import Pinecone
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            return create_response(
                False,
                "PINECONE_API_KEY not configured",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if not pc.has_index(index_name):
            return create_response(
                False,
                f"Pinecone index '{index_name}' does not exist",
                code=status.HTTP_404_NOT_FOUND
            )
        
        index = pc.Index(index_name)
        
        # Generate embedding for the description
        from backend.utils.embed import get_embeddings
        embeddings = get_embeddings()
        
        description_embedding = await asyncio.to_thread(
            embeddings.embed_query,
            request.description.strip()
        )
        
        # Search Pinecone for exact match on memo/description
        # Use high top_k to ensure we find the exact match if it exists
        query_response = index.query(
            vector=description_embedding,
            top_k=100,  # High number to ensure we find exact matches
            include_metadata=True
        )
        
        # Check for exact match on description, category, and optionally credit/debit (case-insensitive)
        matched_vector_id = None
        matched_vector = None
        
        # Normalize description for comparison (handles case, special chars, whitespace)
        description_normalized = normalize_description_for_comparison(request.description)
        category_lower = request.category.strip().lower()
        
        for match in query_response.get('matches', []):
            metadata = match.get('metadata', {})
            memo = metadata.get("memo", "").strip()
            existing_category = (metadata.get("category", "") or metadata.get("split", "")).strip()
            
            # Normalize memo from Pinecone for comparison
            memo_normalized = normalize_description_for_comparison(memo)
            
            # Check for exact match on description AND category only (credit/debit are ignored)
            description_match = memo_normalized == description_normalized
            category_match = existing_category.lower() == category_lower
            
            if not (description_match and category_match):
                continue  # Skip if description or category doesn't match
            
            # ALL CONDITIONS MATCHED - This is an EXACT MATCH (memo + category only)
            # Set matched_vector and break to check if we should return "already_exists"
            matched_vector_id = match.get('id')
            matched_vector = match
            
            log_message(
                "info", 
                f"Found EXACT MATCH for description '{request.description}' "
                f"with category '{request.category}' (vector_id={matched_vector_id})"
            )
            break  # Found exact match, exit loop
        
        # IMPORTANT: First check if EXACT MATCH exists (description + category only, credit/debit ignored)
        # If exact match found, return "already_exists" unless force_update=true
        if matched_vector_id and matched_vector:
            log_message(
                "info",
                f"EXACT MATCH FOUND: description='{request.description}', category='{request.category}'"
                f", force_update={request.force_update}"
            )
            
            # If force_update is False, return "already exists" message FIRST
            # This is the key: we must check for exact match BEFORE updating
            if not request.force_update:
                log_message(
                    "info",
                    f"Returning 'already_exists' for exact match (force_update=false)"
                )
                return create_response(
                    True,
                    f"Data already exists with this description and category",
                    {
                        "action": "already_exists",
                        "vector_id": matched_vector_id,
                        "index_name": index_name,
                        "description": request.description,
                        "category": request.category,
                        "message": f"This combination of description and category already exists in Pinecone. Click 'Confirm' to update it anyway."
                    },
                    code=status.HTTP_200_OK  # 200 because it's a valid response, not an error
                )
            
            # If force_update is True, proceed with update
            # Get current metadata
            current_metadata = matched_vector.get('metadata', {})
            
            # Use the same embedding since description hasn't changed
            original_vector = description_embedding
            
            # Update metadata with new category (keep existing credit/debit values)
            updated_metadata = current_metadata.copy()
            updated_metadata['category'] = request.category.strip()
            updated_metadata['split'] = request.category.strip()  # Also update 'split' for backward compatibility
            
            # Update the vector in Pinecone using upsert
            index.upsert(
                vectors=[{
                    'id': matched_vector_id,
                    'values': original_vector,
                    'metadata': updated_metadata
                }]
            )
            
            log_message(
                "info",
                f"Force updated vector {matched_vector_id} with category '{request.category}' "
                f"for description '{request.description}'"
            )
            
            return create_response(
                True,
                f"Successfully updated transaction category to '{request.category}'",
                {
                    "action": "updated",
                    "vector_id": matched_vector_id,
                    "index_name": index_name,
                    "description": request.description,
                    "category": request.category,
                    "previous_category": current_metadata.get('category') or current_metadata.get('split', '')
                }
            )
        
        # Check if description exists but with different category
        # Only proceed if it's NOT an exact match (different category)
        description_match_found = False
        different_category_vector_id = None
        different_category = None
        different_category_vector = None
        
        for match in query_response.get('matches', []):
            # Skip if this was already matched as exact match in previous loop
            if match.get('id') == matched_vector_id:
                continue
                
            metadata = match.get('metadata', {})
            memo = metadata.get("memo", "").strip()
            
            # Normalize memo from Pinecone for comparison
            memo_normalized = normalize_description_for_comparison(memo)
            
            # Must match description (using normalized versions)
            if memo_normalized != description_normalized:
                continue
                
            existing_category_check = (metadata.get("category", "") or metadata.get("split", "")).strip()
            
            # Check if category matches - if it matches, it should have been caught in exact match above
            if existing_category_check.lower() == category_lower:
                # Category matches - this is an exact match we should have caught
                continue  # Skip, should have been caught in first loop
            
            # Only proceed if description matches but category is different
            description_match_found = True
            different_category_vector_id = match.get('id')
            different_category = existing_category_check
            different_category_vector = match
            break
        
        # If description exists but category is different, update it
        if description_match_found and different_category_vector_id and different_category_vector:
            current_metadata = different_category_vector.get('metadata', {})
            
            # Update metadata with new category and credit/debit
            updated_metadata = current_metadata.copy()
            updated_metadata['category'] = request.category.strip()
            updated_metadata['split'] = request.category.strip()
            # Keep existing credit/debit values (not updated from request)
            
            # Update the vector
            index.upsert(
                vectors=[{
                    'id': different_category_vector_id,
                    'values': description_embedding,
                    'metadata': updated_metadata
                }]
            )
            
            log_message(
                "info",
                f"Updated vector {different_category_vector_id} category from '{different_category}' "
                f"to '{request.category}' for description '{request.description}'"
            )
            
            return create_response(
                True,
                f"Successfully updated transaction category from '{different_category}' to '{request.category}'",
                {
                    "action": "updated",
                    "vector_id": different_category_vector_id,
                    "index_name": index_name,
                    "description": request.description,
                    "category": request.category,
                    "previous_category": different_category
                }
            )
        
        # If not found, add new vector
        from uuid import uuid4
        new_vector_id = str(uuid4())
        
        # Create metadata for new vector
        new_metadata = {
            "memo": request.description.strip(),
            "category": request.category.strip(),
            "split": request.category.strip(),  # Also include 'split' for backward compatibility
            "client_name": client_name,
            "name": "",  # Empty name field
            "credit": "",  # Empty credit (not provided in request)
            "debit": ""   # Empty debit (not provided in request)
        }
        
        # Add the new vector to Pinecone
        index.upsert(
            vectors=[{
                'id': new_vector_id,
                'values': description_embedding,
                'metadata': new_metadata
            }]
        )
        
        log_message(
            "info",
            f"Created new vector {new_vector_id} with category '{request.category}' "
            f"for description '{request.description}'"
        )
        
        return create_response(
            True,
            f"Successfully created new transaction entry with category '{request.category}'",
            {
                "action": "created",
                "vector_id": new_vector_id,
                "index_name": index_name,
                "description": request.description,
                "category": request.category
            }
        )
        
    except Exception as e:
        log_message("error", f"Update transaction category endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error updating transaction category: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# -------------------------------------------------------------------------
# GL Status and Unique Data Endpoints
# -------------------------------------------------------------------------

@router.get("/gl_status")
async def get_gl_status(client_id: str):
    """
    Get GL processing status for a client.
    
    Args:
        client_id: Client identifier (required)
    
    Returns:
        JSON response with:
            - client_id: Client identifier
            - client_name: Client name
            - status: Processing status ("processing", "success", "failed")
            - created_at: When status was created
            - updated_at: When status was last updated
    """
    try:
        from backend.services.mongodb import get_client_status
        
        if not client_id or not client_id.strip():
            return create_response(
                False,
                "client_id is required",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message("info", f"Getting GL status for client_id={client_id}")
        
        status_record = await get_client_status(client_id)
        
        if not status_record:
            return create_response(
                False,
                f"GL status not found for client_id: {client_id}",
                code=status.HTTP_404_NOT_FOUND
            )
        
        # Add id field for convenience (serialization handled by create_response)
        if "_id" in status_record:
            status_record["id"] = str(status_record["_id"])
        
        return create_response(
            True,
            "GL status retrieved successfully",
            data=status_record
        )
        
    except Exception as e:
        log_message("error", f"Get GL status endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error getting GL status: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.get("/gl_unique_data")
async def get_gl_unique_data(client_id: Optional[str] = None):
    """
    Get client unique/duplicate data from GL processing.
    
    Args:
        client_id: Client identifier (optional, if not provided returns all records)
    
    Returns:
        JSON response with list of unique data records:
            - client_id: Client identifier
            - client_name: Client name
            - pinecone_index_name: Pinecone index name
            - duplicate_data: List of duplicate records (same memo + same split)
            - duplicate_count: Number of duplicate groups
            - created_at: When record was created
    """
    try:
        from backend.services.mongodb import get_client_unique_data
        
        log_message("info", f"Getting GL unique data for client_id={client_id or 'all'}")
        
        unique_data_records = await get_client_unique_data(client_id)
        
        # Serialize all records (handles datetime and ObjectId)
        serialized_records = []
        for record in unique_data_records:
            # Add id field for convenience
            if "_id" in record:
                record["id"] = str(record["_id"])
            serialized_records.append(serialize_for_json(record))
        
        return create_response(
            True,
            f"Retrieved {len(serialized_records)} unique data record(s)",
            data={
                "records": serialized_records,
                "count": len(serialized_records)
            }
        )
        
    except Exception as e:
        log_message("error", f"Get GL unique data endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error getting GL unique data: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# -------------------------------------------------------------------------
# Client Bank Extract Data Endpoints
# -------------------------------------------------------------------------

@router.get("/client_bank_extract_data")
async def get_client_bank_extract_data_endpoint(user_id: str):
    """
    Get client bank extract data from MongoDB based on user_id.
    
    This endpoint retrieves the stored bank statement processing results
    (CSV content, banks, accounts, categories) that were saved from the /process endpoint.
    
    Args:
        user_id: User identifier (required, from user table)
    
    Returns:
        JSON response with bank extract data records:
            - user_id: User identifier
            - client_name: Client name
            - data: Processed bank extract data (banks, accounts, csv_content, coa_categories)
            - created_at: When record was created
            - updated_at: When record was last updated
    """
    try:
        from backend.services.mongodb import get_client_bank_extract_data
        
        if not user_id or not user_id.strip():
            return create_response(
                False,
                "user_id is required",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        log_message("info", f"Getting bank extract data for user_id={user_id}")
        
        extract_data_records = await get_client_bank_extract_data(user_id)
        
        # Serialize all records (handles datetime and ObjectId)
        serialized_records = []
        for record in extract_data_records:
            # Add id field for convenience
            if "_id" in record:
                record["id"] = str(record["_id"])
            serialized_records.append(serialize_for_json(record))
        
        return create_response(
            True,
            f"Retrieved {len(serialized_records)} bank extract data record(s)",
            data={
                "records": serialized_records,
                "count": len(serialized_records)
            }
        )
        
    except Exception as e:
        log_message("error", f"Get client bank extract data endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error getting client bank extract data: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
