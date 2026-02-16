import os
import csv
import re
import json
import asyncio
from typing import  Dict, Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.utils.logger import log_message
from newversion.services.rule_base_mongodb import get_database, validate_client_exists
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from backend.utils.file_processor import FileProcessor
from bson import ObjectId

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GOOGLE_API_KEY")
)

def validate_loan_document(loan_data: Dict[str, Any]) -> None:
    """
    Validates whether extracted data represents a real loan document.
    Raises ValueError if invalid.
    """

    # Fields that indicate a REAL loan
    financial_fields = [
        loan_data.get("funding_provided"),
        loan_data.get("payment"),
        loan_data.get("finance_charge"),
        loan_data.get("total_payment_amount"),
        loan_data.get("annual_percentage_rate"),
    ]

    # 1️⃣ At least ONE financial value must exist
    has_financial_data = any(f is not None for f in financial_fields)

    # 2️⃣ Must have either a payment OR funding amount
    has_core_amount = (
        loan_data.get("payment") is not None
        or loan_data.get("funding_provided") is not None
    )

    # 3️⃣ Legal name alone is NOT enough
    if not has_financial_data or not has_core_amount:
        raise ValueError(
            "Document is not a valid loan document: missing financial loan data"
        )

async def extract_loan_details(file, client_id: str) -> Dict[str, Any]:
    """
    Extract loan details from a PDF document using AI and save to MongoDB.
    
    Extracts:
    - Legal Name
    - Funding Provided
    - Annual Percentage Rate (APR)
    - Finance Charge
    - Total Payment Amount
    - Payment
    - Account No.
    
    Args:
        file: UploadFile object containing the loan PDF document.
        client_id: Client identifier for MongoDB storage.
        
    Returns:
        Dict with extracted loan details, MongoDB record ID, or error status.
    """
    try:
        log_message("info", "Starting loan details extraction from PDF")
        
        # Extract text from PDF (with OCR fallback for image-based PDFs)
        
        try:
            # Try standard extraction first, fallback to OCR if needed
            pdf_text = await FileProcessor.read_content_with_fallback(file)
        except Exception as e:
            log_message("error", f"PDF extraction failed: {e}")
            return {
                "status": "failed",
                "error": f"Failed to extract text from PDF: {str(e)}"
            }

        # Log first 500 chars for debugging (don't log full text as it might be huge)
        preview = pdf_text[:500] if pdf_text else ""
        log_message("info", f"Extracted {len(pdf_text)} characters from PDF. Preview: {preview}...")

        
        if not pdf_text or len(pdf_text.strip()) < 50:
            return {
                "status": "failed",
                "error": "Could not extract sufficient text from PDF. The document may be empty or corrupted."
            }
        
        log_message("info", f"Extracted {len(pdf_text)} characters from PDF")
        
        # Use LLM to extract structured loan details
        # Limit to first 8000 chars to avoid token limits
        pdf_text_limited = pdf_text[:8000]
        extraction_prompt = f"""Extract the following loan details from the document text below. 
Return ONLY a valid JSON object with these exact keys (use null if not found):
- "legal_name": The legal name of the borrower/entity (as string)
- "funding_provided": The total funding/loan amount provided (as number or string)
- "annual_percentage_rate": The APR percentage (as number or string with %)
- "finance_charge": The total finance charge amount (as number or string)
- "total_payment_amount": The total payment amount over the loan term (as number or string)
- "payment": The regular payment amount (as number or string)
- "account_no": The account number (as string)
- "due_date": The first payment due date. Look for: "Due Date", "Payment Due", "Finance Agreement dated [date]", "Equipment Finance Schedule dated [date]", "Master Equipment Finance Agreement dated [date]", or any date after "dated" in finance/equipment context (as string in format YYYY-MM-DD or MM/DD/YYYY or any date format found)
- "start_date": The loan start date (as string in format YYYY-MM-DD or MM/DD/YYYY or any date format found) - if not found, will be calculated from due_date

Document text:
{pdf_text_limited}

Return ONLY the JSON object, no other text."""

        try:
            # Invoke LLM with structured output
            response = await asyncio.to_thread(
                llm.invoke,
                extraction_prompt
            )
            
            # Extract JSON from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_text = response_text.strip()
            log_message("info", f"Response text: {response_text}")
            
            # Try to extract JSON from the response
            
            loan_data = None
            
            # First, try parsing the entire response as JSON
            try:
                loan_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    try:
                        loan_data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # If still no JSON, try to find the first balanced JSON object
                if loan_data is None:
                    brace_count = 0
                    start_idx = response_text.find('{')
                    if start_idx != -1:
                        for i in range(start_idx, len(response_text)):
                            if response_text[i] == '{':
                                brace_count += 1
                            elif response_text[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response_text[start_idx:i+1]
                                    try:
                                        loan_data = json.loads(json_str)
                                        break
                                    except json.JSONDecodeError:
                                        continue
            
            if loan_data is None:
                raise json.JSONDecodeError("Could not extract valid JSON from LLM response", response_text, 0)
            
            log_message("info", f"Successfully extracted loan details: {loan_data}")
            
            # Helper function to clean payment value
            def clean_payment_value(value):
                """Clean payment value by removing $, commas, /month, and other formatting."""
                if not value:
                    return None
                # Convert to string if not already
                value_str = str(value)
                # Remove dollar sign, commas, spaces, and /month or /mo suffixes
                cleaned = re.sub(r'[\$,\s]', '', value_str)
                cleaned = re.sub(r'/month|/mo|/monthly', '', cleaned, flags=re.IGNORECASE)
                # Extract just the numeric value (with decimal)
                numeric_match = re.search(r'(\d+\.?\d*)', cleaned)
                if numeric_match:
                    return numeric_match.group(1)
                return cleaned if cleaned else None
            
            # Prepare loan data for MongoDB
            payment_raw = loan_data.get("payment")
            payment_cleaned = clean_payment_value(payment_raw)
            
            # Calculate start_date from due_date + 45 days if due_date is available
            
            
            due_date_str = loan_data.get("due_date")
            start_date_str = loan_data.get("start_date")
            
            # Track if due_date was extracted via regex fallback
            due_date_from_regex = False
            
            # If due_date is still null, try to extract from the raw PDF text using regex
            if not due_date_str:
                log_message("info", "due_date not found in LLM response, trying regex extraction from PDF text")
                # Try to find dates in the PDF text that match our patterns
                due_date_patterns = [
                    # Text date formats first
                    r"finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"equipment\s+finance\s+schedule[^.]*dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"master\s+equipment\s+finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"due\s+date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"payment\s+due[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"
                ]
                for pattern in due_date_patterns:
                    match = re.search(pattern, pdf_text, re.IGNORECASE)
                    if match:
                        due_date_str = match.group(1).strip()
                        due_date_from_regex = True
                        log_message("info", f"Extracted due_date from PDF text using regex: {due_date_str}")
                        break
            
            # If we have due_date, always calculate start_date = due_date + 45 days
            # This overrides any LLM-provided start_date to ensure accuracy
            if due_date_str:
                try:
                    due_date = date_parser.parse(due_date_str)
                    calculated_start_date = due_date + relativedelta(days=45)
                    start_date_str = calculated_start_date.strftime("%Y-%m-%d")
                    if due_date_from_regex:
                        log_message("info", f"Calculated start_date from regex-extracted due_date: {due_date_str} + 45 days = {start_date_str} (overriding LLM start_date: {loan_data.get('start_date')})")
                    else:
                        log_message("info", f"Calculated start_date from due_date: {due_date_str} + 45 days = {start_date_str}")
                except Exception as e:
                    log_message("warning", f"Could not parse due_date to calculate start_date: {e}")
                    log_message("warning", f"due_date_str was: {due_date_str}")
            
            loan_details = {
                "legal_name": loan_data.get("legal_name"),
                "funding_provided": loan_data.get("funding_provided"),
                "annual_percentage_rate": loan_data.get("annual_percentage_rate"),
                "finance_charge": loan_data.get("finance_charge"),
                "total_payment_amount": loan_data.get("total_payment_amount"),
                "payment": payment_cleaned,
                "account_no": loan_data.get("account_no"),
                "start_date": start_date_str
            }
            
            # Validate client_id exists (foreign key validation)
            client_exists = await validate_client_exists(client_id)
            if not client_exists:
                log_message("warning", f"Client {client_id} does not exist, but proceeding with loan save")
                # Optionally, you can return an error here:
                # return {
                #     "status": "failed",
                #     "error": f"Client with ID '{client_id}' does not exist. Please create the client first."
                # }
            
            # Save to MongoDB
            try:
                db = get_database()
                collection = db["client_loans"]
                
                # Convert client_id string to ObjectId for foreign key reference
                client_object_id = ObjectId(client_id)
                
                # Create document to insert
                loan_document = {
                    "client_id": client_object_id,  # ObjectId foreign key to clients._id
                    "pdf_filename": file.filename if hasattr(file, 'filename') else None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    **loan_details
                }

                try:
                    validate_loan_document(loan_details)
                except ValueError as e:
                    log_message("warning", f"Loan validation failed: {e}")
                    return {
                        "status": "failed",
                        "error": str(e),
                        "reason": "NOT_A_LOAN_DOCUMENT"
                    }
                
                # Insert into MongoDB
                result = await collection.insert_one(loan_document)
                log_message("info", f"Saved loan details to MongoDB with ID: {result.inserted_id}")
                
                return {
                    "status": "success",
                    "data": {
                        "id": str(result.inserted_id),
                        "client_id": client_id,
                        **loan_details
                    }
                }
            except Exception as e:
                log_message("error", f"Failed to save to MongoDB: {e}")
                # Return extracted data even if MongoDB save fails
                return {
                    "status": "success",
                    "data": {
                        "client_id": client_id,
                        **loan_details,
                        "mongodb_error": f"Failed to save to MongoDB: {str(e)}"
                    }
                }
            
        except json.JSONDecodeError as e:
            log_message("error", f"Failed to parse LLM response as JSON: {e}")
            log_message("error", f"LLM response was: {response_text[:500]}")
            
            # Fallback: try regex-based extraction
            regex_result = await _extract_loan_details_regex(pdf_text, client_id, file)
            return regex_result
            
        except Exception as e:
            log_message("error", f"LLM extraction failed: {e}")
            # Fallback to regex extraction
            regex_result = await _extract_loan_details_regex(pdf_text, client_id, file)
            return regex_result
            
    except Exception as e:
        log_message("error", f"Loan extraction failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {
            "status": "failed",
            "error": str(e)
        }

async def _extract_loan_details_regex(pdf_text: str, client_id: str, file) -> Dict[str, Any]:
    """
    Fallback regex-based extraction for loan details.
    
    Args:
        pdf_text: Extracted text from PDF.
        client_id: Client identifier for MongoDB storage.
        file: UploadFile object (for filename).
        
    Returns:
        Dict with extracted loan details and MongoDB record ID.
    """
    try:
        log_message("info", "Using regex fallback for loan details extraction")
        
        # Patterns for different loan fields
        patterns = {
            "legal_name": [
                r"legal\s+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|borrower|account|loan|funding)",
                r"borrower[:\s]+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)",
                r"borrower[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)",
                r"name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account\s+number|loan|funding)",
                r"entity\s+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)",
                r"company\s+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)"
            ],
            "funding_provided": [
                r"funding\s+provided[:\s]*\$?([\d,]+\.?\d*)",
                r"loan\s+amount[:\s]*\$?([\d,]+\.?\d*)",
                r"principal[:\s]*\$?([\d,]+\.?\d*)",
                r"amount\s+financed[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "annual_percentage_rate": [
                r"annual\s+percentage\s+rate[:\s]*([\d.]+)\s*%?",
                r"apr[:\s]*([\d.]+)\s*%?",
                r"interest\s+rate[:\s]*([\d.]+)\s*%?"
            ],
            "finance_charge": [
                r"finance\s+charge[:\s]*\$?([\d,]+\.?\d*)",
                r"total\s+finance\s+charge[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "total_payment_amount": [
                r"total\s+payment[:\s]*\$?([\d,]+\.?\d*)",
                r"total\s+of\s+payments[:\s]*\$?([\d,]+\.?\d*)",
                r"total\s+amount\s+payable[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "payment": [
                r"monthly\s+payment[:\s]*\$?([\d,]+\.?\d*)",
                r"payment\s+amount[:\s]*\$?([\d,]+\.?\d*)",
                r"regular\s+payment[:\s]*\$?([\d,]+\.?\d*)",
                r"payment[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "account_no": [
                r"account\s+(?:number|no\.?)[:\s]*([\d\-X]+)",
                r"account\s+#[:\s]*([\d\-X]+)",
                r"acct\s+(?:number|no\.?)[:\s]*([\d\-X]+)"
            ],
            "due_date": [
                # PRIORITY: Text date formats with full month names (e.g., "July 29, 2025", "December 12, 2023")
                # Finance Agreement dated patterns - highest priority
                r"finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"master\s+equipment\s+finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"equipment\s+finance\s+schedule[^.]*dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"equipment\s+finance\s+schedule\s+no[^.]*dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Generic "dated" pattern for finance/equipment context
                r"(?:finance|equipment|schedule|agreement|master).*?dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Simple "dated" pattern (catches "dated December 12, 2023" anywhere)
                r"dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Text date formats with "due date" labels
                r"due\s+date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"payment\s+due[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"first\s+payment\s+due[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Fallback: Numeric date formats (only if text dates not found)
                r"due\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"payment\s+due[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"first\s+payment\s+due[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"first\s+due[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"payment\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"
            ],
            "start_date": [
                r"start\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"loan\s+start[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"date\s+of\s+loan[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"origination\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"beginning\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"effective\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"start[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s*(?:start|beginning|origination)"
            ]
        }
        
        extracted = {}
        pdf_text_lower = pdf_text.lower()
        
        # Log a sample of the text for debugging
        log_message("info", f"PDF text sample (first 1000 chars): {pdf_text[:1000]}")
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                # For legal_name and due_date (which may contain month names), search in original text to preserve case
                # For other fields, use lowercase text
                search_text = pdf_text if field in ["legal_name", "due_date"] else pdf_text_lower
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    log_message("info", f"Found {field} with pattern: {pattern[:50]}... -> value: {value}")
                    
                    # For due_date: Only accept text dates with month names (e.g., "July 29, 2025"), reject numeric-only dates
                    if field == "due_date":
                        # Check if it's a text date (contains month name) or numeric date
                        month_names = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                        if not re.search(month_names, value, re.IGNORECASE):
                            # This is a numeric date, skip it and continue to next pattern
                            log_message("info", f"Skipping numeric date format for due_date: {value}, looking for text date format")
                            continue
                        # It's a text date, accept it
                        log_message("info", f"Accepted text date format for due_date: {value}")
                    
                    # Clean up the value
                    if field == "payment":
                        # Remove dollar sign, commas, spaces, and /month or /mo suffixes
                        value = re.sub(r'[\$,\s]', '', value)
                        value = re.sub(r'/month|/mo|/monthly', '', value, flags=re.IGNORECASE)
                        # Extract just the numeric value (with decimal)
                        numeric_match = re.search(r'(\d+\.?\d*)', value)
                        if numeric_match:
                            value = numeric_match.group(1)
                    elif field in ["funding_provided", "finance_charge", "total_payment_amount"]:
                        # Remove commas from numbers
                        value = value.replace(",", "")
                    elif field == "legal_name":
                        # Clean up legal name - remove extra whitespace
                        value = re.sub(r'\s+', ' ', value).strip()
                    extracted[field] = value
                    break
        
        log_message("info", f"Regex extraction found: {extracted}")
        
        # Calculate start_date from due_date + 45 days if due_date is available
        due_date_str = extracted.get("due_date")
        start_date_str = extracted.get("start_date")
        
        # If we have due_date but no start_date, calculate start_date = due_date + 45 days
        if due_date_str and not start_date_str:
            try:
                due_date = date_parser.parse(due_date_str)
                calculated_start_date = due_date + relativedelta(days=45)
                start_date_str = calculated_start_date.strftime("%Y-%m-%d")
                log_message("info", f"Calculated start_date from due_date: {due_date_str} + 45 days = {start_date_str}")
            except Exception as e:
                log_message("warning", f"Could not parse due_date to calculate start_date: {e}")
        
        # Prepare loan data
        loan_details = {
            "legal_name": extracted.get("legal_name"),
            "funding_provided": extracted.get("funding_provided"),
            "annual_percentage_rate": extracted.get("annual_percentage_rate"),
            "finance_charge": extracted.get("finance_charge"),
            "total_payment_amount": extracted.get("total_payment_amount"),
            "payment": extracted.get("payment"),
            "account_no": extracted.get("account_no"),
            "start_date": start_date_str
        }
        
        # Save to MongoDB
        try:
            
            db = get_database()
            collection = db["client_loans"]
            
            # Convert client_id string to ObjectId for foreign key reference
            client_object_id = ObjectId(client_id)
            
            # Create document to insert
            loan_document = {
                "client_id": client_object_id,  # ObjectId foreign key to clients._id
                "pdf_filename": file.filename if hasattr(file, 'filename') else None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                **loan_details
            }
            
            # Insert into MongoDB
            result = await collection.insert_one(loan_document)
            log_message("info", f"Saved loan details to MongoDB (regex fallback) with ID: {result.inserted_id}")
            
            return {
                "status": "success",
                "data": {
                    "id": str(result.inserted_id),
                    "client_id": str(client_object_id),  # Return ObjectId as string
                    **loan_details
                }
            }
        except Exception as e:
            log_message("error", f"Failed to save to MongoDB (regex fallback): {e}")
            # Return extracted data even if MongoDB save fails
            return {
                "status": "success",
                "data": {
                    "client_id": client_id,
                    **loan_details,
                    "mongodb_error": f"Failed to save to MongoDB: {str(e)}"
                }
            }
        
    except Exception as e:
        log_message("error", f"Regex extraction failed: {e}")
        return {
            "status": "failed",
            "error": f"Failed to extract loan details: {str(e)}"
        }