"""
Gemini PDF Extractor Service

Uses Google Gemini 3 Flash Preview model to extract bank transactions directly from PDF files.
Passes PDF file directly to Gemini API and receives structured JSON output.
"""

import os
import io
import base64
import json
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.utils.logger import log_message

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    log_message("warning", "google-genai not installed. Install with: pip install google-genai")

try:
    import pymupdf as fitz  # PyMuPDF (use pymupdf to avoid conflict with other 'fitz' package)
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    fitz = None
    log_message("warning", "PyMuPDF not installed. Install with: pip install pymupdf")


def _is_quota_error(e: Exception) -> bool:
    """Return True if the exception is a quota/rate-limit error (e.g. 429 RESOURCE_EXHAUSTED)."""
    msg = (str(e) or "").lower()
    name = type(e).__name__
    cause = getattr(e, "__cause__", None)
    cause_msg = (str(cause) or "").lower() if cause else ""
    combined = f"{name} {msg} {cause_msg}"
    return (
        "429" in combined
        or "quota" in combined
        or "resource exhausted" in combined
        or "resource_exhausted" in combined
        or "rate limit" in combined
    )


class GeminiPDFExtractor:
    """Extract bank transactions from PDF using Gemini API with direct PDF upload."""
    
    def __init__(self):
        """Initialize Gemini model for PDF extraction."""
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3-flash-preview"
        log_message("info", f"Initialized Gemini PDF Extractor with {self.model_name} model using google-genai client")
    
    @staticmethod
    def get_pdf_bytes(file) -> bytes:
        """
        Get PDF file bytes.
        
        Args:
            file: UploadFile object containing PDF
            
        Returns:
            PDF file bytes
        """
        try:
            file.file.seek(0)
            pdf_bytes = file.file.read()
            file.file.seek(0)
            return pdf_bytes
        except Exception as e:
            log_message("error", f"Failed to read PDF file: {e}")
            raise Exception(f"PDF file reading failed: {e}")
    
    @staticmethod
    def filter_transaction_history_pages(pdf_bytes: bytes) -> bytes:
        """
        Filter PDF to include only pages containing transaction section headers.
        Supports multiple bank formats:
        - Wells Fargo: "Transaction history" OR "Credits"/"Electronic deposits/bank credits" and "Debits"/"Electronic debits/bank debits" (format_01)
        - Bank of America: "Deposits and other credits", "Withdrawals and other debits", "Service fees" 
        - BMO Format 1: "Deposits and Other Credits", "Withdrawals and Other Debits"
        - BMO Format 2: "Monthly Activity Details", "POSTED" transactions (transaction list format)
        - Chase: "Deposits and Additions", "Checks Paid", "ATM & Debit Card Withdrawals", "Electronic Withdrawals", "Other Withdrawals", "Fees"
        
        Uses regex to search for these headers (case-insensitive) - only includes pages with the headers.
        
        Args:
            pdf_bytes: Original PDF file bytes
            
        Returns:
            Filtered PDF bytes containing only pages with transaction section headers
        """
        if not FITZ_AVAILABLE:
            log_message("warning", "PyMuPDF not available, returning original PDF")
            return pdf_bytes
        
        try:
            log_message("info", "Filtering PDF pages for transaction sections")
            
            # Open PDF from bytes
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            total_pages = len(doc)
            log_message("info", f"Total pages in PDF: {total_pages}")
            
           
            # Pattern 1: Wells Fargo - "Transaction history"
            pattern_wf = re.compile(r'transaction\s+history', re.IGNORECASE)
            
            # Pattern 2: Bank of America / BMO Format 1 - "Deposits and other credits"
            pattern_credits = re.compile(r'deposits\s+and\s+other\s+credits', re.IGNORECASE)
            
            # Pattern 3: Bank of America / BMO Format 1 - "Withdrawals and other debits"
            pattern_debits = re.compile(r'withdrawals\s+and\s+other\s+debits', re.IGNORECASE)
            
            # Pattern 4: Service fees section (Bank of America, BMO, and other banks)
            pattern_service_fees = re.compile(r'service\s+fees', re.IGNORECASE)
            
            # Pattern 5: BMO Format 2 - "Monthly Activity Details"
            pattern_bmo_monthly = re.compile(r'monthly\s+activity\s+details', re.IGNORECASE)
            
            # Pattern 6: BMO Format 2 - "POSTED" transactions (transaction list format)
            pattern_bmo_posted = re.compile(r'\bPOSTED\b', re.IGNORECASE)
            
            # Pattern 7: Chase - "Deposits and Additions" (credits)
            pattern_chase_deposits = re.compile(r'deposits\s+and\s+additions', re.IGNORECASE)
            
            # Pattern 8: Chase - "Checks Paid" (debits)
            pattern_chase_checks = re.compile(r'checks\s+paid', re.IGNORECASE)
            
            # Pattern 9: Chase - "ATM & Debit Card Withdrawals" (debits)
            pattern_chase_atm = re.compile(r'atm\s*[&]\s*debit\s+card\s+withdrawals', re.IGNORECASE)
            
            # Pattern 10: Chase - "Electronic Withdrawals" (debits) - flexible pattern to handle spacing variations
            pattern_chase_electronic = re.compile(r'electronic\s+withdrawals?', re.IGNORECASE)
            
            # Pattern 11: Chase - "Other Withdrawals" (debits) - flexible pattern to handle spacing variations
            pattern_chase_other = re.compile(r'other\s+withdrawals?', re.IGNORECASE)
            
            # Pattern 12: Chase - "Fees" (debits)
            pattern_chase_fees = re.compile(r'\bfees\b', re.IGNORECASE)
            
            # Pattern 13: Wells Fargo - "Credits" header (format_01) - match as standalone word or header
            pattern_wf_credits = re.compile(r'\bcredits\b', re.IGNORECASE)
            
            # Pattern 14: Wells Fargo - "Debits" header (format_01) - match as standalone word or header
            pattern_wf_debits = re.compile(r'\bdebits\b', re.IGNORECASE)
            
            # Pattern 15: Wells Fargo - "Electronic deposits/bank credits" (format_01)
            pattern_wf_electronic_credits = re.compile(r'electronic\s+deposits\s*/\s*bank\s+credits', re.IGNORECASE)
            
            # Pattern 16: Wells Fargo - "Electronic debits/bank debits" (format_01)
            pattern_wf_electronic_debits = re.compile(r'electronic\s+debits\s*/\s*bank\s+debits', re.IGNORECASE)
            
            # First pass: Detect if this is a Wells Fargo statement
            is_wells_fargo = False
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text("text")
                if pattern_wf.search(page_text) or pattern_wf_credits.search(page_text) or pattern_wf_debits.search(page_text) or pattern_wf_electronic_credits.search(page_text) or pattern_wf_electronic_debits.search(page_text):
                    is_wells_fargo = True
                    break
            
            # Find pages containing transaction section headers
            matching_pages = []
            page_headers = {}  # Track which header was found on each page
            bmo_posted_start_page = None  # Track where BMO POSTED section starts
            
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                # Check for transaction section headers
                found_header = None
                
                if is_wells_fargo:
                    # For Wells Fargo: Check for "Transaction history" OR Credits/Debits sections (format_01)
                    if pattern_wf.search(page_text):
                        found_header = "Transaction history"
                    elif pattern_wf_credits.search(page_text) or pattern_wf_electronic_credits.search(page_text):
                        found_header = "Credits"
                    elif pattern_wf_debits.search(page_text) or pattern_wf_electronic_debits.search(page_text):
                        found_header = "Debits"
                else:
                    # For other banks: Check all headers
                    if pattern_wf.search(page_text):
                        found_header = "Transaction history"
                    elif pattern_credits.search(page_text):
                        found_header = "Deposits and other credits"
                    elif pattern_debits.search(page_text):
                        found_header = "Withdrawals and other debits"
                    elif pattern_service_fees.search(page_text):
                        found_header = "Service fees"
                    elif pattern_bmo_monthly.search(page_text):
                        found_header = "Monthly Activity Details"
                    elif pattern_bmo_posted.search(page_text):
                        # Check if this page has transaction-like content (dates, amounts, descriptions)
                        # This helps identify BMO transaction list format
                        has_transactions = bool(re.search(r'\$\d+[,\d]*\.\d{2}', page_text) and 
                                               re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}', page_text, re.IGNORECASE))
                        if has_transactions:
                            found_header = "POSTED transactions"
                            if bmo_posted_start_page is None:
                                bmo_posted_start_page = page_num
                    elif pattern_chase_deposits.search(page_text):
                        found_header = "Deposits and Additions"
                    elif pattern_chase_checks.search(page_text):
                        found_header = "Checks Paid"
                    elif pattern_chase_atm.search(page_text):
                        found_header = "ATM & Debit Card Withdrawals"
                    elif pattern_chase_electronic.search(page_text):
                        found_header = "Electronic Withdrawals"
                    elif pattern_chase_other.search(page_text):
                        found_header = "Other Withdrawals"
                    elif pattern_chase_fees.search(page_text):
                        found_header = "Fees"
                
                if found_header:
                    matching_pages.append(page_num)
                    page_headers[page_num] = found_header
                    log_message("info", f"Found '{found_header}' header on page {page_num + 1}")
            
            if is_wells_fargo:
                headers_found = [page_headers.get(p, "unknown") for p in matching_pages]
                log_message("info", f"Wells Fargo statement detected - included {len(matching_pages)} page(s) with headers: {headers_found}")
                
                # For Wells Fargo Credits/Debits format, include continuation pages with transaction data
                wf_credits_debits_start_pages = [p for p in matching_pages if page_headers.get(p) in ["Credits", "Debits"]]
                if wf_credits_debits_start_pages:
                    log_message("info", f"Wells Fargo Credits/Debits format detected, checking for continuation pages")
                    # Process all pages from the first Credits/Debits page to the end, or until we hit a non-transaction section
                    first_wf_page = min(wf_credits_debits_start_pages)
                    last_wf_page = max(wf_credits_debits_start_pages)
                    
                    for page_num in range(first_wf_page + 1, total_pages):
                        if page_num in matching_pages:
                            continue  # Already included
                        
                        page = doc[page_num]
                        page_text = page.get_text("text")
                        
                        # Check if this is another Credits/Debits section header (include it)
                        if pattern_wf_credits.search(page_text) or pattern_wf_electronic_credits.search(page_text):
                            matching_pages.append(page_num)
                            page_headers[page_num] = "Credits"
                            log_message("info", f"Found 'Credits' header on page {page_num + 1}, including it")
                            last_wf_page = page_num  # Update last WF page
                            continue
                        elif pattern_wf_debits.search(page_text) or pattern_wf_electronic_debits.search(page_text):
                            matching_pages.append(page_num)
                            page_headers[page_num] = "Debits"
                            log_message("info", f"Found 'Debits' header on page {page_num + 1}, including it")
                            last_wf_page = page_num  # Update last WF page
                            continue
                        
                        # Check for transaction data FIRST (prioritize including pages with transactions)
                        has_amount = bool(re.search(r'[\$]?\s*\d+[,\d]*\.\d{2}', page_text))
                        has_date = bool(re.search(r'\d{1,2}/\d{1,2}', page_text))
                        has_content = len(page_text.strip()) > 200
                        has_transactions = has_amount and (has_date or has_content)
                        
                        # If page has transaction data, include it even if there's a minor header
                        if has_transactions:
                            matching_pages.append(page_num)
                            log_message("info", f"Including Wells Fargo continuation page {page_num + 1} (has transaction data)")
                            last_wf_page = page_num  # Update last WF page
                            continue
                        
                        # Be more lenient: include pages that are near WF sections or have any transaction-like content
                        # Only stop if we're clearly in a completely different section with no transaction data
                        
                        # Check if this looks like a summary/end section (not transaction data)
                        is_summary_section = bool(
                            re.search(r'(summary|total|ending\s+balance|account\s+summary)', page_text, re.IGNORECASE) and
                            not has_transactions
                        )
                        
                        # If we're within 3 pages of the last WF section, be very lenient - include if there's any content
                        if page_num <= last_wf_page + 3:
                            if has_content and not is_summary_section:
                                matching_pages.append(page_num)
                                log_message("info", f"Including Wells Fargo continuation page {page_num + 1} (near WF section, has content)")
                                last_wf_page = page_num
                                continue
                        
                        # Only stop if:
                        # 1. We hit a clear summary/end section with no transactions, OR
                        # 2. We're far from WF sections AND have no transaction data AND it's clearly not a transaction page
                        if is_summary_section:
                            log_message("info", f"Stopping Wells Fargo continuation at page {page_num + 1} (hit summary section)")
                            break
                        
                        if page_num > last_wf_page + 3 and not has_transactions and not has_content:
                            log_message("info", f"Stopping Wells Fargo continuation at page {page_num + 1} (no transaction data, far from WF sections)")
                            break
                    
                    # Sort pages after adding continuation pages
                    matching_pages = sorted(matching_pages)
                    log_message("info", f"Wells Fargo Credits/Debits section complete: {len(matching_pages)} total page(s) included")
            
            # For BMO Format 2 (POSTED transactions), include all continuation pages with transaction data
            if bmo_posted_start_page is not None:
                log_message("info", f"BMO POSTED format detected starting at page {bmo_posted_start_page + 1}, checking for continuation pages")
                # Check subsequent pages for transaction data
                for page_num in range(bmo_posted_start_page + 1, total_pages):
                    if page_num in matching_pages:
                        continue  # Already included
                    
                    page = doc[page_num]
                    page_text = page.get_text("text")
                    
                    # More flexible transaction detection for BMO Format 2
                    # Check for amounts (with $ or without, positive or negative)
                    has_amount = bool(re.search(r'[\$]?\s*-?\d+[,\d]*\.\d{2}', page_text))
                    
                    # Check for dates in various formats
                    # Format 1: "Feb 13, 2025" or "Feb 13 2025"
                    date_pattern1 = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}', page_text, re.IGNORECASE)
                    # Format 2: "MM/DD/YYYY" or "MM/DD/YY"
                    date_pattern2 = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', page_text)
                    # Format 3: "MM-DD-YYYY" or "MM-DD-YY"
                    date_pattern3 = re.search(r'\d{1,2}-\d{1,2}-\d{2,4}', page_text)
                    
                    has_date = bool(date_pattern1 or date_pattern2 or date_pattern3)
                    
                    # Check if page has substantial content (likely a transaction page)
                    has_content = len(page_text.strip()) > 200
                    
                    # If page has amounts and (dates OR substantial content), consider it a transaction page
                    has_transactions = has_amount and (has_date or has_content)
                    
                    log_message("info", f"Checking page {page_num + 1}: has_amount={has_amount}, has_date={has_date}, has_content={has_content}, has_transactions={has_transactions}")
                    
                    if has_transactions:
                        matching_pages.append(page_num)
                        log_message("info", f"Including BMO POSTED continuation page {page_num + 1}")
                    else:
                        # Stop if we hit a page without transactions (likely end of transaction section)
                        log_message("info", f"Stopping BMO POSTED continuation at page {page_num + 1} (no transaction data)")
                        break
                
                # Sort pages after adding continuation pages
                matching_pages = sorted(matching_pages)
                log_message("info", f"BMO POSTED section complete: {len(matching_pages)} total page(s) included")
            
            # For Chase bank, add continuation page logic for all withdrawal sections
            chase_withdrawal_sections = ["Deposits and Additions", "Checks Paid", "ATM & Debit Card Withdrawals", 
                                         "Electronic Withdrawals", "Other Withdrawals", "Fees"]
            chase_start_pages = [p for p in matching_pages if page_headers.get(p) in chase_withdrawal_sections]
            if chase_start_pages and not is_wells_fargo:
                log_message("info", f"Chase bank statement detected with {len(chase_start_pages)} section(s), checking for continuation pages")
                # Process each Chase section to find continuation pages
                for start_page in sorted(chase_start_pages):
                    section_name = page_headers.get(start_page)
                    log_message("info", f"Processing Chase section '{section_name}' starting at page {start_page + 1}")
                    
                    # Check subsequent pages for continuation of this section
                    for page_num in range(start_page + 1, total_pages):
                        if page_num in matching_pages:
                            # Check if this is a new section header
                            page = doc[page_num]
                            page_text = page.get_text("text")
                            
                            # If we hit another Chase section header, stop continuation for current section
                            if (pattern_chase_deposits.search(page_text) or 
                                pattern_chase_checks.search(page_text) or
                                pattern_chase_atm.search(page_text) or
                                pattern_chase_electronic.search(page_text) or
                                pattern_chase_other.search(page_text) or
                                pattern_chase_fees.search(page_text)):
                                log_message("info", f"Stopping Chase '{section_name}' continuation at page {page_num + 1} (hit new section header)")
                                break
                            continue  # Already included, skip
                        
                        page = doc[page_num]
                        page_text = page.get_text("text")
                        
                        # Stop if we hit a different bank's section header
                        if (pattern_wf.search(page_text) or 
                            pattern_credits.search(page_text) or
                            pattern_debits.search(page_text) or
                            pattern_bmo_monthly.search(page_text)):
                            log_message("info", f"Stopping Chase '{section_name}' continuation at page {page_num + 1} (hit different bank section)")
                            break
                        
                        # Check for transaction data (dates and amounts)
                        has_amount = bool(re.search(r'[\$]?\s*\d+[,\d]*\.\d{2}', page_text))
                        has_date = bool(re.search(r'\d{1,2}/\d{1,2}', page_text))
                        has_content = len(page_text.strip()) > 200
                        
                        # Check if this page continues the current Chase section
                        # Look for transaction-like content without a new section header
                        if has_amount and (has_date or has_content):
                            # Make sure it's not a new section header
                            if not (pattern_chase_deposits.search(page_text) or 
                                   pattern_chase_checks.search(page_text) or
                                   pattern_chase_atm.search(page_text) or
                                   pattern_chase_electronic.search(page_text) or
                                   pattern_chase_other.search(page_text) or
                                   pattern_chase_fees.search(page_text)):
                                matching_pages.append(page_num)
                                log_message("info", f"Including Chase '{section_name}' continuation page {page_num + 1}")
                            else:
                                log_message("info", f"Stopping Chase '{section_name}' continuation at page {page_num + 1} (hit new section header)")
                                break
                        else:
                            # Stop if we hit a page without transactions (likely end of section)
                            log_message("info", f"Stopping Chase '{section_name}' continuation at page {page_num + 1} (no transaction data)")
                            break
                
                # Sort pages after adding continuation pages
                matching_pages = sorted(matching_pages)
                headers_found = [page_headers.get(p, "continuation") for p in matching_pages]
                log_message("info", f"Chase bank sections complete: {len(matching_pages)} total page(s) included with headers: {headers_found}")
            
            if not matching_pages:
                log_message("warning", "No pages found with transaction section headers, returning original PDF")
                doc.close()
                return pdf_bytes
            
            log_message("info", f"Filtered to {len(matching_pages)} page(s) with transaction section headers: {[p+1 for p in matching_pages]}")
            
            # Create new PDF with only matching pages
            new_doc = fitz.open()
            for page_num in matching_pages:
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Save filtered PDF to bytes
            filtered_pdf_bytes = new_doc.tobytes()
            
            # Clean up
            new_doc.close()
            doc.close()
            
            log_message("info", f"Created filtered PDF with {len(matching_pages)} page(s), size: {len(filtered_pdf_bytes)} bytes")
            
            return filtered_pdf_bytes
            
        except Exception as e:
            log_message("error", f"Failed to filter PDF pages: {e}")
            import traceback
            log_message("error", f"Traceback: {traceback.format_exc()}")
            log_message("warning", "Returning original PDF due to filtering error")
            return pdf_bytes
    
    @staticmethod
    def filter_credit_card_pages(pdf_bytes: bytes) -> bytes:
        """
        Filter PDF to include only pages containing credit card transaction sections.
        Supports multiple credit card formats:
        - Chase: "ACCOUNT ACTIVITY" section with +/- amounts
        - Bank of America: "PAYMENTS AND OTHER CREDITS", "PURCHASES AND OTHER CHARGES", "FEES AND ADJUSTMENTS", "CASH ADVANCES"
        - Wells Fargo: "Transaction Details" section
        
        Args:
            pdf_bytes: Original PDF file bytes
            
        Returns:
            Filtered PDF bytes containing only pages with credit card transaction sections
        """
        if not FITZ_AVAILABLE:
            log_message("warning", "PyMuPDF not available, returning original PDF")
            return pdf_bytes
        
        try:
            log_message("info", "Filtering PDF pages for credit card transaction sections")
            
            # Open PDF from bytes
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            total_pages = len(doc)
            log_message("info", f"Total pages in PDF: {total_pages}")
            
            # Pattern 1: Chase - "ACCOUNT ACTIVITY"
            pattern_chase_activity = re.compile(r'account\s+activity', re.IGNORECASE)
            
            # Pattern 2: Bank of America - "PAYMENTS AND OTHER CREDITS"
            pattern_boa_payments = re.compile(r'payments\s+and\s+other\s+credits', re.IGNORECASE)
            
            # Pattern 3: Bank of America - "PURCHASES AND OTHER CHARGES"
            pattern_boa_purchases = re.compile(r'purchases\s+and\s+other\s+charges', re.IGNORECASE)
            
            # Pattern 4: Bank of America - "FEES AND ADJUSTMENTS"
            pattern_boa_fees = re.compile(r'fees\s+and\s+adjustments', re.IGNORECASE)
            
            # Pattern 5: Bank of America - "CASH ADVANCES"
            pattern_boa_cash = re.compile(r'cash\s+advances', re.IGNORECASE)
            
            # Pattern 6: Bank of America - "CREDITS AND RETURNS"
            pattern_boa_returns = re.compile(r'credits\s+and\s+returns', re.IGNORECASE)
            
            # Pattern 7: Wells Fargo - "Transaction Details"
            pattern_wf_details = re.compile(r'transaction\s+details', re.IGNORECASE)
            
            # Find pages containing credit card transaction section headers
            matching_pages = []
            page_headers = {}  # Track which header was found on each page
            
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                # Check for credit card transaction section headers
                found_header = None
                
                if pattern_chase_activity.search(page_text):
                    found_header = "ACCOUNT ACTIVITY"
                elif pattern_boa_payments.search(page_text):
                    found_header = "PAYMENTS AND OTHER CREDITS"
                elif pattern_boa_purchases.search(page_text):
                    found_header = "PURCHASES AND OTHER CHARGES"
                elif pattern_boa_fees.search(page_text):
                    found_header = "FEES AND ADJUSTMENTS"
                elif pattern_boa_cash.search(page_text):
                    found_header = "CASH ADVANCES"
                elif pattern_boa_returns.search(page_text):
                    found_header = "CREDITS AND RETURNS"
                elif pattern_wf_details.search(page_text):
                    found_header = "Transaction Details"
                
                if found_header:
                    matching_pages.append(page_num)
                    page_headers[page_num] = found_header
                    log_message("info", f"Found '{found_header}' section on page {page_num + 1}")
            
            # Include continuation pages that have transaction data (dates and amounts)
            if matching_pages:
                last_included_page = max(matching_pages)
                
                # Check pages after the last header for continuation
                for page_num in range(last_included_page + 1, total_pages):
                    page = doc[page_num]
                    page_text = page.get_text("text")
                    
                    # Check if page has transaction-like content (dates MM/DD and amounts)
                    has_dates = bool(re.search(r'\d{1,2}/\d{1,2}', page_text))
                    has_amounts = bool(re.search(r'\$\d+[,\d]*\.\d{2}', page_text))
                    
                    # Stop if we hit summary sections
                    is_summary = bool(re.search(r'(total\s+fees\s+charged|total\s+interest\s+charged|year-to-date\s+totals|interest\s+charges|annual\s+percentage\s+rate|payment\s+information)', page_text, re.IGNORECASE))
                    
                    if has_dates and has_amounts and not is_summary:
                        matching_pages.append(page_num)
                        page_headers[page_num] = "continuation"
                        log_message("info", f"Including continuation page {page_num + 1} with transaction data")
                    else:
                        # Stop at first page without transactions
                        log_message("info", f"Stopping at page {page_num + 1} (no more transaction data)")
                        break
            
            if not matching_pages:
                log_message("warning", "No pages found with credit card transaction sections, returning original PDF")
                doc.close()
                return pdf_bytes
            
            log_message("info", f"Filtered to {len(matching_pages)} page(s) with credit card transaction sections: {[p+1 for p in matching_pages]}")
            
            # Create new PDF with only matching pages
            new_doc = fitz.open()
            for page_num in sorted(matching_pages):
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Save filtered PDF to bytes
            filtered_pdf_bytes = new_doc.tobytes()
            
            # Clean up
            new_doc.close()
            doc.close()
            
            log_message("info", f"Created filtered PDF with {len(matching_pages)} page(s), size: {len(filtered_pdf_bytes)} bytes")
            
            return filtered_pdf_bytes
            
        except Exception as e:
            log_message("error", f"Failed to filter credit card PDF pages: {e}")
            import traceback
            log_message("error", f"Traceback: {traceback.format_exc()}")
            log_message("warning", "Returning original PDF due to filtering error")
            return pdf_bytes
    
    @staticmethod
    def _load_prompt_from_file(prompt_filename: str) -> str:
        """
        Load prompt text from file in the prompts directory.
        
        Args:
            prompt_filename: Name of the prompt file to load
            
        Returns:
            Prompt text as string
        """
        try:
            # Get the directory where this file is located
            current_dir = Path(__file__).parent
            prompt_file_path = current_dir / "prompts" / prompt_filename
            
            # Read the prompt file
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            
            log_message("info", f"Loaded prompt from: {prompt_file_path}")
            return prompt_text
            
        except FileNotFoundError:
            log_message("error", f"Prompt file not found: {prompt_filename}")
            raise FileNotFoundError(f"Prompt file not found: {prompt_filename}")
        except Exception as e:
            log_message("error", f"Failed to load prompt file {prompt_filename}: {e}")
            raise Exception(f"Failed to load prompt file: {e}")
    
    def create_extraction_prompt(self, is_credit_card: bool = False) -> str:
        """
        Create prompt for Gemini to extract transactions from bank statement or credit card statement.
        
        Args:
            is_credit_card: Whether the statement is a credit card statement (default: False)
        
        Returns:
            Prompt string for transaction extraction
        """
        if is_credit_card:
            return self._create_credit_card_prompt()
        
        return self._load_prompt_from_file("bank_statement_extraction_prompt.txt")
    
    def _create_credit_card_prompt(self) -> str:
        """
        Create prompt for Gemini to extract transactions from credit card statement.
        
        Returns:
            Prompt string for credit card transaction extraction
        """
        return self._load_prompt_from_file("credit_card_extraction_prompt.txt")
    
    async def extract_from_pdf(self, pdf_bytes: bytes, is_credit_card: bool = False) -> Dict[str, Any]:
        """
        Extract transactions from PDF file using Gemini API directly.
        Filters PDF to include only pages containing transaction section headers (e.g., "Transaction history", 
        "Credits"/"Debits" for Wells Fargo format_01) before extraction.
        
        Args:
            pdf_bytes: PDF file bytes
            
        Returns:
            Dict with extracted transactions
        """
        try:
            log_message("info", f"Extracting transactions from PDF using Gemini (PDF size: {len(pdf_bytes)} bytes)")
            
            # Filter PDF to include only pages with "Transaction history" or credit card activity sections
            if is_credit_card:
                # For credit cards, filter for ACCOUNT ACTIVITY, Transaction Details, etc.
                filtered_pdf_bytes = self.filter_credit_card_pages(pdf_bytes)
            else:
                # For bank statements, filter for Transaction history
                filtered_pdf_bytes = self.filter_transaction_history_pages(pdf_bytes)
            
            if len(filtered_pdf_bytes) != len(pdf_bytes):
                log_message("info", f"PDF filtered: {len(pdf_bytes)} bytes -> {len(filtered_pdf_bytes)} bytes")
            else:
                log_message("info", "PDF filtering did not reduce size (all pages may contain transaction sections or filtering unavailable)")
            
            prompt = self.create_extraction_prompt(is_credit_card=is_credit_card)
            
            # Upload PDF to Gemini using temporary file
            log_message("info", "Preparing PDF for Gemini API")
            
            # Create a temporary file to upload (use filtered PDF)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(filtered_pdf_bytes)
                temp_file_path = temp_file.name
            rate_limit_backoff_seconds = [10, 20, 40]
            rate_limit_retries_used = 0
            response = None
            last_error = None

            try:
                max_attempts = 1 + len(rate_limit_backoff_seconds)
                for attempt in range(max_attempts):
                    uploaded_file = None
                    try:
                        # Upload PDF file to Gemini (requires file path)
                        log_message("info", "Uploading PDF to Gemini File API")
                        uploaded_file = await asyncio.to_thread(
                            self.client.files.upload,
                            file=temp_file_path  # google-genai accepts path/bytes; mime type inferred
                        )
                        
                        log_message("info", f"PDF uploaded to Gemini. File name: {uploaded_file.name}")
                        
                        # Wait for file to be processed
                        import time
                        max_wait = 30  # Maximum 30 seconds wait
                        wait_time = 0
                        while uploaded_file.state.name == "PROCESSING" and wait_time < max_wait:
                            log_message("info", f"Waiting for file to be processed... ({wait_time}s)")
                            time.sleep(2)
                            wait_time += 2
                            uploaded_file = self.client.files.get(name=uploaded_file.name)
                        
                        if uploaded_file.state.name == "FAILED":
                            raise Exception("File upload failed")
                        
                        if uploaded_file.state.name == "PROCESSING":
                            log_message("warning", "File still processing, proceeding anyway...")
                        
                        # Invoke Gemini with uploaded PDF file
                        log_message("info", "Invoking Gemini API with uploaded PDF file")
                        response = await asyncio.to_thread(
                            self.client.models.generate_content,
                            model=self.model_name,
                            contents=[prompt, uploaded_file]
                        )
                        break
                    except Exception as call_error:
                        last_error = call_error
                        if _is_quota_error(call_error) and attempt < len(rate_limit_backoff_seconds):
                            wait_seconds = rate_limit_backoff_seconds[attempt]
                            rate_limit_retries_used += 1
                            log_message(
                                "warning",
                                f"Gemini PDF extraction hit rate limit. Retrying in {wait_seconds} seconds "
                                f"(attempt {attempt + 1}/{max_attempts - 1})."
                            )
                            await asyncio.sleep(wait_seconds)
                            continue
                        raise
                    finally:
                        # Clean up uploaded file
                        if uploaded_file is not None:
                            try:
                                self.client.files.delete(name=uploaded_file.name)
                                log_message("info", "Cleaned up uploaded PDF file from Gemini")
                            except Exception as cleanup_error:
                                log_message("warning", f"Failed to cleanup uploaded file: {cleanup_error}")

                if response is None:
                    if last_error:
                        raise last_error
                    raise Exception("Gemini PDF extraction did not return a response")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    log_message("warning", f"Failed to delete temporary file: {e}")
            
            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            response_text = response_text.strip()
            
            log_message("info", f"Gemini response received: {len(response_text)} characters")
            log_message("debug", f"Gemini response preview: {response_text[:500]}")
            
            # Save extracted text to file
            self._save_extracted_text(response_text)
            
            # Parse JSON from response
            transactions_data = self._parse_json_response(response_text)
            transactions_data["rate_limit_retries_used"] = rate_limit_retries_used
            
            # Save JSON to local file
            self._save_json_to_file(transactions_data)
            
            return transactions_data
            
        except Exception as e:
            log_message("error", f"Gemini extraction failed: {e}")
            import traceback
            log_message("error", f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Gemini PDF extraction failed: {e}")
    
    @staticmethod
    def _save_extracted_text(text: str, filename: Optional[str] = None) -> str:
        """
        Save Gemini extracted text to a file.
        
        Args:
            text: Text content to save
            filename: Optional custom filename (if None, uses timestamp)
            
        Returns:
            Path to saved file
        """
        try:
            # Create categorized output directory if it doesn't exist
            output_dir = Path("data/output/gemini_text")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gemini_extracted_text_{timestamp}.txt"
            
            # Ensure .txt extension
            if not filename.endswith('.txt'):
                filename = f"{filename}.txt"
            
            file_path = output_dir / filename
            
            # Save text to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Gemini PDF Extraction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(text)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write(f"Total characters: {len(text)}\n")
                f.write("=" * 80 + "\n")
            
            log_message("info", f"Saved Gemini extracted text to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            log_message("error", f"Failed to save extracted text to file: {e}")
            # Don't raise, just log the error
            return ""
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from Gemini response, handling markdown code blocks.
        
        Args:
            response_text: Raw response text from Gemini
            
        Returns:
            Parsed JSON dict
        """
        try:
            # Try parsing directly
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
            # Try extracting from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try finding first balanced JSON object
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
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                continue
            
            raise json.JSONDecodeError("Could not extract valid JSON from Gemini response", response_text, 0)
            
        except Exception as e:
            log_message("error", f"Failed to parse JSON from Gemini response: {e}")
            log_message("error", f"Response text: {response_text[:1000]}")
            raise Exception(f"Failed to parse Gemini response as JSON: {e}")
    
    def _save_json_to_file(self, data: Dict[str, Any]) -> str:
        """
        Save extracted JSON data to a local file.
        
        Args:
            data: Extracted transaction data as dict
        
        Returns:
            Path to saved JSON file
        """
        try:
            # Create categorized output directory if it doesn't exist
            output_dir = Path("data/output/gemini_json")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_extracted_transactions_{timestamp}.json"
            file_path = output_dir / filename
            
            # Save JSON to file with proper formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log_message("info", f"Saved Gemini extracted JSON to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            log_message("error", f"Failed to save JSON to file: {e}")
            return ""
    
    async def extract_transactions(self, file, is_credit_card: bool = False) -> Dict[str, Any]:
        """
        Main method to extract transactions from PDF file.
        
        Args:
            file: UploadFile object containing PDF
            is_credit_card: Whether the PDF is a credit card statement (default: False)
            
        Returns:
            Dict with extracted transactions in standard format
        """
        try:
            statement_type = "credit card" if is_credit_card else "bank"
            log_message("info", f"Starting Gemini-based PDF transaction extraction for {statement_type} statement")
            
            # Get PDF bytes
            pdf_bytes = self.get_pdf_bytes(file)
            
            if not pdf_bytes:
                raise Exception("No PDF data found")
            
            # Extract transactions using Gemini with filtered PDF
            result = await self.extract_from_pdf(pdf_bytes, is_credit_card=is_credit_card)
            
            # Validate and normalize the result
            if not result.get("transactions"):
                log_message("warning", "No transactions found in Gemini extraction result")
                result["transactions"] = []
            result["rate_limit_retries_used"] = int(result.get("rate_limit_retries_used") or 0)
            
            # Ensure all transactions have proper structure
            normalized_transactions = []
            for txn in result.get("transactions", []):
                # Ensure credit and debit are properly set
                credit = txn.get("credit")
                debit = txn.get("debit")
                
                # Convert to float if string
                if credit and isinstance(credit, str):
                    try:
                        credit = float(re.sub(r'[^\d.]', '', credit))
                    except:
                        credit = None
                
                if debit and isinstance(debit, str):
                    try:
                        debit = float(re.sub(r'[^\d.]', '', debit))
                    except:
                        debit = None
                
                # Ensure only one is set
                if credit and debit:
                    log_message("warning", f"Transaction has both credit and debit, using credit: {txn.get('description')}")
                    debit = None
                
                normalized_txn = {
                    "date": txn.get("date"),
                    "post_date": txn.get("post_date"),
                    "description": txn.get("description", "").strip(),
                    "credit": credit,
                    "debit": debit,
                    "bank": txn.get("bank") or result.get("bank", "Unknown"),
                    # Preserve account_number from transaction (for multi-account statements)
                    "account_number": txn.get("account_number") or result.get("account_number"),
                    # Preserve any payee_name returned by the model/prompt; fallback to None
                    "payee_name": txn.get("payee_name")
                }
                
                # Only add if it has a valid amount
                if normalized_txn["credit"] or normalized_txn["debit"]:
                    normalized_transactions.append(normalized_txn)
            
            result["transactions"] = normalized_transactions
            
            log_message("info", f"Successfully extracted {len(normalized_transactions)} transactions using Gemini")
            
            return result
            
        except Exception as e:
            log_message("error", f"Gemini transaction extraction failed: {e}")
            # Re-raise quota/429 errors so the API can return quota error message (not empty data).
            if _is_quota_error(e):
                raise
            # Other errors: return safe empty result so caller can continue.
            return {
                "bank": "Unknown",
                "account_number": None,
                "start_date": None,
                "end_date": None,
                "rate_limit_retries_used": 0,
                "transactions": [],
            }
