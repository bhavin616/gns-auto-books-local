"""
Bank Detector Module

Provides utility to detect the bank and statement format from a PDF file.
"""

import io
from typing import Dict
import pdfplumber
from backend.utils.logger import log_message


async def detect_bank(file) -> Dict[str, str]:
    """
    Detects the bank and statement format from a PDF file.

    Args:
        file: UploadFile object containing the PDF bank statement.

    Returns:
        Dict[str, str]: A dictionary with keys:
            - "bank": Name of the bank or "unknown"
            - "format": Statement format (e.g., "format_01", "format_02", "credit_card")
    """
    try:
        # Read PDF bytes
        pdf_bytes = await file.read()
        file.file.seek(0)
        pdf_stream = io.BytesIO(pdf_bytes)

        # Extract text from all pages
        all_lines = []
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                all_lines.extend(lines)

        # Normalize text once
        text_lower = "\n".join(all_lines).lower()
        header_lower = "\n".join(all_lines[:50]).lower()  # Use first 50 lines as header

        # ------------------------------
        # BANK DETECTION LOGIC
        # Check Wells Fargo FIRST (more specific patterns) before Chase Bank
        # ------------------------------

        # Wells Fargo - Check FIRST to avoid false matches with Chase
        wf_core = ["wells fargo", "wellsfargo.com", "wells fargo bank, n.a."]
        wf_in_header = any(k in header_lower for k in wf_core)
        wf_in_body = any(k in text_lower for k in wf_core)
        if wf_in_header or wf_in_body:
            wf_cc_keys = [
                "transaction details",
                "annual percentage rate",
                "finance charges",
                "cash advances",
                "minimum payment",
                "transaction finance charges"
            ]
            wf_02_keys = [
                "primary account number",
                "ending daily balance summary",
                "service charge summary",
                "interest summary",
                "deposits and withdrawals summary"
            ]
            if any(k in text_lower for k in wf_cc_keys):
                return {"bank": "Wells Fargo", "format": "credit_card"}
            if any(k in text_lower for k in wf_02_keys):
                return {"bank": "Wells Fargo", "format": "format_02"}
            return {"bank": "Wells Fargo", "format": "format_01"}

        # Bank of America
        boa_core = [
            "bank of america",
            "bankofamerica.com",
            "bank of america, n.a.",
            "p.o. box 15284",
            "wilmington, de"
        ]
        if any(k in header_lower for k in boa_core):
            boa_format01_keys = [
                "business advantage fundamentals",
                "business advantage relationship",
                "business checking",
                "deposit balances"
            ]
            boa_cc_keys = [
                "fees charged",
                "interest charged",
                "totals year-to-date",
                "transaction finder",
                "rewards summary",
                "payment information",
                "minimum payment"
            ]
            if any(k in text_lower for k in boa_cc_keys):
                return {"bank": "Bank of America", "format": "credit_card"}
            if any(k in text_lower for k in boa_format01_keys):
                return {"bank": "Bank of America", "format": "format_01"}
            return {"bank": "Bank of America", "format": "format_01"}  # fallback

        # Chase Bank - Check AFTER Wells Fargo to avoid false matches
        chase_core = ["jpmorgan chase", "chase.com", "chase bank"]
        if any(k in header_lower for k in chase_core):
            chase_cc_keys = [
                "transactions this cycle",
                "late fee warning",
                "total interest charged",
                "reference:"
            ]
            if any(k in text_lower for k in chase_cc_keys):
                return {"bank": "Chase Bank", "format": "credit_card"}
            return {"bank": "Chase Bank", "format": "format_01"}

        # Unknown bank
        return {"bank": "unknown"}

    except Exception as e:
        log_message("error", f"Bank detection failed: {e}")
        raise RuntimeError(f"Bank detection failed: {e}")
