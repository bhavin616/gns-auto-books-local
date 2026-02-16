import re
from typing import List, Dict, Optional, Tuple, Any
from backend.utils.logger import log_message
from backend.utils.parser_helper import ParserHelper
from backend.services.gemini_pdf_extractor import GeminiPDFExtractor


class UniversalBankService:
    """Universal bank transaction extraction service using Gemini for all banks and formats."""

    @staticmethod
    async def extract_transactions_universal(file, bank_name: str) -> Dict[str, Any]:
        """
        Universal extraction function that uses Gemini to extract transactions from any bank statement.
        
        This function works for all banks (Wells Fargo, Chase, Bank of America, etc.) 
        and all formats (checking, savings, etc.) except credit cards.
        
        Args:
            file: UploadFile object containing PDF bank statement
            bank_name: Name of the bank (for metadata)
            
        Returns:
            Dict with extracted transactions in standard format:
            {
                "bank": str,
                "account_number": str | None,
                "start_date": str | None,
                "end_date": str | None,
                "transactions": List[Dict]
            }
        """
        try:
            log_message("info", f"Using Gemini universal extraction for bank: {bank_name}")
            
            # Initialize Gemini extractor
            extractor = GeminiPDFExtractor()
            
            # Extract transactions using Gemini
            result = await extractor.extract_transactions(file)
            
            # Ensure bank name is set
            if result.get("bank"):
                result["bank"] = bank_name
            else:
                result["bank"] = bank_name
            
            # Ensure all transactions have bank name
            for txn in result.get("transactions", []):
                if not txn.get("bank"):
                    txn["bank"] = bank_name
            
            log_message("info", f"Universal extraction completed: {len(result.get('transactions', []))} transactions extracted")
            
            return result
            
        except Exception as e:
            log_message("error", f"Universal extraction failed for {bank_name}: {e}")
            raise Exception(f"Universal extraction failed for {bank_name}: {e}")


class UniversalCreditCardService:
    """Universal credit card transaction extraction service using Gemini for all banks."""

    @staticmethod
    async def extract_credit_card_transactions_universal(file, bank_name: str) -> Dict[str, Any]:
        """
        Universal extraction function that uses Gemini to extract transactions from any credit card statement.
        
        This function works for all banks (Wells Fargo, Chase, Bank of America, etc.)
        and automatically identifies credits and debits based on statement patterns:
        - Chase: Uses ACCOUNT ACTIVITY section, + for credits, - for debits
        - Bank of America: Identifies transaction types by sections (purchases/charges, payments/credits)
        - Wells Fargo: Credits are positive amounts, charges/debits are negative
        
        Args:
            file: UploadFile object containing PDF credit card statement
            bank_name: Name of the bank (for metadata)
            
        Returns:
            Dict with extracted transactions in standard format:
            {
                "bank": str,
                "account_number": str | None,
                "start_date": str | None,
                "end_date": str | None,
                "transactions": List[Dict]
            }
        """
        try:
            log_message("info", f"Using Gemini universal extraction for credit card: {bank_name}")
            
            # Initialize Gemini extractor with credit card specific prompt
            extractor = GeminiPDFExtractor()
            
            # Extract transactions using Gemini with credit card context
            result = await extractor.extract_transactions(file, is_credit_card=True)
            
            # Ensure bank name is set
            if result.get("bank"):
                result["bank"] = bank_name
            else:
                result["bank"] = bank_name
            
            # Ensure all transactions have bank name and proper credit/debit classification
            for txn in result.get("transactions", []):
                if not txn.get("bank"):
                    txn["bank"] = bank_name
                
                # Normalize credit/debit based on amount signs and keywords
                amount = txn.get("amount")
                description = txn.get("description", "").lower()
                
                # Handle credit/debit classification
                if amount is not None:
                    # Payments, returns, and credits are positive (credit)
                    if any(keyword in description for keyword in ["payment", "credit", "return", "refund"]):
                        txn["credit"] = abs(amount)
                        txn["debit"] = None
                    # Purchases, charges, fees are negative (debit)
                    elif any(keyword in description for keyword in ["purchase", "charge", "fee", "interest"]):
                        txn["credit"] = None
                        txn["debit"] = abs(amount)
                    # Use sign convention: positive = credit, negative = debit
                    elif amount >= 0:
                        txn["credit"] = amount
                        txn["debit"] = None
                    else:
                        txn["credit"] = None
                        txn["debit"] = abs(amount)
            
            log_message("info", f"Universal credit card extraction completed: {len(result.get('transactions', []))} transactions extracted")
            
            return result
            
        except Exception as e:
            log_message("error", f"Universal credit card extraction failed for {bank_name}: {e}")
            raise Exception(f"Universal credit card extraction failed for {bank_name}: {e}")

