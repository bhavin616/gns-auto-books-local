from typing import Dict, Any, Callable, Awaitable
from backend.services.bank_services import (
    UniversalBankService,
    UniversalCreditCardService
)
from backend.utils.logger import log_message
from backend.utils.file_processor import FileProcessor


class TransactionExtractor:
    """Main extraction router for multi-bank transaction formats."""

    # ============================================================
    #   BANK FORMAT → EXTRACTION HANDLERS (DISPATCH TABLES)
    # ============================================================

    WELLS_FARGO_HANDLERS = {
        "format_01": [
            ("universal", UniversalBankService.extract_transactions_universal),
        ],
        "format_02": [
            ("universal", UniversalBankService.extract_transactions_universal),
        ],
        "credit_card": [
            ("universal", UniversalCreditCardService.extract_credit_card_transactions_universal),
        ],
    }

    CHASE_HANDLERS = {
        "format_01": [
            ("universal", UniversalBankService.extract_transactions_universal),
        ],
        "credit_card": [
            ("universal", UniversalCreditCardService.extract_credit_card_transactions_universal),
        ],
    }

    BOA_HANDLERS = {
        "format_01": [
            ("universal", UniversalBankService.extract_transactions_universal),
        ],
        "credit_card": [
            ("universal", UniversalCreditCardService.extract_credit_card_transactions_universal),
        ],
    }

    BANK_ROUTER = {
        "Wells Fargo": WELLS_FARGO_HANDLERS,
        "Chase Bank": CHASE_HANDLERS,
        "Bank of America": BOA_HANDLERS,
    }

    # ============================================================
    #                     EXTRACTION LOGIC
    # ============================================================

    @staticmethod
    async def call(detection_result: Dict[str, Any], file) -> Dict[str, Any]:
        """Routes bank statements to the correct service extractors.

        Args:
            detection_result: Output from detect_bank() with keys:
                - bank: bank name
                - format: detected document format
            file: Uploaded file for transaction extraction.

        Returns:
            Dict containing extracted data, path to saved CSV, and count.

        Raises:
            Exception: If extraction fails at any stage.
        """
        try:
            bank = detection_result.get("bank")
            format_name = detection_result.get("format")

            if not bank or not format_name:
                raise ValueError("Detection result missing 'bank' or 'format'.")

            if bank not in TransactionExtractor.BANK_ROUTER:
                raise ValueError(f"Unsupported bank '{bank}'.")

            handlers = TransactionExtractor.BANK_ROUTER[bank].get(format_name)
            if not handlers:
                raise ValueError(f"Unsupported format '{format_name}' for bank '{bank}'.")

            data = None
            last_error = None

            # Try handlers in the order of priority
            for read_type, handler in handlers:

                log_message(
                    "info",
                    f"Running handler {handler.__name__} for bank={bank}, format={format_name}"
                )

                try:
                    # Handle universal Gemini extraction (takes file directly)
                    if read_type == "universal":
                        # Universal function takes file and bank_name directly
                        data = await handler(file, bank)
                        
                        # Log what was returned for debugging
                        if data:
                            transaction_count = len(data.get("transactions", []))
                            log_message(
                                "info",
                                f"Handler {handler.__name__} returned {transaction_count} transactions"
                            )
                            
                            # Check if we got valid data with transactions
                            if transaction_count > 0:
                                break
                            else:
                                log_message(
                                    "warning",
                                    f"Handler {handler.__name__} returned data but with 0 transactions. "
                                    f"Result keys: {list(data.keys()) if data else 'None'}"
                                )
                        else:
                            log_message(
                                "warning",
                                f"Handler {handler.__name__} returned None or empty data"
                            )
                except Exception as handler_error:
                    log_message(
                        "error",
                        f"Handler {handler.__name__} raised exception: {handler_error}"
                    )
                    last_error = handler_error
                    # Continue to next handler if available
                    continue

            # Check if we got valid data with transactions
            if not data:
                error_msg = f"No data returned from any handler for bank='{bank}', format='{format_name}'."
                if last_error:
                    error_msg += f" Last error: {last_error}"
                raise ValueError(error_msg)
            
            transaction_count = len(data.get("transactions", []))
            if transaction_count == 0:
                # Provide more detailed error message
                error_msg = (
                    f"No transactions extracted for bank='{bank}', format='{format_name}'. "
                    f"Extraction completed but returned 0 transactions. "
                    f"This could mean: (1) The PDF has no transactions in the selected period, "
                    f"(2) The PDF format is not recognized, or (3) The extraction failed silently. "
                    f"Result structure: {list(data.keys()) if data else 'None'}"
                )
                log_message("error", error_msg)
                raise ValueError(error_msg)

            # Save results
            output_path = await FileProcessor.save_transactions_to_csv(data=data)

            response = {
                "output_path": output_path,
                "no_of_transactions": len(data.get("transactions", [])),
                "extracted_transactions": data
            }

            log_message("info", f"Extraction successful for bank={bank}, format={format_name}")

            return response

        except Exception as e:
            log_message("error", f"Transaction extraction failed: {e}")
            raise Exception(f"Extraction of Transactions failed: {e}") from e
