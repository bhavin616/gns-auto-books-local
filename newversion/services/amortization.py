"""
Amortization Schedule Service

Helper service to calculate loan amortization schedules using numpy-financial.
"""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import parser as date_parser
from typing import List, Dict, Any, Optional
import re
from backend.services.mongodb import get_database
from backend.utils.logger import log_message


def generate_schedule(
    principal: float,
    annual_rate: float,
    months: int,
    start_date: datetime,
    monthly_payment: float
) -> List[Dict[str, Any]]:
    """
    Generate an amortization schedule for a loan.
    
    Args:
        principal: The loan principal amount
        annual_rate: Annual interest rate as a decimal (e.g., 0.05 for 5%)
        months: Total number of months for the loan term
        start_date: The start date of the loan (first payment date)
    
    Returns:
        List of dictionaries, each containing:
        - payment_date: datetime object for the payment date
        - total_payment: Total payment amount for the period
        - interest_expense: Interest portion of the payment
        - principal_liability: Principal portion of the payment
        - ending_balance: Remaining balance after the payment
    """
    # Convert annual rate to monthly rate
    monthly_rate = annual_rate / 12.0
    pmt = -monthly_payment  # cash outflow

    schedule = []

    for month in range(1, months + 1):

        interest = principal * monthly_rate
        principal_paid = monthly_payment - interest

        # Final adjustment
        if month == months:
            principal_paid = principal
            monthly_payment = interest + principal_paid

        ending_balance = principal - principal_paid

        schedule.append({
            "payment_date": start_date + relativedelta(months=month - 1),
            "total_payment": round(monthly_payment, 2),
            "interest_expense": round(interest, 2),
            "principal_liability": round(principal_paid, 2),
            "ending_balance": round(max(ending_balance, 0), 2)
        })

        principal = ending_balance

    return schedule

async def find_loan_by_transaction(
    transaction_date: datetime,
    memo: str,
    amount: float,
    client_id: str
) -> Optional[Dict[str, Any]]:
    """
    Find a loan from MongoDB based on transaction details.
    
    This function searches for loans by matching:
    - Client ID (required)
    - Account ID extracted from memo
    - Transaction amount with loan payment amount (exact match)
    
    Args:
        transaction_date: Date of the transaction
        memo: Transaction memo/description (must contain account ID/number)
        amount: Transaction amount
        client_id: Client ID to filter loans
    
    Returns:
        Dictionary containing:
        - loan: The matched loan document from MongoDB
        - schedule: Amortization schedule if loan details are sufficient
        - match_reason: Explanation of why this loan was matched
        Returns None if no matching loan is found
    """
    try:
        from bson import ObjectId
        
        db = get_database()
        collection = db["client_loans"]
        
        # Convert client_id to ObjectId for querying
        try:
            client_object_id = ObjectId(client_id)
        except Exception as e:
            log_message("error", f"Invalid client_id format: {client_id}, error: {e}")
            return None
        
        # Extract all account IDs from memo
        # Look for patterns like "ID:XXXXX" or "CO ID:XXXXX"
        account_ids = []
        if memo:
            # Extract IDs following "ID:" or "CO ID:" patterns (alphanumeric after colon)
            # Pattern matches: ID: followed by alphanumeric characters
            id_patterns = re.findall(r'(?:CO\s+)?ID:([A-Z0-9]+)', memo, re.IGNORECASE)
            if id_patterns:
                account_ids = [id_pattern.strip() for id_pattern in id_patterns if id_pattern.strip()]
                log_message("info", f"Extracted account IDs from memo: {account_ids}")
            
            # Fallback: if no ID: pattern found, look for numeric patterns (4+ digits)
            if not account_ids:
                numeric_patterns = re.findall(r'\d{4,}', memo)
                if numeric_patterns:
                    account_ids = numeric_patterns
                    log_message("info", f"Extracted numeric account IDs from memo: {account_ids}")
        
        if not account_ids:
            log_message("warning", f"Could not extract account ID from memo: {memo}")
            return None
        
        # Build query with AND conditions (must match all)
        # Use $or for account_no to match any of the extracted account IDs
        account_no_conditions = []
        for acc_id in account_ids:
            account_no_conditions.append({
                "account_no": {
                    "$regex": acc_id,
                    "$options": "i"
                }
            })
        
        # Use $and to combine regular field queries with $expr
        final_query = {
            "$and": [
                {"client_id": client_object_id},
                {
                    "$or": account_no_conditions
                },
                {
                    "$expr": {
                        "$eq": [
                            {"$toDouble": {"$ifNull": ["$payment", "0"]}},
                            amount
                        ]
                    }
                }
            ]
        }

        log_message("info", f"Executing loan query: {final_query}")
        
        # Execute query
        loans = await collection.find(final_query).to_list(length=100)
        
        if not loans:
            log_message("info", f"No loans found matching: client_id={client_id}, account_ids={account_ids}, amount={amount}")
            return None
        
        # If multiple matches, find the best match based on exact account ID match and payment amount
        best_match = None
        best_score = 0
        
        for loan in loans:
            score = 0
            match_reasons = []
            
            # Score by exact account ID match (highest priority)
            # Check against all extracted account IDs and use the best match
            loan_account = str(loan.get("account_no", ""))
            matched_account_id = None
            account_match_found = False
            
            # First, check for exact matches (highest priority)
            for acc_id in account_ids:
                if loan_account == acc_id:
                    score += 20
                    matched_account_id = acc_id
                    match_reasons.append(f"exact_account_id_match:{acc_id}")
                    account_match_found = True
                    break
            
            # If no exact match, check for contains match
            if not account_match_found:
                for acc_id in account_ids:
                    if acc_id in loan_account:
                        score += 15
                        matched_account_id = acc_id
                        match_reasons.append(f"account_id_contains_match:{acc_id}")
                        account_match_found = True
                        break
            
            # Score by payment amount match (exact match)
            try:
                loan_payment = float(str(loan.get("payment", 0) or 0).replace(",", "").replace("$", "").strip())
                if loan_payment == amount:
                    score += 10
                    match_reasons.append("exact_payment_amount_match")
            except (ValueError, TypeError):
                pass
            
            # Score by client ID match (should always match, but verify)
            loan_client_id = loan.get("client_id")
            if loan_client_id:
                if str(loan_client_id) == str(client_object_id) or str(loan_client_id) == client_id:
                    score += 10
                    match_reasons.append("client_id_match")
            
            if score > best_score:
                best_score = score
                best_match = loan
                best_match["_match_reasons"] = match_reasons
        
        log_message("info", f"Best match score: {best_score} for loan: {best_match.get('_id') if best_match else 'None'}")
        if not best_match:
            log_message("warning", f"Found {len(loans)} loans but none scored high enough")
            return None
        
        # Convert ObjectId to string for JSON serialization
        if "_id" in best_match:
            best_match["_id"] = str(best_match["_id"])
        if "client_id" in best_match and hasattr(best_match["client_id"], "__str__"):
            best_match["client_id"] = str(best_match["client_id"])
        
        # Try to generate amortization schedule if we have sufficient data
        schedule = None
        try:
            funding_provided = best_match.get("funding_provided")
            annual_rate_str = best_match.get("annual_percentage_rate", "")
            start_date_str = best_match.get("start_date")
            
            if funding_provided and annual_rate_str and start_date_str:
                # Parse funding amount
                principal = float(str(funding_provided).replace(",", "").replace("$", "").strip())
                
                # Parse annual rate (handle percentage format)
                rate_str = str(annual_rate_str).replace("%", "").strip()
                annual_rate = float(rate_str) / 100.0
                
                # Parse start date
                if isinstance(start_date_str, str):
                    loan_start = date_parser.parse(start_date_str)
                else:
                    loan_start = start_date_str
                
                # Estimate months from total_payment_amount and payment
                # Or use a default if not available
                payment_str = best_match.get("payment", "0")
                try:
                    monthly_payment = float(str(payment_str).replace(",", "").replace("$", "").strip())
                    if monthly_payment > 0:
                        # Estimate months: total_payment_amount / monthly_payment
                        total_payment_str = best_match.get("total_payment_amount", "0")
                        total_payment = float(str(total_payment_str).replace(",", "").replace("$", "").strip())
                        if total_payment > 0:
                            months = int(round(total_payment / monthly_payment))
                        else:
                            months = 60  # Default to 5 years
                    else:
                        months = 60
                except (ValueError, TypeError):
                    months = 60
                
                log_message("info", f"Generating schedule: principal={principal}, annual_rate={annual_rate}, months={months}, start_date={loan_start}, monthly_payment={monthly_payment}")
                # Generate full schedule
                full_schedule = generate_schedule(principal, annual_rate, months, loan_start, monthly_payment)
                log_message("info", f"Generated amortization schedule for loan {best_match.get('_id')} with {len(full_schedule)} entries")
                
                # Filter schedule to only return the entry for the transaction month
                # Match by year and month of the transaction_date
                transaction_year_month = (transaction_date.year, transaction_date.month)
                schedule_entry = None
                
                for entry in full_schedule:
                    payment_date = entry.get("payment_date")
                    if isinstance(payment_date, str):
                        payment_date = date_parser.parse(payment_date)
                    if payment_date.year == transaction_date.year and payment_date.month == transaction_date.month:
                        schedule_entry = entry
                        log_message("info", f"Found schedule entry for transaction month: {transaction_date.strftime('%Y-%m')} -> {payment_date}")
                        break
                
                if schedule_entry:
                    schedule = schedule_entry
                else:
                    log_message("warning", f"No schedule entry found for transaction date: {transaction_date.strftime('%Y-%m-%d')}")
                    schedule = None
        except Exception as e:
            log_message("warning", f"Could not generate schedule for loan: {e}")
        
        result = {
            "loan": best_match,
            "schedule": schedule,
            "match_reason": ", ".join(best_match.get("_match_reasons", []))
        }
        
        log_message("info", f"Found matching loan: {best_match.get('_id')} with score {best_score}")
        return result
        
    except Exception as e:
        log_message("error", f"Error finding loan by transaction: {e}")
        return None

