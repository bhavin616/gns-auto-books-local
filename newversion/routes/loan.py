from fastapi import APIRouter, UploadFile, File, status,  Form
from fastapi.responses import JSONResponse
from newversion.interactors import loan as loan_interactors
from backend.utils.logger import log_message
from datetime import datetime
from newversion.services.rule_base_mongodb import validate_client_exists

router = APIRouter(prefix="/api/v2/loan-amortization", tags=["Loan"])

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

@router.post("/extract_loan_details")
async def extract_loan_details(
    file: UploadFile = File(...),
    client_id: str = Form(...)
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
    
    Returns:
        JSON response with extracted loan details and MongoDB record ID.
    """
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
        client_exists = await validate_client_exists(client_id)
        if not client_exists:
            return create_response(
                False,
                f"Client with ID '{client_id}' does not exist. Please create the client first.",
                code=status.HTTP_400_BAD_REQUEST
            )
        
        # Extract loan details
        result = await loan_interactors.extract_loan_details(file, client_id)
        
        if result.get("status") == "failed":
            return create_response(
                False,
                result.get("error", "Failed to extract loan details"),
                code=status.HTTP_400_BAD_REQUEST
            )
        
        return create_response(
            True,
            "Successfully extracted loan details and saved to MongoDB",
            result.get("data", {})
        )
        
    except Exception as e:
        log_message("error", f"Loan extraction endpoint failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return create_response(
            False,
            f"Error extracting loan details: {e}",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

