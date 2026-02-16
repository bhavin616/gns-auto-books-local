"""
Organizer PDF Routes - FastAPI Router

Endpoints for Tax Organizer PDF extraction and mapping to categories from MongoDB.
"""

import re
from fastapi import APIRouter, UploadFile, File, status
from fastapi.responses import JSONResponse

from backend.utils.logger import log_message
from newversion.services.organizer_extractor import OrganizerExtractor

router = APIRouter(prefix="/api", tags=["Organizer"])


def create_response(success: bool, message: str, data=None, code=status.HTTP_200_OK):
    """Return JSON in the format expected by the client (success, message, data)."""
    return JSONResponse(
        content={
            "success": success,
            "message": message,
            "data": data,
        },
        status_code=code,
    )


@router.post("/organizer/extract")
async def extract_organizer_pdf(file: UploadFile = File(...)):
    """
    Extract document labels and client details from a Tax Organizer PDF,
    then map each document label to categories/subcategories from MongoDB using Gemini.

    - **file**: Tax Organizer PDF file.

    Returns:
        - success: bool
        - message: str
        - data:
            - mapped: list of { documentLabel, categoryId, categoryName, subcategoryId, subcategoryName, priority }
            - unmatched: list of document labels that could not be mapped
            - clientDetails: { name, spouseName, ssn, address, phone, email, taxYear }
    """
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return create_response(
                False,
                "Please upload a PDF file.",
                data=None,
                code=status.HTTP_400_BAD_REQUEST,
            )
        log_message("info", f"Organizer extract request: file={file.filename}")
        extractor = OrganizerExtractor()
        result = await extractor.extract_and_map(file)
        data = result.get("data")
        if data and data.get("mapped") is not None:
            exclude_label = "Last year's income tax return(s)"
            exclude_subcategory_re = re.compile(
                r"last\s+year['\s]*(?:s)?\s*income\s+tax\s+return",
                re.IGNORECASE,
            )

            def should_exclude(m: dict) -> bool:
                label = m.get("documentLabel") or ""
                sub_name = m.get("subcategoryName") or ""
                if exclude_label in label:
                    return True
                if exclude_subcategory_re.search(sub_name):
                    return True
                return False

            data = {
                **data,
                "mapped": [m for m in data["mapped"] if not should_exclude(m)],
            }
        return create_response(
            result.get("success", True),
            result.get("message", "Organizer PDF parsed and mapped successfully."),
            data,
        )
    except ValueError as e:
        log_message("warning", f"Organizer extract validation error: {e}")
        return create_response(False, str(e), data=None, code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        log_message("error", f"Organizer extract failed: {e}")
        import traceback
        log_message("error", traceback.format_exc())
        return create_response(
            False,
            f"Error processing organizer PDF: {e}",
            data=None,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
