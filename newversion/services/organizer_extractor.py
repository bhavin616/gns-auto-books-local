"""
Organizer PDF Extractor Service

Uploads the user's PDF directly to the LLM (Gemini), passes categories/subcategories
from MongoDB in the prompt, and gets back the full result in one response:
mapped, unmatched, clientDetails. Uses gemini-3-flash-preview.
"""

import io
import os
import re
import json
import asyncio
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

from backend.utils.logger import log_message
from newversion.services.rule_base_mongodb import get_organizer_categories

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
    log_message("warning", "PyMuPDF not installed. Install with: pip install PyMuPDF")


ORGANIZER_MODEL = "gemini-3-flash-preview"
ORGANIZER_CHECKLIST_MARKER = "CLIENT ORGANIZER CHECKLIST"


def _load_prompt(prompt_filename: str) -> str:
    """Load prompt text from newversion/prompts/."""
    current_dir = Path(__file__).parent.parent  # newversion/
    path = current_dir / "prompts" / prompt_filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_json_from_response(text: str) -> Dict[str, Any]:
    """Parse JSON from model response, handling markdown code blocks."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("Could not extract JSON from response")


def _trim_pdf_from_organizer_checklist(pdf_bytes: bytes) -> bytes:
    """
    Find the first page containing "CLIENT ORGANIZER CHECKLIST" and return a PDF
    containing only from that page to the end. Raises ValueError if marker not found.
    If PyMuPDF unavailable, returns the original bytes. Runs in thread (blocking).
    """
    if not FITZ_AVAILABLE or not pdf_bytes:
        return pdf_bytes
    try:
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError("Invalid PDF: no pages in document.")
        marker = ORGANIZER_CHECKLIST_MARKER
        start_page = None
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            if marker in text:
                start_page = page_num
                log_message("info", f"Found '{marker}' on page {page_num + 1}, trimming to pages {page_num + 1}-{total_pages}")
                break
        if start_page is None:
            doc.close()
            raise ValueError(f"Invalid PDF: 'Please upload a valid Tax Organizer PDF file.'")
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page, to_page=total_pages - 1)
        trimmed = new_doc.tobytes()
        new_doc.close()
        doc.close()
        log_message("info", f"Trimmed PDF: {total_pages - start_page} page(s), {len(trimmed)} bytes")
        return trimmed
    except ValueError:
        raise  # Invalid PDF (marker not found / no pages) — let route return 400
    except Exception as e:
        log_message("error", f"PDF trim failed: {e}")
        return pdf_bytes


def _upload_pdf_and_wait_ready(client, tmp_path: str):
    """Upload PDF to Gemini and poll until ready. Runs in thread. Returns uploaded file."""
    uploaded = client.files.upload(file=tmp_path)
    for _ in range(15):
        uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "PROCESSING":
            break
        time.sleep(2)
    if uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini file upload failed")
    return uploaded


def _build_categories_list(categories: List[Dict[str, Any]]) -> str:
    """Format categories for the prompt with header so subcategoryId and subcategoryName are explicit."""
    if not categories:
        return "(no categories)"
    header = "categoryId | categoryName | subcategoryId | subcategoryName | priority"
    lines = [header]
    for c in categories:
        cat_id = c.get("categoryId", "")
        cat_name = c.get("categoryName", "")
        sub_id = c.get("subcategoryId", "")
        sub_name = c.get("subcategoryName", "")
        pri = c.get("priority", "")
        lines.append(f"{cat_id} | {cat_name} | {sub_id} | {sub_name} | {pri}")
    return "\n".join(lines)


def _normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure mapped, unmatched, clientDetails have the exact shape and string types."""
    mapped = data.get("mapped") or []
    if not isinstance(mapped, list):
        mapped = []
    out_mapped = []
    for item in mapped:
        if not isinstance(item, dict):
            continue
        out_mapped.append({
            "documentLabel": str(item.get("documentLabel", "")),
            "categoryId": str(item.get("categoryId", "")),
            "categoryName": str(item.get("categoryName", "")),
            "subcategoryId": str(item.get("subcategoryId", "")),
            "subcategoryName": str(item.get("subcategoryName", "")),
            "priority": str(item.get("priority", "")),
        })
    unmatched = data.get("unmatched") or []
    if not isinstance(unmatched, list):
        unmatched = []
    out_unmatched = [str(x).strip() for x in unmatched if x]

    cd = data.get("clientDetails") or {}
    if not isinstance(cd, dict):
        cd = {}
    out_cd = {
        "name": str(cd.get("name", "")),
        "spouseName": str(cd.get("spouseName", "")),
        "ssn": str(cd.get("ssn", "")),
        "address": str(cd.get("address", "")),
        "phone": str(cd.get("phone", "")),
        "email": str(cd.get("email", "")),
        "taxYear": str(cd.get("taxYear", "")),
    }
    return {
        "mapped": out_mapped,
        "unmatched": out_unmatched,
        "clientDetails": out_cd,
    }


class OrganizerExtractor:
    """Upload PDF to LLM with categories from MongoDB; get full result in one response."""

    def __init__(self):
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai required. pip install google-genai")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = ORGANIZER_MODEL
        log_message("info", f"OrganizerExtractor initialized with model {self.model_name}")

    @staticmethod
    def get_pdf_bytes(file) -> bytes:
        """Read PDF bytes from UploadFile."""
        file.file.seek(0)
        data = file.file.read()
        file.file.seek(0)
        return data

    async def extract_and_map(self, file) -> Dict[str, Any]:
        """
        Upload PDF to LLM, pass categories from MongoDB, get result in exact output format.

        Flow:
        1. Fetch categories and trim PDF (from "CLIENT ORGANIZER CHECKLIST" to end) in parallel.
        2. Upload trimmed PDF to Gemini and wait until ready.
        3. Build prompt with categories and call model once file is ready.
        4. Parse LLM response as { mapped, unmatched, clientDetails } and return.
        """
        pdf_bytes = self.get_pdf_bytes(file)
        if not pdf_bytes:
            raise ValueError("No PDF data")

        # 1) Run categories fetch and PDF trim (organizer checklist to end) in parallel
        categories, trimmed_pdf_bytes = await asyncio.gather(
            get_organizer_categories(),
            asyncio.to_thread(_trim_pdf_from_organizer_checklist, pdf_bytes),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(trimmed_pdf_bytes)
            tmp_path = tmp.name
        try:
            # 2) Upload trimmed PDF to Gemini and wait until ready
            uploaded = await asyncio.to_thread(
                _upload_pdf_and_wait_ready, self.client, tmp_path
            )

            # 3) Build prompt and call LLM once file is ready
            categories_list = _build_categories_list(categories)
            prompt_template = _load_prompt("organizer_extraction_prompt.txt")
            prompt = prompt_template.replace("{categories_list}", categories_list)

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[prompt, uploaded],
            )
            try:
                self.client.files.delete(name=uploaded.name)
            except Exception:
                pass
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        raw = response.text if hasattr(response, "text") else str(response)
        parsed = _parse_json_from_response(raw)
        # LLM may return full structure { success, message, data: { mapped, unmatched, clientDetails } }
        inner_data = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
        normalized = _normalize_data(inner_data)

        return {
            "success": True,
            "message": "Organizer PDF parsed and mapped successfully.",
            "data": normalized,
        }
