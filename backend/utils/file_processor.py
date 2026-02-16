"""
FileProcessor Module

Provides utility functions to read PDF content, extract transaction text,
and save transactions to CSV files. Handles both sorted and unsorted PDF
text extraction. Includes helper to remove illegal characters for CSV safety.
"""

import re
import io
import os
import fitz
import pandas as pd
from PIL import Image
from backend.utils.logger import log_message
from typing import Optional, Dict


class FileProcessor:
    """Utility class for processing bank statement PDF files and CSV outputs."""

    @staticmethod
    async def read_content_unsorted(file) -> str:
        """
        Extract text from a PDF without sorting by coordinates.

        Args:
            file: UploadFile object representing a PDF.

        Returns:
            str: Extracted PDF text.

        Raises:
            Exception: If PDF reading fails.
        """
        try:
            pdf_bytes = await file.read()
            file.file.seek(0)
            pdf_stream = io.BytesIO(pdf_bytes)

            text = []
            with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
                for page in doc:
                    text.append(page.get_text("text"))

            extracted_text = "\n".join(text)
            return extracted_text

        except Exception as e:
            log_message("error", f"Unsorted PDF extraction failed: {e}")
            raise Exception(f"Unsorted PDF extraction failed: {e}")

    @staticmethod
    async def read_content_sorted(file) -> str:
        """
        Extract text from a PDF sorted by vertical and horizontal positions.

        Args:
            file: UploadFile object representing a PDF.

        Returns:
            str: Extracted, sorted PDF text.

        Raises:
            Exception: If PDF reading fails.
        """
        try:
            pdf_bytes = await file.read()
            file.file.seek(0)
            pdf_stream = io.BytesIO(pdf_bytes)

            all_blocks = []
            with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
                for page in doc:
                    blocks = page.get_text("blocks")
                    # Sort by y (top) then x (left)
                    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

                    for b in blocks:
                        block_text = (b[4] or "").strip()
                        if block_text:
                            all_blocks.append(block_text)

            extracted_text = "\n".join(all_blocks)
            return extracted_text

        except Exception as e:
            log_message("error", f"Sorted PDF extraction failed: {e}")
            raise Exception(f"Sorted PDF extraction failed: {e}")

    @staticmethod
    async def remove_illegal_chars(text: Optional[str]) -> Optional[str]:
        """
        Remove illegal control characters from a string for safe CSV writing.

        Args:
            text: Input string.

        Returns:
            str: Cleaned string without illegal characters.
        """
        try:
            if not isinstance(text, str):
                return text

            illegal_char_re = re.compile(r"[\000-\010\013\014\016-\037]")
            return illegal_char_re.sub("", text)

        except Exception as e:
            log_message("error", f"Illegal character removal failed: {e}")
            raise Exception(f"Illegal character removal failed: {e}")

    @staticmethod
    async def save_transactions_to_csv(data: Dict, file_name: str = "transactions.csv") -> Optional[str]:
        """
        Save transactions to a CSV file after cleaning.

        Args:
            data: Dictionary containing 'transactions' key with list of transactions.
            file_name: Name of the CSV file to save.

        Returns:
            str: Full path to saved CSV file, or None if no transactions exist.

        Raises:
            Exception: If CSV writing fails.
        """
        try:
            txns = data.get("transactions", [])
            if not txns:
                log_message("warning", "No transactions to save in CSV.")
                return None

            if not file_name.lower().endswith(".csv"):
                file_name = f"{file_name}.csv"

            # Create categorized output directory for CSV files
            output_dir = os.path.join("data", "output", "csv")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)

            df = pd.DataFrame(txns)

            # Ensure all required columns exist
            expected_cols = ["date", "post_date", "description", "credit", "debit", "bank"]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None

            df = df[expected_cols]

            # Clean cells for illegal characters
            for col in df.columns:
                df[col] = df[col].apply(lambda x: x if not isinstance(x, str) else re.sub(r"[\000-\010\013\014\016-\037]", "", x))

            df.to_csv(output_path, index=False)

            return output_path

        except Exception as e:
            log_message("error", f"Saving transactions to CSV failed: {e}")
            raise Exception(f"Saving transactions to CSV failed: {e}")

    @staticmethod
    def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            img: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        from PIL import ImageEnhance, ImageFilter
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Apply slight denoising
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img

    @staticmethod
    async def read_content_with_ocr(file) -> str:
        """
        Extract text from a PDF using OCR (Optical Character Recognition).
        This method works for image-based (scanned) PDFs.
        
        Converts each PDF page to an image and uses OCR to extract text.
        Includes image preprocessing to improve OCR accuracy.
        
        Args:
            file: UploadFile object representing a PDF.
            
        Returns:
            str: Extracted PDF text from OCR.
            
        Raises:
            Exception: If OCR extraction fails.
        """
        try:
            # Import pytesseract (optional dependency)
            try:
                import pytesseract
            except ImportError:
                raise Exception(
                    "pytesseract is not installed. Install it with: pip install pytesseract\n"
                    "Also install Tesseract OCR:\n"
                    "- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "- Linux: sudo apt-get install tesseract-ocr\n"
                    "- macOS: brew install tesseract"
                )
            
            # Auto-detect Tesseract path on Windows if not in PATH
            import platform
            if platform.system() == 'Windows':
                default_tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                if os.path.exists(default_tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = default_tesseract_path
                    log_message("info", f"Auto-detected Tesseract at: {default_tesseract_path}")
            
            # Check if Tesseract is installed and accessible
            try:
                version = pytesseract.get_tesseract_version()
                log_message("info", f"Tesseract OCR version: {version}")
            except Exception as e:
                # Try to set path manually if auto-detection failed
                if platform.system() == 'Windows':
                    possible_paths = [
                        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            try:
                                version = pytesseract.get_tesseract_version()
                                log_message("info", f"Found Tesseract at: {path}, version: {version}")
                                break
                            except:
                                continue
                    else:
                        error_msg = (
                            "Tesseract OCR is not installed or not accessible.\n\n"
                            "Tesseract was not found in PATH or default locations.\n"
                            "Please ensure Tesseract is installed at:\n"
                            "  C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n\n"
                            "Or add Tesseract to your system PATH:\n"
                            "  1. Open System Properties → Environment Variables\n"
                            "  2. Add C:\\Program Files\\Tesseract-OCR to PATH\n"
                            "  3. Restart your terminal/IDE\n\n"
                            f"Original error: {str(e)}"
                        )
                        raise Exception(error_msg)
                else:
                    error_msg = (
                        "Tesseract OCR is not installed or not in your PATH.\n\n"
                        "Installation instructions:\n"
                        "Linux:\n"
                        "  sudo apt-get update\n"
                        "  sudo apt-get install tesseract-ocr\n\n"
                        "macOS:\n"
                        "  brew install tesseract\n\n"
                        f"Original error: {str(e)}"
                    )
                    raise Exception(error_msg)
            
            pdf_bytes = await file.read()
            file.file.seek(0)
            pdf_stream = io.BytesIO(pdf_bytes)
            
            log_message("info", "Starting OCR extraction from PDF")
            
            all_text = []
            
            with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
                total_pages = len(doc)
                log_message("info", f"PDF has {total_pages} pages, converting to images for OCR")
                
                # Check if PDF has actual text content first
                has_text = False
                for page in doc:
                    if page.get_text("text").strip():
                        has_text = True
                        break
                
                if has_text:
                    log_message("warning", "PDF appears to have text content. OCR may not be necessary.")
                
                for page_num, page in enumerate(doc):
                    try:
                        # Try multiple DPI settings for better quality
                        dpi_settings = [
                            (3.0, 3.0),  # ~216 DPI - high quality
                            (2.5, 2.5),  # ~180 DPI - medium-high
                            (2.0, 2.0),  # ~144 DPI - medium
                        ]
                        
                        page_text = None
                        last_error = None
                        
                        for zoom_x, zoom_y in dpi_settings:
                            try:
                                # Convert PDF page to image with higher DPI
                                mat = fitz.Matrix(zoom_x, zoom_y)
                                pix = page.get_pixmap(matrix=mat, alpha=False)
                                
                                # Convert to PIL Image
                                img_data = pix.tobytes("png")
                                img = Image.open(io.BytesIO(img_data))
                                
                                # Preprocess image for better OCR
                                img = FileProcessor._preprocess_image_for_ocr(img)
                                
                                # Try multiple OCR configurations
                                ocr_configs = [
                                    r'--oem 3 --psm 6',  # Uniform block of text
                                    r'--oem 3 --psm 3',  # Fully automatic page segmentation
                                    r'--oem 3 --psm 11', # Sparse text
                                    r'--oem 3 --psm 1',  # Automatic page segmentation with OSD
                                ]
                                
                                for ocr_config in ocr_configs:
                                    try:
                                        text = pytesseract.image_to_string(img, config=ocr_config)
                                        if text and len(text.strip()) > 10:
                                            page_text = text
                                            log_message("info", f"Page {page_num + 1}/{total_pages}: OCR extracted {len(text)} characters (zoom={zoom_x}x, config={ocr_config})")
                                            break
                                    except Exception as e:
                                        last_error = e
                                        continue
                                
                                if page_text:
                                    break
                                    
                            except Exception as e:
                                last_error = e
                                continue
                        
                        if page_text and page_text.strip():
                            all_text.append(page_text.strip())
                        else:
                            log_message("warning", f"Page {page_num + 1}/{total_pages}: OCR extracted no text. Last error: {last_error}")
                            
                    except Exception as e:
                        log_message("error", f"OCR failed for page {page_num + 1}: {e}")
                        import traceback
                        log_message("error", f"Traceback: {traceback.format_exc()}")
                        # Continue with next page even if one fails
                        continue
            
            extracted_text = "\n\n".join(all_text)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                # Provide more helpful error message
                error_msg = (
                    "OCR extraction produced no usable text. Possible reasons:\n"
                    "1. PDF images may be too low quality or corrupted\n"
                    "2. PDF might be password-protected\n"
                    "3. Images might be too dark, blurry, or have poor contrast\n"
                    "4. Tesseract OCR might not be properly installed\n"
                    f"Total pages processed: {total_pages}"
                )
                raise Exception(error_msg)
            
            log_message("info", f"OCR extraction completed: {len(extracted_text)} total characters extracted from {len(all_text)} pages")
            return extracted_text
            
        except Exception as e:
            log_message("error", f"OCR PDF extraction failed: {e}")
            raise Exception(f"OCR extraction failed: {e}")

    @staticmethod
    async def read_content_with_fallback(file) -> str:
        """
        Extract text from PDF with automatic fallback to OCR if standard extraction fails.
        
        Tries standard text extraction first, then falls back to OCR for image-based PDFs.
        Also tries alternative extraction methods before OCR.
        
        Args:
            file: UploadFile object representing a PDF.
            
        Returns:
            str: Extracted PDF text.
            
        Raises:
            Exception: If all methods fail.
        """
        try:
            # Method 1: Try standard text extraction
            log_message("info", "Attempting standard PDF text extraction")
            try:
                text = await FileProcessor.read_content_unsorted(file)
                if text and len(text.strip()) > 50:
                    log_message("info", f"Standard extraction successful: {len(text)} characters")
                    return text
                else:
                    log_message("warning", f"Standard extraction returned insufficient text ({len(text) if text else 0} chars)")
            except Exception as e:
                log_message("warning", f"Standard extraction failed: {e}")
            
            # Method 2: Try sorted extraction (sometimes works when unsorted doesn't)
            log_message("info", "Attempting sorted PDF text extraction")
            try:
                file.file.seek(0)
                text = await FileProcessor.read_content_sorted(file)
                if text and len(text.strip()) > 50:
                    log_message("info", f"Sorted extraction successful: {len(text)} characters")
                    return text
                else:
                    log_message("warning", f"Sorted extraction returned insufficient text ({len(text) if text else 0} chars)")
            except Exception as e:
                log_message("warning", f"Sorted extraction failed: {e}")
            
            # Method 3: If all standard methods fail, try OCR
            log_message("info", "All standard extraction methods failed, falling back to OCR")
            file.file.seek(0)  # Reset file pointer
            try:
                ocr_text = await FileProcessor.read_content_with_ocr(file)
                
                if ocr_text and len(ocr_text.strip()) > 10:
                    log_message("info", f"OCR extraction successful: {len(ocr_text)} characters")
                    return ocr_text
                else:
                    log_message("error", "OCR extraction returned insufficient text")
            except Exception as e:
                log_message("error", f"OCR extraction failed: {e}")
                # Don't raise here, let the final exception handle it
            
            # If all methods failed
            raise Exception(
                "All PDF extraction methods failed (standard, sorted, and OCR). "
                "The PDF may be corrupted, password-protected, or have very poor image quality."
            )
                
        except Exception as e:
            log_message("error", f"PDF extraction with fallback failed: {e}")
            raise Exception(f"PDF extraction failed: {e}")
