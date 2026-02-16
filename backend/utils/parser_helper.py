"""
ParserHelper Module

Provides utility functions to extract dates and normalize text from bank statements.
"""

import re
from datetime import datetime
from dateutil import parser
from typing import Optional, Tuple
from backend.utils.logger import log_message


class ParserHelper:
    """Helper class for parsing dates and cleaning text in bank statements."""

    @staticmethod
    def parse_statement_period(text: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Extracts the statement period from text and infers the year.
        Supports multiple date formats.

        Args:
            text (str): The input text containing the statement period.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[int]]: Start date (YYYY-MM-DD),
            End date (YYYY-MM-DD), and inferred year. Returns (None, None, None) if not found.
        """
        try:
            # Try multiple date format patterns
            patterns = [
                # Format: "January 1, 2024 - January 31, 2024"
                r"([A-Za-z]+\s+\d{1,2},\s*\d{4})\s*-\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
                # Format: "Jan 1, 2024 - Jan 31, 2024"
                r"([A-Za-z]{3}\s+\d{1,2},\s*\d{4})\s*-\s*([A-Za-z]{3}\s+\d{1,2},\s*\d{4})",
                # Format: "01/01/2024 - 01/31/2024" or "1/1/2024 - 1/31/2024"
                r"(\d{1,2}/\d{1,2}/\d{4})\s*-\s*(\d{1,2}/\d{1,2}/\d{4})",
                # Format: "for January 1, 2024 to January 31, 2024"
                r"for\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\s+to\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        start_str = match.group(1).strip()
                        end_str = match.group(2).strip()
                        
                        # Parse dates
                        if '/' in start_str:
                            # Handle MM/DD/YYYY format
                            start_parts = start_str.split('/')
                            end_parts = end_str.split('/')
                            if len(start_parts) == 3 and len(end_parts) == 3:
                                start_date = datetime(int(start_parts[2]), int(start_parts[0]), int(start_parts[1]))
                                end_date = datetime(int(end_parts[2]), int(end_parts[0]), int(end_parts[1]))
                            else:
                                continue
                        else:
                            # Handle text format (e.g., "January 1, 2024")
                            start_date = parser.parse(start_str)
                            end_date = parser.parse(end_str)
                        
                        year = start_date.year
                        log_message("info", f"Parsed statement period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, year={year}")
                        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), year
                    except Exception as e:
                        log_message("warning", f"Failed to parse date strings '{match.group(1)}' or '{match.group(2)}': {e}")
                        continue
            
            log_message("warning", "No statement period pattern matched in text")
            return None, None, None

        except Exception as e:
            log_message("error", f"Failed to parse statement period: {e}")
            return None, None, None

    @staticmethod
    def ymd_from_mmdd(mmdd: str, year: Optional[int]) -> Optional[str]:
        """
        Converts a MM/DD string to a full YYYY-MM-DD date using the provided year.

        Args:
            mmdd (str): Date string in MM/DD format.
            year (Optional[int]): Year to use for conversion.

        Returns:
            Optional[str]: Full date string in YYYY-MM-DD format, or None if invalid.
        """
        if not mmdd or not year:
            return None
        try:
            month, day = map(int, mmdd.split("/"))
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except Exception as e:
            log_message("warning", f"Failed to convert MM/DD '{mmdd}' to YMD: {e}")
            return None

    @staticmethod
    def clean_text(s: str) -> str:
        """
        Normalizes inner whitespace while keeping content intact.

        Args:
            s (str): Input string.

        Returns:
            str: Cleaned string.
        """
        try:
            return re.sub(r"\s+", " ", s.strip())
        except Exception as e:
            log_message("warning", f"Failed to clean text '{s}': {e}")
            return s
