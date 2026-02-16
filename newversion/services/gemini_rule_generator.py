"""
Gemini Rule Generator Service

Uses Google Gemini 3 Flash Preview model to generate categorization rules
based on GL, COA, and nature of business.
"""

import os
import json
import re
import asyncio
import csv
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
from backend.utils.logger import log_message
from newversion.utils import is_quota_error

try:
    import google.genai as genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai_types = None
    log_message("warning", "google-genai not installed. Install with: pip install google-genai")


class GeminiRuleGenerator:
    """Generate categorization rules using Gemini API."""

    GENERIC_TRIGGER_WORDS = {
        "DIRECT PAY",
        "PAYMENT",
        "WIRE TRANSFER",
        "WIRE",
        "ACH",
        "ACH PAYMENT",
        "ACH TRANSFER",
        "CASH CHECK",
        "TRANSFER",
        "DEPOSIT",
    }
    
    def __init__(self):
        """Initialize Gemini model for rule generation."""
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        self.client = genai.Client(api_key=api_key)
        # Model priority: Try Gemini 3 preview first, fallback to 1.5 Flash
        self.model_name = "gemini-3-flash-preview"  # Preferred: Gemini 3 preview
        self.fallback_models = [
            "gemini-1.5-flash",   # Fast and reliable fallback
            "gemini-1.5-pro"      # More capable but slower fallback
        ]
        log_message("info", f"Initialized Gemini Rule Generator with {self.model_name} model")

    @staticmethod
    def _load_prompt_template() -> str:
        """
        Load prompt template from file.
        
        Returns:
            Prompt template as string
        """
        try:
            # Get the directory where this file is located
            # File is at: newversion/services/gemini_rule_generator.py
            # Prompt is at: newversion/prompts/rule_generation_prompt.txt
            current_dir = Path(__file__).parent.parent  # This is newversion/
            prompt_file_path = current_dir / "prompts" / "rule_generation_prompt.txt"
            
            # If not found, try alternative path (relative to project root)
            if not prompt_file_path.exists():
                # Try from project root
                project_root = Path(__file__).parent.parent.parent
                prompt_file_path = project_root / "newversion" / "prompts" / "rule_generation_prompt.txt"
            
            if not prompt_file_path.exists():
                raise FileNotFoundError(f"Prompt file not found at: {prompt_file_path}")
            
            # Read the prompt file
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            
            log_message("info", f"Loaded rule generation prompt from: {prompt_file_path}")
            return prompt_text
            
        except FileNotFoundError:
            log_message("error", f"Prompt file not found: rule_generation_prompt.txt")
            raise FileNotFoundError(f"Prompt file not found: rule_generation_prompt.txt")
        except Exception as e:
            log_message("error", f"Failed to load prompt file: {e}")
            raise Exception(f"Failed to load prompt file: {e}")
    
    def _create_prompt(
        self,
        client_name: str,
        nature_of_business: str,
        coa_data: str,
        gl_data: str
    ) -> str:
        """
        Create prompt for Gemini to generate rules.
        
        Args:
            client_name: Client name
            nature_of_business: Nature of business
            coa_data: Chart of Accounts data
            gl_data: General Ledger data
            
        Returns:
            Formatted prompt string
        """
        template = self._load_prompt_template()
        
        prompt = template.format(
            client_name=client_name,
            nature_of_business=nature_of_business,
            coa_data=coa_data,
            gl_data=gl_data
        )
        
        return prompt
    
    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse JSON from Gemini response, handling markdown code blocks.
        If response is truncated, attempts to recover complete rule objects.
        
        Args:
            response_text: Raw response text from Gemini
            
        Returns:
            List of rule dictionaries
        """
        try:
            # Remove markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Try to find JSON array in response (greedy: first [ to last ])
            array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if array_match:
                response_text = array_match.group(0)
            
            # Parse JSON
            rules = json.loads(response_text)
            
            if not isinstance(rules, list):
                log_message("warning", "Gemini response is not a list, wrapping in list")
                rules = [rules] if rules else []
            
            log_message("info", f"Successfully parsed {len(rules)} rules from Gemini response")
            return rules
            
        except json.JSONDecodeError as e:
            log_message("warning", f"Initial JSON parse failed (possible truncation): {e}")
            # Try to recover complete rule objects from truncated JSON
            recovered = self._recover_rules_from_truncated_json(response_text)
            if recovered:
                log_message("info", f"Recovered {len(recovered)} complete rules from truncated response")
                return recovered
            log_message("error", f"Response text: {response_text[:500]}")
            raise json.JSONDecodeError("Could not extract valid JSON array from Gemini response", response_text, 0)
        except Exception as e:
            log_message("error", f"Failed to parse JSON from Gemini response: {e}")
            raise Exception(f"Failed to parse Gemini response as JSON: {e}")
    
    def _recover_rules_from_truncated_json(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extract complete rule objects from truncated JSON (e.g. when Gemini hits output limit).
        Finds the last complete rule object and returns all complete rules before it.
        """
        # Ensure we start from the opening bracket of the array
        start = response_text.find('[')
        if start == -1:
            return []
        text = response_text[start:]
        rules = []
        remaining = text
        if remaining.startswith('['):
            remaining = remaining[1:]
        # Find complete {...} blocks (balanced braces)
        i = 0
        while i < len(remaining):
            if remaining[i] != '{':
                i += 1
                continue
            depth = 0
            start_brace = i
            j = i
            while j < len(remaining):
                if remaining[j] == '{':
                    depth += 1
                elif remaining[j] == '}':
                    depth -= 1
                    if depth == 0:
                        chunk = remaining[start_brace:j + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict) and 'rule_id' in obj and 'account_name' in obj:
                                rules.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
                j += 1
            else:
                break
        return rules
    
    @staticmethod
    def _normalize_account_id(account_id: str) -> str:
        """
        Normalize account_id by removing common prefixes.
        
        Examples:
            "REVENUE-4000" -> "4000"
            "COGS-5000" -> "5000"
            "EXP-6100" -> "6100"
            "5000" -> "5000"
        """
        if not account_id:
            return ""
        
        # Remove common prefixes
        prefixes = [
            "REVENUE-", "REV-", "INC-",
            "COGS-", "COG-",
            "EXP-", "EXPENSE-",
            "LIAB-", "LIABILITY-",
            "ASSET-", "AST-",
            "EQUITY-", "EQ-",
            "TAX-",
            "AMA-"
        ]
        
        account_id_upper = account_id.upper()
        for prefix in prefixes:
            if account_id_upper.startswith(prefix):
                return account_id[len(prefix):]
        
        return account_id
    
    @staticmethod
    def _calculate_trigger_similarity(triggers1: set, triggers2: set) -> float:
        """
        Calculate similarity between two sets of triggers (Jaccard similarity).
        
        Returns:
            Float between 0 and 1, where 1 means identical
        """
        if not triggers1 or not triggers2:
            return 0.0
        
        intersection = len(triggers1.intersection(triggers2))
        union = len(triggers1.union(triggers2))
        
        return intersection / union if union > 0 else 0.0
    
    def _final_deduplication(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Final deduplication pass to remove any remaining duplicates.
        
        Checks for:
        1. Exact duplicate account_name + category_type combinations
        2. Very similar trigger sets (>90% overlap)
        
        Args:
            rules: List of rules after initial merge
            
        Returns:
            Final deduplicated list of rules
        """
        if not rules:
            return rules
        
        log_message("info", f"Running final deduplication on {len(rules)} rules")
        
        unique_rules = []
        seen_combinations = {}
        
        for rule in rules:
            account_name = rule.get('account_name', '').strip().lower()
            category_type = rule.get('category_type', '').strip().lower()
            triggers = set([t.upper().strip() for t in rule.get('trigger_payee_contains', [])])
            
            # Create a key based on account name and category
            key = f"{account_name}|{category_type}"
            
            if key in seen_combinations:
                # Check if this is truly a duplicate
                existing_rule = seen_combinations[key]
                existing_triggers = set([t.upper().strip() for t in existing_rule.get('trigger_payee_contains', [])])
                
                # Calculate overlap
                similarity = self._calculate_trigger_similarity(triggers, existing_triggers)
                
                if similarity > 0.5:  # More than 50% overlap = likely duplicate
                    log_message("info", f"Removing duplicate rule for '{account_name}' (similarity: {similarity:.2f})")
                    # Merge triggers into existing rule
                    existing_triggers.update(triggers)
                    existing_rule['trigger_payee_contains'] = sorted(list(existing_triggers))
                    continue
            
            # Add to unique rules
            seen_combinations[key] = rule
            unique_rules.append(rule)
        
        removed_count = len(rules) - len(unique_rules)
        if removed_count > 0:
            log_message("info", f"Final deduplication removed {removed_count} duplicate rules")
        
        return unique_rules
    
    def _merge_rules(self, all_rules: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Merge rules from multiple chunks, removing duplicates and consolidating.
        
        Uses multi-level deduplication:
        1. Normalize account_ids to handle format variations
        2. Group by normalized account_id + account_name
        3. Detect semantic duplicates based on trigger similarity
        
        Args:
            all_rules: List of rule lists from each chunk
            
        Returns:
            Merged and deduplicated list of rules
        """
        try:
            # Flatten all rules
            flat_rules = []
            for chunk_rules in all_rules:
                flat_rules.extend(chunk_rules)
            
            log_message("info", f"Merging {len(flat_rules)} total rules from {len(all_rules)} chunks")
            
            # Step 1: Group by normalized account_id and account_name
            rule_groups = {}
            for rule in flat_rules:
                account_id = rule.get('account_id', '')
                account_name = rule.get('account_name', '').strip()
                
                # Normalize account_id to handle variations
                normalized_id = self._normalize_account_id(account_id)
                
                # Create key using normalized account_id and account_name
                key = f"{normalized_id}_{account_name.lower()}"
                
                if key not in rule_groups:
                    rule_groups[key] = {
                        'rule_id': rule.get('rule_id'),
                        'account_id': account_id,  # Keep original format
                        'account_name': account_name,
                        'category_type': rule.get('category_type'),
                        'trigger_payee_contains': set(),
                        'logic': rule.get('logic', ''),
                        'vendor_code': (str(rule.get('vendor_code') or '').strip() or None),
                        'ach_vendor_only': bool(rule.get('ach_vendor_only') or False),
                    }
                else:
                    if not rule_groups[key].get('vendor_code'):
                        maybe_code = str(rule.get('vendor_code') or '').strip()
                        if maybe_code:
                            rule_groups[key]['vendor_code'] = maybe_code
                    if bool(rule.get('ach_vendor_only') or False):
                        rule_groups[key]['ach_vendor_only'] = True
                
                # Merge trigger keywords
                triggers = rule.get('trigger_payee_contains', [])
                if isinstance(triggers, list):
                    for trigger in triggers:
                        if trigger:
                            rule_groups[key]['trigger_payee_contains'].add(str(trigger).upper().strip())
            
            log_message("info", f"After account-based grouping: {len(rule_groups)} rule groups")
            
            # Step 2: Detect and merge semantic duplicates
            # (Rules with high trigger overlap even if account names differ slightly)
            merged_groups = []
            processed_keys = set()
            
            for key1, group1 in rule_groups.items():
                if key1 in processed_keys:
                    continue
                
                # Check for similar rules
                for key2, group2 in rule_groups.items():
                    if key1 == key2 or key2 in processed_keys:
                        continue
                    
                    # Calculate trigger similarity
                    similarity = self._calculate_trigger_similarity(
                        group1['trigger_payee_contains'],
                        group2['trigger_payee_contains']
                    )
                    
                    # If >70% overlap and same normalized account_id, merge them
                    if similarity > 0.7:
                        norm_id1 = self._normalize_account_id(group1['account_id'])
                        norm_id2 = self._normalize_account_id(group2['account_id'])
                        
                        if norm_id1 == norm_id2:
                            log_message("info", f"Merging duplicate rules: {key1} + {key2} (similarity: {similarity:.2f})")
                            # Merge triggers into group1
                            group1['trigger_payee_contains'].update(group2['trigger_payee_contains'])
                            processed_keys.add(key2)
                
                merged_groups.append(group1)
                processed_keys.add(key1)
            
            # Convert back to list format
            final_rules = []
            for group in merged_groups:
                merged_rule = {
                    'rule_id': group['rule_id'],
                    'trigger_payee_contains': sorted(list(group['trigger_payee_contains'])),
                    'account_id': group['account_id'],
                    'account_name': group['account_name'],
                    'category_type': group['category_type'],
                    'logic': group['logic'],
                    'vendor_code': group.get('vendor_code'),
                    'ach_vendor_only': bool(group.get('ach_vendor_only') or False),
                }
                final_rules.append(merged_rule)
            
            log_message("info", f"Final result: {len(final_rules)} unique rules (removed {len(flat_rules) - len(final_rules)} duplicates)")
            return final_rules
            
        except Exception as e:
            log_message("error", f"Failed to merge rules: {e}")
            # If merging fails, return all rules flattened
            flat_rules = []
            for chunk_rules in all_rules:
                flat_rules.extend(chunk_rules)
            return flat_rules
    
    @staticmethod
    def _count_gl_records(gl_data: str) -> int:
        """
        Count actual data records in GL file.
        Handles CSV format properly by counting data rows.
        
        Args:
            gl_data: GL file content as string
            
        Returns:
            Number of data records
        """
        try:
            # Clean data first - remove NUL bytes and normalize
            cleaned_data = gl_data.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
            
            # Check if it's CSV format
            is_csv = ',' in cleaned_data and '\n' in cleaned_data
            
            if is_csv:
                # Try to parse as CSV
                import csv as csv_module
                import io
                
                # Count CSV rows (excluding empty rows)
                csv_reader = csv_module.reader(io.StringIO(cleaned_data))
                row_count = 0
                for row in csv_reader:
                    # Skip empty rows
                    if any(cell.strip() for cell in row):
                        row_count += 1
                
                # Subtract 1 if first row looks like a header (has common header words)
                if row_count > 0:
                    csv_reader = csv_module.reader(io.StringIO(cleaned_data))
                    first_row = next(csv_reader, None)
                    if first_row:
                        header_keywords = ['date', 'description', 'payee', 'amount', 'account', 'debit', 'credit', 'type']
                        first_row_lower = ' '.join(first_row).lower()
                        if any(keyword in first_row_lower for keyword in header_keywords):
                            row_count -= 1
                
                log_message("info", f"Counted {row_count} data records in GL file (CSV format)")
                return row_count
            else:
                # For non-CSV files, count non-empty lines
                lines = [line for line in cleaned_data.split('\n') if line.strip()]
                log_message("info", f"Counted {len(lines)} lines in GL file (non-CSV format)")
                return len(lines)
                
        except Exception as e:
            log_message("warning", f"Failed to count records properly, using line count: {e}")
            # Fallback: count lines
            cleaned_data = gl_data.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
            lines = [line for line in cleaned_data.split('\n') if line.strip()]
            return len(lines)
    
    @staticmethod
    def _split_gl_into_chunks(gl_data: str, chunk_size: int = 2000) -> List[str]:
        """
        Split GL data into chunks for processing.
        Handles CSV format properly by splitting by records, not lines.
        
        Args:
            gl_data: GL file content as string
            chunk_size: Number of records per chunk
            
        Returns:
            List of GL data chunks
        """
        try:
            # Clean data first - remove NUL bytes and normalize
            cleaned_data = gl_data.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
            
            # Check if it's CSV format
            is_csv = ',' in cleaned_data and '\n' in cleaned_data
            
            if is_csv:
                # Parse CSV and split by records
                import csv as csv_module
                import io
                
                csv_reader = csv_module.reader(io.StringIO(cleaned_data))
                rows = list(csv_reader)
                
                # Check if first row is header
                has_header = False
                header = None
                if rows:
                    header_keywords = ['date', 'description', 'payee', 'amount', 'account', 'debit', 'credit', 'type']
                    first_row_lower = ' '.join(rows[0]).lower()
                    if any(keyword in first_row_lower for keyword in header_keywords):
                        has_header = True
                        header = rows[0]
                        data_rows = rows[1:]
                    else:
                        data_rows = rows
                else:
                    data_rows = []
                
                # Filter out empty rows
                data_rows = [row for row in data_rows if any(cell.strip() for cell in row)]
                
                total_records = len(data_rows)
                log_message("info", f"Splitting GL data: {total_records} records into chunks of {chunk_size}")
                
                chunks = []
                for i in range(0, total_records, chunk_size):
                    chunk_rows = data_rows[i:i + chunk_size]
                    if has_header and i == 0:
                        # Include header in first chunk
                        chunk_rows = [header] + chunk_rows
                    
                    # Convert back to CSV string
                    output = io.StringIO()
                    csv_writer = csv_module.writer(output)
                    csv_writer.writerows(chunk_rows)
                    chunk = output.getvalue()
                    chunks.append(chunk)
                    log_message("info", f"Created chunk {len(chunks)}: records {i+1} to {min(i+chunk_size, total_records)}")
                
                log_message("info", f"Split GL data into {len(chunks)} chunks")
                return chunks
            else:
                # For non-CSV files, split by lines
                lines = [line for line in cleaned_data.split('\n') if line.strip()]
                total_lines = len(lines)
                log_message("info", f"Splitting GL data: {total_lines} lines into chunks of {chunk_size}")
                
                chunks = []
                for i in range(0, total_lines, chunk_size):
                    chunk = '\n'.join(lines[i:i + chunk_size])
                    chunks.append(chunk)
                    log_message("info", f"Created chunk {len(chunks)}: lines {i+1} to {min(i+chunk_size, total_lines)}")
                
                log_message("info", f"Split GL data into {len(chunks)} chunks")
                return chunks
            
        except Exception as e:
            log_message("error", f"Failed to split GL data into chunks: {e}")
            # If chunking fails, return original data as single chunk
            return [gl_data]
    
    @staticmethod
    def _calculate_optimal_chunk_size(gl_line_count: int) -> int:
        """
        Calculate optimal chunk size based on GL data size.
        
        Args:
            gl_line_count: Total number of lines in GL data
            
        Returns:
            Optimal chunk size in lines
        """
        if gl_line_count <= 1000:
            # Small files: no chunking needed
            return gl_line_count
        elif gl_line_count <= 5000:
            # Medium files: 1500 lines per chunk
            return 1500
        elif gl_line_count <= 10000:
            # Large files: 2000 lines per chunk
            return 2000
        elif gl_line_count <= 20000:
            # Very large files: 2500 lines per chunk
            return 2500
        else:
            # Extremely large files: 3000 lines per chunk
            return 3000
    
    async def _generate_rules_for_chunk(
        self,
        client_name: str,
        nature_of_business: str,
        coa_data: str,
        gl_chunk: str,
        chunk_num: int,
        total_chunks: int
    ) -> List[Dict[str, Any]]:
        """
        Generate rules for a single GL chunk.
        
        Args:
            client_name: Client name
            nature_of_business: Nature of business
            coa_data: Chart of Accounts data
            gl_chunk: GL data chunk
            chunk_num: Current chunk number (1-indexed)
            total_chunks: Total number of chunks
            
        Returns:
            List of rule dictionaries
        """
        try:
            # Add context about chunking
            chunk_context = f"\n\nNote: This is chunk {chunk_num} of {total_chunks} from the GL data. Focus on patterns in this portion of the data."
            gl_chunk_with_context = gl_chunk + chunk_context
            
            # Create prompt
            prompt = self._create_prompt(
                client_name=client_name,
                nature_of_business=nature_of_business,
                coa_data=coa_data,
                gl_data=gl_chunk_with_context
            )
            
            # Invoke Gemini with fallback model support
            models_to_try = [self.model_name] + self.fallback_models
            response = None
            last_error = None
            
            for model in models_to_try:
                try:
                    log_message("info", f"Chunk {chunk_num}/{total_chunks}: Trying model {model}")
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(max_output_tokens=16384)
                    )
                    log_message("info", f"Chunk {chunk_num}/{total_chunks}: Successfully used model {model}")
                    break  # Success, exit loop
                except Exception as model_error:
                    last_error = model_error
                    error_msg = str(model_error)
                    if "404" in error_msg or "NOT_FOUND" in error_msg:
                        log_message("warning", f"Model {model} not found, trying next fallback...")
                        continue  # Try next model
                    else:
                        # Other error, re-raise
                        raise
            
            if response is None:
                raise Exception(f"All models failed. Last error: {last_error}")
            
            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            response_text = response_text.strip()
            
            log_message("info", f"Chunk {chunk_num}/{total_chunks} response received: {len(response_text)} characters")
            
            # Parse JSON from response
            rules = self._parse_json_response(response_text)
            
            log_message("info", f"Chunk {chunk_num}/{total_chunks} generated {len(rules)} rules")
            
            return rules
            
        except Exception as e:
            log_message("error", f"Failed to generate rules for chunk {chunk_num}: {e}")
            # Quota/429: re-raise immediately so API can return quota error response
            if is_quota_error(e):
                raise
            # Other errors: return empty list and let caller continue with other chunks
            return []

    @staticmethod
    def _normalize_value(value: Any) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _parse_coa_reference(cls, coa_data: str) -> Dict[str, Any]:
        """
        Parse chart of accounts text and return fast lookup sets.
        """
        out = {
            "account_ids": set(),
            "account_names": set(),
            "account_types": set(),
        }
        if not coa_data or not coa_data.strip():
            return out

        cleaned = coa_data.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line for line in cleaned.split("\n") if line.strip()]
        if len(lines) < 2:
            return out

        delimiter = ","
        try:
            sample = "\n".join(lines[:20])
            delimiter = csv.Sniffer().sniff(sample).delimiter
        except Exception:
            delimiter = ","

        reader = csv.DictReader(io.StringIO(cleaned), delimiter=delimiter)
        fieldnames = [cls._normalize_value(name) for name in (reader.fieldnames or [])]
        if not fieldnames:
            return out

        def find_column(candidates: List[str]) -> Optional[str]:
            for candidate in candidates:
                for field in fieldnames:
                    if field == candidate:
                        return field
            for candidate in candidates:
                for field in fieldnames:
                    if candidate in field:
                        return field
            return None

        id_field = find_column(["account_id", "account id", "id", "account number", "number", "acct"])
        name_field = find_column(["account_name", "account name", "name"])
        type_field = find_column(["category_type", "category type", "type", "account type", "classification"])

        for row in reader:
            if not isinstance(row, dict):
                continue
            lowered = {cls._normalize_value(k): str(v or "").strip() for k, v in row.items()}
            if id_field:
                value = lowered.get(id_field, "").strip()
                if value:
                    out["account_ids"].add(value.lower())
            if name_field:
                value = lowered.get(name_field, "").strip()
                if value:
                    out["account_names"].add(value.lower())
            if type_field:
                value = lowered.get(type_field, "").strip()
                if value:
                    out["account_types"].add(value.lower())

        return out

    @classmethod
    def evaluate_generation_health(
        cls,
        rules: List[Dict[str, Any]],
        coa_data: str,
    ) -> Dict[str, Any]:
        """
        Build generation quality checks and warning messages.
        """
        coa_lookup = cls._parse_coa_reference(coa_data)

        low_rule_count = len(rules) < 30
        missing_account_mappings: List[Dict[str, Any]] = []
        generic_trigger_rules: List[Dict[str, Any]] = []
        overlap_map: Dict[str, set] = {}

        for rule in rules:
            rule_id = rule.get("rule_id")
            account_id = str(rule.get("account_id") or "").strip()
            account_name = str(rule.get("account_name") or "").strip()
            category_type = str(rule.get("category_type") or "").strip()
            triggers = [str(t).strip() for t in (rule.get("trigger_payee_contains") or []) if str(t).strip()]

            # Missing chart of accounts binding
            if account_id and account_name:
                id_in_coa = account_id.lower() in coa_lookup["account_ids"]
                name_in_coa = account_name.lower() in coa_lookup["account_names"]
                if not (id_in_coa or name_in_coa):
                    missing_account_mappings.append(
                        {
                            "rule_id": rule_id,
                            "account_id": account_id,
                            "account_name": account_name,
                        }
                    )

            # Generic trigger risk
            generic_triggers = []
            for trigger in triggers:
                normalized_trigger = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]", " ", trigger.upper())).strip()
                if normalized_trigger in cls.GENERIC_TRIGGER_WORDS:
                    generic_triggers.append(trigger)
            if generic_triggers:
                generic_trigger_rules.append(
                    {
                        "rule_id": rule_id,
                        "account_name": account_name,
                        "generic_triggers": generic_triggers,
                    }
                )

            # Trigger overlaps by account
            for trigger in triggers:
                normalized_trigger = re.sub(r"\s+", " ", trigger.upper()).strip()
                if not normalized_trigger:
                    continue
                if normalized_trigger not in overlap_map:
                    overlap_map[normalized_trigger] = set()
                overlap_map[normalized_trigger].add(account_name or category_type or "unknown")

        overlapping_triggers = []
        for trigger, accounts in overlap_map.items():
            if len(accounts) > 1:
                overlapping_triggers.append(
                    {
                        "trigger": trigger,
                        "accounts": sorted(list(accounts)),
                    }
                )

        warnings: List[str] = []
        if low_rule_count:
            warnings.append(
                f"Only {len(rules)} rules were generated. This may be too low for strong coverage."
            )
        if missing_account_mappings:
            warnings.append(
                f"{len(missing_account_mappings)} rule(s) reference account identifiers or names not found in chart of accounts."
            )
        if generic_trigger_rules:
            warnings.append(
                f"{len(generic_trigger_rules)} rule(s) include generic trigger wording that can cause mis-categorization."
            )
        if overlapping_triggers:
            warnings.append(
                f"{len(overlapping_triggers)} trigger phrase(s) appear in more than one account category."
            )

        return {
            "rule_count": len(rules),
            "low_rule_count": low_rule_count,
            "missing_account_mappings": missing_account_mappings,
            "generic_trigger_rules": generic_trigger_rules,
            "overlapping_triggers": overlapping_triggers,
            "warnings": warnings,
            "warning_count": len(warnings),
        }
    
    async def generate_rules(
        self,
        client_name: str,
        nature_of_business: str,
        coa_data: str,
        gl_data: str
    ) -> List[Dict[str, Any]]:
        """
        Generate categorization rules using Gemini API.
        
        Args:
            client_name: Client name
            nature_of_business: Nature of business
            coa_data: Chart of Accounts data
            gl_data: General Ledger data
            
        Returns:
            List of rule dictionaries
        """
        try:
            # Count GL records to determine if chunking is needed
            gl_record_count = self._count_gl_records(gl_data)
            log_message("info", f"GL data contains {gl_record_count} records")
            
            if gl_record_count > 1000:
                # Large file: process in chunks
                chunk_size = self._calculate_optimal_chunk_size(gl_record_count)
                log_message("info", f"GL data is large ({gl_record_count} records), processing in chunks of {chunk_size}")
                
                # Split GL into chunks
                gl_chunks = self._split_gl_into_chunks(gl_data, chunk_size)
                total_chunks = len(gl_chunks)
                log_message("info", f"Split GL into {total_chunks} chunks")
                
                # Process chunks concurrently with rate limiting
                all_rules = []
                batch_size = 1 # Process 3 chunks at a time to avoid rate limits
                
                tasks = []
                for i, chunk in enumerate(gl_chunks, 1):
                    task = self._generate_rules_for_chunk(
                        client_name=client_name,
                        nature_of_business=nature_of_business,
                        coa_data=coa_data,
                        gl_chunk=chunk,
                        chunk_num=i,
                        total_chunks=total_chunks
                    )
                    tasks.append(task)
                
                # Process in batches
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    log_message("info", f"Processing batch {i//batch_size + 1} ({len(batch)} chunks)")
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, Exception):
                            # Quota error: re-raise immediately so API returns 429 right away
                            if is_quota_error(result):
                                raise result
                            log_message("error", f"Chunk processing error: {result}")
                        elif isinstance(result, list):
                            all_rules.append(result)
                    
                    # Small delay between batches to avoid rate limits
                    if i + batch_size < len(tasks):
                        await asyncio.sleep(20)
                
                # Merge rules from all chunks
                log_message("info", f"Merging rules from {len(all_rules)} chunks")
                merged_rules = self._merge_rules(all_rules)
                
                # Final deduplication pass
                final_rules = self._final_deduplication(merged_rules)
                
                log_message("info", f"Successfully generated {len(final_rules)} unique rules for client: {client_name}")
                return final_rules
                
            else:
                # Process as single chunk (small files)
                log_message("info", f"GL data has {gl_record_count} records, processing as single chunk")
                
                prompt = self._create_prompt(
                    client_name=client_name,
                    nature_of_business=nature_of_business,
                    coa_data=coa_data,
                    gl_data=gl_data
                )
                
                log_message("info", "Invoking Gemini API for rule generation")
                
                # Invoke Gemini with fallback model support
                models_to_try = [self.model_name] + self.fallback_models
                response = None
                last_error = None
                
                for model in models_to_try:
                    try:
                        log_message("info", f"Trying model {model}")
                        response = await asyncio.to_thread(
                            self.client.models.generate_content,
                            model=model,
                            contents=prompt,
                            config=genai_types.GenerateContentConfig(max_output_tokens=16384)
                        )
                        log_message("info", f"Successfully used model {model}")
                        break  # Success, exit loop
                    except Exception as model_error:
                        last_error = model_error
                        error_msg = str(model_error)
                        if "404" in error_msg or "NOT_FOUND" in error_msg:
                            log_message("warning", f"Model {model} not found, trying next fallback...")
                            continue  # Try next model
                        else:
                            # Other error, re-raise
                            raise
                
                if response is None:
                    raise Exception(f"All models failed. Last error: {last_error}")
                
                # Extract response text
                response_text = response.text if hasattr(response, 'text') else str(response)
                response_text = response_text.strip()
                
                log_message("info", f"Gemini response received: {len(response_text)} characters")
                log_message("debug", f"Gemini response preview: {response_text[:500]}")
                
                # Parse JSON from response
                rules = self._parse_json_response(response_text)
                
                # Final deduplication pass (even for single chunk)
                final_rules = self._final_deduplication(rules)
                
                log_message("info", f"Successfully generated {len(final_rules)} unique rules for client: {client_name}")
                
                return final_rules
            
        except Exception as e:
            log_message("error", f"Gemini rule generation failed: {e}")
            import traceback
            log_message("error", f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Gemini rule generation failed: {e}")
