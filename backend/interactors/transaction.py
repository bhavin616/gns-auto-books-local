import os
import csv
import re
import json
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from io import StringIO, BytesIO
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.services.pinecone import create_vector_db, retrieve_documents, load_vectordb, query_pinecone_with_embedding
from backend.utils.logger import log_message
from backend.utils.embed import get_embeddings
from backend.utils.bank_detector import detect_bank
from backend.utils.transaction_extractor import TransactionExtractor
import pandas as pd

# OpenAI for embeddings (RAG approach)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    openai_client = None
    log_message("warning", "OpenAI library not installed. Install with: pip install openai")


# ============================================================
#                   GLOBAL INITIALIZATION
# ============================================================

load_dotenv()

embeddings = get_embeddings()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================================================
#                       HELPERS
# ============================================================

async def identify_ledger_category_column(
    df: pd.DataFrame,
    potential_category_columns: List[str]
) -> Optional[str]:
    """
    Use Gemini AI to identify which column contains the actual ledger category/account information.
    
    When multiple columns like "category", "account", "split" are present, this function
    analyzes sample data from each column to determine which one contains the actual
    ledger category information that should be used for categorization.
    
    Args:
        df: pandas DataFrame with GL data
        potential_category_columns: List of column names that might contain category information
    
    Returns:
        Column name that contains the ledger category, or None if unable to determine
    """
    if not potential_category_columns or len(potential_category_columns) <= 1:
        # If only one or no potential columns, return the first one or None
        return potential_category_columns[0] if potential_category_columns else None
    
    try:
        # Get sample data from each potential category column (first 10 non-null values)
        column_samples = {}
        for col in potential_category_columns:
            if col in df.columns:
                sample_values = df[col].dropna().head(10).tolist()
                if sample_values:
                    # Convert to strings and limit length
                    sample_strs = [str(v).strip()[:100] for v in sample_values[:10]]
                    column_samples[col] = sample_strs
        
        if len(column_samples) <= 1:
            # Only one column has data, use it
            return list(column_samples.keys())[0] if column_samples else None
        
        # Create prompt for Gemini
        columns_info = []
        for col, samples in column_samples.items():
            columns_info.append(f"Column '{col}': {samples}")
        
        prompt = f"""You are analyzing a General Ledger (GL) file to identify which column contains the actual ledger category/account information.

The ledger category is the account code or account name from the Chart of Accounts that categorizes the transaction (e.g., "Office Supplies", "Rent Expense", "Accounts Payable", "4010", "5000-Office Supplies").

Here are sample values from potential category columns:

{chr(10).join(columns_info)}

Analyze these columns and determine which one contains the actual ledger category/account information that should be used for transaction categorization.

Consider:
1. Ledger categories are typically account names, account codes, or account descriptions from the Chart of Accounts
2. They are used to categorize transactions for accounting purposes
3. They are NOT transaction descriptions, vendor names, or memo fields
4. They represent the GL account that the transaction should be posted to

Respond with ONLY the column name (exactly as provided) that contains the ledger category information.
If you cannot determine, respond with "UNKNOWN".

Column name:"""
        
        log_message("info", f"Using Gemini to identify ledger category column from {len(column_samples)} potential columns")
        
        # Call Gemini
        response = await asyncio.to_thread(llm.invoke, prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        # Extract column name from response
        identified_column = None
        for col in potential_category_columns:
            if col.lower() in response_text.lower():
                identified_column = col
                break
        
        # If exact match not found, try to extract from response
        if not identified_column:
            # Try to find column name in response
            response_lower = response_text.lower()
            for col in potential_category_columns:
                if col.lower() in response_lower or response_lower.startswith(col.lower()):
                    identified_column = col
                    break
        
        if identified_column:
            log_message("info", f"Gemini identified '{identified_column}' as the ledger category column")
            return identified_column
        else:
            log_message("warning", f"Gemini could not clearly identify category column. Response: {response_text}")
            # Fallback: use the first column with data
            return list(column_samples.keys())[0] if column_samples else None
            
    except Exception as e:
        log_message("error", f"Error using Gemini to identify category column: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        # Fallback: return first potential column
        return potential_category_columns[0] if potential_category_columns else None


async def parse_gl_data_from_dataframe(df: pd.DataFrame, client_name: str) -> List[Dict[str, Any]]:
    """Parse GL data from pandas DataFrame with flexible column name matching.
    
    Uses Gemini AI to intelligently identify the ledger category column when multiple
    category/account columns are present.
    
    Supports various column name formats:
    - Description: memo, description, transaction description, details, notes, narrative, etc.
    - Category: split, category, account, account name, account code, etc.
    - Amount: credit, debit, amount (positive=credit, negative=debit), credit amount, debit amount, etc.
    - Date: date, transaction date, posting date, etc.
    - Name: name, account name, vendor, payee, etc.

    Args:
        df: pandas DataFrame with GL data.
        client_name: Name of the client.

    Returns:
        List of cleaned rows with all GL fields including date, name, memo, split, credit, debit.
    """
    cleaned = []
    
    # Normalize column names (case-insensitive, strip whitespace)
    df.columns = df.columns.str.strip().str.lower()
    
    # Log detected columns for debugging
    log_message("info", f"GL file columns detected: {list(df.columns)}")
    
    # Map column names with extensive flexible matching
    date_col = None
    name_col = None
    memo_col = None
    split_col = None
    debit_col = None
    credit_col = None
    amount_col = None  # Single amount column (positive=credit, negative=debit)
    
    # Define possible column name variations
    date_variations = ['date', 'transaction date', 'posting date', 'entry date', 'trans date', 'doc date']
    name_variations = ['name', 'account name', 'vendor', 'payee', 'customer', 'supplier', 'company name']
    memo_variations = ['memo', 'description', 'transaction description', 'details', 'notes', 'narrative', 
                       'transaction details', 'particulars', 'remarks', 'comment', 'reference', 'trans desc']
    split_variations = ['split', 'category', 'account', 'account name', 'account code', 'account number',
                       'account description', 'gl account', 'chart of account', 'coa', 'account title']
    debit_variations = ['debit', 'debit amount', 'dr', 'dr amount', 'withdrawal', 'expense', 'payment']
    credit_variations = ['credit', 'credit amount', 'cr', 'cr amount', 'deposit', 'income', 'receipt']
    amount_variations = ['amount', 'transaction amount', 'value', 'total', 'net amount', 'balance']
    
    # First pass: Look for exact matches (highest priority)
    for col in df.columns:
        col_lower = col.lower().strip()
        if date_col is None and col_lower in date_variations:
            date_col = col
        if name_col is None and col_lower in name_variations:
            name_col = col
        if memo_col is None and col_lower in memo_variations:
            memo_col = col
        if split_col is None and col_lower in split_variations:
            split_col = col
        if debit_col is None and col_lower in debit_variations:
            debit_col = col
        if credit_col is None and col_lower in credit_variations:
            credit_col = col
        if amount_col is None and col_lower in amount_variations:
            amount_col = col
    
    # Second pass: Look for substring matches (if not found in first pass)
    for col in df.columns:
        col_lower = col.lower().strip()
        if date_col is None and any(variation in col_lower for variation in date_variations):
            date_col = col
        # For 'name', avoid matching 'unnamed' columns
        if name_col is None and any(variation in col_lower for variation in name_variations) and 'unnamed' not in col_lower:
            name_col = col
        if memo_col is None and any(variation in col_lower for variation in memo_variations):
            memo_col = col
        if split_col is None and any(variation in col_lower for variation in split_variations):
            split_col = col
        if debit_col is None and any(variation in col_lower for variation in debit_variations):
            debit_col = col
        if credit_col is None and any(variation in col_lower for variation in credit_variations):
            credit_col = col
        if amount_col is None and any(variation in col_lower for variation in amount_variations):
            amount_col = col
    
    # Third pass: Very flexible matching - look for any column containing keywords
    for col in df.columns:
        col_lower = col.lower().strip()
        if date_col is None and ('date' in col_lower or 'dt' in col_lower):
            date_col = col
        if name_col is None and ('name' in col_lower and 'unnamed' not in col_lower):
            name_col = col
        if memo_col is None and ('desc' in col_lower or 'memo' in col_lower or 'note' in col_lower or 'detail' in col_lower):
            memo_col = col
        if split_col is None and ('split' in col_lower or 'category' in col_lower or 'account' in col_lower):
            split_col = col
        if debit_col is None and ('debit' in col_lower or 'dr' in col_lower):
            debit_col = col
        if credit_col is None and ('credit' in col_lower or 'cr' in col_lower):
            credit_col = col
        if amount_col is None and ('amount' in col_lower or 'amt' in col_lower):
            amount_col = col
    
    # Log detected column mappings
    log_message("info", f"Column mappings - date: {date_col}, name: {name_col}, memo: {memo_col}, split: {split_col}, debit: {debit_col}, credit: {credit_col}, amount: {amount_col}")
    
    # Validate required columns - make memo and split more flexible
    # If memo not found, try to use description or first text column
    if not memo_col:
        # Try to find any text-like column that might contain description
        for col in df.columns:
            col_lower = col.lower().strip()
            # Skip numeric columns, date columns, and already matched columns
            if col not in [date_col, name_col, split_col, debit_col, credit_col, amount_col]:
                # Check if column contains mostly text (not numeric)
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    text_count = sum(1 for v in sample_values if isinstance(v, str) and not str(v).replace('.', '').replace('-', '').replace(',', '').isdigit())
                    if text_count > len(sample_values) * 0.7:  # 70% text values
                        memo_col = col
                        log_message("info", f"Auto-detected memo column: {col} (text-like column)")
                        break
        
        if not memo_col:
            raise ValueError(f"Required column for description/memo not found. Available columns: {list(df.columns)}")
    
    # Collect all potential category/account columns
    potential_category_cols = []
    
    # First, collect columns that match category variations
    for col in df.columns:
        col_lower = col.lower().strip()
        if col not in [date_col, name_col, memo_col, debit_col, credit_col, amount_col]:
            if any(variation in col_lower for variation in split_variations):
                potential_category_cols.append(col)
    
    # If we found potential category columns, use Gemini to identify the correct one
    if len(potential_category_cols) > 1:
        log_message("info", f"Multiple category columns detected: {potential_category_cols}. Using Gemini to identify ledger category column...")
        split_col = await identify_ledger_category_column(df, potential_category_cols)
        if split_col:
            log_message("info", f"Selected '{split_col}' as ledger category column based on Gemini analysis")
        else:
            # Fallback to first column if Gemini couldn't determine
            split_col = potential_category_cols[0]
            log_message("warning", f"Gemini analysis inconclusive, using first column: {split_col}")
    elif len(potential_category_cols) == 1:
        split_col = potential_category_cols[0]
        log_message("info", f"Single category column detected: {split_col}")
    else:
        # If split/category not found, try to use account or category-like column
        for col in df.columns:
            col_lower = col.lower().strip()
            if col not in [date_col, name_col, memo_col, debit_col, credit_col, amount_col]:
                if 'account' in col_lower or 'category' in col_lower or 'split' in col_lower:
                    split_col = col
                    log_message("info", f"Auto-detected split/category column: {col}")
                    break
        
        if not split_col:
            # If still not found, use a default category or make it optional
            log_message("warning", "Split/Category column not found. Using 'Uncategorized' as default.")
            # We'll handle this in the processing loop
    
    # Process each row
    for idx, row in df.iterrows():
        # Get values, handling NaN/empty
        # Date: extract as string, handle various date formats
        date_value = ""
        if date_col and pd.notna(row.get(date_col)):
            date_raw = row.get(date_col)
            # Handle datetime objects and timestamps
            try:
                # Check if it's a pandas Timestamp or datetime object
                if isinstance(date_raw, pd.Timestamp):
                    date_value = date_raw.strftime("%Y-%m-%d")
                elif hasattr(date_raw, 'strftime'):
                    # Handle Python datetime objects
                    date_value = date_raw.strftime("%Y-%m-%d")
                else:
                    # Try to parse as datetime if it's a string
                    date_str = str(date_raw).strip()
                    # Try to parse common date formats
                    try:
                        parsed_date = pd.to_datetime(date_str)
                        date_value = parsed_date.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        # If parsing fails, use the string as-is
                        date_value = date_str
            except Exception as e:
                # Fallback: just convert to string
                date_value = str(date_raw).strip()
                log_message("warning", f"Error parsing date '{date_raw}': {e}. Using string value.")
        
        name = str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else ""
        memo = str(row[memo_col]).strip() if memo_col and pd.notna(row.get(memo_col)) else ""
        
        # Handle category/split - use default if not found
        if split_col and pd.notna(row.get(split_col)):
            category = str(row[split_col]).strip()
        else:
            category = "Uncategorized"  # Default category if not found
        
        # Handle credit/debit amounts - support multiple formats
        debit_val = 0.0
        credit_val = 0.0
        
        # Case 1: Separate credit and debit columns
        if credit_col and debit_col:
            debit_str = str(row[debit_col]).strip() if debit_col and pd.notna(row.get(debit_col)) else "0"
            credit_str = str(row[credit_col]).strip() if credit_col and pd.notna(row.get(credit_col)) else "0"
            
            try:
                debit_val = float(debit_str.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()) if debit_str and debit_str != "NaN" else 0.0
            except (ValueError, AttributeError):
                debit_val = 0.0
            
            try:
                credit_val = float(credit_str.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()) if credit_str and credit_str != "NaN" else 0.0
            except (ValueError, AttributeError):
                credit_val = 0.0
        
        # Case 2: Single amount column (positive = credit, negative = debit)
        elif amount_col and pd.notna(row.get(amount_col)):
            amount_str = str(row[amount_col]).strip()
            try:
                # Clean the amount string
                amount_clean = amount_str.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()
                amount_val = float(amount_clean) if amount_clean and amount_clean != "NaN" else 0.0
                
                if amount_val > 0:
                    credit_val = amount_val
                    debit_val = 0.0
                elif amount_val < 0:
                    debit_val = abs(amount_val)  # Store as positive debit
                    credit_val = 0.0
                else:
                    debit_val = 0.0
                    credit_val = 0.0
            except (ValueError, AttributeError):
                debit_val = 0.0
                credit_val = 0.0
        
        # Case 3: Only credit column (debit = 0)
        elif credit_col and not debit_col:
            credit_str = str(row[credit_col]).strip() if pd.notna(row.get(credit_col)) else "0"
            try:
                credit_val = float(credit_str.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()) if credit_str and credit_str != "NaN" else 0.0
            except (ValueError, AttributeError):
                credit_val = 0.0
            debit_val = 0.0
        
        # Case 4: Only debit column (credit = 0)
        elif debit_col and not credit_col:
            debit_str = str(row[debit_col]).strip() if pd.notna(row.get(debit_col)) else "0"
            try:
                debit_val = float(debit_str.replace(",", "").replace("$", "").replace("(", "").replace(")", "").strip()) if debit_str and debit_str != "NaN" else 0.0
            except (ValueError, AttributeError):
                debit_val = 0.0
            credit_val = 0.0
        
        # Log first few rows for debugging
        if len(cleaned) < 3:
            log_message("info", f"Row {len(cleaned)+1}: name='{name}', memo='{memo}', category='{category}', debit={debit_val}, credit={credit_val}")
        
        # Skip rows with no memo (description is required)
        if not memo or memo.strip() == "":
            continue
        
        # Skip rows with no amount (both credit and debit are zero)
        if debit_val == 0.0 and credit_val == 0.0:
            continue
        
        cleaned.append({
            "date": date_value,
            "name": name,
            "memo": memo,
            "category": category,
            "debit": debit_val,
            "credit": credit_val,
            "client_name": client_name
        })
    
    # Log statistics about Name column
    empty_name_count = sum(1 for row in cleaned if not row["name"])
    log_message("info", f"Parsed {len(cleaned)} rows from DataFrame. Empty names: {empty_name_count}/{len(cleaned)}")
    
    return cleaned


async def _prepare_documents_from_raw(raw: bytes, filename: str, client_name: str) -> List[Document]:
    """
    Prepare LangChain Documents from raw file bytes for GL ingestion.

    Supports CSV and Excel inputs. Raises ValueError on parsing issues or
    missing required columns.
    """
    filename_lower = filename.lower() if filename else ""
    is_excel = filename_lower.endswith(('.xlsx', '.xls', '.xlsm'))

    if is_excel:
        df = pd.read_excel(BytesIO(raw), engine='openpyxl')
        log_message("info", f"Parsed Excel file with {len(df)} rows")
    else:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        df = None
        last_error = None

        for encoding in encodings:
            try:
                df = pd.read_csv(BytesIO(raw), encoding=encoding)
                log_message("info", f"Parsed CSV file with {len(df)} rows (encoding: {encoding})")
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        if df is None:
            raise ValueError(f"Failed to parse CSV file with any encoding. Last error: {last_error}")

    cleaned_data = await parse_gl_data_from_dataframe(df, client_name)

    if not cleaned_data:
        log_message("warning", "No valid data rows found in file")
        raise ValueError("No valid data rows found. Ensure file has a description/memo column and at least one amount column (credit, debit, or amount).")

    documents: List[Document] = []
    for row in cleaned_data:
        # Create page content from memo and category (used for embedding)
        page_content = f"{row['memo']} {row['category']}".strip()
        
        # Format credit/debit as strings in metadata
        # Convert numeric values to string (0.0 will become "0.0", which is fine for numeric comparison)
        debit_val_str = str(row["debit"]) if "debit" in row and row["debit"] is not None else ""
        credit_val_str = str(row["credit"]) if "credit" in row and row["credit"] is not None else ""

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "date": row.get("date", ""),  # Include date in metadata
                    "name": row.get("name", ""),  # Include name in metadata
                    "memo": row["memo"],
                    "category": row["category"],
                    "split": row["category"],  # Also include as "split" for backward compatibility
                    "debit": debit_val_str,
                    "credit": credit_val_str,
                    "client_name": row["client_name"],
                },
            )
        )

    log_message("info", f"Prepared {len(documents)} documents for ingestion.")
    return documents


async def _ingest_documents(documents: List[Document], client_name: str, index_name: str) -> Dict[str, Any]:
    """Ingest prepared documents into Pinecone."""
    resolved_index = index_name.strip()
    if not resolved_index:
        raise ValueError("index_name is required and cannot be empty.")

    log_message("info", f"Using Pinecone index '{resolved_index}' for client='{client_name}'.")

    # Don't auto-create index - it should already exist (created via /create_pinecone_index endpoint)
    await asyncio.to_thread(create_vector_db, documents, resolved_index, client_name, False)
    log_message("info", f"Documents successfully added to Pinecone index '{resolved_index}'.")
    return {"status": "success", "count": len(documents)}


# ============================================================
#                  DOCUMENT INGESTION PIPELINE
# ============================================================

async def add_doc_call(client_name: str, file, index_name: str) -> Dict[str, Any]:
    """Adds GL data (CSV/Excel) as vector documents into Pinecone.
    
    Supports new format: Name, Memo, Split (category), Debit, Credit, ClientName

    Args:
        client_name: Name of the client.
        file: Uploaded file object (CSV or Excel).
        index_name: Explicit Pinecone index name (required).

    Returns:
        Status dict with success or failure.
    """
    try:
        file.file.seek(0)
        raw = await file.read()
        file.file.seek(0)
        filename = file.filename if hasattr(file, 'filename') else ""

        documents = await _prepare_documents_from_raw(raw, filename, client_name)

    except Exception as e:
        log_message("error", f"Document parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": str(e)}

    try:
        return await _ingest_documents(documents, client_name, index_name)

    except Exception as e:
        log_message("error", f"Pinecone ingestion failed: {e}")
        return {"status": "failed", "error": str(e)}


async def add_doc_call_from_bytes(client_name: str, index_name: str, raw: bytes, filename: str) -> Dict[str, Any]:
    """
    Adds GL data (CSV/Excel) as vector documents into Pinecone from raw bytes.
    Intended for background processing where the UploadFile may no longer be available.
    """
    try:
        documents = await _prepare_documents_from_raw(raw, filename, client_name)
    except Exception as e:
        log_message("error", f"Document parsing failed (background): {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": str(e)}

    try:
        return await _ingest_documents(documents, client_name, index_name)
    except Exception as e:
        log_message("error", f"Pinecone ingestion failed (background): {e}")
        return {"status": "failed", "error": str(e)}


# ============================================================
#                CATEGORY PREDICTION FUNCTION
# ============================================================

async def predict_category(
    vector_store: Any,
    description: str,
    client_name: str
) -> str:
    """Predict category using Pinecone similarity search.

    Args:
        vector_store: Loaded vector DB.
        description: Raw description text.
        client_name: Name of client to filter results.

    Returns:
        The predicted category (str).
    """
    try:
        docs = await asyncio.to_thread(retrieve_documents, vector_store, description, client_name)
        if docs and docs[0].metadata:
            return docs[0].metadata.get("category", "unknown")
        return "unknown"

    except Exception as e:
        log_message("error", f"Prediction failed: {e}")
        return "unknown"


# ============================================================
#                   MAIN CATEGORIZATION PIPELINE
# ============================================================

async def call(content: bytes, file_name: str, client_name: str) -> Any:
    """
    Categorizes transactions and writes them to a CSV.
    """
    try:
        # Decode CSV content
        decoded = content.decode("utf-8")
        reader = csv.DictReader(StringIO(decoded))
        rows = list(reader)
        if not rows:
            return {"status": "failed", "error": "CSV is empty"}

        # Load vector store
        vector_store = load_vectordb(index_name="banktransactions", embeddings=embeddings)
        log_message("info", "Vector DB loaded.")

        # Prepare output folder (categorized CSV folder)
        base_dir = Path(__file__).resolve().parent
        output_folder = base_dir / "data" / "output" / "csv"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Unique output filename
        unique_suffix = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_path = output_folder / f"{file_name}_categorized_{unique_suffix}.csv"

        # Semaphore to limit concurrency
        sem = asyncio.Semaphore(15)

        async def process_single_row(row: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                description = str(row.get("description", "")).strip()

                # Pre-filter: if empty OR contains no letters OR standalone 'loan'/'check'
                if (not description or
                    not re.search(r"[a-zA-Z]", description) or
                    re.search(r"\bloan\b", description, re.IGNORECASE) or
                    re.search(r"\bcheck\b", description, re.IGNORECASE)):
                    row["Category"] = "unknown"
                    return row

                # Call the LLM category classifier
                category = await predict_category(vector_store, description, client_name)

                # ======= TAXES SUBCATEGORY LOGIC =======
                if category.lower() == "taxes":
                    desc_upper = description.upper()

                    if "FTB" in desc_upper:
                        row["Category"] = "FTB-Taxes"
                    elif "IRS" in desc_upper:
                        row["Category"] = "IRS-Taxes"
                    elif "CDTFA" in desc_upper:
                        row["Category"] = "CDTFA-Taxes"
                    else:
                        row["Category"] = "Other-Taxes"
                else:
                    row["Category"] = category

                return row

        # Process in batches
        batch_size = 20
        all_processed = []
        for i in range(0, len(rows), batch_size):
            batch = rows[i: i + batch_size]
            processed_batch = await asyncio.gather(*(process_single_row(r) for r in batch))
            all_processed.extend(processed_batch)
            log_message("info", f"Processed rows {i+1} to {i+len(processed_batch)}")

        # Write the fully processed output file
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(all_processed)

        return f"✅ Output saved: {output_path}"

    except Exception as e:
        log_message("error", f"Categorization pipeline failed: {e}")
        return {"status": "failed", "error": str(e)}



# ============================================================
#                 BANK TRANSACTION EXTRACTOR
# ============================================================

async def extract_transactions(file) -> Dict[str, Any]:
    """Detects bank format and extracts transactions.

    Args:
        file: Uploaded bank statement.

    Returns:
        A dict indicating success and parsed transaction data.
    """
    try:
        detection = await detect_bank(file)

        if detection.get("bank") == "unknown":
            return {"status": "failed", "message": "Bank format not recognized", "data": {}}

        transactions = await TransactionExtractor.call(detection, file)

        return {
            "status": "success",
            "message": "Transaction extraction successful",
            "data": transactions
        }

    except Exception as e:
        log_message("error", f"Transaction extraction failed: {e}")
        return {"status": "failed", "error": str(e)}


# ============================================================
#           FILE PARSING
# ============================================================

async def parse_bank_pdf(file: Any, use_gemini: bool = True) -> Dict[str, Any]:
    """
    Parse bank transactions from PDF file.
    
    Uses Gemini 3 Flash Preview with vision capabilities to extract transactions directly from PDF.
    Falls back to regex-based extraction if Gemini fails.
    
    Args:
        file: PDF file containing bank statement
        use_gemini: Whether to use Gemini extraction (default: True)
        
    Returns:
        Dict with transactions list and metadata
    """
    try:
        log_message("info", "Parsing bank PDF file")
        file.file.seek(0)
        
        # Try Gemini extraction first (if enabled)
        if use_gemini:
            try:
                from backend.services.gemini_pdf_extractor import GeminiPDFExtractor
                
                log_message("info", "Attempting Gemini-based PDF extraction")
                extractor = GeminiPDFExtractor()
                result = await extractor.extract_transactions(file)
                
                if result and result.get("transactions"):
                    transactions = result.get("transactions", [])
                    log_message("info", f"Successfully extracted {len(transactions)} transactions using Gemini")
                    
                    # Return in the same format as regex extraction for compatibility
                    return {
                        "status": "success",
                        "transactions": transactions,
                        "metadata": {
                            "bank": result.get("bank", "Unknown"),
                            "account_number": result.get("account_number"),
                            "start_date": result.get("start_date"),
                            "end_date": result.get("end_date"),
                        }
                    }
                else:
                    log_message("warning", "Gemini extraction returned no transactions, falling back to regex method")
            except Exception as gemini_error:
                log_message("warning", f"Gemini extraction failed: {gemini_error}, falling back to regex method")
                import traceback
                log_message("debug", f"Gemini error traceback: {traceback.format_exc()}")
                file.file.seek(0)  # Reset file pointer
        
        # Fallback to existing regex-based extraction
        log_message("info", "Using regex-based extraction method")
        file.file.seek(0)
        
        detection = await detect_bank(file)
        if detection.get("bank") == "unknown":
            return {
                "status": "failed",
                "error": "Bank format not recognized. Supported banks: Wells Fargo, Chase Bank, Bank of America"
            }
        
        if detection.get("bank") == "Chase Bank" and detection.get("format") == "format_01" and detection.get("empty_statement") == True:
            return {
                "status": "failed",
                "error": "This statement shows no transaction activity for the selected period."
            }
        
        file.file.seek(0)
        transactions_data = await TransactionExtractor.call(detection, file)
        
        if not transactions_data or not transactions_data.get("extracted_transactions"):
            return {
                "status": "failed",
                "error": "Failed to extract transactions from PDF"
            }
        
        extracted_data = transactions_data.get("extracted_transactions", {})
        transactions = extracted_data.get("transactions", [])
        
        if not transactions:
            return {
                "status": "failed",
                "error": "No transactions found in PDF"
            }
        
        log_message("info", f"Extracted {len(transactions)} transactions from bank PDF using regex method")
        
        return {
            "status": "success",
            "transactions": transactions,
            "metadata": {
                "bank": extracted_data.get("bank"),
                "account_number": extracted_data.get("account_number"),
                "start_date": extracted_data.get("start_date"),
                "end_date": extracted_data.get("end_date"),
            }
        }
        
    except Exception as e:
        log_message("error", f"Bank PDF parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "Unable to process bank statement. Please ensure the file is a valid PDF from a supported bank."}


async def parse_bank_csv(file: Any) -> Dict[str, Any]:
    """
    Parse bank transactions from CSV file.
    
    Expected CSV columns (Chase Bank format):
    - Posting Date (required): Transaction date
    - Description (required): Transaction description
    - Type (required): Contains amount and transaction type (e.g., "13000 CHECK_DEPOSIT" or "-1148 ACH_DEBIT")
    - Amount, Debit, Credit (alternative): Transaction amount columns
    
    Returns transactions in same format as parse_bank_pdf (with credit/debit fields)
    
    Args:
        file: CSV file containing bank transactions
        
    Returns:
        Dict with transactions list and metadata (same format as parse_bank_pdf)
    """
    try:
        log_message("info", "Parsing bank CSV file")
        
        content = await file.read()
        file.file.seek(0)
        
        # Try different encodings
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(BytesIO(content), encoding=encoding)
                log_message("info", f"CSV loaded: {len(df)} rows, columns: {list(df.columns)} (encoding: {encoding})")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df is None:
            return {"status": "failed", "error": "Failed to decode CSV file with any encoding"}
        
        # Normalize column names (case-insensitive)
        df.columns = df.columns.str.strip().str.lower()
        
        log_message("info", f"Normalized column names: {list(df.columns)}")
        
        # Find required columns (case-insensitive search)
        date_col = None
        description_col = None
        amount_col = None
        type_col = None
        debit_col = None
        credit_col = None
        account_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            # Prioritize "posting date" over generic "date"
            if not date_col and 'posting' in col_lower and 'date' in col_lower:
                date_col = col
            elif not date_col and 'date' in col_lower:
                date_col = col
            if not description_col and 'description' in col_lower:
                description_col = col
            if not type_col and col_lower == 'type':
                type_col = col
            if not amount_col and 'amount' in col_lower:
                amount_col = col
            if not debit_col and 'debit' in col_lower:
                debit_col = col
            if not credit_col and 'credit' in col_lower:
                credit_col = col
            if not account_col and ('account' in col_lower and 'number' in col_lower):
                account_col = col
        
        log_message("info", f"Column mapping: date={date_col}, description={description_col}, type={type_col}, amount={amount_col}, debit={debit_col}, credit={credit_col}")
        
        # Validate required columns
        if not date_col:
            return {"status": "failed", "error": "CSV must have a 'Date' or 'Posting Date' column"}
        
        if not description_col:
            return {"status": "failed", "error": "CSV must have a 'Description' column"}
        
        # Check if we have amount info (Type column, Amount column, or Debit/Credit columns)
        if not type_col and not amount_col and not (debit_col or credit_col):
            return {"status": "failed", "error": "CSV must have either a 'Type', 'Amount', or 'Debit'/'Credit' columns"}
        
        # Extract transactions
        transactions = []
        account_number = None
        
        for idx, row in df.iterrows():
            # Skip empty rows
            if pd.isna(row[date_col]) or not str(row[date_col]).strip():
                continue
            
            date_str = str(row[date_col]).strip()
            description = str(row[description_col]).strip() if pd.notna(row[description_col]) else ""
            
            # Skip if no description
            if not description:
                continue
            
            # Get amount - try different methods
            credit_amount = None
            debit_amount = None
            
            # Method 1: Parse Type column (Chase Bank format: "13000 CHECK_DEPOSIT" or "-1148 ACH_DEBIT")
            if type_col and pd.notna(row[type_col]):
                type_value = str(row[type_col]).strip()
                # Extract number from type (e.g., "13000 CHECK_DEPOSIT" -> 13000)
                import re
                amount_match = re.match(r'^(-?\d+(?:\.\d+)?)', type_value)
                if amount_match:
                    try:
                        amount_value = float(amount_match.group(1))
                        if amount_value >= 0:
                            credit_amount = amount_value
                        else:
                            debit_amount = abs(amount_value)
                    except (ValueError, AttributeError):
                        pass
            
            # Method 2: Check for separate Debit/Credit columns
            if credit_amount is None and debit_amount is None:
                if credit_col and pd.notna(row[credit_col]):
                    try:
                        credit_amount = float(str(row[credit_col]).replace(',', '').replace('$', '').strip())
                    except (ValueError, AttributeError):
                        pass
                
                if debit_col and pd.notna(row[debit_col]):
                    try:
                        debit_amount = float(str(row[debit_col]).replace(',', '').replace('$', '').strip())
                    except (ValueError, AttributeError):
                        pass
            
            # Method 3: Check Amount column (positive = credit, negative = debit)
            if credit_amount is None and debit_amount is None and amount_col and pd.notna(row[amount_col]):
                try:
                    amount_value = float(str(row[amount_col]).replace(',', '').replace('$', '').strip())
                    if amount_value >= 0:
                        credit_amount = amount_value
                    else:
                        debit_amount = abs(amount_value)
                except (ValueError, AttributeError):
                    pass
            
            # Skip transaction if no amount found
            if credit_amount is None and debit_amount is None:
                continue
            
            # Get account number (if available)
            if account_col and pd.notna(row[account_col]):
                account_number = str(row[account_col]).strip()
            
            # Create transaction object (SAME FORMAT AS parse_bank_pdf - with credit/debit fields)
            transaction = {
                "date": date_str,
                "description": description,
            }
            
            # Add credit or debit (not both, not amount)
            if credit_amount is not None and credit_amount > 0:
                transaction["credit"] = credit_amount
            elif debit_amount is not None and debit_amount > 0:
                transaction["debit"] = debit_amount
            
            if account_number:
                transaction["account_number"] = account_number
            
            transactions.append(transaction)
        
        if not transactions:
            return {
                "status": "failed",
                "error": "No valid transactions found in CSV"
            }
        
        log_message("info", f"Extracted {len(transactions)} transactions from bank CSV")
        
        # Extract bank name from filename if possible (simple heuristic)
        bank_name = "Unknown"
        if hasattr(file, 'filename') and file.filename:
            filename_lower = file.filename.lower()
            if 'wells' in filename_lower or 'wf' in filename_lower:
                bank_name = "Wells Fargo"
            elif 'chase' in filename_lower:
                bank_name = "Chase Bank"
            elif 'boa' in filename_lower or 'bofa' in filename_lower or 'bank of america' in filename_lower:
                bank_name = "Bank of America"
        
        return {
            "status": "success",
            "transactions": transactions,
            "metadata": {
                "bank": bank_name,
                "account_number": account_number,
                "start_date": None,
                "end_date": None,
            }
        }
        
    except Exception as e:
        log_message("error", f"Bank CSV parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": f"Unable to process bank CSV file: {str(e)}"}


async def parse_bank_excel(file: Any) -> Dict[str, Any]:
    """
    Parse bank transactions from Excel (XLSX) file.
    
    Expected Excel columns:
    - Date (required): Transaction date
    - Description (required): Transaction description
    - Amount (required): Transaction amount (positive for credits, negative for debits)
    - Cheque No (optional): Check number
    
    Returns transactions in same format as parse_bank_pdf (with credit/debit fields)
    
    Args:
        file: Excel file containing bank transactions
        
    Returns:
        Dict with transactions list and metadata (same format as parse_bank_pdf)
    """
    try:
        log_message("info", "Parsing bank Excel file")
        
        content = await file.read()
        file.file.seek(0)
        
        # Read Excel file with pandas
        try:
            df = pd.read_excel(BytesIO(content), engine='openpyxl')
            log_message("info", f"Excel loaded: {len(df)} rows, columns: {list(df.columns)}")
        except Exception as e:
            log_message("error", f"Failed to read Excel file: {e}")
            return {"status": "failed", "error": f"Failed to read Excel file: {str(e)}"}
        
        # Normalize column names (case-insensitive, strip whitespace)
        df.columns = df.columns.str.strip().str.lower()
        
        log_message("info", f"Normalized column names: {list(df.columns)}")
        
        # Find required columns (case-insensitive search)
        date_col = None
        description_col = None
        amount_col = None
        cheque_col = None
        account_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if not date_col and 'date' in col_lower:
                date_col = col
            if not description_col and 'description' in col_lower:
                description_col = col
            if not amount_col and 'amount' in col_lower:
                amount_col = col
            if not cheque_col and ('cheque' in col_lower or 'check' in col_lower) and 'no' in col_lower:
                cheque_col = col
            if not account_col and ('account' in col_lower and 'number' in col_lower):
                account_col = col
        
        log_message("info", f"Column mapping: date={date_col}, description={description_col}, amount={amount_col}, cheque={cheque_col}, account={account_col}")
        
        # Validate required columns
        if not date_col:
            return {"status": "failed", "error": "Excel file must have a 'Date' column"}
        
        if not description_col:
            return {"status": "failed", "error": "Excel file must have a 'Description' column"}
        
        if not amount_col:
            return {"status": "failed", "error": "Excel file must have an 'Amount' column"}
        
        # Extract transactions
        transactions = []
        account_number = None
        
        for idx, row in df.iterrows():
            try:
                # Skip empty rows
                if pd.isna(row[date_col]) and pd.isna(row[description_col]):
                    continue
                
                # Parse date
                date_value = row[date_col]
                if pd.isna(date_value):
                    log_message("warning", f"Row {idx + 1}: Missing date, skipping")
                    continue
                
                # Handle date parsing
                if isinstance(date_value, str):
                    # Try to parse string date
                    try:
                        from dateutil import parser as date_parser
                        parsed_date = date_parser.parse(date_value)
                        date_str = parsed_date.strftime("%m/%d/%Y")
                    except Exception:
                        log_message("warning", f"Row {idx + 1}: Invalid date format '{date_value}', skipping")
                        continue
                elif isinstance(date_value, (pd.Timestamp, datetime)):
                    date_str = date_value.strftime("%m/%d/%Y")
                else:
                    log_message("warning", f"Row {idx + 1}: Unrecognized date type '{type(date_value)}', skipping")
                    continue
                
                # Parse description
                description = row[description_col]
                if pd.isna(description):
                    description = ""
                else:
                    description = str(description).strip()
                
                if not description:
                    log_message("warning", f"Row {idx + 1}: Empty description, skipping")
                    continue
                
                # Parse amount
                amount_value = row[amount_col]
                if pd.isna(amount_value):
                    log_message("warning", f"Row {idx + 1}: Missing amount, skipping")
                    continue
                
                try:
                    # Convert to float
                    amount = float(amount_value)
                except (ValueError, TypeError):
                    log_message("warning", f"Row {idx + 1}: Invalid amount '{amount_value}', skipping")
                    continue
                
                # Determine credit or debit based on amount sign
                # Positive = Credit, Negative = Debit
                credit_amount = None
                debit_amount = None
                
                if amount > 0:
                    credit_amount = amount
                elif amount < 0:
                    debit_amount = abs(amount)  # Store as positive value
                else:
                    # Skip zero amount transactions
                    log_message("warning", f"Row {idx + 1}: Zero amount, skipping")
                    continue
                
                # Build transaction object
                transaction = {
                    "date": date_str,
                    "description": description,
                }
                
                # Add credit or debit (not both)
                if credit_amount is not None:
                    transaction["credit"] = credit_amount
                elif debit_amount is not None:
                    transaction["debit"] = debit_amount
                
                # Add check number if available
                if cheque_col and not pd.isna(row[cheque_col]):
                    cheque_no = str(row[cheque_col]).strip()
                    if cheque_no:
                        transaction["check_number"] = cheque_no
                
                # Extract account number if available
                if account_col and not pd.isna(row[account_col]):
                    acc_num = str(row[account_col]).strip()
                    if acc_num and not account_number:
                        account_number = acc_num
                    if acc_num:
                        transaction["account_number"] = acc_num
                
                transactions.append(transaction)
                
            except Exception as row_error:
                log_message("warning", f"Row {idx + 1}: Error processing row - {row_error}")
                continue
        
        if not transactions:
            return {
                "status": "failed",
                "error": "No valid transactions found in Excel file"
            }
        
        log_message("info", f"Extracted {len(transactions)} transactions from bank Excel file")
        
        # Extract bank name from filename if possible
        bank_name = "Unknown"
        if hasattr(file, 'filename') and file.filename:
            filename_lower = file.filename.lower()
            if 'wells' in filename_lower or 'wf' in filename_lower:
                bank_name = "Wells Fargo"
            elif 'chase' in filename_lower:
                bank_name = "Chase Bank"
            elif 'boa' in filename_lower or 'bofa' in filename_lower or 'bank of america' in filename_lower:
                bank_name = "Bank of America"
        
        return {
            "status": "success",
            "transactions": transactions,
            "metadata": {
                "bank": bank_name,
                "account_number": account_number,
                "start_date": None,
                "end_date": None,
            }
        }
        
    except Exception as e:
        log_message("error", f"Bank Excel parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": f"Unable to process bank Excel file: {str(e)}"}


async def parse_coa_pdf(file: Any) -> Dict[str, Any]:
    """
    Parse Chart of Accounts from PDF file.
    
    Extracts ALL columns from PDF tables as structured data.
    
    Args:
        file: PDF file containing COA
        
    Returns:
        Dict with COA data as list of dictionaries (all columns preserved)
    """
    try:
        import pdfplumber
        
        log_message("info", "Parsing COA PDF file")
        
        # Read file content
        content = await file.read()
        file.file.seek(0)
        
        coa_data = []
        columns = []
        first_table = True
        
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from page
                tables = page.extract_tables()
                
                for table in tables:
                    if not table:
                        continue
                    
                    # First row might be headers
                    if first_table and len(table) > 0:
                        # Try to detect header row
                        header_row = table[0]
                        if header_row and any(cell and str(cell).strip() for cell in header_row):
                            # Use first row as column names
                            columns = [str(cell).strip() if cell else f"Column_{i+1}" for i, cell in enumerate(header_row)]
                            first_table = False
                            # Skip header row in data
                            table = table[1:]
                        else:
                            # No header detected, use generic column names
                            if len(table) > 0 and len(table[0]) > 0:
                                num_cols = len(table[0])
                                columns = [f"Column_{i+1}" for i in range(num_cols)]
                            first_table = False
                    
                    # Process data rows
                    for row_idx, row in enumerate(table):
                        if not row:
                            # Empty row - create dict with empty values
                            coa_row = {col: "" for col in columns} if columns else {}
                            coa_data.append(coa_row)
                            continue
                        
                        # Clean all cells in row
                        row_clean = [str(cell).strip() if cell else "" for cell in row]
                        
                        # Create dict with all columns
                        coa_row = {}
                        for i, col in enumerate(columns):
                            if i < len(row_clean):
                                coa_row[col] = row_clean[i]
                            else:
                                coa_row[col] = ""
                        
                        # If no columns detected yet, create generic structure
                        if not columns and row_clean:
                            num_cols = len(row_clean)
                            columns = [f"Column_{i+1}" for i in range(num_cols)]
                            coa_row = {col: row_clean[i] if i < len(row_clean) else "" for i, col in enumerate(columns)}
                        
                        coa_data.append(coa_row)
        
        log_message("info", f"Extracted {len(coa_data)} COA entries from PDF with {len(columns)} columns each")
        
        return {
            "status": "success",
            "coa_list": coa_data,
            "columns": columns
        }
        
    except ImportError:
        log_message("error", "pdfplumber not installed. Install with: pip install pdfplumber")
        return {"status": "failed", "error": "PDF parsing requires pdfplumber library"}
    except Exception as e:
        log_message("error", f"COA PDF parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "Unable to process COA file. Please ensure the file is a valid PDF, CSV, or Excel file."}


async def parse_coa_csv(file: Any) -> Dict[str, Any]:
    """
    Parse Chart of Accounts from CSV file.
    
    Extracts ALL columns (Account, Type, Balance, Description, etc.) as structured data.
    
    Args:
        file: CSV file containing COA
        
    Returns:
        Dict with COA data as list of dictionaries (all columns preserved)
    """
    try:
        log_message("info", "Parsing COA CSV file")
        
        content = await file.read()
        file.file.seek(0)
        
        # Read CSV with pandas
        df = pd.read_csv(BytesIO(content))
        
        log_message("info", f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        # Normalize column names (preserve original for display)
        original_columns = list(df.columns)
        df.columns = df.columns.str.strip().str.lower()
        
        # Build COA list with ALL columns as structured data
        coa_list = []
        for idx, row in df.iterrows():
            # Create dict with all columns (use original column names)
            coa_row = {}
            for orig_col, norm_col in zip(original_columns, df.columns):
                value = row[norm_col]
                if pd.isna(value):
                    coa_row[orig_col] = ""
                else:
                    coa_row[orig_col] = str(value).strip()
            
            coa_list.append(coa_row)
        
        log_message("info", f"Extracted {len(coa_list)} COA entries from CSV with {len(original_columns)} columns each")
        
        return {
            "status": "success",
            "coa_list": coa_list,
            "columns": original_columns
        }
        
    except Exception as e:
        log_message("error", f"COA CSV parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "Unable to process COA file. Please ensure the file is a valid CSV format."}


async def parse_coa_excel(file: Any) -> Dict[str, Any]:
    """
    Parse Chart of Accounts from Excel file.
    
    Extracts ALL columns (Account, Type, Balance, Description, etc.) as structured data.
    
    Args:
        file: Excel file (.xlsx, .xls, .xlsm) containing COA
        
    Returns:
        Dict with COA data as list of dictionaries (all columns preserved)
    """
    try:
        log_message("info", "Parsing COA Excel file")
        
        content = await file.read()
        file.file.seek(0)
        
        # Read Excel with pandas (reads first sheet by default)
        df = pd.read_excel(BytesIO(content), engine='openpyxl')
        
        log_message("info", f"Excel loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        # Normalize column names (preserve original for display)
        original_columns = list(df.columns)
        df.columns = df.columns.str.strip().str.lower()
        
        # Build COA list with ALL columns as structured data
        coa_list = []
        for idx, row in df.iterrows():
            # Create dict with all columns (use original column names)
            coa_row = {}
            for orig_col, norm_col in zip(original_columns, df.columns):
                value = row[norm_col]
                if pd.isna(value):
                    coa_row[orig_col] = ""
                else:
                    coa_row[orig_col] = str(value).strip()
            
            coa_list.append(coa_row)
        
        log_message("info", f"Extracted {len(coa_list)} COA entries from Excel with {len(original_columns)} columns each")
        
        return {
            "status": "success",
            "coa_list": coa_list,
            "columns": original_columns
        }
        
    except ImportError as e:
        log_message("error", "openpyxl not installed. Install with: pip install openpyxl")
        return {"status": "failed", "error": "Excel parsing requires openpyxl library"}
    except Exception as e:
        log_message("error", f"COA Excel parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "Unable to process COA file. Please ensure the file is a valid Excel format."}


async def parse_files_by_type(bank_file: Any, bank_file_type: str, 
                               coa_file: Any, coa_file_type: str) -> Dict[str, Any]:
    """
    Main parsing function that routes to appropriate parser based on file type.
    
    Args:
        bank_file: Bank transactions file
        bank_file_type: Type of bank file ('pdf')
        coa_file: COA file
        coa_file_type: Type of COA file ('pdf', 'csv', 'excel')
        
    Returns:
        Dict with parsed bank transactions and COA list
    """
    try:
        log_message("info", f"Parsing files: bank={bank_file_type}, coa={coa_file_type}")
        
        # Parse bank file (always PDF)
        if bank_file_type != 'pdf':
            return {"status": "failed", "error": "Bank file must be a PDF document."}
        
        bank_result = await parse_bank_pdf(bank_file)
        if bank_result.get("status") == "failed":
            return bank_result
        
        # Parse COA file based on type
        coa_result = None
        if coa_file_type == 'pdf':
            coa_result = await parse_coa_pdf(coa_file)
        elif coa_file_type == 'csv':
            coa_result = await parse_coa_csv(coa_file)
        elif coa_file_type == 'excel':
            coa_result = await parse_coa_excel(coa_file)
        else:
            return {"status": "failed", "error": "COA file must be a PDF, CSV, or Excel document."}
        
        if coa_result.get("status") == "failed":
            return coa_result
        
        return {
            "status": "success",
            "bank_transactions": bank_result.get("transactions", []),
            "bank_metadata": bank_result.get("metadata", {}),
            "coa_list": coa_result.get("coa_list", []),
            "summary": {
                "bank_transactions_count": len(bank_result.get("transactions", [])),
                "coa_categories_count": len(coa_result.get("coa_list", []))
            }
        }
        
    except Exception as e:
        log_message("error", f"File parsing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {"status": "failed", "error": "Unable to process files. Please check that both files are valid and try again."}


# ============================================================
#           TRANSACTION STANDARDIZATION
# ============================================================

def standardize_bank_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardize bank transactions to unified format: Date, Memo, Amount.
    Preserve bank name and payee name if provided.
    
    Combines credit/debit into single Amount (positive for credits, negative for debits).
    Preserves memo text as-is without cleaning or refinement.
    
    Args:
        transactions: List of raw transactions with date, post_date, description, credit, debit, bank
        
    Returns:
        List of standardized transactions with Date, Memo, Amount
    """
    standardized = []
    skipped_no_date = 0
    skipped_no_memo = 0
    skipped_no_amount = 0
    
    log_message("info", f"Standardizing {len(transactions)} raw transactions")
    
    for idx, txn in enumerate(transactions):
        date = txn.get("date") or txn.get("post_date")
        if not date:
            skipped_no_date += 1
            if idx < 5:
                log_message("warning", f"Transaction {idx} skipped - no date. Keys: {list(txn.keys())}, date={txn.get('date')}, post_date={txn.get('post_date')}")
            continue
        
        memo = str(txn.get("description", "")).strip()
        if not memo:
            skipped_no_memo += 1
            if idx < 5:
                log_message("warning", f"Transaction {idx} skipped - no description. Date: {date}")
            continue
        
        credit = txn.get("credit")
        debit = txn.get("debit")
        
        if credit is not None:
            amount = float(credit)
        elif debit is not None:
            amount = -float(debit)
        else:
            skipped_no_amount += 1
            if idx < 5:
                log_message("warning", f"Transaction {idx} skipped - no amount. Date: {date}, Memo: {memo[:50]}")
            continue
        
        # Handle payee field - check both payee (from check mapping) and payee_name, with None safety
        payee_value = txn.get("payee") or txn.get("payee_name") or ""
        
        standardized.append({
            "Date": date,
            "Memo": memo,
            "Amount": round(amount, 2),
            "Bank": txn.get("bank", ""),
            "Payee": payee_value if payee_value else ""  # Ensure it's never None
        })
    
    log_message("info", f"Standardized {len(standardized)} transactions (skipped: {skipped_no_date} no date, {skipped_no_memo} no memo, {skipped_no_amount} no amount)")
    return standardized


def create_working_list_for_pinecone(
    standardized_transactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create working list of transactions for Pinecone categorization.
    
    All transactions go through Pinecone (no COA matching).
    Groups transactions into batches of 10 for processing.
    
    Args:
        standardized_transactions: List of {Date, Memo, Amount}
        
    Returns:
        Dict with working_list and batches
    """
    working_list = []
    
    for idx, txn in enumerate(standardized_transactions):
        memo = txn.get("Memo", "").strip()
        amount = txn.get("Amount", 0)
        date = txn.get("Date", "")
        is_check = txn.get("is_check_transaction", False)
        payee_name = txn.get("Payee", "") or txn.get("payee", "")  # Get payee from check mapping CSV
        
        if not memo:
            continue
        
        transaction_item = {
            "transaction_id": str(uuid.uuid4())[:8],
            "date": date,
            "memo": memo,
            "amount": amount,
            "is_check_transaction": is_check,
            "Payee": payee_name.strip() if payee_name else "",  # Include payee for check transaction category lookup
            "payee": payee_name.strip() if payee_name else ""  # Also include lowercase version for compatibility
        }
        
        working_list.append(transaction_item)
    
    batches = []
    batch_size = 10
    for i in range(0, len(working_list), batch_size):
        batches.append(working_list[i:i + batch_size])
    
    log_message("info", f"Created {len(batches)} batches of 10 transactions for Pinecone")
    
    return {
        "working_list": working_list,
        "batches": batches,
        "summary": {
            "total_transactions": len(standardized_transactions),
            "total_batches": len(batches)
        }
    }


def extract_coa_categories(coa_list: List[Dict[str, Any]]) -> List[str]:
    """
    Extract unique categories/account names from COA data.
    
    Primary column: "Account" - This column contains all the categories.
    Falls back to other columns if "Account" is not found.
    
    Args:
        coa_list: List of COA entries (dictionaries with various columns)
        
    Returns:
        List of unique category names (e.g., ["taxes", "loan", "service"])
    """
    if not coa_list:
        return []
    
    categories = set()
    
    # Priority order for column names to extract categories from
    # "Account" is the primary column that contains all categories
    category_columns = ['account', 'account name', 'name', 'category', 'type', 'description']
    
    for coa_item in coa_list:
        if not isinstance(coa_item, dict):
            continue
        
        # Try each column in priority order
        for col_name in category_columns:
            # Find matching key (case-insensitive)
            matching_key = None
            for key in coa_item.keys():
                if key.lower() == col_name.lower():
                    matching_key = key
                    break
            
            if matching_key and coa_item.get(matching_key):
                value = str(coa_item[matching_key]).strip()
                if value and value.lower() not in ['', 'nan', 'none', 'null']:
                    # Clean the value - remove extra whitespace, convert to lowercase
                    cleaned = ' '.join(value.split()).lower()
                    if cleaned:
                        categories.add(cleaned)
                    break  # Found a value, move to next COA item
    
    # Convert to sorted list
    return sorted(list(categories))


async def predict_category_from_coa_llm(
    transaction_description: str,
    transaction_amount: float,
    transaction_date: str,
    coa_categories: List[str]
) -> Optional[str]:
    """
    Use Gemini LLM to predict a category from COA for a transaction that got "Ask My Accountant".
    
    This function only suggests categories that exist in the COA list (from the "Account" column).
    If no suitable category is found, it returns None (which means keep "Ask My Accountant").
    
    Args:
        transaction_description: Transaction memo/description
        transaction_amount: Transaction amount (positive for credit, negative for debit)
        transaction_date: Transaction date string
        coa_categories: List of available COA categories from the "Account" column (from extract_coa_categories)
    
    Returns:
        Category name from COA "Account" column if found, None if no suitable category (keep "Ask My Accountant")
    """
    if not coa_categories or len(coa_categories) == 0:
        log_message("warning", "No COA categories provided for LLM prediction")
        return None
    
    if not transaction_description or not transaction_description.strip():
        log_message("warning", "Empty transaction description for LLM prediction")
        return None
    
    try:
        # Load the prompt template
        from pathlib import Path
        current_dir = Path(__file__).parent.parent
        prompt_file_path = current_dir / "services" / "prompts" / "coa_category_prediction_prompt.txt"
        
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Format the prompt with actual values
        # Convert COA categories list to a readable format
        coa_categories_str = "\n".join([f"- {cat}" for cat in coa_categories])
        
        formatted_prompt = prompt_template.format(
            coa_categories=coa_categories_str,
            transaction_description=transaction_description.strip(),
            transaction_amount=f"{transaction_amount:.2f}",
            transaction_date=transaction_date or "Unknown"
        )
        
        log_message("debug", f"Calling Gemini LLM to predict category from COA for transaction: {transaction_description[:50]}...")
        
        # Call Gemini LLM
        response = await asyncio.to_thread(llm.invoke, formatted_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_text = response_text.strip()
        
        # Check if response is "Ask My Accountant" or empty
        if not response_text or response_text.lower() == "ask my accountant":
            log_message("debug", f"LLM returned 'Ask My Accountant' or empty for transaction: {transaction_description[:50]}")
            return None
        
        # Check if the response matches any COA category (case-insensitive)
        response_lower = response_text.lower().strip()
        for coa_cat in coa_categories:
            if coa_cat.lower() == response_lower:
                log_message("info", f"LLM predicted category '{coa_cat}' from COA for transaction: {transaction_description[:50]}")
                # Return the original case from COA to preserve formatting
                return coa_cat
        
        # If response doesn't match any COA category, log warning and return None
        log_message("warning", f"LLM returned category '{response_text}' which is not in COA list. Keeping 'Ask My Accountant'.")
        return None
        
    except FileNotFoundError:
        log_message("error", f"COA category prediction prompt file not found: {prompt_file_path}")
        return None
    except Exception as e:
        log_message("error", f"Error predicting category from COA using LLM: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return None


def find_direct_coa_match(memo: str, coa_list: List[Any]) -> Optional[str]:
    """
    Simple keyword matching to find direct COA matches.
    
    Returns COA category if found, None if needs AI.
    Handles both structured data (dicts) and legacy string format.
    
    Args:
        memo: Transaction memo text
        coa_list: List of COA entries (can be dicts with all columns or strings)
        
    Returns:
        Matched COA account name or None
    """
    if not memo or not coa_list:
        return None
    
    memo_lower = memo.lower()
    
    # Handle structured data (dicts) or legacy string format
    for coa_item in coa_list:
        if isinstance(coa_item, dict):
            # Structured data - look for Account Name, Account, or Name column
            account_name = ""
            for key in ['account name', 'account', 'name', 'category']:
                if key.lower() in [k.lower() for k in coa_item.keys()]:
                    # Find the matching key (case-insensitive)
                    matching_key = next((k for k in coa_item.keys() if k.lower() == key.lower()), None)
                    if matching_key and coa_item[matching_key]:
                        account_name = str(coa_item[matching_key]).strip()
                        break
            
            if not account_name:
                continue
            
            # Check if memo contains account name keywords
            account_lower = account_name.lower()
            keywords = [kw for kw in account_lower.split() if len(kw) > 3]
            if keywords and any(keyword in memo_lower for keyword in keywords):
                return account_name
        else:
            # Legacy string format
            if not coa_item or str(coa_item).strip() == "":
                continue
                
            coa_lower = str(coa_item).lower()
            
            # Extract category name (after " - " if present)
            if " - " in coa_lower:
                category_name = coa_lower.split(" - ", 1)[1].strip()
            else:
                category_name = coa_lower.strip()
            
            # Check if memo contains category keywords (words longer than 3 chars)
            keywords = [kw for kw in category_name.split() if len(kw) > 3]
            if keywords and any(keyword in memo_lower for keyword in keywords):
                return str(coa_item)
    
    return None


# ============================================================
#           PINECONE SIMILARITY SEARCH
# ============================================================

def generate_openai_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding vector using OpenAI text-embedding-3-small.
    
    Args:
        text: Text to embed (transaction memo)
        
    Returns:
        List of floats representing the embedding vector, or None if failed
    """
    try:
        if not openai_client:
            log_message("error", "OpenAI client not initialized. Check OPENAI_API_KEY in .env")
            return None
        
        if not text or not text.strip():
            return None
        
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text.strip()
        )
        
        return response.data[0].embedding
        
    except Exception as e:
        log_message("error", f"Failed to generate OpenAI embedding: {e}")
        return None


# ============================================================
#           PINECONE CATEGORIZATION
# ============================================================

async def batch_embed_transactions(memos: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple transaction memos in batch.
    
    Uses embed_documents() for efficient batch processing instead of
    calling embed_query() multiple times.
    
    Args:
        memos: List of transaction memo strings
        
    Returns:
        List of embedding vectors (one per memo)
    """
    try:
        if not memos:
            return []
        
        valid_memos = [memo for memo in memos if memo and memo.strip()]
        if not valid_memos:
            return []
        
        embedding_vectors = await asyncio.to_thread(
            embeddings.embed_documents,
            valid_memos
        )
        
        result = []
        memo_idx = 0
        for memo in memos:
            if memo and memo.strip():
                if memo_idx < len(embedding_vectors):
                    result.append(embedding_vectors[memo_idx])
                    memo_idx += 1
                else:
                    result.append([])
            else:
                result.append([])
        
        return result
        
    except Exception as e:
        log_message("error", f"Batch embedding generation failed: {e}")
        return [[] for _ in memos]  # Return empty embeddings on error


async def search_pinecone_by_payee_name(
    payee_name: str,
    client_name: Optional[str] = None,
    index_name: Optional[str] = None,
    memo: Optional[str] = None,
    credit: Optional[str] = None,
    debit: Optional[str] = None
) -> Optional[str]:
    """
    Search Pinecone GL data by payee name to find category.
    
    This function searches the Pinecone index for transactions with matching payee name
    in the 'name' metadata field and returns the category from the first (newest) match.
    
    Uses Pinecone metadata filtering to directly filter by the 'name' field for exact matches.
    Only matches by payee name - does not check memo, credit, or debit.
    
    Args:
        payee_name: Payee name to search for (e.g., from check mapping CSV)
        client_name: Client name for filtering and index resolution (optional)
        index_name: Pre-resolved Pinecone index name (optional, takes precedence)
        memo: Transaction memo (ignored - kept for backward compatibility)
        credit: Credit amount (ignored - kept for backward compatibility)
        debit: Debit amount (ignored - kept for backward compatibility)
    
    Returns:
        Category string if found, None otherwise
    """
    try:
        if not payee_name or not payee_name.strip():
            return None
        
        payee_name_clean = payee_name.strip()
        
        # Resolve index name
        if index_name is None:
            from backend.services.pinecone import get_index_name_from_client
            index_name = get_index_name_from_client(client_name, log_result=False)
        
        # Query Pinecone directly with metadata filter on 'name' field
        # We need to use Pinecone's query API with a filter
        from pinecone import Pinecone
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            log_message("error", "PINECONE_API_KEY not configured")
            return None
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if not pc.has_index(index_name):
            log_message("warning", f"Index '{index_name}' does not exist")
            return None
        
        index = pc.Index(index_name)
        
        # Build filter: exact match on 'name' field (case-sensitive in Pinecone)
        # We'll need to query with a dummy vector and filter by name
        # Generate a dummy embedding (we'll use a zero vector or the payee name embedding)
        payee_embedding = embeddings.embed_query(payee_name_clean)
        
        # Build filter dictionary
        # Check if this is a client-specific index
        is_client_specific_index = index_name and index_name.endswith('-data') and '-' in index_name
        
        # Build filter: name must match exactly (case-sensitive)
        filter_dict = {"name": {"$eq": payee_name_clean}}
        
        # Also add client_name filter if needed (for shared indexes)
        if client_name and not is_client_specific_index:
            filter_dict = {
                "$and": [
                    {"name": {"$eq": payee_name_clean}},
                    {"client_name": {"$eq": client_name}}
                ]
            }
        
        log_message("debug", f"Searching Pinecone for payee name '{payee_name_clean}' with filter: {filter_dict}")
        
        # Query Pinecone with embedding and metadata filter
        # Use a large top_k to ensure we get all matches
        query_response = await asyncio.to_thread(
            index.query,
            vector=payee_embedding,
            top_k=100,  # Large top_k to get all matching records
            include_metadata=True,
            filter=filter_dict
        )
        
        matches = query_response.get('matches', [])
        
        if not matches:
            log_message("debug", f"No Pinecone results found for payee name: {payee_name_clean[:50]}")
            return None
        
        log_message("debug", f"Found {len(matches)} Pinecone matches for payee name: {payee_name_clean[:50]}")
        
        # Extract results with metadata (only match by payee name, ignore memo/credit/debit)
        matching_results = []
        for match in matches:
            metadata = match.get('metadata', {})
            result_name = metadata.get("name", "").strip()
            
            # Double-check name match (case-insensitive comparison for safety)
            if result_name and result_name.lower() == payee_name_clean.lower():
                category = metadata.get("split", "") or metadata.get("category", "")
                
                matching_results.append({
                    "name": result_name,
                    "split": category,
                    "category": category,
                    "date": metadata.get("date", ""),
                    "score": match.get("score", 0.0)
                })
        
        if not matching_results:
            # Try case-insensitive fallback: query without filter and filter in code
            log_message("debug", f"No exact case-sensitive match, trying case-insensitive search for: {payee_name_clean[:50]}")
            
            # Query with larger top_k and filter in code (case-insensitive)
            query_response_fallback = await asyncio.to_thread(
                index.query,
                vector=payee_embedding,
                top_k=100,
                include_metadata=True,
                filter={"client_name": {"$eq": client_name}} if client_name and not is_client_specific_index else None
            )
            
            payee_lower = payee_name_clean.lower()
            for match in query_response_fallback.get('matches', []):
                metadata = match.get('metadata', {})
                result_name = metadata.get("name", "").strip()
                if result_name and result_name.lower() == payee_lower:
                    category = metadata.get("split", "") or metadata.get("category", "")
                    
                    matching_results.append({
                        "name": result_name,
                        "split": category,
                        "category": category,
                        "date": metadata.get("date", ""),
                        "score": match.get("score", 0.0)
                    })
            
            if not matching_results:
                log_message("debug", f"No payee name match found (case-sensitive or case-insensitive) in Pinecone for: {payee_name_clean[:50]}")
                return None
        
        # Sort by date (newest first) - only match by payee name, no memo/credit/debit checking
        from datetime import datetime
        
        def parse_date(date_str):
            """Parse date string and return a sortable value"""
            if not date_str:
                return None
            try:
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                    try:
                        return datetime.strptime(str(date_str).strip(), fmt)
                    except ValueError:
                        continue
                return None
            except:
                return None
        
        # Add parsed dates for sorting
        for match in matching_results:
            match['_parsed_date'] = parse_date(match.get('date', ''))
        
        # Sort by date (newest first) - simple sorting, no memo/amount prioritization
        matching_results.sort(
            key=lambda x: x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
            reverse=True
        )
        
        # Get category from newest match
        best_match = matching_results[0]
        category = best_match.get("split", "").strip() or best_match.get("category", "").strip()
        
        if category:
            log_message("debug", f"Found category '{category}' from GL for payee name '{payee_name_clean[:50]}' (newest match by date)")
            return category
        
        log_message("debug", f"Found payee match but no category in GL for: {payee_name[:50]}")
        return None
        
    except Exception as e:
        log_message("error", f"Error searching Pinecone by payee name '{payee_name}': {e}")
        return None


async def parallel_pinecone_query(
    embedding_vector: List[float],
    client_name: Optional[str] = None,
    top_k: int = 1,
    index_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query Pinecone asynchronously for a single embedding vector.
    
    Wraps the synchronous query_pinecone_with_embedding in asyncio.to_thread
    to allow parallel execution.
    
    Args:
        embedding_vector: Embedding vector to query
        client_name: Client name for filtering
        top_k: Number of results to return
        index_name: Pre-computed index name (avoids repeated lookups)
        
    Returns:
        List of Pinecone results (same format as query_pinecone_with_embedding)
    """
    try:
        if not embedding_vector:
            return []
        
        results = await asyncio.to_thread(
            query_pinecone_with_embedding,
            embedding_vector=embedding_vector,
            client_name=client_name,
            top_k=top_k,
            index_name=index_name
        )
        
        return results if results else []
        
    except Exception as e:
        log_message("error", f"Pinecone query failed: {e}")
        return []


async def categorize_with_pinecone(
    transactions_batch: List[Dict[str, Any]], 
    client_name: Optional[str] = None,
    index_name: Optional[str] = None,
    llm: bool = False,
    coa_categories: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Optimized Pinecone categorization using batch embeddings and parallel queries.
    
    For each transaction, searches Pinecone and uses FIRST result's category.
    Prioritizes exact memo matches - if an exact match exists, it will be ranked first.
    Now uses batch embedding generation and parallel Pinecone queries for speed.
    
    If llm=True and category is "Ask My Accountant", uses Gemini LLM to predict
    a category from the provided COA categories.
    
    Args:
        transactions_batch: List of transactions to categorize
        client_name: Client name for filtering Pinecone results (optional, used if index_name not provided)
        index_name: Pre-resolved Pinecone index name (optional, takes precedence over client_name)
        llm: If True, use LLM to predict category from COA when "Ask My Accountant" (default: False)
        coa_categories: List of COA categories for LLM prediction (required if llm=True)
        
    Returns:
        List of categorized transactions with {date, bank_memo, credit, debit, category}
    """
    try:
        if not transactions_batch:
            return []
        
        # Prepare transaction data
        transaction_data = []
        memos = []
        
        for txn in transactions_batch:
            memo = txn.get('memo', '').strip()
            date = txn.get('date', '')
            amount = txn.get('amount', 0)
            is_check = txn.get('is_check_transaction', False)
            payee_name = txn.get('Payee', '') or txn.get('payee', '')  # Get payee from check mapping CSV
            
            transaction_data.append({
                'memo': memo,
                'date': date,
                'amount': amount,
                'is_check_transaction': is_check,
                'payee_name': payee_name.strip() if payee_name else ''
            })
            memos.append(memo)
        
        # Use provided index_name if available, otherwise resolve from client_name
        if index_name:
            cached_index_name = index_name
            log_message("info", f"Using provided Pinecone index name: {cached_index_name}")
        else:
            from backend.services.pinecone import get_index_name_from_client
            cached_index_name = get_index_name_from_client(client_name, log_result=False)
            if client_name:
                log_message("info", f"Resolved Pinecone index name from client_name '{client_name}': {cached_index_name}")
        
        embedding_vectors = await batch_embed_transactions(memos)
        
        semaphore = asyncio.Semaphore(30)
        
        async def query_with_semaphore(embedding_vec: List[float], memo: str, amount: float) -> List[Dict[str, Any]]:
            async with semaphore:
                normalized_input_memo = memo.strip().lower()
                
                # Normalize check memos for comparison (handles "Check 3340" vs "CHECK # 3340")
                def normalize_check_memo(m):
                    """Normalize check memo for comparison - removes #, normalizes spaces, extracts check number"""
                    if not m:
                        return ""
                    # Remove #, normalize spaces, lowercase
                    normalized = m.replace("#", "").replace("  ", " ").strip().lower()
                    # Remove common check prefixes if present (for comparison only)
                    for prefix in ["check", "chk", "ch"]:
                        if normalized.startswith(prefix + " "):
                            # Keep the number part
                            normalized = normalized[len(prefix):].strip()
                    return normalized
                
                normalized_input_memo_check = normalize_check_memo(memo)
                
                # Calculate credit/debit for matching
                if amount >= 0:
                    input_credit = f"{amount:.2f}"
                    input_debit = ""
                else:
                    input_credit = ""
                    input_debit = f"{abs(amount):.2f}"
                
                # Query with top_k=50 to find exact matches
                results = await parallel_pinecone_query(
                    embedding_vector=embedding_vec,
                    client_name=client_name,
                    top_k=50,  # Query more results to find exact match
                    index_name=cached_index_name
                )
                
                # Check for exact match first (memo + credit/debit)
                if results and len(results) > 0:
                    exact_matches = []  # Collect ALL exact matches
                    memo_only_matches = []  # Collect ALL memo-only matches
                    partial_memo_matches = []  # Collect partial memo matches (contains or is contained)
                    
                    for result in results:
                        # Safely convert memo to string and strip
                        result_memo_raw = result.get("memo", "")
                        result_memo = str(result_memo_raw).strip().lower() if result_memo_raw else ""
                        normalized_result_memo_check = normalize_check_memo(result_memo_raw) if result_memo_raw else ""
                        
                        # Convert credit/debit to string first, then strip (handles float, None, or string)
                        result_credit_raw = result.get("credit", "")
                        if result_credit_raw is not None and result_credit_raw != "":
                            result_credit = str(result_credit_raw).strip()
                        else:
                            result_credit = ""
                        
                        result_debit_raw = result.get("debit", "")
                        if result_debit_raw is not None and result_debit_raw != "":
                            result_debit = str(result_debit_raw).strip()
                        else:
                            result_debit = ""
                        
                        # Check for exact memo match (both regular and normalized check memo)
                        # First try exact match, then try normalized check memo match
                        memo_exact_match = (result_memo == normalized_input_memo)
                        if not memo_exact_match and normalized_result_memo_check and normalized_input_memo_check:
                            # Try normalized check memo comparison (handles "Check 3340" vs "CHECK # 3340")
                            memo_exact_match = (normalized_result_memo_check == normalized_input_memo_check)
                        
                        # Check for partial memo match (one contains the other)
                        # This helps with "Deposit" matching "Deposit Bar Sales" or vice versa
                        memo_partial_match = False
                        if not memo_exact_match and result_memo and normalized_input_memo:
                            # Check if one memo contains the other (must be at least 4 chars to avoid false positives like "Inc")
                            if len(result_memo) >= 4 and len(normalized_input_memo) >= 4:
                                memo_partial_match = (
                                    result_memo in normalized_input_memo or 
                                    normalized_input_memo in result_memo
                                )
                        
                        memo_match = memo_exact_match or memo_partial_match
                        
                        # Check amount match (credit or debit)
                        # For checks, match if absolute amount matches regardless of credit/debit
                        amount_match = False
                        try:
                            # Get absolute amounts from both sides
                            input_amount = 0.0
                            if input_credit:
                                input_amount = float(input_credit)
                            elif input_debit:
                                input_amount = float(input_debit)
                            
                            result_amount = 0.0
                            if result_credit:
                                result_amount = float(result_credit)
                            elif result_debit:
                                result_amount = float(result_debit)
                            
                            # Match if absolute amounts are the same (handles credit vs debit for checks)
                            if abs(input_amount) > 0 and abs(result_amount) > 0:
                                if abs(abs(input_amount) - abs(result_amount)) < 0.01:  # Allow small rounding differences
                                    amount_match = True
                        except (ValueError, TypeError):
                            # Fallback to original logic
                            if input_credit and result_credit:
                                if input_credit == result_credit:
                                    amount_match = True
                            elif input_debit and result_debit:
                                if input_debit == result_debit:
                                    amount_match = True
                        
                        # Perfect match: memo + amount - collect all exact matches
                        if memo_match and amount_match:
                            exact_match = result.copy()  # Make a copy to avoid modifying original
                            exact_match["_is_exact_match"] = True  # Flag to indicate exact match
                            exact_match["_is_partial_memo"] = memo_partial_match  # Track if this was a partial match
                            exact_matches.append(exact_match)
                        # Memo-only match (exact or partial)
                        elif memo_match:
                            if memo_partial_match:
                                result_copy = result.copy()
                                result_copy["_is_partial_memo"] = True
                                memo_only_matches.append(result_copy)
                            else:
                                memo_only_matches.append(result)
                    
                    # If exact matches found, sort by date and return the newest one
                    if exact_matches:
                        from datetime import datetime
                        
                        def parse_date(date_str):
                            """Parse date string and return a sortable value"""
                            if not date_str:
                                return None
                            try:
                                # Try various date formats
                                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                                    try:
                                        return datetime.strptime(str(date_str).strip(), fmt)
                                    except ValueError:
                                        continue
                                return None
                            except:
                                return None
                        
                        # Parse dates for all exact matches
                        for match in exact_matches:
                            match['_parsed_date'] = parse_date(match.get('date', ''))
                        
                        # Sort by date (newest first)
                        exact_matches.sort(
                            key=lambda x: x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
                            reverse=True
                        )
                        
                        newest_exact = exact_matches[0]
                        log_message("debug", f"Found exact match for '{memo[:30]}'")
                        return [newest_exact]  # Return only the newest exact match
                    
                    # If memo-only matches found, filter by score >= 0.800 (same as test_single_transaction)
                    # and return the newest one
                    if memo_only_matches:
                        # Filter memo-only matches by score >= 0.800 (same threshold as test API)
                        high_score_memo_matches = [r for r in memo_only_matches if r.get("score", 0.0) >= 0.800]
                        
                        if high_score_memo_matches:
                            from datetime import datetime
                            
                            def parse_date(date_str):
                                """Parse date string and return a sortable value"""
                                if not date_str:
                                    return None
                                try:
                                    for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                                        try:
                                            return datetime.strptime(str(date_str).strip(), fmt)
                                        except ValueError:
                                            continue
                                    return None
                                except:
                                    return None
                            
                            # Parse dates for high-score memo-only matches
                            for match in high_score_memo_matches:
                                match['_parsed_date'] = parse_date(match.get('date', ''))
                            
                            # Sort by date (newest first)
                            high_score_memo_matches.sort(
                                key=lambda x: x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
                                reverse=True
                            )
                            
                            newest_memo = high_score_memo_matches[0]
                            log_message("debug", f"Found memo-only match with score >= 0.800 for '{memo[:30]}'")
                            return [newest_memo]
                        # If memo-only matches exist but none meet score threshold, continue to semantic matching
                
                # No exact or memo-only match found, return top 20 semantic results for score checking
                if results and len(results) > 0:
                    top_20 = results[:20]  # Return top 20 results for score checking and date sorting
                    return top_20
                
                return []
        
        async def empty_result():
            return []
        
        query_tasks = [
            query_with_semaphore(emb_vec, memos[idx], transaction_data[idx]['amount']) if emb_vec else empty_result()
            for idx, emb_vec in enumerate(embedding_vectors)
        ]
        
        pinecone_results_list = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        categorized = []
        for idx, txn_data in enumerate(transaction_data):
            memo = txn_data['memo']
            date = txn_data['date']
            amount = txn_data['amount']
            is_check = txn_data.get('is_check_transaction', False)
            
            if amount >= 0:
                credit = f"{amount:.2f}"
                debit = ""
            else:
                credit = ""
                debit = f"{abs(amount):.2f}"
            
            category = "Ask My Accountant"  # Default category
            payee_name = ""  # Initialize payee name
            
            # For check transactions: First try exact match, then fall back to payee name search
            check_transaction_processed = False
            if is_check:
                # Get payee name from transaction data (from check mapping CSV)
                txn_payee = transaction_data[idx].get('payee_name', '').strip()
                
                # First priority: Check for exact match in Pinecone results (memo + credit/debit)
                if idx < len(pinecone_results_list):
                    result = pinecone_results_list[idx]
                    
                    if not isinstance(result, Exception) and result and len(result) > 0:
                        # Look through ALL results to find exact match (not just first one)
                        exact_match_found = None
                        for res in result:
                            is_exact_match = res.get("_is_exact_match", False)
                            if is_exact_match:
                                exact_match_found = res
                                log_message("debug", f"Check transaction: Found exact match flag in result - memo: {res.get('memo', '')[:50]}, category: {res.get('category', '')[:50]}")
                                break
                        
                        # If exact match found, use that category (highest priority)
                        if exact_match_found:
                            category = exact_match_found.get("split", "").strip() or exact_match_found.get("category", "").strip()
                            if not category:
                                category = "Ask My Accountant"
                            payee_name = exact_match_found.get("name", "").strip() or txn_payee
                            log_message("info", f"Check transaction: Found EXACT memo+amount match in GL - using category '{category}' for memo: {memo[:50]}")
                            check_transaction_processed = True
                        else:
                            log_message("debug", f"Check transaction: No exact match flag found in {len(result)} Pinecone results for memo: {memo[:50]}")
                
                # Second priority: If no exact match, search by payee name
                if not check_transaction_processed and txn_payee:
                    log_message("debug", f"Check transaction: No exact memo match found, searching by payee name '{txn_payee}'...")
                    payee_category = await search_pinecone_by_payee_name(
                        payee_name=txn_payee,
                        client_name=client_name,
                        index_name=cached_index_name
                    )
                    
                    if payee_category:
                        category = payee_category
                        payee_name = txn_payee
                        log_message("info", f"Check transaction: Found category '{category}' from GL by payee name '{txn_payee}' for memo: {memo[:50]}")
                        check_transaction_processed = True
                    else:
                        log_message("debug", f"Check transaction: No category found in GL for payee '{txn_payee}' - using default 'Ask My Accountant' for memo: {memo[:50]}")
                
                # If still not processed, use default
                if not check_transaction_processed:
                    log_message("debug", f"Check transaction: No exact match and no payee name - using 'Ask My Accountant' for memo: {memo[:50]}")
                
                categorized.append({
                    "date": date,
                    "bank_memo": memo,
                    "credit": credit,
                    "debit": debit,
                    "category": category,
                    "payee_name": payee_name
                })
                continue  # Skip regular Pinecone processing for this transaction (already handled above)
            
            if idx < len(pinecone_results_list):
                result = pinecone_results_list[idx]
                
                if isinstance(result, Exception):
                    log_message("error", f"Pinecone query error for '{memo}': {result}")
                    # Keep default "Ask My Accountant" on error
                elif result and len(result) > 0:
                    first_result = result[0]
                    
                    # Check if this is an exact match (memo + credit/debit)
                    is_exact_match = first_result.get("_is_exact_match", False)
                    
                    if is_exact_match:
                        # Exact match found - use category regardless of score
                        category = first_result.get("split", "").strip() or first_result.get("category", "").strip()
                        if not category:
                            category = "Ask My Accountant"
                        # Get name from Pinecone metadata for payee - ONLY for exact matches
                        payee_name = first_result.get("name", "").strip()
                        log_message("debug", f"Using category from exact match: {category} for memo: {memo[:50]}")
                        if payee_name:
                            log_message("debug", f"Payee from Pinecone exact match: {payee_name} for memo: {memo[:50]}")
                    else:
                        # No exact match - use same logic as test_single_transaction
                        # IMPORTANT: This logic only applies to regular transactions, NOT check transactions
                        # Check transactions are handled separately above and should have already been skipped with continue
                        if is_check:
                            # Safety check: If this is a check transaction, skip this logic
                            log_message("warning", f"Check transaction reached regular processing logic - this should not happen. Using 'Ask My Accountant' for memo: {memo[:50]}")
                            category = "Ask My Accountant"
                        else:
                            # Filter by score >= 0.800 (same threshold as test API)
                            high_score_results = [r for r in result if r.get("score", 0.0) >= 0.800]
                            
                            if high_score_results:
                                # Parse dates and sort by newest first
                                from datetime import datetime
                                
                                def parse_date(date_str):
                                    """Parse date string and return a sortable value"""
                                    if not date_str:
                                        return None
                                    try:
                                        # Try various date formats
                                        for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                                            try:
                                                return datetime.strptime(str(date_str).strip(), fmt)
                                            except ValueError:
                                                continue
                                        return None
                                    except:
                                        return None
                                
                                # Add parsed dates for sorting
                                for r in high_score_results:
                                    r['_parsed_date'] = parse_date(r.get('date', ''))
                                
                                # Sort by full date including year (newest first), then by score if dates are equal or missing
                                high_score_results.sort(
                                    key=lambda x: (
                                        x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
                                        x.get('score', 0.0)
                                    ),
                                    reverse=True
                                )
                                
                                # Use the newest high-score result (prioritizes most recent year/date)
                                newest_result = high_score_results[0]
                                category = newest_result.get("split", "").strip() or newest_result.get("category", "").strip()
                                if not category:
                                    category = "Ask My Accountant"
                                # NOTE: Do NOT set payee_name here - only use Pinecone payee_name for exact matches
                                # payee_name will remain empty and fall through to description extraction
                                
                                score = newest_result.get("score", 0.0)
                                log_message("debug", f"Using category from semantic match for '{memo[:30]}' (score={score:.3f}): {category} (payee_name not used from similarity match)")
                            else:
                                # No results with sufficient score (score < 0.8)
                                # IMPORTANT: This top-5 logic ONLY applies to regular transactions, NOT check transactions
                                # Check transactions should use exact match or payee name search only
                                if is_check:
                                    # Safety check: If this is a check transaction, use default
                                    log_message("warning", f"Check transaction reached top-5 logic - this should not happen. Using 'Ask My Accountant' for memo: {memo[:50]}")
                                    category = "Ask My Accountant"
                                else:
                                    # Take top 5 similar results and use the best one from those
                                    top_5_results = result[:5] if len(result) >= 5 else result
                                    
                                    if top_5_results:
                                        from datetime import datetime
                                        
                                        def parse_date(date_str):
                                            """Parse date string and return a sortable value"""
                                            if not date_str:
                                                return None
                                            try:
                                                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                                                    try:
                                                        return datetime.strptime(str(date_str).strip(), fmt)
                                                    except ValueError:
                                                        continue
                                                return None
                                            except:
                                                return None
                                        
                                        # Add parsed dates for sorting
                                        for r in top_5_results:
                                            r['_parsed_date'] = parse_date(r.get('date', ''))
                                        
                                        # Sort by score (highest first), then by date (newest first)
                                        top_5_results.sort(
                                            key=lambda x: (
                                                x.get('score', 0.0),
                                                x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min
                                            ),
                                            reverse=True
                                        )
                                        
                                        # Use the best result from top 5 (highest score, newest date)
                                        best_result = top_5_results[0]
                                        category = best_result.get("split", "").strip() or best_result.get("category", "").strip()
                                        if not category:
                                            category = "Ask My Accountant"
                                        
                                        score = best_result.get("score", 0.0)
                                        log_message("debug", f"Using category from top 5 similar results for '{memo[:30]}' (score={score:.3f}): {category}")
                                    else:
                                        # No results at all
                                        top_score = first_result.get("score", 0.0) if first_result else 0.0
                                        log_message("debug", f"Best match score {top_score:.3f} below threshold and no top 5 results for '{memo[:30]}', using 'Ask My Accountant'")
                                        category = "Ask My Accountant"
                else:
                    # No results found, keep default "Ask My Accountant"
                    log_message("debug", f"No Pinecone results found, using 'Ask My Accountant' for memo: {memo[:50]}")
            
            # Fallback: If category is still "Ask My Accountant", try name-based search
            if category == "Ask My Accountant" and not is_check:
                try:
                    # Extract name from description
                    extracted_name = _extract_payee_from_description(memo)
                    
                    if extracted_name and extracted_name.strip():
                        log_message("debug", f"Attempting name-based search for memo '{memo[:50]}' with extracted name: '{extracted_name}'")
                        
                        # Search by name using optimized helper function
                        name_match_result = await _search_by_name_optimized(
                            extracted_name=extracted_name.strip(),
                            client_name=client_name,
                            index_name=cached_index_name
                        )
                        
                        if name_match_result and name_match_result.get("category"):
                            category = name_match_result.get("category")
                            log_message("debug", f"Found category from name-based search: {category} for memo '{memo[:50]}'")
                except Exception as name_search_error:
                    log_message("warning", f"Error in name-based fallback search for memo '{memo[:50]}': {name_search_error}")
                    # Keep "Ask My Accountant" on error
            
            # LLM Fallback: ONLY if category is "Ask My Accountant" and llm=True, use Gemini to predict from COA
            # CRITICAL: NEVER overwrite a good category (not "Ask My Accountant") - only apply LLM when category is already "Ask My Accountant"
            # IMPORTANT: Do NOT apply LLM prediction for check transactions - only for regular transactions
            if category == "Ask My Accountant" and llm and coa_categories and not is_check:
                try:
                    log_message("debug", f"Attempting LLM-based COA prediction for memo '{memo[:50]}' (category is 'Ask My Accountant')")
                    predicted_category = await predict_category_from_coa_llm(
                        transaction_description=memo,
                        transaction_amount=amount,
                        transaction_date=date,
                        coa_categories=coa_categories
                    )
                    
                    if predicted_category:
                        # Only update if we got a valid prediction
                        category = predicted_category
                        log_message("info", f"LLM predicted category '{category}' from COA for memo '{memo[:50]}' - category updated in categorize_with_pinecone")
                    else:
                        # LLM couldn't find a match, keep "Ask My Accountant"
                        log_message("debug", f"LLM could not find suitable category in COA for memo '{memo[:50]}', keeping 'Ask My Accountant'")
                except Exception as llm_error:
                    log_message("warning", f"Error in LLM-based COA prediction for memo '{memo[:50]}': {llm_error}")
                    # Keep "Ask My Accountant" on error
            elif category == "Ask My Accountant" and is_check:
                log_message("debug", f"Skipping LLM prediction for check transaction: '{memo[:50]}' - keeping 'Ask My Accountant'")
            elif category != "Ask My Accountant" and llm:
                # IMPORTANT: If category is NOT "Ask My Accountant", preserve it - do NOT run LLM
                log_message("debug", f"Category is '{category}' (not 'Ask My Accountant') for memo '{memo[:50]}' - preserving category, skipping LLM prediction")
            
            # Log category before adding to categorized list (for debugging)
            if category and category != "Ask My Accountant" and category != "Uncategorized":
                log_message("debug", f"Adding to categorized list with category '{category}' for memo '{memo[:50]}'")
            
            categorized.append({
                "date": date,
                "bank_memo": memo,
                "credit": credit,
                "debit": debit,
                "category": category,
                "payee_name": payee_name  # Add payee name from Pinecone
            })
        
        return categorized
        
    except Exception as e:
        log_message("error", f"Pinecone categorization failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        # Fallback: create CSV rows for all transactions
        return _create_fallback_csv_results(transactions_batch, str(e))


async def _search_by_name_optimized(
    extracted_name: str,
    client_name: Optional[str] = None,
    index_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Optimized name-based search in Pinecone.
    
    Logic:
    1. Generate embedding for extracted name
    2. Query Pinecone (top_k=50)
    3. First check for exact name matches (case-insensitive)
    4. If exact matches found, use newest by date
    5. If no exact matches, use similar matches with score >= 0.8, newest by date
    
    Args:
        extracted_name: Name extracted from transaction description
        client_name: Client name for filtering
        index_name: Pinecone index name
        
    Returns:
        Dict with category and match info, or None if no match found
    """
    try:
        if not extracted_name or not extracted_name.strip():
            return None
        
        # Generate embedding for the extracted name
        name_embedding = await asyncio.to_thread(
            embeddings.embed_query,
            extracted_name.strip()
        )
        
        if not name_embedding:
            return None
        
        # Query Pinecone with name embedding
        from backend.services.pinecone import query_pinecone_with_embedding
        
        name_results = await asyncio.to_thread(
            query_pinecone_with_embedding,
            name_embedding,
            client_name,
            50,  # Query top 50 results
            index_name
        )
        
        if not name_results or len(name_results) == 0:
            return None
        
        # Normalize extracted name for comparison
        normalized_extracted_name = extracted_name.strip().lower()
        
        # Step 1: Check for exact name matches first (case-insensitive)
        exact_name_matches = []
        for r in name_results:
            result_name = str(r.get("name", "")).strip().lower()
            if result_name == normalized_extracted_name:
                exact_name_matches.append(r)
        
        # Step 2: If exact matches found, use those (sort by date, newest first)
        # Otherwise, use similar matches with score >= 0.8
        if exact_name_matches:
            name_matches_to_use = exact_name_matches
            log_message("debug", f"Found {len(exact_name_matches)} exact name match(es) for '{extracted_name}'")
        else:
            # No exact match - use similar matches with score >= 0.8
            name_matches_to_use = [r for r in name_results if r.get("score", 0.0) >= 0.800]
            log_message("debug", f"No exact name match, found {len(name_matches_to_use)} similar name match(es) (score >= 0.8) for '{extracted_name}'")
        
        if not name_matches_to_use:
            return None
        
        # Step 3: Sort by date (newest first) and use the newest one
        from datetime import datetime
        
        def parse_date(date_str):
            """Parse date string and return a sortable value"""
            if not date_str:
                return None
            try:
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                    try:
                        return datetime.strptime(str(date_str).strip(), fmt)
                    except ValueError:
                        continue
                return None
            except:
                return None
        
        # Parse dates for all matches
        for match in name_matches_to_use:
            match['_parsed_date'] = parse_date(match.get('date', ''))
        
        # Sort by date (newest first)
        name_matches_to_use.sort(
            key=lambda x: x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
            reverse=True
        )
        
        # Use the newest match
        best_match = name_matches_to_use[0]
        category = best_match.get("split", "").strip() or best_match.get("category", "").strip()
        
        if not category:
            return None
        
        return {
            "category": category,
            "match_type": "exact" if exact_name_matches else "similar",
            "score": best_match.get("score", 0.0),
            "name": best_match.get("name", "")
        }
        
    except Exception as e:
        log_message("warning", f"Error in optimized name search for '{extracted_name}': {e}")
        return None


def _extract_payee_from_description(description: str) -> str:
    """
    Extract payee name from transaction description using simple text parsing.
    
    This is a fallback method when Pinecone doesn't have a name field.
    Extracts the person or company that sent/received the payment (NOT the client name).
    
    Priority:
    1. Business/company names that appear BEFORE transaction codes/numbers (in middle) - HIGHEST PRIORITY
    2. Business names at the start
    3. Human names (person names) at the end (but avoid client names - skip if contains INC, LLC, CORP, CARRIER, etc.)
    
    Args:
        description: Transaction description/memo text
        
    Returns:
        Extracted payee name, or empty string if not found
    """
    if not description or not description.strip():
        return ""
    
    desc = description.strip()
    
    # Generic business terms to skip
    generic_business_terms = {
        'BUSINESS', 'AMERICAN', 'EXPRESS', 'COMPANY', 'CORP', 'CORPORATION',
        'INC', 'LLC', 'LTD', 'BANK', 'BANKING', 'FINANCIAL', 'SERVICES',
        'ACH', 'DEBIT', 'CREDIT', 'PAYMENT', 'TRANSFER', 'PMT', 'WIRE',
        'ONLINE', 'BILLPAY', 'CHECK', 'CHK', 'WF', 'TO', 'FROM', 'AND', 'THE',
        'STO', 'PAYMENTS', 'PAYMENT', 'SALE', 'SALES', 'REGISTRATION'
    }
    
    # Pattern 1: Payment to/from
    payment_match = re.search(r'(?:payment|pay)\s+(?:to|from)\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s|$|,|\.)', desc, re.IGNORECASE)
    if payment_match:
        payee = payment_match.group(1).strip()
        if len(payee) > 2 and len(payee) < 100:  # Reasonable length
            return payee
    
    # Pattern 2: Transfer to/from
    transfer_match = re.search(r'transfer\s+(?:to|from)\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s|$|,|\.)', desc, re.IGNORECASE)
    if transfer_match:
        payee = transfer_match.group(1).strip()
        if len(payee) > 2 and len(payee) < 100:
            return payee
    
    # Pattern 3: ACH DEBIT/CREDIT followed by name
    ach_match = re.search(r'ACH\s+(?:DEBIT|CREDIT|PAYMENT)\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s|$|,|\.)', desc, re.IGNORECASE)
    if ach_match:
        payee = ach_match.group(1).strip()
        if len(payee) > 2 and len(payee) < 100:
            return payee
    
    # Pattern 4: Check number followed by name
    check_match = re.search(r'check\s+\d+\s+([A-Z][A-Za-z\s&,\.]+?)(?:\s|$|,|\.)', desc, re.IGNORECASE)
    if check_match:
        payee = check_match.group(1).strip()
        if len(payee) > 2 and len(payee) < 100:
            return payee
    
    words = desc.split()
    
    # Pattern 5: HIGHEST PRIORITY - Extract business/company name in MIDDLE (before transaction codes/numbers)
    # Look for business names that appear after initial terms but before numbers/codes
    # Example: "BUSINESS TO BUSINESS ACH LOVES TRAVEL STO PAYMENTS 241231 ..." -> "LOVES TRAVEL"
    # Find where numbers/codes start (transaction codes are usually long numbers)
    number_start_idx = None
    for i, word in enumerate(words):
        # Check if word is a number or contains mostly digits (transaction code)
        if word.replace('.', '').replace(',', '').isdigit() or (len(word) >= 6 and sum(c.isdigit() for c in word) >= len(word) * 0.7):
            number_start_idx = i
            break
    
    # If we found where numbers start, extract business name before that
    if number_start_idx and number_start_idx > 3:
        # Look for business name between position 3-4 and number_start_idx
        # Skip initial generic terms like "BUSINESS", "TO", "BUSINESS", "ACH"
        business_name_parts = []
        
        # Start from position 3-4 (after "BUSINESS TO BUSINESS ACH" or similar)
        for i in range(3, number_start_idx):
            word = words[i]
            word_upper = word.upper()
            
            # Skip generic transaction terms
            if word_upper in {'STO', 'PAYMENTS', 'PAYMENT', 'PMT', 'ACH', 'DEBIT', 'CREDIT', 'SALE', 'SALES', 'REGISTRATION'}:
                continue
            # Skip generic business terms
            if word_upper in generic_business_terms:
                continue
            # Skip if it's all caps abbreviation (likely not a business name)
            if word.isupper() and len(word) <= 3:
                continue
            # Skip if it's a number
            if word.replace('.', '').replace(',', '').isdigit() or any(c.isdigit() for c in word):
                continue
            # Take capitalized words (business names)
            if word and word[0].isupper():
                business_name_parts.append(word)
                if len(business_name_parts) >= 3:  # Limit to 3 words
                    break
            else:
                # Stop if we hit non-capitalized word
                if business_name_parts:
                    break
        
        if business_name_parts:
            payee = ' '.join(business_name_parts).strip()
            if len(payee) > 2 and len(payee) < 100:
                return payee
    
    # Pattern 6: Extract business name at START (before any transaction terms)
    # Example: "TRIUMPH FINANCE Payment 241231 ..." -> "TRIUMPH FINANCE"
    if len(words) > 0:
        start_name_parts = []
        for i, word in enumerate(words[:5]):  # Check first 5 words
            word_upper = word.upper()
            # Stop if we hit transaction terms
            if word_upper in {'PAYMENT', 'PAYMENTS', 'ACH', 'DEBIT', 'CREDIT', 'TRANSFER', 'PMT', 'SALE', 'SALES', 'REGISTRATION'}:
                break
            # Skip generic terms
            if word_upper in generic_business_terms:
                continue
            # Skip if it's all caps abbreviation
            if word.isupper() and len(word) <= 3:
                continue
            # Skip if it's a number
            if word.replace('.', '').replace(',', '').isdigit() or any(c.isdigit() for c in word):
                break
            # Take capitalized words
            if word and word[0].isupper():
                start_name_parts.append(word)
                if len(start_name_parts) >= 3:  # Limit to 3 words
                    break
            else:
                if start_name_parts:
                    break
        
        if start_name_parts:
            payee = ' '.join(start_name_parts).strip()
            if len(payee) > 2 and len(payee) < 100:
                return payee
    
    # Pattern 7: Fallback - Extract human names at the END (but skip client names)
    # Only check if no business name found in middle/start
    # Skip names that contain business suffixes (likely client names)
    client_name_indicators = {'INC', 'LLC', 'LTD', 'CORP', 'CORPORATION', 'CARRIER', 'INC.'}
    
    for start_from_end in range(2, min(5, len(words) + 1)):  # Try 2, 3, or 4 words from the end
        if len(words) < start_from_end:
            continue
        
        # Get the last N words
        candidate_words = words[-start_from_end:]
        candidate_text = ' '.join(candidate_words).upper()
        
        # Skip if it contains client name indicators (likely a client name, not payee)
        if any(indicator in candidate_text for indicator in client_name_indicators):
            continue
        
        # Check if this looks like a human name (not generic business terms)
        is_human_name = True
        name_parts = []
        
        for word in candidate_words:
            word_upper = word.upper()
            # Skip if it's a generic business term
            if word_upper in generic_business_terms:
                is_human_name = False
                break
            # Skip if it's all uppercase abbreviation (likely not a name)
            if word.isupper() and len(word) <= 3:
                is_human_name = False
                break
            # Skip if it's a number or contains digits
            if word.replace('.', '').replace(',', '').isdigit() or any(c.isdigit() for c in word):
                is_human_name = False
                break
            # Check if it has proper capitalization (first letter uppercase)
            if word and word[0].isupper():
                name_parts.append(word)
            else:
                is_human_name = False
                break
        
        # If we found a potential human name (2-4 words with proper capitalization)
        if is_human_name and len(name_parts) >= 2:
            payee = ' '.join(name_parts).strip()
            if len(payee) > 2 and len(payee) < 100:
                return payee
    
    return ""


def _create_fallback_csv_results(transactions_batch: List[Dict[str, Any]], error_msg: str) -> List[Dict[str, Any]]:
    """Create fallback CSV results when AI fails"""
    fallback = []
    for txn in transactions_batch:
        amount = txn.get("amount", 0)
        if amount >= 0:
            credit = str(amount)
            debit = ""
        else:
            credit = ""
            debit = str(abs(amount))
        
        fallback.append({
            "date": txn.get("date", ""),
            "bank_memo": txn.get("memo", ""),
            "credit": credit,
            "debit": debit,
            "category": "Uncategorized"
        })
    return fallback


async def process_all_batches_with_pinecone(
    standardized_transactions: List[Dict[str, Any]],
    working_list_result: Dict[str, Any],
    client_name: Optional[str] = None,
    client_id: Optional[str] = None,
    account_number: Optional[str] = None,
    index_name: Optional[str] = None,
    llm: bool = False,
    coa_categories: Optional[List[str]] = None
) -> str:
    """
    Process all batches of transactions through Pinecone matching and return CSV string.
    Uses only Pinecone (GL data) - no COA matching.
    Now processes batches in parallel for better performance.
    
    Args:
        standardized_transactions: ALL standardized transactions (to ensure we include everything)
        working_list_result: Result from create_working_list_for_pinecone()
        client_name: Client name for filtering Pinecone results (optional, used if index_name not provided)
        client_id: Client ID for loan transaction splitting (optional)
        account_number: Account number for CSV output (optional)
        index_name: Pre-resolved Pinecone index name (optional, takes precedence over client_name)
        
    Returns:
        CSV string with all transactions: Date, Bank_Memo, Credit, Debit, Category, Bank, Payee
    """
    try:
        batches = working_list_result.get("batches", [])
        working_list = working_list_result.get("working_list", [])
        
        log_message("info", f"Processing {len(batches)} batches ({len(working_list)} transactions) through Pinecone - parallel mode")
        
        batch_semaphore = asyncio.Semaphore(10)
        
        async def process_batch_with_semaphore(batch: List[Dict[str, Any]], batch_num: int) -> List[Dict[str, Any]]:
            async with batch_semaphore:
                if batch_num % 10 == 0 or batch_num == len(batches):
                    log_message("info", f"Processing batch {batch_num}/{len(batches)}")
                return await categorize_with_pinecone(
                    batch, 
                    client_name=client_name, 
                    index_name=index_name,
                    llm=llm,
                    coa_categories=coa_categories
                )
        
        batch_tasks = [
            process_batch_with_semaphore(batch, idx + 1)
            for idx, batch in enumerate(batches)
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        all_categorized = []
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                log_message("error", f"Batch {idx + 1} failed: {result}")
                fallback = _create_fallback_csv_results(batches[idx], str(result))
                all_categorized.extend(fallback)
            else:
                all_categorized.extend(result)
        
        # CRITICAL FIX: Use list matching by index instead of dict to preserve duplicate transactions
        # The all_categorized list is in same order as standardized_transactions (via working_list)
        # This ensures each transaction keeps its unique amount even with same date/memo
        
        # Updated CSV header to include bank, payee, and account_number fields
        # Columns: Date, Description (bank memo), Credit, Debit, Category, Bank, Payee, AccountNumber
        csv_rows = ["Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber"]
        
        # Use provided account_number (passed from grouping logic)
        # This account_number should be the same for all transactions in this batch
        default_account_number = str(account_number or "").strip()
        
        if default_account_number:
            log_message("info", f"Using account_number '{default_account_number}' for CSV output")
        else:
            log_message("warning", "No account_number provided - CSV will have empty AccountNumber column")
        
        # Import date parser and loan finder
        from dateutil import parser as date_parser
        from backend.services.amortization import find_loan_by_transaction
        
        def escape_csv(value):
            if value is None:
                return ""
            value_str = str(value)
            if ',' in value_str or '"' in value_str or '\n' in value_str:
                return '"' + value_str.replace('"', '""') + '"'
            return value_str
        
        for idx, txn in enumerate(standardized_transactions):
            original_memo = txn.get("Memo", "")
            date = txn.get("Date", "")
            amount = txn.get("Amount", 0)
            
            # Match by index instead of (date, memo) key to preserve duplicates
            if idx < len(all_categorized):
                result = all_categorized[idx]
            else:
                result = {
                    "date": date,
                    "bank_memo": original_memo,
                    "category": "Uncategorized"
                }
            
            category = result.get('category', 'Uncategorized')
            bank_memo = result.get('bank_memo', original_memo)
            
            # IMPORTANT: Preserve category from categorize_with_pinecone (which includes LLM predictions)
            # CRITICAL: If category is NOT "Ask My Accountant", preserve it - do NOT change it
            original_category_from_pinecone = category
            
            # Log the category we received from categorize_with_pinecone (which includes LLM predictions)
            if category and category != "Ask My Accountant" and category != "Uncategorized":
                log_message("debug", f"Category from categorize_with_pinecone for '{original_memo[:50]}': '{category}' - PRESERVING this category")
            
            bank_name = txn.get("Bank", "")
            
            # 4-Tier Payee Extraction Logic:
            # 1. FIRST: Check if transaction already has payee (from check mapping CSV)
            # 2. Second: Try to get payee from Pinecone results (name field in metadata)
            # 3. Third: Extract from description if not found in Pinecone
            # 4. Fourth: Use "Ask My Accountant" if still not found
            
            payee_name = ""
            
            # Tier 1: Check transaction for payee (from check mapping CSV)
            transaction_payee = txn.get("Payee") or ""
            if transaction_payee and transaction_payee.strip():
                payee_name = transaction_payee.strip()
                log_message("debug", f"Payee from check mapping: {payee_name} for memo: {original_memo[:50]}")
            
            # Override category for check transactions: First try exact match, then fall back to payee name search
            # IMPORTANT: For non-check transactions, preserve the category from categorize_with_pinecone (which includes LLM predictions)
            if txn.get("is_check_transaction"):
                check_transaction_processed = False
                
                # First priority: Check if result already has an exact match category (from categorize_with_pinecone)
                # The categorize_with_pinecone function already checks for exact matches, so if category is not "Ask My Accountant"
                # and not "Uncategorized", it might be from an exact match
                if category and category != "Ask My Accountant" and category != "Uncategorized":
                    # Check if this came from an exact match by checking if payee_name is set from Pinecone
                    pinecone_payee = result.get('payee_name', '').strip()
                    if pinecone_payee:
                        # This is likely from an exact match - keep the category
                        log_message("info", f"Check transaction: Using category '{category}' from exact match in GL for memo: {original_memo[:50]}")
                        check_transaction_processed = True
                
                # Second priority: If no exact match, search by payee name
                if not check_transaction_processed and payee_name:
                    log_message("debug", f"Check transaction: No exact match found, searching by payee name '{payee_name}'...")
                    try:
                        # Resolve index name
                        if index_name is None:
                            from backend.services.pinecone import get_index_name_from_client
                            resolved_index_name = get_index_name_from_client(client_name, log_result=False)
                        else:
                            resolved_index_name = index_name
                        
                        # Search Pinecone GL by payee name (no memo/credit/debit checking)
                        payee_category = await search_pinecone_by_payee_name(
                            payee_name=payee_name,
                            client_name=client_name,
                            index_name=resolved_index_name
                        )
                        
                        if payee_category:
                            category = payee_category
                            log_message("info", f"Check transaction: Found category '{category}' from GL by payee name '{payee_name}' for memo: {original_memo[:50]}")
                            check_transaction_processed = True
                        else:
                            category = "Ask My Accountant"
                            log_message("debug", f"Check transaction: No category found in GL for payee '{payee_name}' - using default 'Ask My Accountant' for memo: {original_memo[:50]}")
                    except Exception as e:
                        log_message("error", f"Error searching GL by payee name '{payee_name}' for check transaction: {e}")
                        category = "Ask My Accountant"
                elif not check_transaction_processed:
                    category = "Ask My Accountant"
                    log_message("debug", f"Check transaction: No exact match and no payee name - category set to 'Ask My Accountant' for memo: {original_memo[:50]}")
            
            # Tier 2: Check Pinecone results for name field (ONLY if exact match was found)
            # Note: payee_name from Pinecone is only set when there's an exact match (memo + all fields match)
            # Similarity/score-based matches do NOT provide payee_name - will fall through to description extraction
            if not payee_name:
                pinecone_payee = result.get('payee_name', '').strip()
                if pinecone_payee:
                    payee_name = pinecone_payee
                    log_message("debug", f"Payee from Pinecone exact match: {payee_name} for memo: {original_memo[:50]}")
            
            # Tier 3: If not found in Pinecone, try to extract from description
            if not payee_name:
                extracted_payee = _extract_payee_from_description(original_memo)
                if extracted_payee:
                    payee_name = extracted_payee
                    log_message("debug", f"Payee extracted from description: {payee_name} for memo: {original_memo[:50]}")
            
            # Tier 4: If still not found, use "Ask My Accountant"
            if not payee_name:
                payee_name = "Ask My Accountant"
                log_message("debug", f"Payee not found, using default 'Ask My Accountant' for memo: {original_memo[:50]}")
            
            # CRITICAL: Preserve category from categorize_with_pinecone for non-check transactions
            # If categorize_with_pinecone found a good category (not "Ask My Accountant"), preserve it
            # Only check transactions can have their category changed (for payee name search)
            if not txn.get("is_check_transaction") and original_category_from_pinecone and original_category_from_pinecone != "Ask My Accountant":
                # Restore original category if it was a good category (not "Ask My Accountant")
                category = original_category_from_pinecone
                log_message("debug", f"Preserving original category '{category}' for non-check transaction '{original_memo[:50]}' (from categorize_with_pinecone)")
            
            # Use account_number from parameter (passed when processing grouped transactions)
            txn_account_number = str(account_number or default_account_number).strip()
            
            # Check if category contains "Loan" and client_id is available
            should_split_loan = client_id and "loan" in category.lower()
            
            if should_split_loan:
                try:
                    # Parse transaction date
                    transaction_date = date_parser.parse(date)
                    
                    # Find loan details
                    loan_result = await find_loan_by_transaction(
                        transaction_date=transaction_date,
                        memo=original_memo,
                        amount=abs(amount),
                        client_id=client_id
                    )
                    
                    if loan_result and loan_result.get("schedule"):
                        schedule = loan_result.get("schedule")
                        interest_expense = schedule.get("interest_expense", 0)
                        principal_liability = schedule.get("principal_liability", 0)
                        
                        # Only split if we have valid amounts
                        if interest_expense > 0 and principal_liability > 0:
                            # First transaction: Principal with original category
                            if amount >= 0:
                                principal_credit = str(principal_liability)
                                principal_debit = ""
                            else:
                                principal_credit = ""
                                # Store debit as negative amount as requested
                                principal_debit = str(-principal_liability)
                            
                            csv_rows.append(
                                f"{escape_csv(date)},"
                                f"{escape_csv(bank_memo)},"
                                f"{escape_csv(principal_credit)},"
                                f"{escape_csv(principal_debit)},"
                                f"{escape_csv(category)},"
                                f"{escape_csv(bank_name)},"
                                f"{escape_csv(payee_name)},"
                                f"{escape_csv(txn_account_number)}"
                            )
                            
                            # Second transaction: Interest with "Interest" category
                            if amount >= 0:
                                interest_credit = str(interest_expense)
                                interest_debit = ""
                            else:
                                interest_credit = ""
                                interest_debit = str(-interest_expense)
                            
                            csv_rows.append(
                                f"{escape_csv(date)},"
                                f"{escape_csv(bank_memo)},"
                                f"{escape_csv(interest_credit)},"
                                f"{escape_csv(interest_debit)},"
                                f"{escape_csv('Interest')},"
                                f"{escape_csv(bank_name)},"
                                f"{escape_csv(payee_name)},"
                                f"{escape_csv(txn_account_number)}"
                            )
                            
                            log_message("info", f"Split loan transaction: {original_memo} - Principal: {principal_liability}, Interest: {interest_expense}")
                            continue  # Skip the original transaction
                        else:
                            log_message("warning", f"Loan schedule found but invalid amounts: interest={interest_expense}, principal={principal_liability}")
                    else:
                        log_message("warning", f"Loan transaction found but no schedule available: {original_memo}")
                except Exception as e:
                    log_message("error", f"Error splitting loan transaction {original_memo}: {e}")
                    # Fall through to add original transaction
            
            # Default: add transaction as-is (or if loan splitting failed)
            if amount >= 0:
                credit_val = str(amount)
                debit_val = ""
            else:
                credit_val = ""
                # Store debits as negative amounts as requested
                debit_val = str(amount)
            
            # Log final category before adding to CSV (for debugging LLM predictions)
            if category and category != "Ask My Accountant" and category != "Uncategorized":
                log_message("debug", f"Final category for CSV output '{original_memo[:50]}': '{category}'")
            
            csv_rows.append(
                f"{escape_csv(date)},"
                f"{escape_csv(bank_memo)},"
                f"{escape_csv(credit_val)},"
                f"{escape_csv(debit_val)},"
                f"{escape_csv(category)},"
                f"{escape_csv(bank_name)},"
                f"{escape_csv(payee_name)},"
                f"{escape_csv(txn_account_number)}"
            )
        
        csv_string = "\n".join(csv_rows)
        log_message("info", f"Generated CSV with {len(standardized_transactions)} transactions")
        
        # Save CSV to local file in categorized folder
        try:
            output_dir = Path("data/output/csv")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transactions_categorized_{timestamp}.csv"
            file_path = output_dir / filename
            
            # Write CSV to file
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_string)
            
            log_message("info", f"Saved CSV file to: {file_path}")
        except Exception as save_error:
            log_message("warning", f"Failed to save CSV to file: {save_error}")
            # Don't fail the whole operation if file save fails
        
        return csv_string
        
    except Exception as e:
        log_message("error", f"Batch processing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        
        # Fallback: return CSV with all transactions as Uncategorized
        csv_rows = ["Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber"]
        for txn in standardized_transactions:
            amount = txn.get('Amount', 0)
            if amount >= 0:
                credit = str(amount)
                debit = ""
            else:
                credit = ""
                # Store debits as negative amounts as requested
                debit = str(amount)
            
            txn_account_number = str(account_number or default_account_number).strip()
            
            # Handle None values in fallback CSV
            txn_payee = txn.get('Payee') or ""
            
            csv_rows.append(
                f"{txn.get('Date', '')},"
                f"{txn.get('Memo', '')},"  # Description
                f"{credit},"
                f"{debit},"
                f"Uncategorized,"
                f"{txn.get('Bank', '')},"
                f"{txn_payee},"
                f"{txn_account_number}"
            )
        return "\n".join(csv_rows)


async def process_all_batches_with_pinecone_streaming(
    standardized_transactions: List[Dict[str, Any]],
    working_list_result: Dict[str, Any],
    client_name: Optional[str] = None,
    client_id: Optional[str] = None,
    account_number: Optional[str] = None,
    index_name: Optional[str] = None,
    user_id: Optional[str] = None,
    stream_callback: Optional[Callable] = None,
    llm: bool = False,
    coa_categories: Optional[List[str]] = None
):
    """
    Process batches incrementally and yield results as they complete (streaming version).
    
    This function processes batches one by one and calls stream_callback for each completed batch.
    This allows the frontend to receive results incrementally without waiting for all processing.
    
    Args:
        standardized_transactions: ALL standardized transactions
        working_list_result: Result from create_working_list_for_pinecone()
        client_name: Client name for filtering Pinecone results
        client_id: Client ID for loan transaction splitting
        account_number: Account number for CSV output
        index_name: Pre-resolved Pinecone index name
        user_id: User ID for storing batches in MongoDB
        stream_callback: Async function to call for each completed batch
                         Signature: async def callback(batch_num: int, total_batches: int, batch_csv: str, batch_transactions: List[Dict])
    
    Yields:
        Tuple of (batch_num, total_batches, batch_csv_string, batch_transactions)
    """
    try:
        batches = working_list_result.get("batches", [])
        working_list = working_list_result.get("working_list", [])
        
        log_message("info", f"Processing {len(batches)} batches ({len(working_list)} transactions) through Pinecone - streaming mode")
        
        # CSV header
        csv_header = "Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber"
        default_account_number = str(account_number or "").strip()
        
        # Import date parser and loan finder
        from dateutil import parser as date_parser
        from backend.services.amortization import find_loan_by_transaction
        
        def escape_csv(value):
            if value is None:
                return ""
            value_str = str(value)
            if ',' in value_str or '"' in value_str or '\n' in value_str:
                return '"' + value_str.replace('"', '""') + '"'
            return value_str
        
        # Process batches one by one (not in parallel for streaming)
        all_categorized = []
        all_csv_rows = [csv_header]
        
        for batch_idx, batch in enumerate(batches):
            batch_num = batch_idx + 1
            total_batches = len(batches)
            
            try:
                # Process this batch
                batch_result = await categorize_with_pinecone(
                    batch, 
                    client_name=client_name, 
                    index_name=index_name,
                    llm=llm,
                    coa_categories=coa_categories
                )
                all_categorized.extend(batch_result)
                
                # Generate CSV rows for this batch
                batch_csv_rows = []
                batch_transactions = []
                
                # Map batch results to transactions
                # The batch contains working_list items, we need to map them to standardized_transactions
                for batch_item_idx, batch_item in enumerate(batch):
                    # Find corresponding transaction in standardized_transactions
                    # Use the working_list index to find the transaction
                    working_list_idx = batch_idx * 10 + batch_item_idx
                    if working_list_idx < len(standardized_transactions):
                        txn = standardized_transactions[working_list_idx]
                        
                        # Get categorization result
                        if batch_item_idx < len(batch_result):
                            result = batch_result[batch_item_idx]
                        else:
                            result = {
                                "date": txn.get("Date", ""),
                                "bank_memo": txn.get("Memo", ""),
                                "category": "Uncategorized"
                            }
                        
                        category = result.get('category', 'Uncategorized')
                        bank_memo = result.get('bank_memo', txn.get("Memo", ""))
                        
                        bank_name = txn.get("Bank", "")
                        date = txn.get("Date", "")
                        amount = txn.get("Amount", 0)
                        
                        # Payee extraction (same logic as non-streaming version)
                        payee_name = ""
                        transaction_payee = txn.get("Payee") or ""
                        if transaction_payee and transaction_payee.strip():
                            payee_name = transaction_payee.strip()
                        elif not payee_name:
                            pinecone_payee = result.get('payee_name', '').strip()
                            if pinecone_payee:
                                payee_name = pinecone_payee
                        if not payee_name:
                            extracted_payee = _extract_payee_from_description(txn.get("Memo", ""))
                            if extracted_payee:
                                payee_name = extracted_payee
                        if not payee_name:
                            payee_name = "Ask My Accountant"
                        
                        txn_account_number = str(account_number or default_account_number).strip()
                        
                        # Loan splitting logic (same as non-streaming)
                        should_split_loan = client_id and "loan" in category.lower()
                        
                        if should_split_loan:
                            try:
                                transaction_date = date_parser.parse(date)
                                loan_result = await find_loan_by_transaction(
                                    transaction_date=transaction_date,
                                    memo=txn.get("Memo", ""),
                                    amount=abs(amount),
                                    client_id=client_id
                                )
                                
                                if loan_result and loan_result.get("schedule"):
                                    schedule = loan_result.get("schedule")
                                    interest_expense = schedule.get("interest_expense", 0)
                                    principal_liability = schedule.get("principal_liability", 0)
                                    
                                    if interest_expense > 0 and principal_liability > 0:
                                        # Principal transaction
                                        if amount >= 0:
                                            principal_credit = str(principal_liability)
                                            principal_debit = ""
                                        else:
                                            principal_credit = ""
                                            principal_debit = str(-principal_liability)
                                        
                                        batch_csv_rows.append(
                                            f"{escape_csv(date)},{escape_csv(bank_memo)},{escape_csv(principal_credit)},{escape_csv(principal_debit)},{escape_csv(category)},{escape_csv(bank_name)},{escape_csv(payee_name)},{escape_csv(txn_account_number)}"
                                        )
                                        
                                        # Interest transaction
                                        if amount >= 0:
                                            interest_credit = str(interest_expense)
                                            interest_debit = ""
                                        else:
                                            interest_credit = ""
                                            interest_debit = str(-interest_expense)
                                        
                                        batch_csv_rows.append(
                                            f"{escape_csv(date)},{escape_csv(bank_memo)},{escape_csv(interest_credit)},{escape_csv(interest_debit)},{escape_csv('Interest')},{escape_csv(bank_name)},{escape_csv(payee_name)},{escape_csv(txn_account_number)}"
                                        )
                                        
                                        batch_transactions.append({
                                            "date": date,
                                            "description": bank_memo,
                                            "credit": principal_credit if amount >= 0 else "",
                                            "debit": principal_debit if amount < 0 else "",
                                            "category": category,
                                            "bank": bank_name,
                                            "payee": payee_name,
                                            "accountNumber": txn_account_number
                                        })
                                        batch_transactions.append({
                                            "date": date,
                                            "description": bank_memo,
                                            "credit": interest_credit if amount >= 0 else "",
                                            "debit": interest_debit if amount < 0 else "",
                                            "category": "Interest",
                                            "bank": bank_name,
                                            "payee": payee_name,
                                            "accountNumber": txn_account_number
                                        })
                                        continue
                            except Exception as e:
                                log_message("error", f"Error splitting loan transaction: {e}")
                        
                        # Default transaction
                        if amount >= 0:
                            credit_val = str(amount)
                            debit_val = ""
                        else:
                            credit_val = ""
                            debit_val = str(amount)
                        
                        csv_row = f"{escape_csv(date)},{escape_csv(bank_memo)},{escape_csv(credit_val)},{escape_csv(debit_val)},{escape_csv(category)},{escape_csv(bank_name)},{escape_csv(payee_name)},{escape_csv(txn_account_number)}"
                        batch_csv_rows.append(csv_row)
                        all_csv_rows.append(csv_row)
                        
                        batch_transactions.append({
                            "date": date,
                            "description": bank_memo,
                            "credit": credit_val,
                            "debit": debit_val,
                            "category": category,
                            "bank": bank_name,
                            "payee": payee_name,
                            "accountNumber": txn_account_number
                        })
                
                # Create batch CSV string
                batch_csv = "\n".join([csv_header] + batch_csv_rows)
                
                # Call stream callback if provided
                if stream_callback:
                    try:
                        await stream_callback(batch_num, total_batches, batch_csv, batch_transactions)
                    except Exception as e:
                        log_message("error", f"Stream callback failed for batch {batch_num}: {e}")
                
                # Yield batch result
                yield (batch_num, total_batches, batch_csv, batch_transactions)
                
            except Exception as e:
                log_message("error", f"Batch {batch_num} failed: {e}")
                # Create fallback for this batch
                fallback = _create_fallback_csv_results(batch, str(e))
                all_categorized.extend(fallback)
                
                # Generate fallback CSV rows
                fallback_csv_rows = []
                fallback_transactions = []
                for fallback_item in fallback:
                    # Find corresponding transaction
                    batch_item_idx = fallback.index(fallback_item)
                    working_list_idx = batch_idx * 10 + batch_item_idx
                    if working_list_idx < len(standardized_transactions):
                        txn = standardized_transactions[working_list_idx]
                        date = txn.get("Date", "")
                        bank_memo = fallback_item.get("bank_memo", txn.get("Memo", ""))
                        credit = fallback_item.get("credit", "")
                        debit = fallback_item.get("debit", "")
                        category = fallback_item.get("category", "Uncategorized")
                        bank_name = txn.get("Bank", "")
                        payee_name = txn.get("Payee") or "Ask My Accountant"
                        txn_account_number = str(account_number or default_account_number).strip()
                        
                        csv_row = f"{escape_csv(date)},{escape_csv(bank_memo)},{escape_csv(credit)},{escape_csv(debit)},{escape_csv(category)},{escape_csv(bank_name)},{escape_csv(payee_name)},{escape_csv(txn_account_number)}"
                        fallback_csv_rows.append(csv_row)
                        all_csv_rows.append(csv_row)
                        
                        fallback_transactions.append({
                            "date": date,
                            "description": bank_memo,
                            "credit": credit,
                            "debit": debit,
                            "category": category,
                            "bank": bank_name,
                            "payee": payee_name,
                            "accountNumber": txn_account_number
                        })
                
                batch_csv = "\n".join([csv_header] + fallback_csv_rows)
                
                if stream_callback:
                    try:
                        await stream_callback(batch_num, total_batches, batch_csv, fallback_transactions)
                    except Exception as e:
                        log_message("error", f"Stream callback failed for fallback batch {batch_num}: {e}")
                
                yield (batch_num, total_batches, batch_csv, fallback_transactions)
        
        # Final CSV is accumulated in all_csv_rows during processing
        # No need to return it - the generator yields batches as they complete
        log_message("info", f"Generated streaming CSV with {len(standardized_transactions)} transactions")
        
    except Exception as e:
        log_message("error", f"Streaming batch processing failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        
        # Fallback: yield CSV with all transactions as Uncategorized
        csv_rows = ["Date,Description,Credit,Debit,Category,Bank,Payee,AccountNumber"]
        for txn in standardized_transactions:
            amount = txn.get('Amount', 0)
            if amount >= 0:
                credit = str(amount)
                debit = ""
            else:
                credit = ""
                debit = str(amount)
            
            txn_account_number = str(account_number or "").strip()
            txn_payee = txn.get('Payee') or "Ask My Accountant"
            
            csv_rows.append(
                f"{txn.get('Date', '')},{txn.get('Memo', '')},{credit},{debit},Uncategorized,{txn.get('Bank', '')},{txn_payee},{txn_account_number}"
            )
        fallback_csv = "\n".join(csv_rows)
        # Yield fallback as a single batch
        yield (1, 1, fallback_csv, [])


# ============================================================
#           TESTING FUNCTION
# ============================================================

async def test_single_transaction(
    transaction_memo: str,
    client_id: Optional[str] = None,
    client_name: Optional[str] = None,
    index_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test function to query Pinecone for a single transaction and return top 25 results.
    
    This follows the same logic as /process endpoint:
    1. Query top 25 similar records from Pinecone
    2. If exact memo match found, use that (newest by date)
    3. If no exact match: filter by score >= 0.800, sort by date (newest first), use first
    4. If no high-score results (>= 0.800), return "Ask My Accountant"
    5. Return all 25 results with the selected category
    
    Args:
        transaction_memo: Transaction memo/description to test
        client_id: Client ID (for reference)
        client_name: Client name (for reference)
        index_name: Pinecone index name to query (required)
        
    Returns:
        Dict with status and data containing:
            - transaction_memo: Input transaction memo
            - client_id: Client ID used
            - client_name: Client name used
            - pinecone_index: Pinecone index name queried
            - selected_category: The category selected based on the logic
            - all_25_results: List of all 25 similar records with details (sorted by similarity score)
            - exact_match_found: Whether exact memo match was found
            - high_score_used: Whether high score (>= 0.800) was used
            - ask_accountant: Whether "Ask My Accountant" category was assigned
    """
    try:
        log_message("info", f"Testing transaction: memo='{transaction_memo}', client_id={client_id}, index_name={index_name}")
        
        if not transaction_memo or not transaction_memo.strip():
            return {
                "status": "failed",
                "error": "Transaction memo cannot be empty"
            }
        
        if not index_name:
            return {
                "status": "failed",
                "error": "Pinecone index_name is required"
            }
        
        memo = transaction_memo.strip().lower()
        
        # Generate embedding using HuggingFace (same as storage)
        embedding = embeddings.embed_query(memo)
        
        if not embedding:
            return {
                "status": "failed",
                "error": "Failed to generate embedding"
            }
        
        log_message("info", f"Generated embedding (dimension: {len(embedding)})")
        
        from backend.services.pinecone import query_pinecone_with_embedding
        from datetime import datetime
        
        # Query top 25 results from Pinecone (same as /process endpoint logic)
        pinecone_results = query_pinecone_with_embedding(
            embedding_vector=embedding,
            client_name=client_name,
            top_k=25,  # Query top 25 results
            index_name=index_name
        )
        
        log_message("info", f"Pinecone returned {len(pinecone_results)} results")
        
        total_results_count = len(pinecone_results)
        selected_category = "Ask My Accountant"  # Default
        exact_match_found = False
        high_score_used = False
        ask_accountant = True
        all_25_results = []  # Initialize empty list
        
        def parse_date(date_str):
            """Parse date string and return datetime or None"""
            if not date_str:
                return None
            try:
                # Try various date formats (same as /process)
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                    try:
                        return datetime.strptime(str(date_str).strip(), fmt)
                    except ValueError:
                        continue
                # Try dateutil parser as fallback
                from dateutil import parser as date_parser
                return date_parser.parse(str(date_str))
            except Exception:
                return None
        
        all_25_results = []
        
        # Initialize variables
        exact_memo_matches = []
        high_score_results = []
        selected_result = None  # Store the selected result for comparison
        
        if pinecone_results and len(pinecone_results) > 0:
            # Step 1: Check for exact memo match (case-insensitive, trimmed)
            # Since we don't have amount in test API, we check for exact memo match only
            
            # Normalize input memo for comparison
            normalized_input_memo = memo.strip().lower()
            
            for result in pinecone_results:
                result_memo = result.get("memo", "").strip().lower()
                if result_memo == normalized_input_memo:
                    exact_memo_matches.append(result)
            
            # Step 2: If exact memo matches found, sort by date and use newest
            if exact_memo_matches:
                # Parse dates for all exact memo matches
                for match in exact_memo_matches:
                    match['_parsed_date'] = parse_date(match.get('date', ''))
                
                # Sort by date (newest first)
                exact_memo_matches.sort(
                    key=lambda x: x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
                    reverse=True
                )
                
                newest_exact = exact_memo_matches[0]
                selected_result = newest_exact  # Store selected result
                selected_category = newest_exact.get("split", "").strip() or newest_exact.get("category", "").strip()
                if not selected_category:
                    selected_category = "Ask My Accountant"
                
                exact_match_found = True
                ask_accountant = (selected_category == "Ask My Accountant")
                match_date = newest_exact.get('date', 'N/A')
                if len(exact_memo_matches) > 1:
                    log_message("info", f"Found {len(exact_memo_matches)} exact memo matches, selected newest (date={match_date}): {selected_category}")
                else:
                    log_message("info", f"Found exact memo match (date={match_date}): {selected_category}")
            
            # Step 3: If no exact memo match, filter by score >= 0.800 and sort by date
            else:
                # Filter results with score >= 0.800
                high_score_results = [r for r in pinecone_results if r.get("score", 0.0) >= 0.800]
                
                if high_score_results:
                    # Parse dates for high score results
                    for r in high_score_results:
                        r['_parsed_date'] = parse_date(r.get('date', ''))
                    
                    # Sort by date (newest first), then by score if dates are equal
                    high_score_results.sort(
                        key=lambda x: (
                            x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
                            x.get('score', 0.0)
                        ),
                        reverse=True
                    )
                    
                    # Use the newest high-score result
                    newest_result = high_score_results[0]
                    selected_result = newest_result  # Store selected result
                    selected_category = newest_result.get("split", "").strip() or newest_result.get("category", "").strip()
                    if not selected_category:
                        selected_category = "Ask My Accountant"
                    
                    high_score_used = True
                    ask_accountant = (selected_category == "Ask My Accountant")
                    top_score = newest_result.get("score", 0.0)
                    result_date = newest_result.get("date", "N/A")
                    log_message("info", f"Using high-score match (score={top_score:.3f}, date={result_date}): {selected_category}")
                else:
                    # No results with score >= 0.800
                    top_score = pinecone_results[0].get("score", 0.0) if pinecone_results else 0.0
                    log_message("info", f"Best score {top_score:.3f} below 0.8 threshold, using 'Ask My Accountant'")
                    selected_category = "Ask My Accountant"
                    ask_accountant = True
            
            # Format all 25 results with details
            for idx, result in enumerate(pinecone_results[:25], 1):
                date_obj = parse_date(result.get('date', ''))
                date_str = result.get('date', '') if result.get('date') else None
                
                # Check if this result is the selected one (compare by memo to handle modified dicts)
                is_selected = False
                if selected_result:
                    # Compare by memo to handle cases where dict was modified (e.g., _parsed_date added)
                    result_memo = result.get("memo", "").strip().lower()
                    selected_memo = selected_result.get("memo", "").strip().lower()
                    if result_memo == selected_memo:
                        # Also compare score to ensure it's the same result
                        if abs(result.get("score", 0.0) - selected_result.get("score", 0.0)) < 0.0001:
                            is_selected = True
                
                # Check if this is an exact match (memo matches input)
                is_exact = False
                if exact_memo_matches:
                    result_memo_normalized = result.get("memo", "").strip().lower()
                    for exact_match in exact_memo_matches:
                        if exact_match.get("memo", "").strip().lower() == result_memo_normalized:
                            is_exact = True
                            break
                
                all_25_results.append({
                    "rank": idx,
                    "memo": result.get("memo", ""),
                    "name": result.get("name", ""),
                    "category": result.get("split", "") or result.get("category", ""),
                    "similarity_score": round(result.get("score", 0.0), 4),
                    "date": date_str,
                    "is_exact_match": is_exact,
                    "is_high_score": result.get("score", 0.0) >= 0.800,
                    "is_selected": is_selected
                })
        
        return {
            "status": "success",
            "data": {
                "transaction_memo": transaction_memo,
                "client_id": client_id,
                "client_name": client_name,
                "pinecone_index": index_name,
                "total_results": total_results_count,
                "selected_category": selected_category,
                "exact_match_found": exact_match_found,
                "high_score_used": high_score_used,
                "ask_accountant": ask_accountant,
                "all_25_results": all_25_results
            }
        }
        
    except Exception as e:
        log_message("error", f"Test transaction failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {
            "status": "failed",
            "error": str(e)
        }


async def test_name_search(
    name: str,
    client_id: Optional[str] = None,
    client_name: Optional[str] = None,
    index_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test function to query Pinecone by name and return top 50 results.
    
    This follows the same logic as name-based fallback in /process endpoint:
    1. Generate embedding for the name
    2. Query top 50 similar records from Pinecone
    3. First check for exact name matches (case-insensitive)
    4. If exact matches found, use newest by date
    5. If no exact matches, use similar matches with score >= 0.8, newest by date
    6. Return all 50 results with the selected category
    
    Args:
        name: Name to test
        client_id: Client ID (for reference)
        client_name: Client name (for reference)
        index_name: Pinecone index name to query (required)
        
    Returns:
        Dict with status and data containing:
            - input_name: Input name
            - client_id: Client ID used
            - client_name: Client name used
            - pinecone_index: Pinecone index name queried
            - selected_category: The category selected based on the logic
            - match_type: "exact" or "similar" (or None if no match)
            - all_50_results: List of all 50 similar records with details (sorted by similarity score)
            - ask_accountant: Whether "Ask My Accountant" category was assigned
            - total_results: Number of results found
    """
    try:
        log_message("info", f"Testing name search: name='{name}', client_id={client_id}, index_name={index_name}")
        
        if not name or not name.strip():
            return {
                "status": "failed",
                "error": "Name cannot be empty"
            }
        
        if not index_name:
            return {
                "status": "failed",
                "error": "Pinecone index_name is required"
            }
        
        name_clean = name.strip()
        
        # Generate embedding for the name
        name_embedding = await asyncio.to_thread(
            embeddings.embed_query,
            name_clean
        )
        
        if not name_embedding:
            return {
                "status": "failed",
                "error": "Failed to generate embedding"
            }
        
        log_message("info", f"Generated embedding for name (dimension: {len(name_embedding)})")
        
        from backend.services.pinecone import query_pinecone_with_embedding
        from datetime import datetime
        
        # Query top 50 results from Pinecone
        name_results = await asyncio.to_thread(
            query_pinecone_with_embedding,
            name_embedding,
            client_name,
            50,  # Query top 50 results
            index_name
        )
        
        log_message("info", f"Pinecone returned {len(name_results)} results for name search")
        
        total_results_count = len(name_results)
        selected_category = "Ask My Accountant"  # Default
        match_type = None
        ask_accountant = True
        all_50_results = []
        selected_result = None
        
        def parse_date(date_str):
            """Parse date string and return datetime or None"""
            if not date_str:
                return None
            try:
                # Try various date formats
                for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%y"]:
                    try:
                        return datetime.strptime(str(date_str).strip(), fmt)
                    except ValueError:
                        continue
                # Try dateutil parser as fallback
                from dateutil import parser as date_parser
                return date_parser.parse(str(date_str))
            except Exception:
                return None
        
        if name_results and len(name_results) > 0:
            # Normalize extracted name for comparison
            normalized_input_name = name_clean.strip().lower()
            
            # Step 1: Check for exact name matches first (case-insensitive)
            exact_name_matches = []
            for r in name_results:
                result_name = str(r.get("name", "")).strip().lower()
                if result_name == normalized_input_name:
                    exact_name_matches.append(r)
            
            # Step 2: If exact matches found, use those (sort by date, newest first)
            # Otherwise, use similar matches with score >= 0.8
            if exact_name_matches:
                name_matches_to_use = exact_name_matches
                match_type = "exact"
                log_message("info", f"Found {len(exact_name_matches)} exact name match(es) for '{name_clean}'")
            else:
                # No exact match - use similar matches with score >= 0.8
                name_matches_to_use = [r for r in name_results if r.get("score", 0.0) >= 0.800]
                match_type = "similar" if name_matches_to_use else None
                log_message("info", f"No exact name match, found {len(name_matches_to_use)} similar name match(es) (score >= 0.8) for '{name_clean}'")
            
            if name_matches_to_use:
                # Parse dates for all matches
                for match in name_matches_to_use:
                    match['_parsed_date'] = parse_date(match.get('date', ''))
                
                # Sort by date (newest first)
                name_matches_to_use.sort(
                    key=lambda x: x['_parsed_date'] if x['_parsed_date'] is not None else datetime.min,
                    reverse=True
                )
                
                # Use the newest match
                selected_result = name_matches_to_use[0]
                selected_category = selected_result.get("split", "").strip() or selected_result.get("category", "").strip()
                if not selected_category:
                    selected_category = "Ask My Accountant"
                
                ask_accountant = (selected_category == "Ask My Accountant")
                top_score = selected_result.get("score", 0.0)
                result_date = selected_result.get("date", "N/A")
                result_name = selected_result.get("name", "")
                log_message("info", f"Using {match_type} name match (score={top_score:.3f}, date={result_date}, GL name='{result_name}'): {selected_category}")
            else:
                # No matches found
                top_score = name_results[0].get("score", 0.0) if name_results else 0.0
                log_message("info", f"Best score {top_score:.3f} below threshold, using 'Ask My Accountant'")
                selected_category = "Ask My Accountant"
                ask_accountant = True
                match_type = None
            
            # Format all 50 results with details
            for idx, result in enumerate(name_results[:50], 1):
                date_obj = parse_date(result.get('date', ''))
                date_str = result.get('date', '') if result.get('date') else None
                
                # Check if this result is the selected one
                is_selected = False
                if selected_result:
                    result_name_check = result.get("name", "").strip().lower()
                    selected_name_check = selected_result.get("name", "").strip().lower()
                    if result_name_check == selected_name_check:
                        # Also compare score to ensure it's the same result
                        if abs(result.get("score", 0.0) - selected_result.get("score", 0.0)) < 0.0001:
                            is_selected = True
                
                # Check if this is an exact name match
                is_exact = False
                result_name_check = str(result.get("name", "")).strip().lower()
                if result_name_check == normalized_input_name:
                    is_exact = True
                
                all_50_results.append({
                    "rank": idx,
                    "memo": result.get("memo", ""),
                    "name": result.get("name", ""),
                    "category": result.get("split", "") or result.get("category", ""),
                    "similarity_score": round(result.get("score", 0.0), 4),
                    "date": date_str,
                    "is_exact_match": is_exact,
                    "is_high_score": result.get("score", 0.0) >= 0.800,
                    "is_selected": is_selected
                })
        
        return {
            "status": "success",
            "data": {
                "input_name": name,
                "client_id": client_id,
                "client_name": client_name,
                "pinecone_index": index_name,
                "total_results": total_results_count,
                "selected_category": selected_category,
                "match_type": match_type,
                "ask_accountant": ask_accountant,
                "all_50_results": all_50_results
            }
        }
        
    except Exception as e:
        log_message("error", f"Test name search failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {
            "status": "failed",
            "error": str(e)
        }


# ============================================================
#                   LOAN DETAILS EXTRACTION
# ============================================================

async def extract_loan_details(file, client_id: str) -> Dict[str, Any]:
    """
    Extract loan details from a PDF document using AI and save to MongoDB.
    
    Extracts:
    - Legal Name
    - Funding Provided
    - Annual Percentage Rate (APR)
    - Finance Charge
    - Total Payment Amount
    - Payment
    - Account No.
    
    Args:
        file: UploadFile object containing the loan PDF document.
        client_id: Client identifier for MongoDB storage.
        
    Returns:
        Dict with extracted loan details, MongoDB record ID, or error status.
    """
    try:
        log_message("info", "Starting loan details extraction from PDF")
        
        # Extract text from PDF (with OCR fallback for image-based PDFs)
        from backend.utils.file_processor import FileProcessor
        try:
            # Try standard extraction first, fallback to OCR if needed
            pdf_text = await FileProcessor.read_content_with_fallback(file)
        except Exception as e:
            log_message("error", f"PDF extraction failed: {e}")
            return {
                "status": "failed",
                "error": f"Failed to extract text from PDF: {str(e)}"
            }

        # Log first 500 chars for debugging (don't log full text as it might be huge)
        preview = pdf_text[:500] if pdf_text else ""
        log_message("info", f"Extracted {len(pdf_text)} characters from PDF. Preview: {preview}...")

        
        if not pdf_text or len(pdf_text.strip()) < 50:
            return {
                "status": "failed",
                "error": "Could not extract sufficient text from PDF. The document may be empty or corrupted."
            }
        
        log_message("info", f"Extracted {len(pdf_text)} characters from PDF")
        
        # Use LLM to extract structured loan details
        # Limit to first 8000 chars to avoid token limits
        pdf_text_limited = pdf_text[:8000]
        extraction_prompt = f"""Extract the following loan details from the document text below. 
Return ONLY a valid JSON object with these exact keys (use null if not found):
- "legal_name": The legal name of the borrower/entity (as string)
- "funding_provided": The total funding/loan amount provided (as number or string)
- "annual_percentage_rate": The APR percentage (as number or string with %)
- "finance_charge": The total finance charge amount (as number or string)
- "total_payment_amount": The total payment amount over the loan term (as number or string)
- "payment": The regular payment amount (as number or string)
- "account_no": The account number (as string)
- "due_date": The first payment due date. Look for: "Due Date", "Payment Due", "Finance Agreement dated [date]", "Equipment Finance Schedule dated [date]", "Master Equipment Finance Agreement dated [date]", or any date after "dated" in finance/equipment context (as string in format YYYY-MM-DD or MM/DD/YYYY or any date format found)
- "start_date": The loan start date (as string in format YYYY-MM-DD or MM/DD/YYYY or any date format found) - if not found, will be calculated from due_date

Document text:
{pdf_text_limited}

Return ONLY the JSON object, no other text."""

        try:
            # Invoke LLM with structured output
            response = await asyncio.to_thread(
                llm.invoke,
                extraction_prompt
            )
            
            # Extract JSON from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_text = response_text.strip()
            log_message("info", f"Response text: {response_text}")
            
            # Try to extract JSON from the response
            import json
            loan_data = None
            
            # First, try parsing the entire response as JSON
            try:
                loan_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    try:
                        loan_data = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # If still no JSON, try to find the first balanced JSON object
                if loan_data is None:
                    brace_count = 0
                    start_idx = response_text.find('{')
                    if start_idx != -1:
                        for i in range(start_idx, len(response_text)):
                            if response_text[i] == '{':
                                brace_count += 1
                            elif response_text[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response_text[start_idx:i+1]
                                    try:
                                        loan_data = json.loads(json_str)
                                        break
                                    except json.JSONDecodeError:
                                        continue
            
            if loan_data is None:
                raise json.JSONDecodeError("Could not extract valid JSON from LLM response", response_text, 0)
            
            log_message("info", f"Successfully extracted loan details: {loan_data}")
            
            # Helper function to clean payment value
            def clean_payment_value(value):
                """Clean payment value by removing $, commas, /month, and other formatting."""
                if not value:
                    return None
                # Convert to string if not already
                value_str = str(value)
                # Remove dollar sign, commas, spaces, and /month or /mo suffixes
                cleaned = re.sub(r'[\$,\s]', '', value_str)
                cleaned = re.sub(r'/month|/mo|/monthly', '', cleaned, flags=re.IGNORECASE)
                # Extract just the numeric value (with decimal)
                numeric_match = re.search(r'(\d+\.?\d*)', cleaned)
                if numeric_match:
                    return numeric_match.group(1)
                return cleaned if cleaned else None
            
            # Prepare loan data for MongoDB
            payment_raw = loan_data.get("payment")
            payment_cleaned = clean_payment_value(payment_raw)
            
            # Calculate start_date from due_date + 45 days if due_date is available
            from dateutil import parser as date_parser
            from dateutil.relativedelta import relativedelta
            
            due_date_str = loan_data.get("due_date")
            start_date_str = loan_data.get("start_date")
            
            # Track if due_date was extracted via regex fallback
            due_date_from_regex = False
            
            # If due_date is still null, try to extract from the raw PDF text using regex
            if not due_date_str:
                log_message("info", "due_date not found in LLM response, trying regex extraction from PDF text")
                # Try to find dates in the PDF text that match our patterns
                due_date_patterns = [
                    # Text date formats first
                    r"finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"equipment\s+finance\s+schedule[^.]*dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"master\s+equipment\s+finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"due\s+date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                    r"payment\s+due[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})"
                ]
                for pattern in due_date_patterns:
                    match = re.search(pattern, pdf_text, re.IGNORECASE)
                    if match:
                        due_date_str = match.group(1).strip()
                        due_date_from_regex = True
                        log_message("info", f"Extracted due_date from PDF text using regex: {due_date_str}")
                        break
            
            # If we have due_date, always calculate start_date = due_date + 45 days
            # This overrides any LLM-provided start_date to ensure accuracy
            if due_date_str:
                try:
                    due_date = date_parser.parse(due_date_str)
                    calculated_start_date = due_date + relativedelta(days=45)
                    start_date_str = calculated_start_date.strftime("%Y-%m-%d")
                    if due_date_from_regex:
                        log_message("info", f"Calculated start_date from regex-extracted due_date: {due_date_str} + 45 days = {start_date_str} (overriding LLM start_date: {loan_data.get('start_date')})")
                    else:
                        log_message("info", f"Calculated start_date from due_date: {due_date_str} + 45 days = {start_date_str}")
                except Exception as e:
                    log_message("warning", f"Could not parse due_date to calculate start_date: {e}")
                    log_message("warning", f"due_date_str was: {due_date_str}")
            
            loan_details = {
                "legal_name": loan_data.get("legal_name"),
                "funding_provided": loan_data.get("funding_provided"),
                "annual_percentage_rate": loan_data.get("annual_percentage_rate"),
                "finance_charge": loan_data.get("finance_charge"),
                "total_payment_amount": loan_data.get("total_payment_amount"),
                "payment": payment_cleaned,
                "account_no": loan_data.get("account_no"),
                "start_date": start_date_str
            }
            
            # Validate client_id exists (foreign key validation)
            from backend.services.mongodb import get_database, validate_client_exists
            from datetime import datetime
            
            client_exists = await validate_client_exists(client_id)
            if not client_exists:
                log_message("warning", f"Client {client_id} does not exist, but proceeding with loan save")
                # Optionally, you can return an error here:
                # return {
                #     "status": "failed",
                #     "error": f"Client with ID '{client_id}' does not exist. Please create the client first."
                # }
            
            # Save to MongoDB
            try:
                db = get_database()
                collection = db["client_loans"]
                
                # Convert client_id string to ObjectId for foreign key reference
                from bson import ObjectId
                client_object_id = ObjectId(client_id)
                
                # Create document to insert
                loan_document = {
                    "client_id": client_object_id,  # ObjectId foreign key to clients._id
                    "pdf_filename": file.filename if hasattr(file, 'filename') else None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    **loan_details
                }
                
                # Insert into MongoDB
                result = await collection.insert_one(loan_document)
                log_message("info", f"Saved loan details to MongoDB with ID: {result.inserted_id}")
                
                return {
                    "status": "success",
                    "data": {
                        "id": str(result.inserted_id),
                        "client_id": client_id,
                        **loan_details
                    }
                }
            except Exception as e:
                log_message("error", f"Failed to save to MongoDB: {e}")
                # Return extracted data even if MongoDB save fails
                return {
                    "status": "success",
                    "data": {
                        "client_id": client_id,
                        **loan_details,
                        "mongodb_error": f"Failed to save to MongoDB: {str(e)}"
                    }
                }
            
        except json.JSONDecodeError as e:
            log_message("error", f"Failed to parse LLM response as JSON: {e}")
            log_message("error", f"LLM response was: {response_text[:500]}")
            
            # Fallback: try regex-based extraction
            regex_result = await _extract_loan_details_regex(pdf_text, client_id, file)
            return regex_result
            
        except Exception as e:
            log_message("error", f"LLM extraction failed: {e}")
            # Fallback to regex extraction
            regex_result = await _extract_loan_details_regex(pdf_text, client_id, file)
            return regex_result
            
    except Exception as e:
        log_message("error", f"Loan extraction failed: {e}")
        import traceback
        log_message("error", f"Traceback: {traceback.format_exc()}")
        return {
            "status": "failed",
            "error": str(e)
        }


async def _extract_loan_details_regex(pdf_text: str, client_id: str, file) -> Dict[str, Any]:
    """
    Fallback regex-based extraction for loan details.
    
    Args:
        pdf_text: Extracted text from PDF.
        client_id: Client identifier for MongoDB storage.
        file: UploadFile object (for filename).
        
    Returns:
        Dict with extracted loan details and MongoDB record ID.
    """
    try:
        log_message("info", "Using regex fallback for loan details extraction")
        
        # Patterns for different loan fields
        patterns = {
            "legal_name": [
                r"legal\s+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|borrower|account|loan|funding)",
                r"borrower[:\s]+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)",
                r"borrower[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)",
                r"name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account\s+number|loan|funding)",
                r"entity\s+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)",
                r"company\s+name[:\s]+([A-Za-z0-9\s,\.&\-']+?)(?:\n|$|account|loan|funding)"
            ],
            "funding_provided": [
                r"funding\s+provided[:\s]*\$?([\d,]+\.?\d*)",
                r"loan\s+amount[:\s]*\$?([\d,]+\.?\d*)",
                r"principal[:\s]*\$?([\d,]+\.?\d*)",
                r"amount\s+financed[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "annual_percentage_rate": [
                r"annual\s+percentage\s+rate[:\s]*([\d.]+)\s*%?",
                r"apr[:\s]*([\d.]+)\s*%?",
                r"interest\s+rate[:\s]*([\d.]+)\s*%?"
            ],
            "finance_charge": [
                r"finance\s+charge[:\s]*\$?([\d,]+\.?\d*)",
                r"total\s+finance\s+charge[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "total_payment_amount": [
                r"total\s+payment[:\s]*\$?([\d,]+\.?\d*)",
                r"total\s+of\s+payments[:\s]*\$?([\d,]+\.?\d*)",
                r"total\s+amount\s+payable[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "payment": [
                r"monthly\s+payment[:\s]*\$?([\d,]+\.?\d*)",
                r"payment\s+amount[:\s]*\$?([\d,]+\.?\d*)",
                r"regular\s+payment[:\s]*\$?([\d,]+\.?\d*)",
                r"payment[:\s]*\$?([\d,]+\.?\d*)"
            ],
            "account_no": [
                r"account\s+(?:number|no\.?)[:\s]*([\d\-X]+)",
                r"account\s+#[:\s]*([\d\-X]+)",
                r"acct\s+(?:number|no\.?)[:\s]*([\d\-X]+)"
            ],
            "due_date": [
                # PRIORITY: Text date formats with full month names (e.g., "July 29, 2025", "December 12, 2023")
                # Finance Agreement dated patterns - highest priority
                r"finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"master\s+equipment\s+finance\s+agreement\s+dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"equipment\s+finance\s+schedule[^.]*dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"equipment\s+finance\s+schedule\s+no[^.]*dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Generic "dated" pattern for finance/equipment context
                r"(?:finance|equipment|schedule|agreement|master).*?dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Simple "dated" pattern (catches "dated December 12, 2023" anywhere)
                r"dated\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Text date formats with "due date" labels
                r"due\s+date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"payment\s+due[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                r"first\s+payment\s+due[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
                # Fallback: Numeric date formats (only if text dates not found)
                r"due\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"payment\s+due[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"first\s+payment\s+due[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"first\s+due[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"payment\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"
            ],
            "start_date": [
                r"start\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"loan\s+start[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"date\s+of\s+loan[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"origination\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"beginning\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"effective\s+date[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"start[:\s]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
                r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s*(?:start|beginning|origination)"
            ]
        }
        
        extracted = {}
        pdf_text_lower = pdf_text.lower()
        
        # Log a sample of the text for debugging
        log_message("info", f"PDF text sample (first 1000 chars): {pdf_text[:1000]}")
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                # For legal_name and due_date (which may contain month names), search in original text to preserve case
                # For other fields, use lowercase text
                search_text = pdf_text if field in ["legal_name", "due_date"] else pdf_text_lower
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    log_message("info", f"Found {field} with pattern: {pattern[:50]}... -> value: {value}")
                    
                    # For due_date: Only accept text dates with month names (e.g., "July 29, 2025"), reject numeric-only dates
                    if field == "due_date":
                        # Check if it's a text date (contains month name) or numeric date
                        month_names = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                        if not re.search(month_names, value, re.IGNORECASE):
                            # This is a numeric date, skip it and continue to next pattern
                            log_message("info", f"Skipping numeric date format for due_date: {value}, looking for text date format")
                            continue
                        # It's a text date, accept it
                        log_message("info", f"Accepted text date format for due_date: {value}")
                    
                    # Clean up the value
                    if field == "payment":
                        # Remove dollar sign, commas, spaces, and /month or /mo suffixes
                        value = re.sub(r'[\$,\s]', '', value)
                        value = re.sub(r'/month|/mo|/monthly', '', value, flags=re.IGNORECASE)
                        # Extract just the numeric value (with decimal)
                        numeric_match = re.search(r'(\d+\.?\d*)', value)
                        if numeric_match:
                            value = numeric_match.group(1)
                    elif field in ["funding_provided", "finance_charge", "total_payment_amount"]:
                        # Remove commas from numbers
                        value = value.replace(",", "")
                    elif field == "legal_name":
                        # Clean up legal name - remove extra whitespace
                        value = re.sub(r'\s+', ' ', value).strip()
                    extracted[field] = value
                    break
        
        log_message("info", f"Regex extraction found: {extracted}")
        
        # Calculate start_date from due_date + 45 days if due_date is available
        from dateutil import parser as date_parser
        from dateutil.relativedelta import relativedelta
        
        due_date_str = extracted.get("due_date")
        start_date_str = extracted.get("start_date")
        
        # If we have due_date but no start_date, calculate start_date = due_date + 45 days
        if due_date_str and not start_date_str:
            try:
                due_date = date_parser.parse(due_date_str)
                calculated_start_date = due_date + relativedelta(days=45)
                start_date_str = calculated_start_date.strftime("%Y-%m-%d")
                log_message("info", f"Calculated start_date from due_date: {due_date_str} + 45 days = {start_date_str}")
            except Exception as e:
                log_message("warning", f"Could not parse due_date to calculate start_date: {e}")
        
        # Prepare loan data
        loan_details = {
            "legal_name": extracted.get("legal_name"),
            "funding_provided": extracted.get("funding_provided"),
            "annual_percentage_rate": extracted.get("annual_percentage_rate"),
            "finance_charge": extracted.get("finance_charge"),
            "total_payment_amount": extracted.get("total_payment_amount"),
            "payment": extracted.get("payment"),
            "account_no": extracted.get("account_no"),
            "start_date": start_date_str
        }
        
        # Save to MongoDB
        try:
            from backend.services.mongodb import get_database
            from datetime import datetime
            from bson import ObjectId
            
            db = get_database()
            collection = db["client_loans"]
            
            # Convert client_id string to ObjectId for foreign key reference
            client_object_id = ObjectId(client_id)
            
            # Create document to insert
            loan_document = {
                "client_id": client_object_id,  # ObjectId foreign key to clients._id
                "pdf_filename": file.filename if hasattr(file, 'filename') else None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                **loan_details
            }
            
            # Insert into MongoDB
            result = await collection.insert_one(loan_document)
            log_message("info", f"Saved loan details to MongoDB (regex fallback) with ID: {result.inserted_id}")
            
            return {
                "status": "success",
                "data": {
                    "id": str(result.inserted_id),
                    "client_id": str(client_object_id),  # Return ObjectId as string
                    **loan_details
                }
            }
        except Exception as e:
            log_message("error", f"Failed to save to MongoDB (regex fallback): {e}")
            # Return extracted data even if MongoDB save fails
            return {
                "status": "success",
                "data": {
                    "client_id": client_id,
                    **loan_details,
                    "mongodb_error": f"Failed to save to MongoDB: {str(e)}"
                }
            }
        
    except Exception as e:
        log_message("error", f"Regex extraction failed: {e}")
        return {
            "status": "failed",
            "error": f"Failed to extract loan details: {str(e)}"
        }