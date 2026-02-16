"""
MongoDB Connection Service

Provides MongoDB database connection and utilities for async operations.
"""

import os
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from backend.utils.logger import log_message
from dotenv import load_dotenv

load_dotenv()

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "cpa_backend_migration")

# Global MongoDB client
_client: AsyncIOMotorClient = None
_database = None


def get_mongodb_client() -> AsyncIOMotorClient:
    """
    Get or create MongoDB client (singleton pattern).
    
    Returns:
        AsyncIOMotorClient: MongoDB client instance
    """
    global _client
    if _client is None:
        try:
            _client = AsyncIOMotorClient(MONGODB_URL)
            log_message("info", f"Connected to MongoDB at {MONGODB_URL}")
        except Exception as e:
            log_message("error", f"Failed to connect to MongoDB: {e}")
            raise
    return _client


def get_database():
    """
    Get MongoDB database instance.
    
    Returns:
        Database: MongoDB database instance
    """
    global _database
    if _database is None:
        client = get_mongodb_client()
        _database = client[MONGODB_DB_NAME]
        log_message("info", f"Using database: {MONGODB_DB_NAME}")
    return _database


async def close_mongodb_connection():
    """
    Close MongoDB connection.
    """
    global _client
    if _client:
        _client.close()
        _client = None
        log_message("info", "MongoDB connection closed")


async def validate_client_exists(client_id: str) -> bool:
    """
    Validate that a client exists in the clients collection by _id (ObjectId).
    
    Args:
        client_id: Client ObjectId string to validate
        
    Returns:
        bool: True if client exists, False otherwise
    """
    try:
        from bson import ObjectId
        
        db = get_database()
        clients_collection = db["clients"]
        
        # Try to convert to ObjectId and check if client exists by _id
        try:
            client_object_id = ObjectId(client_id)
            client = await clients_collection.find_one({"_id": client_object_id})
            
            if client:
                log_message("info", f"Client validation: {client_id} exists (by _id)")
                return True
            else:
                log_message("warning", f"Client validation: {client_id} does not exist (by _id)")
                return False
        except Exception as e:
            log_message("error", f"Invalid ObjectId format: {client_id}, error: {e}")
            return False
            
    except Exception as e:
        log_message("error", f"Error validating client: {e}")
        return False


async def create_client_indexes():
    """
    Create indexes on the clients collection for better performance.
    """
    try:
        db = get_database()
        clients_collection = db["clients"]
        
        # Create unique index on client_id
        await clients_collection.create_index("client_id", unique=True)
        log_message("info", "Created unique index on clients.client_id")
        
    except Exception as e:
        log_message("warning", f"Failed to create client indexes (may already exist): {e}")


async def create_client_loan_indexes():
    """
    Create indexes on the client_loans collection for better performance.
    """
    try:
        db = get_database()
        client_loans_collection = db["client_loans"]
        
        # Create index on client_id for faster queries
        await client_loans_collection.create_index("client_id")
        log_message("info", "Created index on client_loans.client_id")
        
        # Create compound index on client_id and created_at for sorting
        await client_loans_collection.create_index([("client_id", 1), ("created_at", -1)])
        log_message("info", "Created compound index on client_loans (client_id, created_at)")
        
    except Exception as e:
        log_message("warning", f"Failed to create client_loan indexes (may already exist): {e}")


async def get_client_by_id(client_id: str) -> dict:
    """
    Get client from MongoDB by client_id (tries both _id as ObjectId and client_id as string).
    
    Args:
        client_id: Client identifier (can be ObjectId string or client_id string)
        
    Returns:
        dict: Client document if found, None otherwise
    """
    try:
        from bson import ObjectId
        
        db = get_database()
        clients_collection = db["clients"]
        
        # First try to find by _id (ObjectId)
        try:
            client_object_id = ObjectId(client_id)
            client = await clients_collection.find_one({"_id": client_object_id})
            if client:
                log_message("info", f"Found client by _id: {client_id}")
                return client
        except Exception:
            # Not a valid ObjectId, continue to try client_id field
            pass
        
        # Try to find by client_id field (string)
        client = await clients_collection.find_one({"client_id": client_id})
        if client:
            log_message("info", f"Found client by client_id field: {client_id}")
            return client
        
        log_message("warning", f"Client not found: {client_id}")
        return None
        
    except Exception as e:
        log_message("error", f"Error getting client by id: {e}")
        return None


async def update_client_pinecone_index(client_id: str, pinecone_index_name: str) -> bool:
    """
    Update client's pinecone_index_name in MongoDB.
    
    Args:
        client_id: Client identifier (can be ObjectId string or client_id string)
        pinecone_index_name: Pinecone index name to store
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        from bson import ObjectId
        from datetime import datetime
        
        db = get_database()
        clients_collection = db["clients"]
        
        # Build query - try both _id and client_id
        query = {}
        try:
            client_object_id = ObjectId(client_id)
            query["_id"] = client_object_id
        except Exception:
            query["client_id"] = client_id
        
        # Update the client document
        update_result = await clients_collection.update_one(
            query,
            {
                "$set": {
                    "pinecone_index_name": pinecone_index_name,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if update_result.modified_count > 0:
            log_message("info", f"Updated pinecone_index_name for client {client_id}: {pinecone_index_name}")
            return True
        elif update_result.matched_count > 0:
            log_message("info", f"Client {client_id} already has pinecone_index_name: {pinecone_index_name}")
            return True
        else:
            log_message("warning", f"Client {client_id} not found for update")
            return False
        
    except Exception as e:
        log_message("error", f"Error updating client pinecone_index: {e}")
        return False


# -------------------------------------------------------------------------
# Client Pinecone Index Collection Functions
# -------------------------------------------------------------------------

async def get_client_pinecone_index(client_id: str) -> dict:
    """
    Get client Pinecone index information from client_pinecone_index collection.
    
    Args:
        client_id: Client identifier (can be ObjectId string or client_id string)
        
    Returns:
        dict: Client Pinecone index document if found, None otherwise
    """
    try:
        db = get_database()
        collection = db["client_pinecone_index"]
        
        # Find by client_id
        result = await collection.find_one({"client_id": client_id})
        
        if result:
            log_message("info", f"Found client_pinecone_index for client_id: {client_id}")
            return result
        else:
            log_message("warning", f"Client Pinecone index not found for client_id: {client_id}")
            return None
        
    except Exception as e:
        log_message("error", f"Error getting client Pinecone index: {e}")
        return None


async def create_or_update_client_pinecone_index(
    client_id: str,
    client_name: str,
    pinecone_index_name: Optional[str] = None
) -> bool:
    """
    Create or update client Pinecone index information in client_pinecone_index collection.
    
    Args:
        client_id: Client identifier
        client_name: Client name
        pinecone_index_name: Pinecone index name (optional, if None or empty, only stores client_id and name)
        
    Returns:
        bool: True if operation was successful, False otherwise
    """
    try:
        from datetime import datetime
        
        db = get_database()
        collection = db["client_pinecone_index"]
        
        # Check if record exists
        existing = await collection.find_one({"client_id": client_id})
        
        # Build update/create document - only include pinecone_index_name if provided
        if existing:
            # Update existing record
            update_operations = {
                "$set": {
                    "name": client_name,
                    "updated_at": datetime.utcnow()
                }
            }
            
            # Only set pinecone_index_name if provided (not None and not empty)
            if pinecone_index_name:
                update_operations["$set"]["pinecone_index_name"] = pinecone_index_name
            else:
                # If pinecone_index_name is None/empty, remove it from the document
                update_operations["$unset"] = {"pinecone_index_name": ""}
            
            update_result = await collection.update_one(
                {"client_id": client_id},
                update_operations
            )
            
            if update_result.modified_count > 0 or update_result.matched_count > 0:
                log_message(
                    "info",
                    f"Updated client_pinecone_index for client_id={client_id}, "
                    f"index_name={pinecone_index_name or '(not set)'}"
                )
                return True
            else:
                log_message("warning", f"Failed to update client_pinecone_index for client_id={client_id}")
                return False
        else:
            # Create new record
            new_record = {
                "client_id": client_id,
                "name": client_name,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Only include pinecone_index_name if provided
            if pinecone_index_name:
                new_record["pinecone_index_name"] = pinecone_index_name
            
            result = await collection.insert_one(new_record)
            
            if result.inserted_id:
                log_message(
                    "info",
                    f"Created client_pinecone_index for client_id={client_id}, index_name={pinecone_index_name}"
                )
                return True
            else:
                log_message("warning", f"Failed to create client_pinecone_index for client_id={client_id}")
                return False
        
    except Exception as e:
        log_message("error", f"Error creating/updating client Pinecone index: {e}")
        return False


async def create_client_pinecone_index_indexes():
    """
    Create indexes on the client_pinecone_index collection for better performance.
    """
    try:
        db = get_database()
        collection = db["client_pinecone_index"]
        
        # Create unique index on client_id (one index per client)
        await collection.create_index("client_id", unique=True)
        log_message("info", "Created unique index on client_pinecone_index.client_id")
        
        # Create index on pinecone_index_name for faster lookups
        await collection.create_index("pinecone_index_name")
        log_message("info", "Created index on client_pinecone_index.pinecone_index_name")
        
    except Exception as e:
        log_message("warning", f"Failed to create client_pinecone_index indexes (may already exist): {e}")


# -------------------------------------------------------------------------
# Client Status Collection Functions
# -------------------------------------------------------------------------

async def create_or_update_client_status(
    client_id: str,
    client_name: str,
    status: str = "processing"
) -> bool:
    """
    Create or update client GL processing status in client_status collection.
    
    Args:
        client_id: Client identifier
        client_name: Client name
        status: Processing status (default: "processing")
    
    Returns:
        bool: True if operation was successful, False otherwise
    """
    try:
        from datetime import datetime
        
        db = get_database()
        collection = db["client_status"]
        
        # Check if record exists
        existing = await collection.find_one({"client_id": client_id})
        
        if existing:
            # Update existing record
            update_result = await collection.update_one(
                {"client_id": client_id},
                {
                    "$set": {
                        "client_name": client_name,
                        "status": status,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if update_result.modified_count > 0 or update_result.matched_count > 0:
                log_message("info", f"Updated client_status for client_id={client_id}, status={status}")
                return True
            else:
                log_message("warning", f"Failed to update client_status for client_id={client_id}")
                return False
        else:
            # Create new record
            status_doc = {
                "client_id": client_id,
                "client_name": client_name,
                "status": status,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            result = await collection.insert_one(status_doc)
            
            if result.inserted_id:
                log_message("info", f"Created client_status for client_id={client_id}, status={status}")
                return True
            else:
                log_message("warning", f"Failed to create client_status for client_id={client_id}")
                return False
        
    except Exception as e:
        log_message("error", f"Error creating/updating client status: {e}")
        return False


async def get_client_status(client_id: str) -> Optional[dict]:
    """
    Get client GL processing status from client_status collection.
    
    Args:
        client_id: Client identifier
    
    Returns:
        dict: Client status document if found, None otherwise
    """
    try:
        db = get_database()
        collection = db["client_status"]
        
        result = await collection.find_one({"client_id": client_id})
        
        if result:
            log_message("info", f"Found client_status for client_id: {client_id}")
            return result
        else:
            log_message("warning", f"Client status not found for client_id: {client_id}")
            return None
        
    except Exception as e:
        log_message("error", f"Error getting client status: {e}")
        return None


# -------------------------------------------------------------------------
# Client Unique Data Collection Functions
# -------------------------------------------------------------------------

async def insert_client_unique_data(
    client_id: str,
    client_name: str,
    pinecone_index_name: str,
    duplicate_data: List[Dict[str, Any]] = None
) -> bool:
    """
    Insert client unique/duplicate data into client_uniquedata collection.
    
    Args:
        client_id: Client identifier
        client_name: Client name
        pinecone_index_name: Pinecone index name
        duplicate_data: List of duplicate records (each with date, memo, split/category, credit, debit)
    
    Returns:
        bool: True if operation was successful, False otherwise
    """
    try:
        from datetime import datetime
        
        db = get_database()
        collection = db["client_uniquedata"]
        
        if duplicate_data is None:
            duplicate_data = []
        
        # Create document
        unique_doc = {
            "client_id": client_id,
            "client_name": client_name,
            "pinecone_index_name": pinecone_index_name,
            "duplicate_data": duplicate_data,
            "duplicate_count": len(duplicate_data),
            "created_at": datetime.utcnow()
        }
        
        result = await collection.insert_one(unique_doc)
        
        if result.inserted_id:
            log_message("info", f"Inserted client_uniquedata for client_id={client_id}, duplicate_count={len(duplicate_data)}")
            return True
        else:
            log_message("warning", f"Failed to insert client_uniquedata for client_id={client_id}")
            return False
        
    except Exception as e:
        log_message("error", f"Error inserting client unique data: {e}")
        return False


async def get_client_unique_data(client_id: Optional[str] = None) -> List[dict]:
    """
    Get client unique/duplicate data from client_uniquedata collection.
    
    Args:
        client_id: Client identifier (optional, if not provided returns all records)
    
    Returns:
        List[dict]: List of unique data documents
    """
    try:
        db = get_database()
        collection = db["client_uniquedata"]
        
        # Build query
        query = {}
        if client_id:
            query["client_id"] = client_id
        
        # Find all matching records, sorted by created_at descending
        cursor = collection.find(query).sort("created_at", -1)
        results = await cursor.to_list(length=None)
        
        log_message("info", f"Found {len(results)} client_uniquedata record(s) for client_id={client_id or 'all'}")
        return results
        
    except Exception as e:
        log_message("error", f"Error getting client unique data: {e}")
        return []


# -------------------------------------------------------------------------
# Client Bank Extract Data Collection Functions
# -------------------------------------------------------------------------

async def insert_client_bank_extract_data(
    user_id: str,
    client_name: str,
    bank_extract_data: Dict[str, Any]
) -> Optional[str]:
    """
    Insert client bank extract data into client_bankextarct_data collection.
    
    This stores the processed bank statement data (CSV content, banks, accounts, etc.)
    from the /process endpoint.
    
    Args:
        user_id: User identifier (from user table)
        client_name: Client name
        bank_extract_data: Dictionary containing processed bank extract data
            (should include banks list with accounts and csv_content, coa_categories, etc.)
    
    Returns:
        str: Inserted document _id as string if successful, None otherwise
    """
    try:
        from datetime import datetime
        
        db = get_database()
        collection = db["client_bankextarct_data"]
        
        # Create document
        extract_doc = {
            "user_id": user_id,
            "client_name": client_name,
            "data": bank_extract_data,  # Store the entire response data
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await collection.insert_one(extract_doc)
        
        if result.inserted_id:
            log_message("info", f"Inserted client_bankextarct_data for user_id={user_id}, record_id={result.inserted_id}")
            return str(result.inserted_id)
        else:
            log_message("warning", f"Failed to insert client_bankextarct_data for user_id={user_id}")
            return None
            
    except Exception as e:
        log_message("error", f"Error inserting client bank extract data: {e}")
        return None


async def insert_client_bank_extract_batch(
    user_id: str,
    client_name: str,
    batch_num: int,
    total_batches: int,
    batch_csv: str,
    batch_transactions: List[Dict[str, Any]],
    account_key: Optional[str] = None
) -> Optional[str]:
    """
    Insert a single batch of processed transactions into MongoDB for streaming.
    
    This stores individual batches as they complete during streaming processing,
    allowing the frontend to receive and display results incrementally.
    
    Args:
        user_id: User identifier (from user table)
        client_name: Client name
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches
        batch_csv: CSV string for this batch
        batch_transactions: List of transaction dictionaries for this batch
        account_key: Account key identifier (e.g., "BOA_123456")
    
    Returns:
        str: Inserted document _id as string if successful, None otherwise
    """
    try:
        from datetime import datetime
        
        db = get_database()
        collection = db["client_bankextract_batches"]
        
        # Create batch document
        batch_doc = {
            "user_id": user_id,
            "client_name": client_name,
            "account_key": account_key,
            "batch_num": batch_num,
            "total_batches": total_batches,
            "batch_csv": batch_csv,
            "batch_transactions": batch_transactions,
            "transaction_count": len(batch_transactions),
            "created_at": datetime.utcnow()
        }
        
        result = await collection.insert_one(batch_doc)
        
        if result.inserted_id:
            log_message("info", f"Stored batch {batch_num}/{total_batches} for user_id={user_id}, record_id={result.inserted_id}")
            return str(result.inserted_id)
        else:
            log_message("warning", f"Failed to store batch {batch_num}/{total_batches} for user_id={user_id}")
            return None
            
    except Exception as e:
        log_message("error", f"Error storing batch {batch_num}/{total_batches}: {e}")
        return None


async def get_client_bank_extract_data(
    user_id: str,
    limit: Optional[int] = None
) -> List[dict]:
    """
    Get client bank extract data from client_bankextarct_data collection.
    
    Args:
        user_id: User identifier (required, from user table)
        limit: Maximum number of records to return (optional, returns most recent if specified)
    
    Returns:
        List[dict]: List of bank extract data documents, sorted by created_at descending
    """
    try:
        db = get_database()
        collection = db["client_bankextarct_data"]
        
        # Build query - user_id is required
        query = {"user_id": user_id}
        
        # Find all matching records, sorted by created_at descending
        cursor = collection.find(query).sort("created_at", -1)
        
        # Apply limit if specified
        if limit and limit > 0:
            cursor = cursor.limit(limit)
        
        results = await cursor.to_list(length=None)
        
        log_message("info", f"Found {len(results)} client_bankextarct_data record(s) for user_id={user_id}")
        return results
        
    except Exception as e:
        log_message("error", f"Error getting client bank extract data: {e}")
        return []


async def get_client_bank_extract_data_by_id(record_id: str) -> Optional[dict]:
    """
    Get a specific client bank extract data record by _id.
    
    Args:
        record_id: MongoDB document _id (as string)
    
    Returns:
        dict: Bank extract data document if found, None otherwise
    """
    try:
        from bson import ObjectId
        
        db = get_database()
        collection = db["client_bankextarct_data"]
        
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(record_id)
            result = await collection.find_one({"_id": object_id})
            
            if result:
                log_message("info", f"Found client_bankextarct_data record: {record_id}")
                return result
            else:
                log_message("warning", f"Client bank extract data record not found: {record_id}")
                return None
        except Exception as e:
            log_message("error", f"Invalid ObjectId format: {record_id}, error: {e}")
            return None
        
    except Exception as e:
        log_message("error", f"Error getting client bank extract data by id: {e}")
        return None


async def create_client_bank_extract_data_indexes():
    """
    Create indexes on the client_bankextarct_data collection for better performance.
    """
    try:
        db = get_database()
        collection = db["client_bankextarct_data"]
        
        # Create index on user_id for faster queries
        await collection.create_index("user_id")
        log_message("info", "Created index on client_bankextarct_data.user_id")
        
        # Create compound index on user_id and created_at for sorting
        await collection.create_index([("user_id", 1), ("created_at", -1)])
        log_message("info", "Created compound index on client_bankextarct_data (user_id, created_at)")
        
    except Exception as e:
        log_message("warning", f"Failed to create client_bankextarct_data indexes (may already exist): {e}")