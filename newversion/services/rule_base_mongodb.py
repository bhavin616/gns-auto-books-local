"""
MongoDB Service for Client Rule Base

Handles storage and retrieval of client rule base data in MongoDB.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from uuid import uuid4
from backend.services.mongodb import get_database
from backend.utils.logger import log_message
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

_client: AsyncIOMotorClient = None
_database = None

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "cpa_backend_migration")

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

async def create_or_update_client_rule_base(
    client_id: str,
    client_name: str,
    nature_of_business: str,
    rules: List[Dict[str, Any]],
    gl_start_date: Optional[str] = None,
    gl_end_date: Optional[str] = None,
    gl_year: Optional[Any] = None,
) -> bool:
    """
    Create or update client rule base in MongoDB.
    
    Args:
        client_id: Client identifier
        client_name: Client name
        nature_of_business: Nature of business
        rules: List of rule dictionaries
        gl_start_date: GL transaction start date (e.g. 1/1/24)
        gl_end_date: GL transaction end date (e.g. 7/8/24)
        gl_year: GL year (int or list of ints if spanning multiple years)
        
    Returns:
        bool: True if operation was successful, False otherwise
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        
        # Check if record exists
        existing = await collection.find_one({"client_id": client_id})
        
        # Build document
        rule_base_doc = {
            "client_id": client_id,
            "client_name": client_name,
            "nature_of_business": nature_of_business,
            "rules": rules,
            "rule_count": len(rules),
            "updated_at": datetime.utcnow()
        }
        if gl_start_date is not None:
            rule_base_doc["gl_start_date"] = gl_start_date
        if gl_end_date is not None:
            rule_base_doc["gl_end_date"] = gl_end_date
        if gl_year is not None:
            rule_base_doc["gl_year"] = gl_year
        
        if existing:
            # Update existing record (do not overwrite rules_slots here; regenerate uses store_regenerated_rules)
            update_result = await collection.update_one(
                {"client_id": client_id},
                {"$set": rule_base_doc}
            )
            
            if update_result.modified_count > 0 or update_result.matched_count > 0:
                log_message(
                    "info",
                    f"Updated client_rule_base for client_id={client_id}, "
                    f"rule_count={len(rules)}"
                )
                return True
            else:
                log_message("warning", f"Failed to update client_rule_base for client_id={client_id}")
                return False
        else:
            # Create new record: first-time generate stores only rules (no rules_slots = no duplication).
            # rules_slots is added only on first regenerate when we need two slots.
            rule_base_doc["created_at"] = datetime.utcnow()
            result = await collection.insert_one(rule_base_doc)
            
            if result.inserted_id:
                log_message(
                    "info",
                    f"Created client_rule_base for client_id={client_id}, "
                    f"rule_count={len(rules)}"
                )
                return True
            else:
                log_message("warning", f"Failed to create client_rule_base for client_id={client_id}")
                return False
        
    except Exception as e:
        log_message("error", f"Error creating/updating client rule base: {e}")
        return False


async def store_regenerated_rules(
    client_id: str,
    client_name: str,
    nature_of_business: str,
    new_rules: List[Dict[str, Any]],
    gl_start_date: Optional[str] = None,
    gl_end_date: Optional[str] = None,
    gl_year: Optional[Any] = None,
) -> bool:
    """
    Store regenerated rules using two-slot history: index 0 = older, index 1 = newer.
    - First regenerate: store at index 1 (index 0 already has first generate).
    - Second+ regenerate: drop old index 0, move current index 1 to index 0, put new at index 1.
    Also sets `rules` to the latest (new_rules) for categorization and update/delete.
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        doc = await collection.find_one({"client_id": client_id})
        if not doc:
            log_message("warning", f"No client_rule_base for client_id={client_id}")
            return False

        slots = doc.get("rules_slots") or []
        current_rules = doc.get("rules") or []

        # If we have 2 slots, new_slots = [previous index 1, new_rules]. Else new_slots = [previous single slot or current rules, new_rules].
        if len(slots) >= 2:
            new_slots = [slots[1], new_rules]  # drop old index 0, shift 1->0, new at 1
        else:
            # First regenerate: index 0 = existing (from rules_slots[0] or rules), index 1 = new
            older = slots[0] if slots else current_rules
            new_slots = [older, new_rules]

        rule_base_doc = {
            "client_id": client_id,
            "client_name": client_name,
            "nature_of_business": nature_of_business,
            "rules": new_rules,
            "rules_slots": new_slots,
            "rule_count": len(new_rules),
            "updated_at": datetime.utcnow(),
        }
        if gl_start_date is not None:
            rule_base_doc["gl_start_date"] = gl_start_date
        if gl_end_date is not None:
            rule_base_doc["gl_end_date"] = gl_end_date
        if gl_year is not None:
            rule_base_doc["gl_year"] = gl_year

        update_result = await collection.update_one(
            {"client_id": client_id},
            {"$set": rule_base_doc},
        )
        if update_result.matched_count == 0:
            log_message("warning", f"Failed to update client_rule_base for client_id={client_id}")
            return False
        log_message(
            "info",
            f"Stored regenerated rules for client_id={client_id}, slots: [len={len(new_slots[0])}, len={len(new_slots[1])}]",
        )
        return True
    except Exception as e:
        log_message("error", f"Error storing regenerated rules: {e}")
        return False


async def get_client_rule_base(client_id: str) -> Optional[Dict[str, Any]]:
    """
    Get client rule base from MongoDB.
    
    Args:
        client_id: Client identifier
        
    Returns:
        dict: Client rule base document if found, None otherwise
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        
        result = await collection.find_one({"client_id": client_id})
        
        if result:
            log_message("info", f"Found client_rule_base for client_id: {client_id}")
            return result
        else:
            log_message("warning", f"Client rule base not found for client_id: {client_id}")
            return None
        
    except Exception as e:
        log_message("error", f"Error getting client rule base: {e}")
        return None


async def get_client_rule_base_by_name(client_name: str) -> Optional[Dict[str, Any]]:
    """
    Get client rule base from MongoDB by client name (case-insensitive).
    
    Args:
        client_name: Client name
        
    Returns:
        dict: Client rule base document if found, None otherwise
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        
        # Case-insensitive search
        result = await collection.find_one({
            "client_name": {"$regex": f"^{re.escape(client_name)}$", "$options": "i"}
        })
        
        if result:
            log_message("info", f"Found client_rule_base for client_name: {client_name}")
            return result
        else:
            log_message("warning", f"Client rule base not found for client_name: {client_name}")
            return None
        
    except Exception as e:
        log_message("error", f"Error getting client rule base by name: {e}")
        return None


async def get_all_client_rule_bases() -> List[Dict[str, Any]]:
    """
    Get all client rule bases from MongoDB.
    
    Returns:
        List of client rule base documents
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        
        cursor = collection.find({}).sort("created_at", -1)
        results = await cursor.to_list(length=None)
        
        log_message("info", f"Found {len(results)} client rule base record(s)")
        return results
        
    except Exception as e:
        log_message("error", f"Error getting all client rule bases: {e}")
        return []


def _normalize_trigger_values(values: List[Any]) -> List[str]:
    """Normalize trigger strings to uppercase and remove empty values."""
    out: List[str] = []
    for value in values or []:
        text = str(value).upper().strip()
        if text:
            out.append(text)
    return sorted(set(out))


async def get_client_rule_by_id(client_id: str, rule_id: str) -> Optional[Dict[str, Any]]:
    """Get one rule by rule_id for a client."""
    try:
        db = get_database()
        collection = db["client_rule_base"]
        doc = await collection.find_one(
            {"client_id": client_id, "rules.rule_id": rule_id},
            {"rules": {"$elemMatch": {"rule_id": rule_id}}},
        )
        if not doc:
            return None
        rules = doc.get("rules") or []
        return rules[0] if rules else None
    except Exception as e:
        log_message("error", f"Error getting client rule by id: {e}")
        return None


async def create_client_rule(client_id: str, rule_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Create and append one new rule for a client.
    Appends to top-level rules and current rules slot (if slot history exists).
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        doc = await collection.find_one({"client_id": client_id}, {"rules": 1, "rules_slots": 1})
        if not doc:
            return None

        new_rule = {
            "rule_id": str(rule_data.get("rule_id") or uuid4()),
            "trigger_payee_contains": _normalize_trigger_values(rule_data.get("trigger_payee_contains") or []),
            "account_id": str(rule_data.get("account_id") or "").strip(),
            "account_name": str(rule_data.get("account_name") or "").strip(),
            "category_type": str(rule_data.get("category_type") or "Expense").strip(),
            "logic": str(rule_data.get("logic") or "").strip(),
        }

        # Optional fields
        vendor_code = str(rule_data.get("vendor_code") or "").strip()
        if vendor_code:
            new_rule["vendor_code"] = vendor_code
        if rule_data.get("ach_vendor_only") is not None:
            new_rule["ach_vendor_only"] = bool(rule_data.get("ach_vendor_only"))

        if not new_rule["trigger_payee_contains"] or not new_rule["account_name"]:
            return None

        update_doc: Dict[str, Any] = {
            "$push": {"rules": new_rule},
            "$currentDate": {"updated_at": True},
        }

        slots = doc.get("rules_slots") or []
        if slots:
            slot_idx = len(slots) - 1
            update_doc["$push"][f"rules_slots.{slot_idx}"] = new_rule

        update_result = await collection.update_one({"client_id": client_id}, update_doc)
        if update_result.matched_count == 0:
            return None

        updated_doc = await collection.find_one({"client_id": client_id}, {"rules": 1})
        new_count = len((updated_doc or {}).get("rules") or [])
        await collection.update_one({"client_id": client_id}, {"$set": {"rule_count": new_count}})
        return new_rule
    except Exception as e:
        log_message("error", f"Error creating client rule: {e}")
        return None


async def update_client_rule(
    client_id: str,
    rule_id: str,
    updates: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update a single rule inside a client's rule base (by rule_id).
    Updates both the top-level `rules` array and the current slot in `rules_slots`
    so that UI and MongoDB stay in sync after regenerate.
    """
    try:
        if not updates:
            return None

        allowed_fields = {
            "trigger_payee_contains",
            "account_id",
            "account_name",
            "category_type",
            "logic",
            "vendor_code",
            "ach_vendor_only",
        }

        set_doc: Dict[str, Any] = {}
        for k, v in updates.items():
            if k in allowed_fields:
                set_doc[f"rules.$.{k}"] = v

        if not set_doc:
            return None

        db = get_database()
        collection = db["client_rule_base"]

        # Update the matched rule inside rules array
        update_result = await collection.update_one(
            {"client_id": client_id, "rules.rule_id": rule_id},
            {
                "$set": set_doc,
                "$currentDate": {"updated_at": True},
            },
        )

        if update_result.matched_count == 0:
            log_message("warning", f"No matching rule found for client_id={client_id}, rule_id={rule_id}")
            return None

        # Also update the same rule in rules_slots (current slot = last index) so MongoDB view stays in sync
        doc = await collection.find_one(
            {"client_id": client_id},
            {"rules_slots": 1},
        )
        if doc and doc.get("rules_slots"):
            slots = doc["rules_slots"]
            slot_idx = len(slots) - 1  # current slot: 0 if one slot, 1 if two slots
            slot_set = {f"rules_slots.{slot_idx}.$[r].{k}": v for k, v in updates.items() if k in allowed_fields}
            if slot_set:
                await collection.update_one(
                    {"client_id": client_id},
                    {"$set": slot_set, "$currentDate": {"updated_at": True}},
                    array_filters=[{"r.rule_id": rule_id}],
                )

        # Fetch only the updated rule back
        doc = await collection.find_one(
            {"client_id": client_id},
            {
                "client_id": 1,
                "client_name": 1,
                "nature_of_business": 1,
                "rules": {"$elemMatch": {"rule_id": rule_id}},
                "updated_at": 1,
            },
        )

        if not doc:
            return None

        rules = doc.get("rules") or []
        if not rules:
            return None

        updated_rule = rules[0]
        log_message("info", f"Updated rule_id={rule_id} for client_id={client_id}")
        return updated_rule

    except Exception as e:
        log_message("error", f"Error updating client rule: {e}")
        return None


async def delete_client_rule_base(client_id: str) -> bool:
    """
    Delete client rule base from MongoDB.
    
    Args:
        client_id: Client identifier
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        
        result = await collection.delete_one({"client_id": client_id})
        
        if result.deleted_count > 0:
            log_message("info", f"Deleted client_rule_base for client_id: {client_id}")
            return True
        else:
            log_message("warning", f"Client rule base not found for deletion: {client_id}")
            return False
        
    except Exception as e:
        log_message("error", f"Error deleting client rule base: {e}")
        return False


async def delete_client_rule(
    client_id: str,
    rule_id: str,
) -> bool:
    """
    Delete a single rule from a client's rule base by rule_id.
    Removes from both top-level `rules` and the current slot in `rules_slots`.
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]

        # Get current rules_slots so we can also pull from the current slot
        doc = await collection.find_one(
            {"client_id": client_id},
            {"rules_slots": 1},
        )
        slot_idx = None
        if doc and doc.get("rules_slots"):
            slot_idx = len(doc["rules_slots"]) - 1

        pull_doc: Dict[str, Any] = {"rules": {"rule_id": rule_id}}
        if slot_idx is not None:
            pull_doc[f"rules_slots.{slot_idx}"] = {"rule_id": rule_id}

        update_result = await collection.update_one(
            {"client_id": client_id},
            {
                "$pull": pull_doc,
                "$currentDate": {"updated_at": True},
            },
        )

        if update_result.modified_count == 0:
            log_message(
                "warning",
                f"No rule deleted for client_id={client_id}, rule_id={rule_id} (rule may not exist)",
            )
            return False

        # Recalculate rule_count to keep it accurate
        doc = await collection.find_one(
            {"client_id": client_id},
            {"rules": 1},
        )
        if doc is not None:
            new_count = len(doc.get("rules") or [])
            await collection.update_one(
                {"client_id": client_id},
                {"$set": {"rule_count": new_count}},
            )

        log_message("info", f"Deleted rule_id={rule_id} for client_id={client_id}")
        return True

    except Exception as e:
        log_message("error", f"Error deleting client rule: {e}")
        return False


async def add_old_rule_to_current_rules(
    client_id: str,
    rule_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Copy a rule from the old slot (rules_slots[0]) into the current rules (rules_slots[last] and rules).
    Used after regenerate when the client wants to keep a specific old rule in the new rule set.
    Old rule remains unchanged in the old slot.

    Requires rules_slots with at least 2 slots (i.e. client must have run regenerate at least once).

    Returns:
        The added rule dict if successful; None if rule not found in old slot, already in current, or only one slot.
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        doc = await collection.find_one(
            {"client_id": client_id},
            {"rules": 1, "rules_slots": 1},
        )
        if not doc:
            return None

        slots = doc.get("rules_slots") or []
        if len(slots) < 2:
            log_message(
                "warning",
                f"add_old_rule_to_current_rules: client_id={client_id} has no two-slot history (regenerate first)",
            )
            return None

        old_rules = slots[0]
        current_rules = doc.get("rules") or slots[-1]
        slot_idx = len(slots) - 1

        # Find rule in old slot
        rule_doc = None
        for r in old_rules:
            if r.get("rule_id") == rule_id:
                rule_doc = r
                break
        if not rule_doc:
            log_message("warning", f"Rule rule_id={rule_id} not found in old slot for client_id={client_id}")
            return None

        # Skip if already in current rules
        if any((r.get("rule_id") == rule_id for r in current_rules)):
            log_message("info", f"Rule rule_id={rule_id} already in current rules for client_id={client_id}")
            return rule_doc

        # Push to rules and to current slot
        new_count = len(current_rules) + 1
        update_result = await collection.update_one(
            {"client_id": client_id},
            {
                "$push": {
                    "rules": rule_doc,
                    f"rules_slots.{slot_idx}": rule_doc,
                },
                "$set": {
                    "rule_count": new_count,
                    "updated_at": datetime.utcnow(),
                },
            },
        )
        if update_result.modified_count == 0:
            return None
        log_message(
            "info",
            f"Added old rule rule_id={rule_id} to current rules for client_id={client_id}",
        )
        return rule_doc
    except Exception as e:
        log_message("error", f"Error adding old rule to current rules: {e}")
        return None


# -------------------------------------------------------------------------
# Client Bank Statement Data Collection (categorize output)
# -------------------------------------------------------------------------

COLLECTION_CLIENT_BANK_STATEMENT_DATA = "client_bank_statement_data"
COLLECTION_STATEMENT_RUN_OVERRIDES = "statement_run_overrides"


async def store_client_bank_statement_data(
    user_id: str,
    result: Dict[str, Any],
    client_id: Optional[str] = None,
    client_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Store categorized bank statement result in MongoDB by user_id.

    Args:
        user_id: User identifier (data is stored and retrieved by user_id)
        result: Categorize result dict (bank, account_number, transactions, matched_transactions, total_transactions, etc.)
        client_id: Optional client ID used for rules (stored for reference only)
        client_name: Optional client name (stored for display in GET response, same as backend get_client_bank_extract_data)

    Returns:
        Dict with success flag and run_id
    """
    try:
        db = get_database()
        collection = db[COLLECTION_CLIENT_BANK_STATEMENT_DATA]
        assigned_run_id = run_id or str(uuid4())

        doc = {
            "user_id": user_id,
            "run_id": assigned_run_id,
            "bank": result.get("bank", "Unknown"),
            "account_number": result.get("account_number") or "NA",
            "transactions": result.get("transactions") or [],
            "matched_transactions": result.get("matched_transactions", 0),
            "total_transactions": result.get("total_transactions", 0),
            "unmatched_transactions": result.get(
                "unmatched_transactions",
                (result.get("total_transactions", 0) - result.get("matched_transactions", 0)),
            ),
            "source_counts": result.get("source_counts") or {},
            "warnings": result.get("warnings") or [],
            "unknown_vendor_codes": result.get("unknown_vendor_codes") or [],
            "ama_rate": float(result.get("ama_rate") or 0.0),
            "rate_limit_retries_used": int(result.get("rate_limit_retries_used") or 0),
            "created_at": datetime.utcnow(),
        }
        if client_id is not None:
            doc["client_id"] = client_id
        name = client_name if client_name is not None else result.get("client_name")
        if name is not None:
            doc["client_name"] = name

        result_insert = await collection.insert_one(doc)

        if result_insert.inserted_id:
            log_message(
                "info",
                f"Stored client_bank_statement_data for user_id={user_id}, "
                f"transactions={len(doc['transactions'])}",
            )
            return {"success": True, "run_id": assigned_run_id, "record_id": str(result_insert.inserted_id)}
        log_message("warning", f"Failed to store client_bank_statement_data for user_id={user_id}")
        return {"success": False, "run_id": assigned_run_id}

    except Exception as e:
        log_message("error", f"Error storing client bank statement data: {e}")
        return {"success": False, "run_id": run_id}


async def get_client_bank_statements(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all bank statement data documents for a user (by user_id).

    Args:
        user_id: User identifier

    Returns:
        List of stored bank statement documents
    """
    try:
        db = get_database()
        collection = db[COLLECTION_CLIENT_BANK_STATEMENT_DATA]

        cursor = collection.find({"user_id": user_id}).sort("created_at", 1)
        results = await cursor.to_list(length=None)

        log_message("info", f"Found {len(results)} client_bank_statement_data record(s) for user_id={user_id}")
        return results

    except Exception as e:
        log_message("error", f"Error getting client bank statements: {e}")
        return []


async def get_statement_run_by_client_and_run_id(client_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Get one stored statement run by client and run identifier."""
    try:
        db = get_database()
        collection = db[COLLECTION_CLIENT_BANK_STATEMENT_DATA]
        doc = await collection.find_one(
            {"client_id": client_id, "run_id": run_id},
            sort=[("created_at", -1)],
        )
        if doc:
            return doc
        return None
    except Exception as e:
        log_message("error", f"Error getting statement run for client_id={client_id}, run_id={run_id}: {e}")
        return None


async def update_statement_run_transactions(
    client_id: str,
    run_id: str,
    transactions: List[Dict[str, Any]],
    matched_transactions: int,
    unmatched_transactions: int,
    source_counts: Optional[Dict[str, int]] = None,
    warnings: Optional[List[str]] = None,
    unknown_vendor_codes: Optional[List[Dict[str, Any]]] = None,
    ama_rate: Optional[float] = None,
    rate_limit_retries_used: Optional[int] = None,
) -> bool:
    """Update stored run transactions after reapply or one-time override."""
    try:
        db = get_database()
        collection = db[COLLECTION_CLIENT_BANK_STATEMENT_DATA]
        update_result = await collection.update_one(
            {"client_id": client_id, "run_id": run_id},
            {
                "$set": {
                    "transactions": transactions,
                    "matched_transactions": matched_transactions,
                    "unmatched_transactions": unmatched_transactions,
                    "total_transactions": len(transactions),
                    "source_counts": source_counts or {},
                    "warnings": warnings or [],
                    "unknown_vendor_codes": unknown_vendor_codes or [],
                    "ama_rate": float(ama_rate or 0.0),
                    "rate_limit_retries_used": int(rate_limit_retries_used or 0),
                    "updated_at": datetime.utcnow(),
                }
            },
        )
        return update_result.matched_count > 0
    except Exception as e:
        log_message("error", f"Error updating statement run for client_id={client_id}, run_id={run_id}: {e}")
        return False


async def upsert_statement_run_override(
    client_id: str,
    run_id: str,
    transaction_fingerprint: str,
    account_name: str,
    category_type: Optional[str] = None,
    reason: Optional[str] = None,
    user_id: Optional[str] = None,
    action_type: str = "one_time_override",
) -> bool:
    """Insert or update a one-time override for a statement run."""
    try:
        db = get_database()
        collection = db[COLLECTION_STATEMENT_RUN_OVERRIDES]
        now = datetime.utcnow()
        update_doc = {
            "client_id": client_id,
            "run_id": run_id,
            "transaction_fingerprint": transaction_fingerprint,
            "account_name": account_name,
            "category_type": category_type,
            "reason": reason,
            "action_type": action_type,
            "updated_at": now,
        }
        if user_id:
            update_doc["user_id"] = user_id

        result = await collection.update_one(
            {
                "client_id": client_id,
                "run_id": run_id,
                "transaction_fingerprint": transaction_fingerprint,
            },
            {"$set": update_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return (result.matched_count > 0) or (result.upserted_id is not None)
    except Exception as e:
        log_message(
            "error",
            f"Error upserting statement run override for client_id={client_id}, run_id={run_id}: {e}",
        )
        return False


async def get_statement_run_overrides(client_id: str, run_id: str) -> List[Dict[str, Any]]:
    """Get all one-time overrides for a statement run."""
    try:
        db = get_database()
        collection = db[COLLECTION_STATEMENT_RUN_OVERRIDES]
        cursor = collection.find({"client_id": client_id, "run_id": run_id}).sort("created_at", 1)
        return await cursor.to_list(length=None)
    except Exception as e:
        log_message("error", f"Error getting statement run overrides for client_id={client_id}, run_id={run_id}: {e}")
        return []


async def create_client_bank_statement_data_indexes():
    """Create indexes on client_bank_statement_data collection (by user_id)."""
    try:
        db = get_database()
        collection = db[COLLECTION_CLIENT_BANK_STATEMENT_DATA]

        await collection.create_index("user_id")
        log_message("info", "Created index on client_bank_statement_data.user_id")

        await collection.create_index([("user_id", 1), ("created_at", 1)])
        log_message("info", "Created compound index on client_bank_statement_data (user_id, created_at)")

        await collection.create_index([("client_id", 1), ("run_id", 1)])
        log_message("info", "Created compound index on client_bank_statement_data (client_id, run_id)")

    except Exception as e:
        log_message("warning", f"Failed to create client_bank_statement_data indexes (may already exist): {e}")


async def create_statement_run_override_indexes():
    """Create indexes for statement run one-time overrides."""
    try:
        db = get_database()
        collection = db[COLLECTION_STATEMENT_RUN_OVERRIDES]

        await collection.create_index([("client_id", 1), ("run_id", 1)])
        await collection.create_index(
            [("client_id", 1), ("run_id", 1), ("transaction_fingerprint", 1)],
            unique=True,
        )
        log_message("info", "Created indexes on statement_run_overrides")
    except Exception as e:
        log_message("warning", f"Failed to create statement_run_overrides indexes (may already exist): {e}")


async def create_client_rule_base_indexes():
    """
    Create indexes on the client_rule_base collection for better performance.
    """
    try:
        db = get_database()
        collection = db["client_rule_base"]
        
        # Create unique index on client_id (one rule base per client)
        await collection.create_index("client_id", unique=True)
        log_message("info", "Created unique index on client_rule_base.client_id")
        
        # Create index on client_name for faster lookups
        await collection.create_index("client_name")
        log_message("info", "Created index on client_rule_base.client_name")
        
    except Exception as e:
        log_message("warning", f"Failed to create client_rule_base indexes (may already exist): {e}")


# -------------------------------------------------------------------------
# Organizer categories (for Organizer PDF mapping)
# Fetches from separate "categories" and "subcategories" collections per schema.
# -------------------------------------------------------------------------

CATEGORIES_COLLECTION = "categories"
SUBCATEGORIES_COLLECTION = "subcategories"


async def get_organizer_categories() -> List[Dict[str, Any]]:
    """
    Fetch categories and subcategories from MongoDB per schema.

    Category schema: _id (ObjectId), name.
    Subcategory schema: _id (ObjectId), name, categoryId (string ref to category _id).

    Returns a flat list of { categoryId, categoryName, subcategoryId, subcategoryName, priority }
    by joining each subcategory with its parent category. priority is empty (not in schema).
    """
    try:
        db = get_database()
        cats_coll = db[CATEGORIES_COLLECTION]
        subcats_coll = db[SUBCATEGORIES_COLLECTION]

        cats_cursor = cats_coll.find({})
        categories = await cats_cursor.to_list(length=None)
        subcats_cursor = subcats_coll.find({})
        subcategories = await subcats_cursor.to_list(length=None)

        cat_by_id = {}
        for doc in categories:
            if not isinstance(doc, dict):
                continue
            cid = doc.get("_id")
            cid = str(cid) if cid is not None else ""
            cname = str(doc.get("name") or "")
            cat_by_id[cid] = {"categoryId": cid, "categoryName": cname}

        out = []
        seen_cat_ids = set()

        # Subcategory schema: _id, name, categoryId (ref)
        for doc in subcategories:
            if not isinstance(doc, dict):
                continue
            sub_id = str(doc.get("_id")) if doc.get("_id") is not None else ""
            sub_name = str(doc.get("name") or "")
            cat_id_ref = doc.get("categoryId")
            cat_id_ref = str(cat_id_ref) if cat_id_ref is not None else ""
            pri = str(doc.get("priority") or "")
            parent = cat_by_id.get(cat_id_ref) if cat_id_ref else None
            if parent:
                out.append({
                    "categoryId": parent["categoryId"],
                    "categoryName": parent["categoryName"],
                    "subcategoryId": sub_id,
                    "subcategoryName": sub_name,
                    "priority": pri,
                })
            else:
                out.append({
                    "categoryId": str(cat_id_ref or ""),
                    "categoryName": "",
                    "subcategoryId": sub_id,
                    "subcategoryName": sub_name,
                    "priority": pri,
                })
            if cat_id_ref:
                seen_cat_ids.add(cat_id_ref)

        # Include categories that have no subcategories (one row per category, empty subcategory)
        for cid, c in cat_by_id.items():
            if cid and cid not in seen_cat_ids:
                out.append({
                    "categoryId": c["categoryId"],
                    "categoryName": c["categoryName"],
                    "subcategoryId": "",
                    "subcategoryName": "",
                    "priority": "",
                })

        log_message("info", f"Fetched organizer list from categories + subcategories: {len(out)} rows")
        return out
    except Exception as e:
        log_message("error", f"Failed to fetch organizer categories: {e}")
        return []


async def create_organizer_categories_indexes():
    """Create indexes on categories and subcategories (schema: _id, name; subcategories have categoryId)."""
    try:
        db = get_database()
        cats = db[CATEGORIES_COLLECTION]
        subcats = db[SUBCATEGORIES_COLLECTION]
        await subcats.create_index("categoryId")
        log_message("info", "Created indexes on categories and subcategories")
    except Exception as e:
        log_message("warning", f"Failed to create categories/subcategories indexes (may already exist): {e}")
