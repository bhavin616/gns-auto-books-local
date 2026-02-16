"""
Pinecone Vector DB Integration Module

Provides utilities to create, load, and query Pinecone vector stores
for document embeddings.
"""

import os
from uuid import uuid4
from typing import List, Optional, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from backend.utils.embed import get_embeddings
from backend.utils.logger import log_message
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
INDEX_NAME: str = os.getenv("index_name")
EMBEDDINGS = get_embeddings()


def _normalize_client_to_index_name(client_name: str) -> str:
    """
    Generate a default Pinecone index name from a client name.

    This is used as a fallback when the client is not present in
    the explicit mapping in `get_index_name_from_client`.
    """
    safe_name = client_name.lower().strip().replace(" ", "-")
    if not safe_name.endswith("-data"):
        safe_name = f"{safe_name}-data"
    return safe_name


def get_index_name_from_client(client_name: Optional[str] = None, log_result: bool = False) -> str:
    """
    Resolve Pinecone index name from a client name using a deterministic convention.

    This removes the need to hard-code a mapping table. Any new client name
    will automatically map to a valid index name.

    Convention:
        - Lowercase the client name
        - Strip leading/trailing whitespace
        - Replace spaces with '-'
        - Ensure the name ends with '-data'

    Examples:
        "Unity Trucking"   -> "unity-trucking-data"
        "rahman engineering" -> "rahman-engineering-data"

    Args:
        client_name: Client name (case-insensitive). If None or empty, returns default INDEX_NAME.
        log_result: Whether to log the mapping (default: False to reduce log noise)

    Returns:
        Pinecone index name based on client_name (or default INDEX_NAME when not provided).
    """
    if not client_name:
        return INDEX_NAME

    index_name = _normalize_client_to_index_name(client_name)

    if log_result:
        log_message("info", f"Mapped client_name '{client_name}' to index '{index_name}' (convention-based)")

    return index_name


def create_index_for_client(client_name: str, index_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create (or ensure) a Pinecone index for a given client name.

    The index is created with static parameters:
        dimension = 768
        metric    = "cosine"
        spec      = ServerlessSpec(cloud="aws", region="us-east-1")

    The caller only needs to provide `client_name`; all other fields
    remain constant.

    Returns:
        dict with:
            - status: "success" | "failed"
            - index_name: resolved Pinecone index name
            - created: bool (True if a new index was created, False if it already existed)
            - message: human-readable description
    """
    try:
        if not client_name or not client_name.strip():
            return {
                "status": "failed",
                "message": "client_name is required to create a Pinecone index.",
            }

        # Resolve index name from client_name unless an explicit index_name is provided
        if index_name and index_name.strip():
            index_name = index_name.strip()
            log_message(
                "info",
                f"Using provided index_name '{index_name}' for client='{client_name}'.",
            )
        else:
            index_name = get_index_name_from_client(client_name, log_result=True)

        pc = Pinecone(api_key=PINECONE_API_KEY)

        if pc.has_index(index_name):
            log_message("info", f"Pinecone index '{index_name}' already exists (client='{client_name}').")
            return {
                "status": "success",
                "index_name": index_name,
                "created": False,
                "message": f"Index '{index_name}' already exists.",
            }

        # Create the index with static configuration
        log_message(
            "info",
            f"Creating Pinecone index '{index_name}' for client='{client_name}' "
            f"(dimension=768, metric='cosine', cloud='aws', region='us-east-1').",
        )
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        log_message("info", f"Pinecone index '{index_name}' created successfully for client='{client_name}'.")

        return {
            "status": "success",
            "index_name": index_name,
            "created": True,
            "message": f"Index '{index_name}' created successfully.",
        }

    except Exception as e:
        log_message("error", f"Failed to create Pinecone index for client '{client_name}': {e}")
        return {
            "status": "failed",
            "message": str(e),
        }


def create_vector_db(documents: List[Document], index_name: Optional[str] = None, client_name: Optional[str] = None, auto_create: bool = False) -> str:
    """
    Initializes a Pinecone vector store with the specified index name and embeddings.
    
    Checks if the Pinecone index exists. If not and auto_create=True, creates a new index.
    If auto_create=False and index doesn't exist, raises an error.

    Args:
        documents (List[Document]): List of LangChain Document objects to store.
        index_name (str, optional): Name of the Pinecone index to use or create.
        client_name (str, optional): Client name to determine index if index_name not provided.
        auto_create (bool): If True, creates index if it doesn't exist. If False, raises error if index doesn't exist.

    Returns:
        str: The name of the Pinecone index used.

    Raises:
        RuntimeError: If there is an error during Pinecone operations or if index doesn't exist and auto_create=False.
    """
    try:
        # Determine index name from client_name if not provided
        if index_name is None:
            index_name = get_index_name_from_client(client_name)
        
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists
        if not pc.has_index(index_name):
            if auto_create:
                log_message("info", f"Pinecone index '{index_name}' does not exist. Creating...")
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                log_message("info", f"Pinecone index '{index_name}' created successfully.")
            else:
                error_msg = (
                    f"Pinecone index '{index_name}' does not exist. "
                    f"Please create it first using /api/create_pinecone_index endpoint."
                )
                log_message("error", error_msg)
                raise RuntimeError(error_msg)
        else:
            log_message("info", f"Pinecone index '{index_name}' already exists.")

        # Initialize vector store
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=EMBEDDINGS)

        # Add documents with unique IDs
        ids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=ids)
        log_message("info", f"Added {len(documents)} documents to Pinecone index '{index_name}'.")

        return index_name

    except Exception as e:
        log_message("error", f"Failed to create or initialize Pinecone vector store: {e}")
        raise RuntimeError(f"Failed to create or initialize Pinecone vector store: {e}") from e


def retrieve_documents(vector_store: PineconeVectorStore, query: str, client_name: Optional[str] = None) -> List[Document]:
    """
    Retrieves documents from a Pinecone vector store using similarity search.

    Args:
        vector_store (PineconeVectorStore): Initialized Pinecone vector store.
        query (str): Text query for similarity search.
        client_name (str, optional): Optional filter for client_name in metadata.

    Returns:
        List[Document]: List of matched documents.
    """
    try:
        filter_metadata = {'client_name': client_name} if client_name else None
        results = vector_store.similarity_search(query, k=1, filter=filter_metadata)
        return results
    except Exception as e:
        log_message("error", f"Failed to retrieve documents from Pinecone vector store: {e}")
        raise RuntimeError(f"Failed to retrieve documents from Pinecone vector store: {e}") from e


def query_pinecone_with_embedding(
    embedding_vector: List[float],
    client_name: Optional[str] = None,
    top_k: int = 20,
    index_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query Pinecone using embedding vector and return top similar records.
    
    This function is used for RAG (Retrieval-Augmented Generation) to find
    similar historical transactions from Pinecone.
    
    Args:
        embedding_vector: Embedding vector (list of floats) from OpenAI
        client_name: Client name to filter results and determine index (optional)
        top_k: Number of top results to return (default 20)
        index_name: Pinecone index name (optional, will be determined from client_name if not provided)
        
    Returns:
        List of dictionaries with metadata:
        [
            {
                "memo": "historical memo text",
                "name": "vendor name",
                "split": "COA category",
                "score": 0.95  # similarity score
            },
            ...
        ]
    """
    try:
        if not embedding_vector:
            log_message("warning", "Empty embedding vector provided")
            return []
        
        if index_name is None:
            index_name = get_index_name_from_client(client_name, log_result=False)
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if not pc.has_index(index_name):
            log_message("warning", f"Index '{index_name}' does not exist")
            return []
        
        index = pc.Index(index_name)
        
        # First, check if index has data (only if it might be empty)
        total_vectors = -1  # -1 means we couldn't check
        try:
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                log_message("warning", f"Pinecone index '{index_name}' has NO DATA (0 vectors). Please upload GL data first.")
                return []
        except Exception as stats_error:
            log_message("debug", f"Could not check index stats: {stats_error}")
            total_vectors = -1  # Unknown
        
        # Build filter for client_name if provided
        # Only filter if using default index (index might contain multiple clients)
        # If using client-specific index, don't filter (index is already client-specific)
        # 
        # IMPORTANT: Client-specific indexes have names like "clientname-data" (e.g., "barbar-llc-data")
        # Default/shared indexes typically don't follow this pattern
        # We should NEVER filter by client_name for client-specific indexes!
        filter_dict = None
        
        # Check if this is a client-specific index (ends with '-data' and has hyphens in the name)
        # Client-specific indexes: barbar-llc-data, unity-trucking-data, etc.
        # Default indexes: might be "cpa-index", "default", etc. (configure INDEX_NAME correctly in .env)
        is_client_specific_index = index_name and index_name.endswith('-data') and '-' in index_name
        
        # Only apply client_name filter if:
        # 1. client_name is provided
        # 2. We're NOT using a client-specific index (client-specific indexes already contain only that client's data)
        if client_name and not is_client_specific_index:
            # Using default/shared index - filter by client_name
            filter_dict = {"client_name": {"$eq": client_name}}
            log_message("debug", f"Using client_name filter for shared index '{index_name}': {client_name}")
        
        # Query Pinecone with embedding vector
        query_response = index.query(
            vector=embedding_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Log if no matches found
        matches_count = len(query_response.get('matches', []))
        if matches_count == 0:
            log_message("debug", f"Pinecone returned 0 matches for index '{index_name}'")
        
        # Extract results with metadata
        results = []
        for match in query_response.get('matches', []):
            metadata = match.get('metadata', {})
            # Check both "split" and "category" for backward compatibility
            category = metadata.get("split", "") or metadata.get("category", "")
            memo = metadata.get("memo", "")
            score = match.get("score", 0.0)
            
            results.append({
                "memo": memo,
                "name": metadata.get("name", ""),
                "split": category,  # Use "split" as the key for consistency
                "category": category,  # Also include "category" for compatibility
                "credit": metadata.get("credit", ""),  # Include credit for matching
                "debit": metadata.get("debit", ""),  # Include debit for matching
                "date": metadata.get("date", ""),  # Include date for sorting newest first
                "score": score
            })
        
        if len(results) == 0 and total_vectors > 0:
            log_message("warning", f"No Pinecone results found for index='{index_name}'. Index has {total_vectors} vectors but query returned 0 matches. This suggests the embeddings might not match or there's a data issue.")
        
        return results
        
    except Exception as e:
        log_message("error", f"Failed to query Pinecone with embedding: {e}")
        return []


def load_vectordb(index_name: str = INDEX_NAME, embeddings=EMBEDDINGS) -> PineconeVectorStore:
    """
    Loads an existing Pinecone index as a LangChain-compatible vector store.

    Args:
        index_name (str): Name of the Pinecone index to load.
        embeddings: Embedding function compatible with LangChain.

    Returns:
        PineconeVectorStore: Initialized vector store for querying.

    Raises:
        RuntimeError: If the vector store cannot be loaded.
    """
    try:
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        log_message("info", f"Pinecone vector store loaded from index '{index_name}'.")
        return vector_store
    except Exception as e:
        log_message("error", f"Failed to load Pinecone vector store from index '{index_name}': {e}")
        raise RuntimeError(f"Failed to load Pinecone vector store from index '{index_name}': {e}") from e


def delete_all_from_index(index_name: Optional[str] = INDEX_NAME, filter_metadata: Optional[dict] = None) -> dict:
    """
    Deletes all vectors from a Pinecone index, optionally filtered by metadata.

    Args:
        index_name (str, optional): Name of the Pinecone index to delete from.
        filter_metadata (dict, optional): Metadata filter to delete specific records.
                                        If None, deletes all records.

    Returns:
        dict: Status with count of deleted records.

    Raises:
        RuntimeError: If deletion fails.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if not pc.has_index(index_name):
            log_message("warning", f"Index '{index_name}' does not exist.")
            return {"status": "failed", "error": f"Index '{index_name}' does not exist."}
        
        index = pc.Index(index_name)
        
        # Delete all vectors (Pinecone delete_all deletes everything)
        if filter_metadata is None:
            # Delete all vectors in the index
            index.delete(delete_all=True)
            log_message("info", f"Deleted all vectors from index '{index_name}'.")
            return {"status": "success", "message": f"All vectors deleted from index '{index_name}'."}
        else:
            # Delete vectors matching the filter
            index.delete(filter=filter_metadata)
            log_message("info", f"Deleted vectors matching filter {filter_metadata} from index '{index_name}'.")
            return {"status": "success", "message": f"Deleted vectors matching filter from index '{index_name}'."}
            
    except Exception as e:
        log_message("error", f"Failed to delete from Pinecone index '{index_name}': {e}")
        raise RuntimeError(f"Failed to delete from Pinecone index '{index_name}': {e}") from e


def check_index_has_data(index_name: Optional[str] = INDEX_NAME) -> Dict[str, Any]:
    """
    Check if a Pinecone index exists and has any data (vectors).

    Args:
        index_name (str, optional): Name of the Pinecone index to check.

    Returns:
        dict with:
            - status: "success" | "failed"
            - has_data: bool (True if index exists and has vectors, False otherwise)
            - total_vectors: int (number of vectors in the index, 0 if none or index doesn't exist)
            - index_exists: bool (True if index exists, False otherwise)
            - message: Human-readable description
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if not pc.has_index(index_name):
            log_message("info", f"Index '{index_name}' does not exist.")
            return {
                "status": "success",
                "has_data": False,
                "total_vectors": 0,
                "index_exists": False,
                "message": f"Index '{index_name}' does not exist."
            }
        
        index = pc.Index(index_name)
        
        # Get stats to see how many records exist
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        has_data = total_vectors > 0
        
        log_message("info", f"Index '{index_name}' has {total_vectors} vectors (has_data={has_data}).")
        
        return {
            "status": "success",
            "has_data": has_data,
            "total_vectors": total_vectors,
            "index_exists": True,
            "message": f"Index '{index_name}' has {total_vectors} vectors." if has_data else f"Index '{index_name}' exists but has no data."
        }
            
    except Exception as e:
        log_message("error", f"Failed to check Pinecone index '{index_name}': {e}")
        return {
            "status": "failed",
            "has_data": False,
            "total_vectors": 0,
            "index_exists": False,
            "error": str(e)
        }


def delete_all_records(index_name: Optional[str] = INDEX_NAME) -> dict:
    """
    Deletes ALL records from Pinecone index.

    Args:
        index_name (str, optional): Name of the Pinecone index.

    Returns:
        dict: Status with count of deleted records.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        if not pc.has_index(index_name):
            log_message("warning", f"Index '{index_name}' does not exist.")
            return {"status": "failed", "error": f"Index '{index_name}' does not exist."}
        
        index = pc.Index(index_name)
        
        # Get stats to see how many records exist
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        log_message("info", f"Index '{index_name}' has {total_vectors} vectors.")
        
        # Delete ALL vectors from the index
        index.delete(delete_all=True)
        log_message("info", f"Deleted all {total_vectors} vectors from index '{index_name}'.")
        
        return {
            "status": "success", 
            "message": f"Deleted all {total_vectors} vectors from index '{index_name}'. You can now re-upload clean data.",
            "deleted_count": total_vectors
        }
            
    except Exception as e:
        log_message("error", f"Failed to delete all records from Pinecone index '{index_name}': {e}")
        return {"status": "failed", "error": str(e)}


def list_all_indexes() -> Dict[str, Any]:
    """
    List all Pinecone indexes available for the current API key.

    Returns:
        dict with:
            - status: "success" | "failed"
            - indexes: list of index names (on success)
            - message / error: human-readable description
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        description = pc.list_indexes()

        # `description` may be an object; extract index names in a generic way
        index_names: List[str] = []
        for item in description:
            # Newer SDKs typically return objects with `.name`
            name = getattr(item, "name", None)
            if not name and isinstance(item, dict):
                name = item.get("name")
            if name:
                index_names.append(name)

        log_message("info", f"Retrieved {len(index_names)} Pinecone indexes.")

        return {
            "status": "success",
            "indexes": index_names,
            "message": f"Found {len(index_names)} Pinecone indexes.",
        }

    except Exception as e:
        log_message("error", f"Failed to list Pinecone indexes: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }