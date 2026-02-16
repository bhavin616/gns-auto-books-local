"""
Client Pinecone Index Model

MongoDB model for storing client Pinecone index mappings.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId for Pydantic compatibility."""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class ClientPineconeIndex(BaseModel):
    """
    Client Pinecone Index model for MongoDB storage.
    
    Fields:
    - client_id: Unique identifier for the client (foreign key reference to clients collection)
    - name: Client name
    - pinecone_index_name: Pinecone index name associated with this client
    - created_at: Timestamp when record was created
    - updated_at: Timestamp when record was last updated
    """
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    client_id: str = Field(..., description="Client identifier (foreign key reference to clients collection)")
    name: str = Field(..., description="Client name")
    pinecone_index_name: str = Field(..., description="Pinecone index name for this client")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Record last update timestamp")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "client_id": "507f1f77bcf86cd799439011",
                "name": "ABC Company LLC",
                "pinecone_index_name": "abc-company-llc-data"
            }
        }

