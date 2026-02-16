"""
Client Model

MongoDB model for storing client information.
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


class Client(BaseModel):
    """
    Client model for MongoDB storage.
    
    Fields:
    - client_id: Unique identifier for the client (can be custom string or ObjectId)
    - name: Client name
    - email: Client email (optional)
    - pinecone_index_name: Pinecone index name associated with this client (optional)
    - created_at: Timestamp when record was created
    - updated_at: Timestamp when record was last updated
    """
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    client_id: str = Field(..., description="Unique client identifier", unique=True)
    name: Optional[str] = Field(None, description="Client name")
    email: Optional[str] = Field(None, description="Client email")
    pinecone_index_name: Optional[str] = Field(None, description="Pinecone index name for this client")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Record last update timestamp")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "client_id": "client_123",
                "name": "ABC Company LLC",
                "email": "contact@abccompany.com",
                "pinecone_index_name": "abc-company-llc-data"
            }
        }

