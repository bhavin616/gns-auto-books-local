"""
Analytics Event Model

MongoDB model for storing analytics events and user/client activity tracking.
"""

from datetime import datetime
from typing import Optional, Dict, Any
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


class AnalyticsEvent(BaseModel):
    """
    Analytics Event model for MongoDB storage.
    
    Fields:
    - userId: User ID reference (optional, for staff users)
    - clientId: Client ID reference (optional, for clients)
    - eventType: Type of event (tool_usage, feature_access, etc.)
    - category: Event category (ESRP, ALE, Chat, etc.)
    - action: Specific action taken
    - metadata: Additional metadata as flexible dictionary
    - sessionId: Session identifier (optional)
    - ipAddress: User IP address (optional)
    - userAgent: User agent string (optional)
    - deviceInfo: Device information dictionary (optional)
    - duration: Event duration in milliseconds (optional)
    - status: Event status (success, error, pending)
    - errorMessage: Error message if status is error (optional)
    - created_at: Timestamp when record was created
    - updated_at: Timestamp when record was last updated
    """
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    userId: Optional[PyObjectId] = Field(None, description="User ID reference")
    clientId: Optional[PyObjectId] = Field(None, description="Client ID reference")
    eventType: str = Field(..., description="Event type")
    category: str = Field(..., description="Event category")
    action: str = Field(..., description="Event action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    sessionId: Optional[str] = Field(None, description="Session identifier")
    ipAddress: Optional[str] = Field(None, description="IP address")
    userAgent: Optional[str] = Field(None, description="User agent string")
    deviceInfo: Dict[str, Any] = Field(default_factory=dict, description="Device information")
    duration: Optional[int] = Field(None, description="Duration in milliseconds")
    status: Optional[str] = Field(None, description="Event status (success, error, pending)")
    errorMessage: Optional[str] = Field(None, description="Error message if applicable")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Record last update timestamp")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "userId": "507f1f77bcf86cd799439011",
                "eventType": "tool_usage",
                "category": "ESRP",
                "action": "file_processed",
                "metadata": {
                    "fileName": "document.pdf",
                    "fileSize": 1024000
                },
                "sessionId": "sess_abc123",
                "status": "success"
            }
        }


# Valid event types enum
VALID_EVENT_TYPES = [
    'tool_usage',
    'feature_access',
    'document_action',
    'chat_action',
    'login',
    'download',
    'upload',
    'api_call',
    'error',
    'custom'
]

# Valid status enum
VALID_STATUSES = ['success', 'error', 'pending']

