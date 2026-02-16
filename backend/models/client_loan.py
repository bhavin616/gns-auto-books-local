"""
Client Loan Model

MongoDB model for storing client loan details extracted from PDF documents.
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


class ClientLoan(BaseModel):
    """
    Client Loan model for MongoDB storage.
    
    Fields:
    - client_id: Unique identifier for the client (foreign key reference to clients collection)
    - legal_name: Legal name of the borrower/entity
    - funding_provided: Total funding/loan amount provided
    - annual_percentage_rate: APR percentage
    - finance_charge: Total finance charge amount
    - total_payment_amount: Total payment amount over loan term
    - payment: Regular payment amount
    - account_no: Account number
    - created_at: Timestamp when record was created
    - updated_at: Timestamp when record was last updated
    - pdf_filename: Original PDF filename
    """
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    client_id: str = Field(..., description="Client identifier (foreign key reference to clients collection)")
    legal_name: Optional[str] = Field(None, description="Legal name of the borrower/entity")
    funding_provided: Optional[str] = Field(None, description="Total funding/loan amount provided")
    annual_percentage_rate: Optional[str] = Field(None, description="APR percentage")
    finance_charge: Optional[str] = Field(None, description="Total finance charge amount")
    total_payment_amount: Optional[str] = Field(None, description="Total payment amount over loan term")
    payment: Optional[str] = Field(None, description="Regular payment amount")
    account_no: Optional[str] = Field(None, description="Account number")
    start_date: Optional[str] = Field(None, description="Loan start date")
    pdf_filename: Optional[str] = Field(None, description="Original PDF filename")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Record last update timestamp")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "client_id": "client_123",
                "legal_name": "ABC Company LLC",
                "funding_provided": "50000",
                "annual_percentage_rate": "5.5%",
                "finance_charge": "15000",
                "total_payment_amount": "65000",
                "payment": "500",
                "account_no": "123456789",
                "start_date": "2024-01-15",
                "pdf_filename": "loan_agreement.pdf"
            }
        }


class ClientLoanCreate(BaseModel):
    """Schema for creating a new client loan record."""
    client_id: str
    legal_name: Optional[str] = None
    funding_provided: Optional[str] = None
    annual_percentage_rate: Optional[str] = None
    finance_charge: Optional[str] = None
    total_payment_amount: Optional[str] = None
    payment: Optional[str] = None
    account_no: Optional[str] = None
    start_date: Optional[str] = None
    pdf_filename: Optional[str] = None


class ClientLoanResponse(BaseModel):
    """Schema for API response."""
    id: str
    client_id: str
    legal_name: Optional[str] = None
    funding_provided: Optional[str] = None
    annual_percentage_rate: Optional[str] = None
    finance_charge: Optional[str] = None
    total_payment_amount: Optional[str] = None
    payment: Optional[str] = None
    account_no: Optional[str] = None
    start_date: Optional[str] = None
    pdf_filename: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

