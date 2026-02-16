"""
Rule Base Models

Data models for client rule base system.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Rule(BaseModel):
    """Individual rule structure."""
    rule_id: str = Field(..., description="Unique identifier for the rule")
    trigger_payee_contains: List[str] = Field(..., description="List of payee names or keywords that trigger this rule")
    account_id: str = Field(..., description="Account ID from COA")
    account_name: str = Field(..., description="Account name from COA")
    category_type: str = Field(..., description="Category type (e.g., Cost of Goods Sold, Operating Expense)")
    logic: str = Field(..., description="Description of the rule logic")
    vendor_code: Optional[str] = Field(default=None, description="Optional 4 digit vendor code for ACH vendor payment rules")
    ach_vendor_only: Optional[bool] = Field(default=False, description="If true, this rule applies only to ACH vendor payment style transactions")


class ClientRuleBase(BaseModel):
    """Client rule base document structure."""
    client_id: str = Field(..., description="Unique client identifier")
    client_name: str = Field(..., description="Client name")
    nature_of_business: str = Field(..., description="Nature of business")
    rules: List[Rule] = Field(..., description="List of rules for this client")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class RuleGenerationRequest(BaseModel):
    """Request model for rule generation."""
    client_name: str = Field(..., description="Client name")
    nature_of_business: str = Field(..., description="Nature of business")
    gl_data: str = Field(..., description="General Ledger data (text or JSON)")
    coa_data: str = Field(..., description="Chart of Accounts data (text or JSON)")
