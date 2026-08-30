from typing import Literal
from pydantic import BaseModel, Field

class Customer(BaseModel):
    customer_id: str = "C001"
    age: int = Field(35, ge=18, le=100)
    income: float = Field(55000, ge=0)
    loan_amount: float = Field(300000, ge=0)
    outstanding_amount: float = Field(180000, ge=0)
    emi_amount: float = Field(12000, ge=0)
    credit_score: float = Field(690, ge=300, le=900)
    dpd: int = Field(45, ge=0)
    missed_payment_count: int = Field(2, ge=0)
    previous_ptp_count: int = Field(3, ge=0)
    previous_ptp_kept_count: int = Field(2, ge=0)
    collection_attempts: int = Field(5, ge=0)
    successful_contacts: int = Field(3, ge=0)
    recent_payment_amount: float = Field(8000, ge=0)
    days_since_last_payment: int = Field(18, ge=0)
    preferred_channel: Literal["PHONE","SMS","WHATSAPP","EMAIL"] = "PHONE"

class RAGQuery(BaseModel):
    question: str

class MessageRequest(BaseModel):
    customer: Customer
    tone: Literal["professional","empathetic","concise"] = "professional"
