from typing import Literal
from pydantic import BaseModel, Field


class Customer(BaseModel):
    customer_id: str
    customer_name: str
    due_date: str

    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0)
    loan_amount: float = Field(ge=0)
    outstanding_amount: float = Field(ge=0)
    emi_amount: float = Field(ge=0)
    credit_score: float = Field(ge=300, le=900)
    dpd: int = Field(ge=0)

    missed_payment_count: int = Field(ge=0)
    previous_ptp_count: int = Field(ge=0)
    previous_ptp_kept_count: int = Field(ge=0)
    collection_attempts: int = Field(ge=0)
    successful_contacts: int = Field(ge=0)
    recent_payment_amount: float = Field(ge=0)
    days_since_last_payment: int = Field(ge=0)

    preferred_channel: Literal[
        "PHONE", "SMS", "WHATSAPP", "EMAIL"
    ]

class RAGQuery(BaseModel):
    question: str


class MessageRequest(BaseModel):
    customer: Customer

    tone: Literal[
        "professional",
        "empathetic",
        "concise"
    ] = "professional"