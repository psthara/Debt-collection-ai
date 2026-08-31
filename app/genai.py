import os
from google import genai

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def generate_message(customer, action, tone):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Write a short, respectful repayment reminder.

Customer name: {customer.customer_name}
Customer ID: {customer.customer_id}
Outstanding amount: INR {customer.outstanding_amount:,.2f}
EMI amount: INR {customer.emi_amount:,.2f}
Due date: {customer.due_date}
Days past due: {customer.dpd}
Recommended action: {action["action"]}
Preferred channel: {customer.preferred_channel}
Tone: {tone}

Requirements:
- Address the customer by name.
- Mention the outstanding amount.
- Mention the due date and days past due.
- Ask the customer to contact the authorised collection team.
- Do not threaten or shame the customer.
- Return only the final message.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty message")

    return response.text.strip()