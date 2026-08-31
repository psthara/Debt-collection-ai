import os
from google import genai


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)


def generate_message(customer, action, tone):

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    client = genai.Client()

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
- Mention the correct outstanding amount.
- Mention the due date and days past due.
- Ask the customer to contact the authorised collection team.
- Do not threaten or shame the customer.
- Do not invent legal action or discounts.
- Return only the final message.
"""

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt
    )

    message = interaction.output_text.strip()

    if not message:
        raise RuntimeError(
            "Gemini returned an empty message"
        )

    return message