import os

from google import genai
from google.genai import types


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.7-flash"
)


def generate_message(customer, action, tone):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            )
        )
    )

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

    try:
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

    except Exception as exc:

        error_text = str(exc)

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
        ):
            return (
                f"Dear {customer.customer_name}, "
                f"this is a friendly reminder regarding your "
                f"outstanding amount of "
                f"INR {customer.outstanding_amount:,.2f}. "
                f"Your account is currently "
                f"{customer.dpd} days past due. "
                f"Please contact our authorised collection team "
                f"to discuss your repayment options. Thank you."
            )

        raise RuntimeError(
            f"Gemini message generation failed: {exc}"
        ) from exc

    finally:
        client.close()