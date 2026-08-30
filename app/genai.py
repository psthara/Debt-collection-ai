import requests
from .config import OLLAMA_URL, OLLAMA_MODEL

def generate_message(c, action, tone):
    prompt=f"""Write a short respectful debt-collection message.
Never threaten, shame, misrepresent policy, invent legal consequences,
or promise a discount. Encourage contact with the authorized lender.
Tone: {tone}
DPD: {c.dpd}
Outstanding: {c.outstanding_amount}
Recommended action: {action["action"]}"""
    try:
        r=requests.post(f"{OLLAMA_URL}/api/generate",
            json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False},timeout=30)
        r.raise_for_status()
        x=r.json().get("response","").strip()
        if x: return x
    except Exception:
        pass
    return (f"Hello, we are contacting you regarding your outstanding repayment. "
            f"Your account currently shows {c.dpd} days past due. Please contact "
            f"the authorized collection team to discuss available repayment options.")
