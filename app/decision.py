def next_best_action(c, s):
    if c.dpd == 0:
        action = "SEND_STANDARD_PAYMENT_REMINDER"
    elif s["ptp_probability"] >= .65:
        action = "REQUEST_AND_CONFIRM_PROMISE_TO_PAY"
    elif s["payment_probability_30d"] >= .55:
        action = "OFFER_REPAYMENT_PLAN_DISCUSSION"
    elif c.dpd >= 90:
        action = "ESCALATE_FOR_HUMAN_REVIEW"
    else:
        action = "COLLECTION_CALL_AND_HARDSHIP_ASSESSMENT"
    return {
        "action": action,
        "recommended_channel": c.preferred_channel,
        "reason": f"DPD={c.dpd}; payment={s['payment_probability_30d']}; PTP={s['ptp_probability']}."
    }
