import re

# --- The Blocklist: Keywords for filtering promotional content ---
# This list is designed to catch common marketing, social media, and newsletter emails.
PROMOTIONAL_SENDERS = [
    "no-reply@", "newsletter@", "deals@", "noreply@", "support@", "updates@",
    "@linkedin.com", "@facebookmail.com", "@quora.com", "@twitter.com",
    "info@", "contact@", "team@","hello@", "service@", "donotreply@"
]

PROMOTIONAL_SUBJECTS = [
    "sale", "discount", "% off", "offer", "webinar", "exclusive", "free",
    "new post", "commented", "mentioned you", "connection request"
]


def is_important_email(raw_email: dict) -> bool:
    """
    Acts as a gatekeeper to filter out obviously unimportant emails (promotions, etc.)
    before they are passed to the AI for full processing.
    Returns True if the email should be processed, False if it should be discarded.
    """
    try:
        payload = raw_email.get("payload", {})
        headers = payload.get("headers", [])
        
        sender = ""
        subject = ""
        has_unsubscribe_link = False

        for header in headers:
            name = header.get("name").lower()
            if name == "from":
                sender = header.get("value", "").lower()
            elif name == "subject":
                subject = header.get("value", "").lower()
            elif name == "list-unsubscribe":
                # This is a very strong signal that the email is promotional.
                has_unsubscribe_link = True

        # --- Rule 1: Check for Unsubscribe Header ---
        if has_unsubscribe_link:
            print(f"Gatekeeper REJECTED (Unsubscribe Link): {subject}")
            return False

        # --- Rule 2: Check Sender against Blocklist ---
        for promo_sender in PROMOTIONAL_SENDERS:
            if promo_sender in sender:
                print(f"Gatekeeper REJECTED (Sender Match: {promo_sender}): {subject}")
                return False

        # --- Rule 3: Check Subject against Blocklist ---
        for promo_subject in PROMOTIONAL_SUBJECTS:
            # Use regex for whole word matching to avoid false positives (e.g., "free" in "freedom")
            if re.search(r'\b' + promo_subject + r'\b', subject):
                print(f"Gatekeeper REJECTED (Subject Match: {promo_subject}): {subject}")
                return False

        # --- If it passes all checks, it's allowed into the processing queue ---
        print(f"Gatekeeper ACCEPTED: {subject}")
        return True

    except Exception as e:
        print(f"An error occurred in the gatekeeper for email ID {raw_email.get('id', 'N/A')}: {e}")
        # Default to processing the email if the filter fails for some reason
        return True

