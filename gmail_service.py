import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import traceback
from googleapiclient.errors import HttpError
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def mark_emails_as_read(service, message_ids: list, account_email: str):
    """Marks a list of emails as read in a single batch request."""
    if not service or not message_ids:
        return
    try:
        service.users().messages().batchModify(
            userId='me',
            body={'ids': message_ids, 'removeLabelIds': ['UNREAD']}
        ).execute()
        print(f"Successfully marked {len(message_ids)} unimportant emails as read for {account_email}.")
    except HttpError as error:
        # This will now print a very detailed error message if it fails
        print("\n" + "!"*60)
        print(f"❗️ CRITICAL ERROR: Could not mark emails as read for account: {account_email}")
        print(f"❗️ Google API responded with: {error.resp.status} {error.resp.reason}")
        print(f"❗️ Full Error Content: {error.content.decode()}")
        print("!"*60 + "\n")

def get_gmail_service(token_path: str):
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
            
    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except HttpError as error:
        print(f"An error occurred while building the service: {error}")
        return None

def fetch_raw_emails(service, max_results=25):
    try:
        query = "is:unread newer_than:30d"
        results = service.users().messages().list(
            userId="me", labelIds=["INBOX"], q=query, maxResults=max_results
        ).execute()
        
        messages = results.get("messages", [])
        if not messages:
            return []

        print(f"Found {len(messages)} new unread emails. Fetching full content...")
        raw_email_list = []
        for message in messages:
            raw_email = service.users().messages().get(userId="me", id=message["id"]).execute()
            raw_email_list.append(raw_email)
            
            # FIX 1: REMOVED the premature "mark as read" logic from here.
            
        return raw_email_list
    except HttpError as error:
        print(f"An error occurred while fetching raw emails: {error}")
        return []

def get_attachment_data(service, message_id, attachment_id):
    try:
        attachment = service.users().messages().attachments().get(
            userId='me', messageId=message_id, id=attachment_id
        ).execute()
        return attachment['data']
    except HttpError as error:
        print(f"An error occurred while fetching attachment data: {error}")
        return None
