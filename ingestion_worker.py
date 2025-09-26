import traceback
from outlines import Generator, from_ollama
from pydantic import BaseModel, Field
from typing import Optional
import os
from datetime import datetime
import base64
import fitz
import ollama
from bs4 import BeautifulSoup

from database import save_processed_email
from gmail_service import get_gmail_service, get_attachment_data
from outlines.models.ollama import Ollama

# Patch Ollama.generate to handle dict responses from official ollama client
_original_generate = Ollama.generate

def _patched_generate(self, *args, **kwargs):
    response = self.client.chat(model=self.model_name, messages=[{"role": "user", "content": args[0]}])
    # Official ollama returns dict
    if isinstance(response, dict):
        return response["message"]["content"]
    # fallback to original
    return _original_generate(self, *args, **kwargs)

Ollama.generate = _patched_generate

class EmailAnalysis(BaseModel):
    summary: str = Field(None, description="A concise summary of the email's core message.")
    priority: str = Field(..., description="The priority of the email.", pattern=r"^(High|Medium|Low)$")
    action_needed: bool = Field(..., description="Whether an explicit action is required from the user.")
    deadline: Optional[str] = Field(None, description="Any deadline mentioned (YYYY-MM-DD HH:MM:SS), or null if none.")

ALLOWED_MIMETYPES = ["application/pdf"]
MAX_ATTACHMENT_SIZE_MB = 5

def _process_attachments(service, message_id, parts):
    # This function remains unchanged
    all_attachment_summaries = []
    if not service or not parts: return None
    for part in parts:
        if part.get("filename") and part.get("body", {}).get("attachmentId"):
            filename, mime_type, size = part.get("filename"), part.get("mimeType"), part.get("body", {}).get("size", 0)
            if mime_type not in ALLOWED_MIMETYPES or size > MAX_ATTACHMENT_SIZE_MB * 1024 * 1024: continue
            try:
                attachment_id = part["body"]["attachmentId"]
                attachment_data_b64 = get_attachment_data(service, message_id, attachment_id)
                if not attachment_data_b64: continue
                attachment_bytes = base64.urlsafe_b64decode(attachment_data_b64)
                attachment_text = ""
                if mime_type == "application/pdf":
                    with fitz.open(stream=attachment_bytes, filetype="pdf") as doc:
                        attachment_text = "".join(page.get_text() for page in doc)
                if not attachment_text.strip(): continue
                prompt = f"Summarize the following text from an attachment named '{filename}':\n---{attachment_text[:4000]}---"
                response = ollama.chat(model=os.getenv("OLLAMA_MODEL", "phi3"), messages=[{'role': 'user', 'content': prompt}])
                summary = response['message']['content'].strip()
                all_attachment_summaries.append(f"Attachment '{filename}': {summary}")
            except Exception as e:
                print(f"Error processing attachment '{filename}': {e}")
    return "\n".join(all_attachment_summaries) if all_attachment_summaries else None


def get_email_body(payload: dict) -> str:
    # This function remains unchanged
    body_parts = []
    if "parts" in payload:
        for part in payload["parts"]:
            body_parts.append(get_email_body(part))
    elif payload.get("mimeType") == "message/rfc822" and "payload" in payload:
        body_parts.append(get_email_body(payload["payload"]))
    elif "data" in payload.get("body", {}):
        data = payload["body"]["data"]
        mime_type = payload.get("mimeType", "")
        try:
            decoded_data = base64.urlsafe_b64decode(data).decode("utf-8", "ignore")
            if "text/html" in mime_type:
                soup = BeautifulSoup(decoded_data, "html.parser")
                body_parts.append(soup.get_text(separator="\n", strip=True))
            elif "text/plain" in mime_type:
                body_parts.append(decoded_data)
        except Exception:
            pass
    return "\n".join(filter(None, body_parts)).strip()


def process_single_email(raw_email: dict, account_id: int, token_path: str):
    email_id = raw_email['id']
    print(f"--- [Thread] Starting processing for Email ID: {email_id} (Account ID: {account_id}) ---")
    
    try:
        # --- ADDED: Each thread creates its own private, isolated service instances ---
        # 1. Private Gmail Service for fetching attachments.
        service = get_gmail_service(token_path)
        if not service:
            print(f"❗️ [Thread] Could not create Gmail service for Email ID {email_id}. Aborting task.")
            return False # Signal failure

        # 2. Private Ollama client (your code was already doing this correctly).
        client = ollama.Client(timeout=60)
        
        payload = raw_email.get("payload", {})
        headers = payload.get("headers", [])
        email_metadata = {"id": email_id, "threadId": raw_email["threadId"]}
        for header in headers:
            name = header.get("name").lower()
            if name == "from": email_metadata["sender"] = header.get("value")
            elif name == "subject": email_metadata["subject"] = header.get("value")
            elif name == "date":
                date_str = header.get("value")
                try:
                    dt_format = '%a, %d %b %Y %H:%M:%S %z (%Z)' if '(' in date_str else '%a, %d %b %Y %H:%M:%S %z'
                    email_metadata["received_at"] = datetime.strptime(date_str, dt_format)
                except ValueError:
                    email_metadata["received_at"] = datetime.now()
         
        email_metadata["full_text"] = get_email_body(payload)
        email_metadata["attachment_summary"] = _process_attachments(service, email_id, payload.get("parts"))

        prompt = f"""You are an expert email analysis agent. Analyze the following email and extract the required information into a valid JSON format.
        
                Email Content:
                ---
                Subject: {email_metadata.get('subject')}
                From: {email_metadata.get('sender')}
                Body:
                {email_metadata.get('full_text')[:4000]}
                ---
                
                You MUST respond with only a single, valid JSON object. Do not include any other text or explanations."""
        
        model_name = os.getenv("OLLAMA_MODEL", "phi3")
        
        model = from_ollama(client, model_name)
        generator = Generator(model, EmailAnalysis)
        result = generator(prompt)
        
        if isinstance(result, EmailAnalysis):
            llm_output = result.model_dump()
        else:
            llm_output = EmailAnalysis.model_validate_json(result).model_dump()

        if email_metadata["full_text"] and len(email_metadata["full_text"].split()) <= 50:
            llm_output["summary"] = email_metadata["full_text"]

        final_email_data = {**email_metadata, **llm_output, "account_id": account_id}
        
        save_processed_email(final_email_data)
        
        # --- THE KEY CHANGE IS HERE ---
        print(f"--- [Thread] Successfully processed Email ID: {email_id}. ---")
        
        # 1. The individual "mark as read" API call is correctly removed.
        # 2. Return True to signal success to the main thread.
        return True

    except Exception as e:
        print(f"\n--- ❗️ [Thread] ERROR processing Email ID {email_id} for account {account_id} ---")
        traceback.print_exc()
        print(f"--- END OF ERROR: Email {email_id} was NOT saved or marked as read. ---\n")
        # 3. Return False to signal failure.
        return False