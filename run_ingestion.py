import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.blocking import BlockingScheduler
import sqlalchemy

from gmail_service import get_gmail_service, fetch_raw_emails, mark_emails_as_read
from ingestion_worker import process_single_email
from gatekeeper import is_important_email
from database_cleanup import cleanup_old_emails
from database import engine, managed_accounts_table

# --- REMOVED ThreadPoolExecutor ---
# You can adjust max_workers based on your remote system's CPU cores. 5 is a good start.
executor = ThreadPoolExecutor(max_workers=5)

def run_ingestion_cycle():
    print(f"\n--- [{time.ctime()}] Running ingestion cycle for all managed accounts ---")
    
    with engine.connect() as connection:
        stmt = sqlalchemy.select(managed_accounts_table)
        accounts_to_process = connection.execute(stmt).fetchall()

    if not accounts_to_process:
        print("No managed accounts found in the database. Run add_account.py to add one.")
        return

    for account in accounts_to_process:
        account_id, account_email, token_path = account
        print(f"\n--- Processing account: {account_email} ---")
        
        service = get_gmail_service(token_path)
        if not service:
            print(f"Could not connect to Gmail for {account_email}. Skipping.")
            continue
            
        raw_emails = fetch_raw_emails(service)
        if not raw_emails:
            print(f"No new emails to process for {account_email}.")
            continue
        
        important_emails = []
        unimportant_email_ids = []
        for email in raw_emails:
            if is_important_email(email):
                important_emails.append(email)
            else:
                unimportant_email_ids.append(email['id'])

        # This part was already correct: batch-marking unimportant emails.
        if unimportant_email_ids:
            mark_emails_as_read(service, unimportant_email_ids, account_email)

        if not important_emails:
            print(f"No important emails found for {account_email}.")
            continue
            
        print(f"Submitting {len(important_emails)} important emails to the processing pool...")
        
        # 1. Submit jobs to the thread pool.
        #    We pass the 'token_path' so each thread can create its own service object.
        future_to_email_id = {
            executor.submit(process_single_email, email, account_id, token_path): email['id']
            for email in important_emails
        }
        
        # --- STABLE PROCESSING & BATCHING LOGIC ---
        successfully_processed_ids = []
        
        # 2. As jobs complete, collect the IDs of the ones that succeeded.
        for future in as_completed(future_to_email_id):
            email_id = future_to_email_id[future]
            try:
                # The worker will return True on success.
                if future.result():
                    successfully_processed_ids.append(email_id)
            except Exception as exc:
                print(f"❗️ Email ID {email_id} generated an exception in the worker: {exc}")
        
       # 3. After all threads have finished, mark the successful emails as read in a single batch.
        if successfully_processed_ids:
            print(f"Marking {len(successfully_processed_ids)} processed emails as read in a single batch...")
            mark_emails_as_read(service, successfully_processed_ids, account_email)
        
        
    print("\n--- ✅ Ingestion cycle complete ---")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_ingestion_cycle, 'interval', minutes=2)
    scheduler.add_job(cleanup_old_emails, 'interval', hours=24)
    
    print("--- AI Email Ingestion & Cleanup Pipeline (Multi-Account) ---")
    print("Scheduler started. Ingestion runs every 2 minutes, cleanup every 24 hours.")
    print("Press Ctrl+C to exit.")
    
    # Run one cycle immediately on startup
    run_ingestion_cycle()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped. Shutting down worker threads...")
        # --- REMOVED executor.shutdown() as it no longer exists ---
        executor.shutdown(wait=True)