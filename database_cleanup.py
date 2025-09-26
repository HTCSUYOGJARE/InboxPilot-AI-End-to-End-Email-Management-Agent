import sqlalchemy
from datetime import datetime, timedelta

from database import engine, emails_table, email_collection

def cleanup_old_emails(days_to_keep=30):
    print(f"\n--- 🧼 Starting cleanup for emails older than {days_to_keep} days ---")
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")

    try:
        with engine.begin() as connection:
            # FIX 2 & 3: Find old emails based on the composite key.
            select_stmt = sqlalchemy.select(emails_table.c.id, emails_table.c.account_id).where(
                emails_table.c.received_at < cutoff_date
            )
            old_emails = connection.execute(select_stmt).fetchall()

            if not old_emails:
                print("No old emails found to delete.")
                return

            print(f"Found {len(old_emails)} old emails to delete from SQLite.")
            
            old_email_ids = [email[0] for email in old_emails]
            delete_stmt = emails_table.delete().where(
                emails_table.c.id.in_(old_email_ids)
            )
            result = connection.execute(delete_stmt)
            print(f"Deleted {result.rowcount} records from SQLite.")

            # FIX 3: Use the efficient $in operator to find all chunks to delete.
            if old_email_ids:
                chunk_ids_to_delete = email_collection.get(
                    where={"email_id": {"$in": old_email_ids}},
                    include=[] # We only need the IDs, not the data
                )['ids']
                
                if chunk_ids_to_delete:
                    email_collection.delete(ids=chunk_ids_to_delete)
                    print(f"Deleted {len(chunk_ids_to_delete)} vector chunks from ChromaDB.")
            
            print("--- ✅ Cleanup successful ---")

    except Exception as e:
        print(f"--- ❌ An error occurred during cleanup: {e} ---")
