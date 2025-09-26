import os
import sqlalchemy
from collections import defaultdict

# --- Important: This will import the tables and also initialize the DBs ---
# --- It will also load the sentence-transformer model, which may take a moment ---
from database import (
    engine,
    chroma_client,
    email_collection,
    managed_accounts_table,
    emails_table,
    people_table,
    contact_emails_table
)

def verify_email_counts(connection, collection):
    """
    Counts unique emails for each account in SQLite and ChromaDB and verifies they match.
    """
    print("--- 1. Verifying Email Counts Per Account ---")
    
    accounts = connection.execute(sqlalchemy.select(managed_accounts_table)).fetchall()
    if not accounts:
        print("No managed accounts found in the database.")
        return

    try:
        chroma_data = collection.get(include=["metadatas"])
        chroma_metadatas = chroma_data['metadatas']
    except Exception as e:
        print(f"Could not fetch data from ChromaDB: {e}")
        return

    chroma_emails_by_account = defaultdict(set)
    for meta in chroma_metadatas:
        if 'account_id' in meta and 'email_id' in meta:
            chroma_emails_by_account[meta['account_id']].add(meta['email_id'])

    all_match = True
    for account in accounts:
        account_id, account_email, _ = account
        print(f"\nVerifying account: {account_email} (ID: {account_id})...")

        stmt = sqlalchemy.select(sqlalchemy.func.count(emails_table.c.id)).where(emails_table.c.account_id == account_id)
        sqlite_count = connection.execute(stmt).scalar_one_or_none() or 0
        chroma_count = len(chroma_emails_by_account.get(account_id, set()))

        print(f"  - SQLite Count:   {sqlite_count} unique emails")
        print(f"  - ChromaDB Count: {chroma_count} unique emails (from chunks)")
        
        if sqlite_count == chroma_count:
            print("  - ✅ Status: Match")
        else:
            print(f"  - ❌ Status: MISMATCH")
            all_match = False
            
    if all_match:
        print("\nVerification Complete: All account counts are consistent between SQLite and ChromaDB.")
    else:
        print("\nVerification Complete: INCONSISTENCIES WERE FOUND.")

def display_sqlite_info(connection):
    """
    Displays the schema and a few sample rows for each table in the SQLite database.
    """
    print("\n" + "---" * 15)
    print("--- 2. SQLite Database Schema and Sample Data ---")

    tables = {
        "managed_accounts": managed_accounts_table,
        "emails": emails_table,
        "people": people_table,
        "contact_emails": contact_emails_table
    }

    for name, table_obj in tables.items():
        print(f"\n--- Table: {name} ---")
        
        print("  Schema:")
        for column in table_obj.columns:
            print(f"    - {column.name} ({column.type})")

        print("  Sample Data:")
        try:
            stmt = table_obj.select().limit(3)
            results = connection.execute(stmt).fetchall()
            if not results:
                print("    No data in this table.")
            else:
                for i, row in enumerate(results):
                    print(f"    Row {i+1}: {dict(row._mapping)}")
        except Exception as e:
            print(f"    Could not fetch data for table {name}: {e}")

# ================================================================= #
# CORRECTED SECTION: This function is now more robust.              #
# ================================================================= #
def display_chromadb_info(collection):
    """
    Displays the format and a few sample entries from the ChromaDB collection.
    """
    print("\n" + "---" * 15)
    print("--- 3. ChromaDB Vector Store Format and Sample Data ---")

    try:
        count = collection.count()
        print(f"Total chunks (documents) in collection: {count}")
        if count == 0:
            print("Collection is empty.")
            return

        # A more robust way to get a few items: get all IDs, then query for the first few.
        all_ids = collection.get(include=[])['ids']
        sample_ids = all_ids[:3] # Get the first 3 IDs

        sample_chunks = collection.get(ids=sample_ids, include=["metadatas", "embeddings"])

        for i in range(len(sample_chunks['ids'])):
            chunk_id = sample_chunks['ids'][i]
            metadata = sample_chunks['metadatas'][i]
            embedding_dim = len(sample_chunks['embeddings'][i]) if sample_chunks['embeddings'] else 'N/A'

            print(f"\n--- Sample Chunk {i+1} ---")
            print(f"  - Chunk ID:      {chunk_id}")
            print(f"  - Metadata:      {metadata}")
            print(f"  - Embedding:     Vector of dimension {embedding_dim}")

    except Exception as e:
        print(f"Could not fetch data from ChromaDB: {e}")
# ================================================================= #


if __name__ == "__main__":
    print("Initializing database connections...")
    with engine.connect() as connection:
        verify_email_counts(connection, email_collection)
        display_sqlite_info(connection)
        display_chromadb_info(email_collection)