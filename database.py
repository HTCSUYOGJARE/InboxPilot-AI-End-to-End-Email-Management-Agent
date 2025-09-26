import os
import re
import chromadb
import sqlalchemy
from sqlalchemy import (Table, Column, Integer, String, MetaData, ForeignKey,
                        Boolean, update, select, Text, DateTime, UniqueConstraint,
                        PrimaryKeyConstraint)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sentence_transformers import SentenceTransformer

# --- 0. Ensure the Database Directory Exists ---
db_folder = "./db"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
    print(f"Created database directory at: {db_folder}")

# --- 1. Initialize the Embedding Model ---
print("Loading sentence-transformer model... (This may take a moment on first run)")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
print("Model loaded successfully.")

# --- 2. ChromaDB (Vector Store for Semantic Search) Setup ---
chroma_client = chromadb.PersistentClient(path=os.path.join(db_folder, "chroma_db"))
email_collection = chroma_client.get_or_create_collection(
    name="emails",
    metadata={"hnsw:space": "cosine"}
)
print("ChromaDB vector store initialized.")

# --- 3. SQLite (Structured Data Store) Setup ---
DB_URL = f"sqlite:///{os.path.join(db_folder, 'emails.db')}"
engine = sqlalchemy.create_engine(DB_URL)
metadata = sqlalchemy.MetaData()

managed_accounts_table = Table(
    "managed_accounts", metadata,
    Column("id", Integer, primary_key=True),
    Column("account_email", String, unique=True, nullable=False),
    Column("token_path", String, unique=True, nullable=False)
)

emails_table = Table(
    "emails", metadata,
    Column("account_id", Integer, ForeignKey("managed_accounts.id", ondelete="CASCADE"), nullable=False),
    Column("id", String, nullable=False), # Gmail's message ID
    Column("threadId", String),
    Column("sender", String),
    Column("subject", String),
    Column("summary", Text),
    Column("full_text", Text),
    Column("received_at", DateTime),
    Column("priority", String),
    Column("action_needed", Boolean),
    Column("deadline", String, nullable=True),
    Column("attachment_summary", Text, nullable=True),
    PrimaryKeyConstraint('account_id', 'id', name='pk_emails')
)

people_table = Table(
    "people", metadata,
    Column("account_id", Integer, ForeignKey("managed_accounts.id", ondelete="CASCADE"), nullable=False),
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("last_seen_at", DateTime, default=sqlalchemy.func.now(), onupdate=sqlalchemy.func.now()),
    UniqueConstraint('account_id', 'name', name='uq_account_name')
)

# ================================================================= #
# CORRECTED SECTION: The constraint is now on the *pair* of columns #
# ================================================================= #
contact_emails_table = Table(
    "contact_emails", metadata,
    Column("id", Integer, primary_key=True),
    Column("person_id", Integer, ForeignKey("people.id", ondelete="CASCADE"), nullable=False),
    Column("email", String, nullable=False),
    Column("is_primary", Boolean, default=False, nullable=False),
    # This ensures an email is unique PER person, not globally.
    UniqueConstraint('person_id', 'email', name='uq_person_email')
)
# ================================================================= #

metadata.create_all(engine)
print("SQLite database initialized.")

def chunk_text(text: str, chunk_size: int = 100, chunk_overlap: int = 20) -> list[str]:
    """Splits a text into overlapping chunks of words."""
    if not text:
        return []
    
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
        
    chunks = []
    start_index = 0
    while start_index < len(words):
        end_index = start_index + chunk_size
        chunk = words[start_index:end_index]
        chunks.append(" ".join(chunk))
        start_index += chunk_size - chunk_overlap
    return chunks

def save_processed_email(email_data: dict):
    """
    Saves processed email data to SQLite and ChromaDB, and updates contact information.
    """
    email_id = email_data.get("id")
    account_id = email_data.get("account_id")
    if not email_id or not account_id:
        print(f"Error: Email data is missing 'id' or 'account_id'. Cannot save.")
        return

    try:
        subject = email_data.get("subject", "")
        full_text = email_data.get("full_text", "")
        text_to_chunk = f"Subject: {subject}\n\nBody: {full_text}"
        chunks = chunk_text(text_to_chunk)
        
        if chunks:
            chunk_vectors = embedding_model.encode(chunks).tolist()
            chunk_ids = [f"{account_id}_{email_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadatas = [{
                "account_id": account_id,
                "email_id": email_id, 
                "chunk_index": i, 
                "chunk_text": chunk_content,
            } for i, chunk_content in enumerate(chunks)]
            
            email_collection.upsert(ids=chunk_ids, embeddings=chunk_vectors, metadatas=chunk_metadatas)

        with engine.begin() as connection:
            stmt = sqlite_insert(emails_table).values(email_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['account_id', 'id'], 
                set_={k: v for k, v in email_data.items() if k not in ['account_id', 'id']}
            )
            connection.execute(stmt)

            sender_str = email_data.get("sender")
            if sender_str:
                try:
                    match = re.match(r'(.*)<(.*)>', sender_str)
                    name, email = (match.group(1).strip().replace('"', '').title(), match.group(2).strip()) if match else (sender_str.split('@')[0].title(), sender_str.strip())
                    if not name: name = email.split('@')[0]

                    stmt_find = select(people_table.c.id).where(people_table.c.name == name, people_table.c.account_id == account_id)
                    person_result = connection.execute(stmt_find).first()
                    
                    if person_result:
                        person_id = person_result[0]
                        stmt_update_person = update(people_table).where(people_table.c.id == person_id).values(last_seen_at=sqlalchemy.func.now())
                        connection.execute(stmt_update_person)
                    else:
                        stmt_insert = sqlite_insert(people_table).values(name=name, account_id=account_id)
                        cursor = connection.execute(stmt_insert)
                        person_id = cursor.lastrowid

                    stmt_insert_email = sqlite_insert(contact_emails_table).values(person_id=person_id, email=email)
                    
                    # This now checks for conflicts on (person_id, email)
                    stmt_do_nothing = stmt_insert_email.on_conflict_do_nothing(index_elements=['person_id', 'email'])
                    connection.execute(stmt_do_nothing)
                    
                    stmt_unset = update(contact_emails_table).where(contact_emails_table.c.person_id == person_id).values(is_primary=False)
                    connection.execute(stmt_unset)
                    stmt_set = update(contact_emails_table).where(contact_emails_table.c.person_id == person_id, contact_emails_table.c.email == email).values(is_primary=True)
                    connection.execute(stmt_set)

                except Exception as e:
                    print(f"Warning: Could not parse or save sender '{sender_str}'. Error: {e}")

    except Exception as e:
        print(f"An error occurred while saving email ID {email_id} for account {account_id}: {e}")
        raise