import sqlalchemy
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from gmail_service import get_gmail_service
from database import engine, managed_accounts_table

def add_new_account():
    """
    A one-time script to authorize a new Google account and save it to the database.
    """
    print("--- Add New Gmail Account ---")
    account_email = input("Enter the Gmail address for the new account: ").strip()
    
    if not account_email:
        print("Email cannot be empty.")
        return

    token_path = f"token_{account_email.replace('@', '_')}.json"
    print(f"\nA browser window will now open for you to authorize: {account_email}")
    print("Please follow the steps to grant permission.")

    service = get_gmail_service(token_path)

    if not service:
        print("\nAuthorization failed. Could not create Gmail service.")
        return

    try:
        with engine.begin() as connection:
            stmt = sqlite_insert(managed_accounts_table).values(
                account_email=account_email,
                token_path=token_path
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=['account_email'],
                set_=dict(token_path=token_path)
            )
            connection.execute(stmt)
            print(f"\n✅ Successfully authorized and saved account: {account_email}")

    except Exception as e:
        print(f"\n❌ An error occurred while saving the account to the database: {e}")


if __name__ == "__main__":
    # This script should be run before the main pipeline to add accounts.
    add_new_account()
