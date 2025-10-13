from app.db.session import engine, Base
from app.models.user import User  # Import all models here


def run_migrations():
    print("Running database migrations...")
    Base.metadata.create_all(bind=engine)
    print("Migrations completed successfully.")


if __name__ == "__main__":
    run_migrations()
