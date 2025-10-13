from app.db.session import engine, Base
from app.models.user import User  # Import all models here


def initialize_database():
    print("Initializing the shared development database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialization completed successfully.")


if __name__ == "__main__":
    initialize_database()
