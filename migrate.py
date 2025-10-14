from app.db.session import engine, Base
from app.models.user import User  # Import User model here
from app.models.dataset import Dataset  # Import Dataset model here
from app.models.columns import DatasetColumn  # Import DatasetColumn model here
from app.models.post import Post  # Import Post model here
from app.models.profile import Profile  # Import Profile model here


def run_migrations():
    print("Running database migrations...")
    Base.metadata.create_all(bind=engine)
    print("Migrations completed successfully.")


if __name__ == "__main__":
    run_migrations()
