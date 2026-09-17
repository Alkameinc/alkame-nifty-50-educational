import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DB_PATH

from storage_reliability import configure_sqlite_engine, create_reliable_engine

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_reliable_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
