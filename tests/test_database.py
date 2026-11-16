from database import get_db


def test_database():
    db_gen = get_db()
    db = next(db_gen)
    assert db is not None
    db.close()
