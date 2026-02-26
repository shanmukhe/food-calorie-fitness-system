import sqlite3

def get_connection():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()
    return conn, cursor


def init_db(cursor, conn):

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password BLOB,
        age INTEGER,
        gender TEXT,
        height REAL,
        weight REAL,
        activity TEXT,
        goal TEXT,
        diabetes INTEGER,
        acidity INTEGER,
        constipation INTEGER,
        obesity INTEGER,
        avatar BLOB,
        is_admin INTEGER DEFAULT 0
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS food_logs (
        username TEXT,
        food TEXT,
        calories REAL,
        date TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weight_logs (
        username TEXT,
        weight REAL,
        date TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS newsletter_subscribers (
        email TEXT PRIMARY KEY,
        date TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS exercise_logs (
        username TEXT,
        exercise TEXT,
        minutes REAL,
        calories_burned REAL,
        date TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_activity (
        username TEXT PRIMARY KEY,
        last_login TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sugar_logs (
        username TEXT,
        craving_level INTEGER,
        trigger TEXT,
        date TEXT
    )
    """)

    conn.commit()

def migrate_schema(cursor, conn):

    def column_exists(table, column):
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        return column in columns

    # Add avatar column if missing
    if not column_exists("users", "avatar"):
        cursor.execute("ALTER TABLE users ADD COLUMN avatar BLOB")

    # Add is_admin column if missing
    if not column_exists("users", "is_admin"):
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")

    conn.commit()