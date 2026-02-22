import os
import sqlite3


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")


class Database:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_users_table()

    def create_users_table(self):
        self.cursor.execute("""
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
                obesity INTEGER
            )
        """)
        self.conn.commit()

    def execute(self, query, params=()):
        self.cursor.execute(query, params)
        self.conn.commit()

    def fetchone(self, query, params=()):
        return self.cursor.execute(query, params).fetchone()

    def fetchall(self, query, params=()):
        return self.cursor.execute(query, params).fetchall()


db = Database()