import sqlite3
import os
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "storage" / "narad_memory.db"

def init_memory_db():
    """Initializes the conversations table."""
    os.makedirs(DB_FILE.parent, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_history (session_id)")
        conn.commit()

init_memory_db()

def add_message(session_id: str, role: str, content: str):
    """Saves a message to the session history."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()

def get_recent_history(session_id: str, limit: int = 6) -> list[dict]:
    """Retrieves the most recent messages in chronological order."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content FROM chat_history 
            WHERE session_id = ? 
            ORDER BY id DESC LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        # Reverse to get chronological order (oldest to newest)
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def clear_session_history(session_id: str):
    """Deletes all history for a specific session."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()