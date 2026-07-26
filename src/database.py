import sqlite3
import os

DB_PATH = "database/sentinel.db"

def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Track uploaded retrain samples
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retraining_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            label TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed INTEGER DEFAULT 0
        )
    """)
    
    # Track training run history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retraining_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            samples_used INTEGER,
            accuracy REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def log_sample(filename: str, filepath: str, label: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO retraining_samples (filename, filepath, label) VALUES (?, ?, ?)",
        (filename, filepath, label)
    )
    conn.commit()
    conn.close()