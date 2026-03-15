# database.py - SQLite operations for support tickets

import os
import sqlite3
import random
from datetime import datetime
from config import DATABASE_PATH


def get_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """Sets up the tickets table and seeds some sample data for testing."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            issue_description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Unresolved',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # pre-loaded tickets so we have something to query against
    seeds = [
        ("650932", "Ramesh Kumar", "Net banking login issue", "Resolved"),
        ("784520", "Anjali Sharma", "Credit card billing dispute", "In Progress"),
        ("901234", "David Chen", "Mobile app crash on transfer", "Unresolved"),
        ("543210", "Priya Patel", "Debit card not working abroad", "Resolved"),
        ("112233", "John Smith", "Loan EMI auto-debit failed", "In Progress"),
    ]

    now = datetime.now().isoformat()
    for tid, name, desc, status in seeds:
        cur.execute(
            "INSERT OR IGNORE INTO support_tickets "
            "(ticket_id, customer_name, issue_description, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tid, name, desc, status, now, now),
        )

    conn.commit()
    conn.close()


def generate_ticket_id():
    """Makes a random 6-digit ID that doesn't already exist."""
    conn = get_connection()
    cur = conn.cursor()
    while True:
        new_id = str(random.randint(100000, 999999))
        cur.execute("SELECT 1 FROM support_tickets WHERE ticket_id = ?", (new_id,))
        if not cur.fetchone():
            break
    conn.close()
    return new_id


def create_ticket(ticket_id, customer_name, issue_description):
    """Inserts a new ticket with 'Unresolved' status."""
    conn = get_connection()
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO support_tickets "
        "(ticket_id, customer_name, issue_description, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'Unresolved', ?, ?)",
        (ticket_id, customer_name, issue_description, now, now),
    )
    conn.commit()
    conn.close()
    return ticket_id


def get_ticket_status(ticket_id):
    """Fetches a single ticket by its ID. Returns dict or None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT ticket_id, customer_name, issue_description, status, created_at "
        "FROM support_tickets WHERE ticket_id = ?",
        (ticket_id,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_tickets():
    """Pulls every ticket, newest first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM support_tickets ORDER BY created_at DESC")
    tickets = [dict(r) for r in cur.fetchall()]
    conn.close()
    return tickets
