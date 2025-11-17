import json
import sqlite3
import ssdeep

DB_PATH = "fuzzy.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS payloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sha256 TEXT,
            fuzzy TEXT,
            size INTEGER
        )
    """)
    conn.commit()
    conn.close()

def insert_payload(sha256, fuzzy, size):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO payloads (sha256, fuzzy, size) VALUES (?, ?, ?)", (sha256, fuzzy, size))
    conn.commit()
    conn.close()

def fuzzy_compare(fuzzy_hash):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT sha256, fuzzy FROM payloads")
    results = []

    for sha256, stored_fuzzy in c.fetchall():
        score = ssdeep.compare(fuzzy_hash, stored_fuzzy)
        if score > 0:
            results.append((sha256, score))

    conn.close()
    return sorted(results, key=lambda x: x[1], reverse=True)

def process_eve(path):
    with open(path, "r") as f:
        for line in f:
            event = json.loads(line)

            # Fichier extrait
            if "fileinfo" in event:
                sha256 = event["fileinfo"].get("sha256")
                payload = event["fileinfo"].get("magic")  # à remplacer par ta vraie source de payload binaire brut

            # HTTP body
            elif "http" in event and "request_body" in event["http"]:
                sha256 = None
                payload = event["http"]["request_body"]

            else:
                continue

            if not payload:
                continue

            fuzzy = ssdeep.hash(payload)
            matches = fuzzy_compare(fuzzy)

            print("\n--- Nouveau payload ---")
            print("Fuzzy:", fuzzy)
            print("Matches:", matches)

            insert_payload(sha256, fuzzy, len(payload))

if __name__ == "__main__":
    init_db()
    process_eve("eve.json")
