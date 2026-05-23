import sqlite3

conn = sqlite3.connect("history.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    model_name TEXT,
    prediction TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()


def save_history(image_path, model_name, prediction, confidence):

    cursor.execute(
        """
        INSERT INTO history(
            image_path,
            model_name,
            prediction,
            confidence
        )
        VALUES(?,?,?,?)
        """,
        (
            image_path,
            model_name,
            prediction,
            confidence
        )
    )

    conn.commit()


def get_history():

    cursor.execute(
        "SELECT * FROM history ORDER BY id DESC"
    )

    return cursor.fetchall()