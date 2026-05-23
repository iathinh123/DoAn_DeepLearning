import sqlite3
import matplotlib.pyplot as plt


def show_statistics():

    conn = sqlite3.connect("history.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT prediction, COUNT(*) FROM history GROUP BY prediction"
    )

    data = cursor.fetchall()

    labels = [x[0] for x in data]
    values = [x[1] for x in data]

    plt.figure(figsize=(6,6))

    plt.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    plt.title("Waste Classification Statistics")

    plt.show()