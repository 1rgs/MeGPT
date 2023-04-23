import os
import sqlite3
import csv

# Get the user's home directory
home = os.path.expanduser("~")

# Path to the SQLite database
db_path = f"{home}/Library/Messages/chat.db"

# Check and update read permission if needed
if not os.access(db_path, os.R_OK):
    try:
        os.chmod(db_path, 0o644)
    except PermissionError as e:
        print(f"Permission Error: {e}")
        print(
            "Please go to System Preferences > Security & Privacy > Privacy > Full Disk Access and give Terminal full disk access and try again."
        )
        exit(1)

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Run the query
query = "SELECT text FROM message WHERE is_from_me = '1' AND length(text) > 1;"
cursor.execute(query)

# Fetch the results
results = cursor.fetchall()

# Save the response to a CSV file
csv_file = "messages.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["text"])
    for row in results:
        writer.writerow(row)

# Close the SQLite connection
cursor.close()
conn.close()

print(f"Results saved to '{csv_file}'")
