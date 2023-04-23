import os
import sqlite3
import json

# Get the user's home directory
home = os.path.expanduser("~")

output_file = "response.json"
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
query = """
WITH messages_with_prev AS (
    SELECT m.ROWID, m.guid, m.text, m.subject, m.country, m.date, chj.chat_id, m.is_from_me,
           LAG(m.is_from_me) OVER (PARTITION BY chj.chat_id ORDER BY m.date) AS prev_is_from_me
    FROM message AS m
    JOIN chat_message_join AS chj ON m.ROWID = chj.message_id
    WHERE LENGTH(m.text) > 0
),
grouped_messages AS (
    SELECT *,
           SUM(CASE WHEN is_from_me != IFNULL(prev_is_from_me, -1) THEN 1 ELSE 0 END) OVER (PARTITION BY chat_id ORDER BY date) AS grp
    FROM messages_with_prev
),
consecutive_messages AS (
    SELECT chat_id, is_from_me, group_concat(text, '\n') AS joined_text, MIN(date) AS min_date
    FROM grouped_messages
    GROUP BY chat_id, is_from_me, grp
),
my_consecutive_messages AS (
    SELECT * FROM consecutive_messages WHERE is_from_me = 1
),
other_consecutive_messages AS (
    SELECT * FROM consecutive_messages WHERE is_from_me = 0
)

SELECT other.joined_text AS prev_text, my.joined_text AS my_text
        FROM my_consecutive_messages AS my
LEFT JOIN other_consecutive_messages AS other ON my.chat_id = other.chat_id AND other.min_date < my.min_date
WHERE other.min_date = (
    SELECT MAX(min_date) FROM other_consecutive_messages AS ocm
    WHERE ocm.chat_id = my.chat_id AND ocm.min_date < my.min_date
)
ORDER BY my.min_date;
"""
cursor.execute(query)

# Fetch the results
results = cursor.fetchall()

print(results[0])
# remove all newlines
results = [tuple(map(lambda x: x.replace("\n", " "), row)) for row in results]


with open(output_file, "w") as f:
    json.dump(
        [{"text": f"person: {row[0]}\nMeGPT: {row[1]}", "label": 0} for row in results],
        f,
        indent=4,
    )

print(f"Saved {len(results)} messages to {output_file}")
