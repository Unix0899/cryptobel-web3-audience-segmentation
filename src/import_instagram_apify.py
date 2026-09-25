"""Instagram: import of the Apify CSV exports into SQLite (TFE Cryptobel, 2025). Original script; local paths replaced by parameters."""
import pandas as pd
import sqlite3
import os
from datetime import datetime

# 📁 Dossier contenant les fichiers CSV Apify
folder_path = os.environ.get("INSTAGRAM_EXPORT_DIR", "data/raw/instagram/")
new_db_path = os.environ.get("INSTAGRAM_DB", "data/raw/instagram_apify_data.db")  # ← nouvelle base

# Connexion à la nouvelle base
conn = sqlite3.connect(new_db_path)
cursor = conn.cursor()
print(f"🆕 Base de données créée : {new_db_path}")

# Création des tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    username TEXT,
    date TEXT,
    caption TEXT,
    url TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    post_id TEXT,
    text TEXT,
    created_at TEXT,
    FOREIGN KEY(post_id) REFERENCES posts(id)
)
""")

# Compteurs
total_posts = 0
total_comments = 0

# Lecture de tous les fichiers .csv
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        print(f"\n📂 Traitement : {file_name}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Erreur de lecture : {e}")
            continue

        added_posts = 0
        added_comments = 0

        for _, row in df.iterrows():
            try:
                post_id = str(row["id"])
                caption = row.get("caption", "")
                if pd.isna(caption): caption = ""

                username = "apify_import"
                date = row["timestamp"] if pd.notna(row["timestamp"]) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                url = row["url"] if pd.notna(row["url"]) else f"https://www.instagram.com/p/{post_id}"

                # Insertion du post
                cursor.execute("""
                    INSERT OR IGNORE INTO posts (id, username, date, caption, url)
                    VALUES (?, ?, ?, ?, ?)
                """, (post_id, username, date, caption, url))
                added_posts += 1

                # Insertion du commentaire principal
                first_comment = row.get("firstComment", "")
                if pd.notna(first_comment) and first_comment.strip():
                    cursor.execute("""
                        INSERT OR IGNORE INTO comments (id, post_id, text, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (
                        f"{post_id}_first",
                        post_id,
                        first_comment.strip(),
                        date
                    ))
                    added_comments += 1

            except Exception as e:
                print(f"⚠️ Problème avec une ligne : {e}")
                continue

        total_posts += added_posts
        total_comments += added_comments

        print(f"✅ {added_posts} posts et {added_comments} commentaires ajoutés.")

# Sauvegarde finale
conn.commit()
conn.close()

print(f"\n📊 Import terminé. Total : {total_posts} posts / {total_comments} commentaires dans {new_db_path}")
