"""Reddit collection (TFE Cryptobel, 2025). Original script; credentials moved to environment variables."""
# On importe les bibliothèques nécessaires :
import os
import praw  # pour accéder à l'API Reddit
import sqlite3  # pour interagir avec la base de données SQLite
from datetime import datetime, timedelta  # pour formater les dates et gérer les plages de temps

# --- Connexion à Reddit via ton app personnelle ---
reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],           # set in .env (never committed)
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],   # set in .env (never committed)
    user_agent=os.environ.get("REDDIT_USER_AGENT", "cryptobel_data_scraper")           # Nom d'identification (libre)
)

# --- Configuration générale du script ---
subreddit_name = "CryptoFr+BitcoinFrance"
  # Nom du subreddit à scraper
keywords =[
    "bitcoin", "ethereum", "crypto", "cryptomonnaie", "blockchain", "web3",
"nft", "metaverse", "defi", "token", "mint", "airdrop", "hodl", "staking",
"wallet", "smart", "contract", "binance", "coinbase", "trading", "bullrun",
"bear", "market", "altcoin", "opensea", "gamefi", "ledger", "shitcoin", "pump", "dump",
"solana", "tokens", "cryptos", "arnaque", "crypto", "beginner", "avis", "tuto", "forum", "belgique",
"france", "québec", "sécurité", "help", "project", "airdrops", "cold", "wallet", "metamask",
"trader", "analyse", "technique", "fondamentale", "investissement", "légal", "education",
"débutant", "investir", "portefeuille", "formation", "projet", "avis",
"problème", "fiscalité", "taxe", "comprendre", "peur", "scam", "recherche", "où", "acheter", "tutoriel"


]  # Mots-clés à chercher
min_score = 3         # Score minimum d’un post pour être retenu
post_limit = 200      # Nombre max de posts à analyser par appel
max_comments = 10     # Nombre max de commentaires à récupérer par post

# --- Stop automatique après le 20 avril 2025 (fenêtre de collecte de l'étude) ---
# Pour réutiliser le script, adapter ou retirer cette date.
date_stop = datetime(2025, 4, 20)
if datetime.now() > date_stop:
    print("⛔ La période de récolte est terminée. Le script s'arrête.")
    exit()

# --- On définit la date minimale : on ne garde que les posts depuis 1 an ---
date_limite = datetime.now() - timedelta(days=365)

# --- Connexion à la base de données SQLite (ou création si elle n'existe pas) ---
conn = sqlite3.connect(os.environ.get("REDDIT_DB", "data/raw/reddit_posts.db"))  # Fichier qui contient la base
cursor = conn.cursor()  # Curseur pour exécuter les requêtes

# --- Création de la table des posts si elle n’existe pas ---
cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        title TEXT,
        score INTEGER,
        created_utc TEXT
    )
""")

# --- Création de la table des commentaires associés aux posts ---
cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id TEXT PRIMARY KEY,
        post_id TEXT,
        body TEXT,
        score INTEGER,
        created_utc TEXT,
        FOREIGN KEY(post_id) REFERENCES posts(id)
    )
""")

# --- Scraping des posts et commentaires ---
subreddit = reddit.subreddit(subreddit_name)
posts = subreddit.new(limit=post_limit)  # On récupère les posts les plus récents (par appel)
added_posts = 0
added_comments = 0

# On parcourt les posts un par un
for post in posts:
    post_date = datetime.fromtimestamp(post.created_utc)
    title_lower = post.title.lower()

    # Vérifie que :
    # - le post contient un mot-clé
    # - il a un score minimum
    # - il a été publié dans l’année écoulée
    if any(kw in title_lower for kw in keywords) and post.score >= min_score and post_date >= date_limite:
        cursor.execute("INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?)", (
            post.id,
            post.title,
            post.score,
            post_date.isoformat()
        ))
        added_posts += 1
        print(f"🧵 Post : {post.title} ({post.score} votes)")

        # On récupère les commentaires (max X)
        post.comments.replace_more(limit=0)
        for comment in post.comments.list()[:max_comments]:
            comment_date = datetime.fromtimestamp(comment.created_utc)
            if len(comment.body.strip()) > 0 and comment_date >= date_limite:
                cursor.execute("INSERT OR IGNORE INTO comments VALUES (?, ?, ?, ?, ?)", (
                    comment.id,
                    post.id,
                    comment.body,
                    comment.score,
                    comment_date.isoformat()
                ))
                added_comments += 1

# --- Sauvegarde et fermeture ---
conn.commit()
conn.close()

# --- Résumé ---
print(f"\n✅ {added_posts} posts ajoutés")
print(f"💬 {added_comments} commentaires ajoutés")
