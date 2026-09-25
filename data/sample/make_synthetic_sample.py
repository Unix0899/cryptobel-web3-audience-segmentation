"""Create small SYNTHETIC SQLite databases with the same schema as the real collection.

    python data/sample/make_synthetic_sample.py

The real Reddit / Instagram messages are personal data and are NOT published. These
databases contain invented French messages generated from templates, only so that
the clustering scripts can be run end to end. Results on this sample mean nothing.
"""
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
rng = random.Random(2025)

SUBJECTS = ["mon wallet", "ma ledger", "la plateforme", "binance", "coinbase", "le bitcoin", "ethereum",
            "les frais de retrait", "le staking", "un airdrop", "la blockchain", "les NFT", "le marché",
            "la sécurité de mes cryptos", "la fiscalité belge", "un smart contract", "la seed phrase"]
PROFILES = {
    "curieux": ["Je débute et je me demande comment sécuriser {s}, quelqu'un peut expliquer ?",
                "Les frais pour transférer depuis {s} sont vraiment élevés, une solution moins chère ?",
                "Comment déclarer {s} aux impôts en Belgique, vous faites comment ?",
                "Je voudrais investir un petit montant, vaut-il mieux passer par {s} ?"],
    "dev": ["J'ai développé un petit outil pour analyser {s}, retour bienvenu sur le code.",
            "Le déploiement de mon contrat sur testnet fonctionne, prochaine étape auditer {s}.",
            "Pour sécuriser {s} je recommande une vérification des permissions avant chaque signature."],
    "idealiste": ["La décentralisation va changer la finance, {s} en est la preuve.",
                  "Je crois vraiment que {s} rendra le système plus transparent pour tout le monde.",
                  "Long terme uniquement, {s} c'est l'avenir malgré la volatilité."],
    "sceptique": ["Encore une arnaque autour de {s}, jamais je n'y mettrai mon argent.",
                  "J'ai perdu gros avec {s}, méfiez-vous des promesses de gains rapides.",
                  "Trop de risques et aucune régulation sérieuse pour {s}, je reste à l'écart."],
}
WEIGHTS = {"curieux": 0.45, "idealiste": 0.3, "sceptique": 0.15, "dev": 0.1}


def message():
    profile = rng.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
    text = rng.choice(PROFILES[profile]).format(s=rng.choice(SUBJECTS))
    if rng.random() < 0.3:
        text += " " + rng.choice(["Merci d'avance !", "Des avis ?", "Franchement déçu.", "Hâte de voir la suite."])
    return text


def when():
    return datetime(2025, 3, 1) + timedelta(minutes=rng.randrange(0, 60 * 24 * 50))


def reddit(path, n_posts=40, n_comments=320):
    path.unlink(missing_ok=True)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE posts (id TEXT PRIMARY KEY, title TEXT, score INTEGER, created_utc TEXT)")
    c.execute("CREATE TABLE comments (id TEXT PRIMARY KEY, post_id TEXT, body TEXT, score INTEGER, "
              "created_utc TEXT, FOREIGN KEY(post_id) REFERENCES posts(id))")
    for i in range(n_posts):
        c.execute("INSERT INTO posts VALUES (?,?,?,?)", (f"p{i:03d}", f"[synthetic] Question sur {rng.choice(SUBJECTS)}",
                                                         rng.randint(3, 80), when().isoformat()))
    for i in range(n_comments):
        c.execute("INSERT INTO comments VALUES (?,?,?,?,?)", (f"c{i:04d}", f"p{rng.randrange(n_posts):03d}", message(),
                                                              rng.randint(-2, 40), when().isoformat()))
    c.commit()
    c.close()


def instagram(path, n_posts=60, n_comments=360):
    path.unlink(missing_ok=True)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE posts (id TEXT PRIMARY KEY, username TEXT, date TEXT, caption TEXT, url TEXT)")
    c.execute("CREATE TABLE comments (id TEXT PRIMARY KEY, post_id TEXT, text TEXT, created_at TEXT, "
              "FOREIGN KEY(post_id) REFERENCES posts(id))")
    for i in range(n_posts):
        c.execute("INSERT INTO posts VALUES (?,?,?,?,?)", (f"ig{i:03d}", "synthetic_account", when().isoformat(),
                                                           f"[synthetic] Actu crypto : {rng.choice(SUBJECTS)}",
                                                           f"https://example.org/synthetic/{i}"))
    for i in range(n_comments):
        c.execute("INSERT INTO comments VALUES (?,?,?,?)", (f"igc{i:04d}", f"ig{rng.randrange(n_posts):03d}", message(),
                                                            when().isoformat()))
    c.commit()
    c.close()


if __name__ == "__main__":
    reddit(HERE / "reddit_sample.db")
    instagram(HERE / "instagram_sample.db")
    print("Synthetic samples written: data/sample/reddit_sample.db, data/sample/instagram_sample.db")
