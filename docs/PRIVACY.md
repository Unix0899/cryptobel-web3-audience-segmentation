# Privacy and what is not published

The thesis worked with messages posted publicly on Reddit and Instagram. Even public posts are personal data
under the GDPR. This repository therefore publishes the **method**, not the **data**.

| Item | Published? | Why |
|---|---|---|
| Collection, cleaning and clustering scripts | Yes | Credentials removed (environment variables), local paths replaced by parameters |
| Aggregated outputs: cluster sizes, sentiment shares, keyword lists, charts | Yes | No message text, no username, no identifier |
| Raw exports (Apify CSV, Reddit CSV / XLSX) | No | Message text and account information |
| SQLite databases and SQL dumps | No | Same |
| Per-message cluster assignments (CSV / DB) | No | Contain the cleaned message text |
| Reddit API credentials | No | Must be revoked if ever exposed; read from `.env`, which is git-ignored |
| X (Twitter) and LinkedIn scrapers | No | Not used in the final analysis |
| Thesis PDF and interview material | No | Contains working documents and personal information |
| Synthetic sample (`data/sample/`) | Yes | Invented messages, generated from templates, only to run the code |

Word clouds and keyword charts show aggregated words across a whole cluster, after removal of URLs and
@mentions. No individual message can be read from them.
