"""Cleaning, CamemBERT sentiment and K-means clustering - Reddit (TFE Cryptobel, 2025).
Original script; input database and output folder made configurable."""
# 📦 CLUSTERING K-MEANS POUR PROFILS CRYPTO-WEB3
# Script spécialisé pour l'identification des profils via K-means
import pandas as pd
import numpy as np
import sqlite3
import re
import emoji
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
from wordcloud import WordCloud
import warnings
import os
from datetime import datetime
from collections import Counter
from tqdm import tqdm

# Supprimer les avertissements
warnings.filterwarnings("ignore")

# Configuration
print("🔶 CLUSTERING K-MEANS POUR PROFILS CRYPTO-WEB3 🔶")
print("=================================================")

# Création d'un dossier pour les résultats
results_folder = os.path.join("results", "runs", f"clusters_crypto_{datetime.now().strftime('%Y%m%d_%H%M')}")
os.makedirs(results_folder, exist_ok=True)
print(f"📁 Les résultats seront sauvegardés dans: {results_folder}")

# 🔄 NLTK
print("🔄 Initialisation de NLTK...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('french'))


# Mots spécifiques aux cryptomonnaies à conserver
crypto_keywords = {'bitcoin', 'btc', 'eth', 'ethereum', 'crypto', 'blockchain', 'nft', 
                   'web3', 'defi', 'token', 'wallet', 'mining', 'miner', 'altcoin', 
                   'staking', 'hodl', 'bull', 'bear', 'fomo', 'dyor', 'sol', 'ada',
                   'crypto', 'cryptomonnaie', 'cryptomonnaies', 'satoshi', 'nakamoto'}

# Mots neutres à exclure du scoring (pas révélateurs d’un profil)
neutral_keywords = {
    'crypto', 'cryptos',
    'cryptomonnaie', 'cryptomonnaies',
    'blockchain',
    'token', 'nft',
    'web3',
    'projet', 'projets',
    'block', 'transactions',
    'web',
    'user', 'client', 'fichier'
}
 # Déclare les mots de bruit
noise_words = {
   # Pronoms et termes de repérage
    'je','j','tu','te','toi','il','elle','on','nous','vous','ils','elles',
    'me','m','te','t','se','s','y','en','lui','eux','leur','leurs',
    'cela','celui','celle','ceux','celles','ça','ce','c','là','ici'

    # Articles (défini, indéfini), prépositions, conjonctions
    'le','la','les','un','une','des','du','de','d','au','aux','à','dans','sur',
    'sous','par','pour','avec','sans','chez','entre','vers','selon',
    'mais','ou','et','donc','or','ni','car','si','que','qu','quoi','quand',
    'pourquoi','comment','où','dont','quel','quelle','quels','quelles'

    # Verbes auxiliaires et usage fréquent
    'être','avoir','aller','venir','faire','dire','voir','savoir','pouvoir',
    'vouloir','falloir','tenir','mettre','prendre','donner','parler','passer',
    'rester','devenir','sembler','laisser','arriver','comprendre','sentir'

    # Adverbes/adjectifs d’intensité ou de fréquence
    'très','trop','plus','moins','bien','mal','peu','quelque','certain','vraiment',
    'juste','simple','possible','encore','déjà','toujours','souvent','parfois','rarement'

    # Rires, abréviations, emojis, mots d’engagement social
    'lol','mdr','ptdr','haha','😉','👍','😂','🌟','🔥','💯',
    'ok','cool','super','top','bcp','beaucoup','svp','stp','merci',
    'please','hello','hey','yo','salut','coucou','bjr','bonsoir'

    # Exemples vus dans les wordclouds : 
    'merci','svp','stp','ok','cool','super','top','bcp','beaucoup',
    'love','like','ty','hey','yo','comme',"maintenant",'aprés','toute','enfin'
    

    # Mots liés à la crypto, trop génériques ou ambigus
    'raydium','france','bitcoin','crypto','ethereum','eth','btc','sol','ada','nft','web3','fomo'


}

# Fusionne avec neutral_keywords
neutral_keywords |= noise_words

# Mise à jour des stop words
stop_words = stop_words - crypto_keywords
all_stop_words = list(stop_words.union(neutral_keywords))



# 🔌 Connexion à la base locale
print("📂 Chargement des données REDDIT...")
conn = sqlite3.connect(os.environ.get("REDDIT_DB", "data/sample/reddit_sample.db"))
df_comments = pd.read_sql_query("SELECT * FROM comments", conn)
conn.close()

print(f"Commentaires chargés: {len(df_comments)}")

# 🧼 Nettoyage du texte adapté aux discussions crypto
def nettoyer_texte_crypto(text):
    if pd.isnull(text):
        return None
    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s#]", "", text)  # Conserver les hashtags (#)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    # on filtre les tokens : pas dans all_stop_words, 4+ caractères, que des lettres
    filtered = [
        t
        for t in tokens
        if t not in all_stop_words and len(t) >= 5 and t.isalpha()
    ]

    if len(filtered) < 2:
        return None

    return " ".join(filtered)

# Application du nettoyage
print("🧹 Nettoyage des textes...")
df_comments["text_clean"] = df_comments["body"].apply(nettoyer_texte_crypto)

# Filtrer les textes nuls
df_comments = df_comments[df_comments["text_clean"].notnull()]

print(f"Commentaires après nettoyage: {len(df_comments)}")

# 🤖 CamemBERT pipeline pour le sentiment
print("🔄 Chargement du modèle d'analyse de sentiment...")
model_name = "cmarkea/distilcamembert-base-sentiment"
sentiment_model = pipeline("text-classification", model=model_name, tokenizer=model_name, device=-1)

# Fonction d'analyse de sentiment
def analyser_sentiment(text):
    try:
        if text is None or len(text) < 5:
            return "neutre", 0.0
        
        result = sentiment_model(text[:512])[0]
        label = result['label']
        
        # Convertir l'étiquette en polarité et sentiment
        if '5 stars' in label or '4 stars' in label:
            sentiment = "positif"
            polarite = 0.7 if '4 stars' in label else 0.9
        elif '1 star' in label or '2 stars' in label:
            sentiment = "négatif"
            polarite = -0.7 if '2 stars' in label else -0.9
        else:
            sentiment = "neutre"
            polarite = 0.0
            
        return sentiment, polarite
    except Exception as e:
        print(f"Erreur d'analyse: {e}")
        return "neutre", 0.0

# Échantillonnage des commentaires pour équilibrer le dataset
sample_size = min(10000, len(df_comments))
print(f"🔍 Échantillonnage de {sample_size} commentaires pour l'analyse...")
df_comments_sample = df_comments.sample(sample_size, random_state=42)

# Calcul des caractéristiques pour les commentaires
print("🔍 Calcul des caractéristiques pour les commentaires...")
sentiments_comments = []
polarites_comments = []

for text in tqdm(df_comments_sample["text_clean"]):
    sentiment, polarite = analyser_sentiment(text)
    sentiments_comments.append(sentiment)
    polarites_comments.append(polarite)

df_comments_sample["sentiment"] = sentiments_comments
df_comments_sample["polarite"] = polarites_comments

# Préparation des données uniquement à partir des commentaires
print("🔄 Préparation du dataset à partir des commentaires uniquement...")
df_combined = pd.DataFrame()
df_combined["text"] = df_comments_sample["text_clean"]
df_combined["sentiment"] = df_comments_sample["sentiment"]
df_combined["polarite"] = df_comments_sample["polarite"]
df_combined["source"] = "comment"

# Extraction des caractéristiques via TF-IDF
print("🔄 Extraction des caractéristiques textuelles (TF-IDF)...")
tfidf_vectorizer = TfidfVectorizer(
    stop_words=all_stop_words,
    max_features=200,  # Limiter les features pour éviter la malédiction de la dimensionnalité
    min_df=5,          # Ignorer les termes qui apparaissent dans moins de 5 documents
    max_df=0.5,        # Ignorer les termes qui apparaissent dans plus de 50% des documents
    ngram_range=(1, 2) # Considérer les unigrammes et bigrammes
)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_combined["text"].fillna(""))

# Convertir la matrice TF-IDF en DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

# Ajouter des caractéristiques supplémentaires
print("🔄 Ajout de caractéristiques supplémentaires...")

# Conversion du sentiment en variable numérique
sentiment_map = {"positif": 1, "neutre": 0, "négatif": -1}
df_combined["sentiment_num"] = df_combined["sentiment"].map(sentiment_map)

# Calculer la longueur du texte
df_combined["text_length"] = df_combined["text"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

# Définition améliorée des profils crypto pour une meilleure segmentation
profile_keywords = {
    'tech_developer': [
    # Langages & outils de dev
    'solidity', 'rust', 'python', 'javascript', 'react', 'typescript', 'go', 'vyper', 'cairo', 'ink!',

    # Environnement blockchain
    'smart contract', 'dapp', 'node', 'rpc', 'ipfs', 'subgraph', 'blockchain indexer', 'oracle', 'parachain', 'testnet',

    # Frameworks & outils
    'hardhat', 'truffle', 'ganache', 'ethers.js', 'web3.js', 'foundry', 'forge', 'thirdweb', 'scaffold-eth', 'chainlink',

    # Sécurité & performance
    'audit', 'reentrancy', 'gas', 'overflow', 'underflow', 'slither', 'echidna', 'mythril', 'formal verification', 'optimisation',

    # Concepts techniques
    'layer2', 'zk-rollup', 'sharding', 'validator', 'consensus', 'interoperability', 'cross-chain', 'zero-knowledge proofs', 'attestation ledger', 'MEV',
    
    # Technologies émergentes
    'modular blockchain', 'zero-knowledge EVM', 'account abstraction', 'AI-powered smart contracts', 'hybrid blockchain', 'tezos', 'hyperledger fabric', 'corda', 'substrate', 'cosmos SDK'
],

  'web3_believer': [
    # 🌱 Valeurs et engagement
    'liberté', 'autonomie', 'décentralisé', 'indépendance', 'résilience',
    'transparence', 'privacy', 'libre', 'open source', 'sans autorité', 'anti-banque',
    'souveraineté numérique', 'anti-censure', 'confiance distribuée', 'accessibilité', 'self-custody',

    # ⚖️ Idéologie et changement de système
    'nouvel ordre', 'révolution', 'système cassé', 'banques corrompues',
    'reprise de contrôle', 'contrôle citoyen', 'fin des intermédiaires', 'éthique',
    'débancarisation', 'débanqué', 'anti-establishment', 'money revolution', 'anti-GAFAM', 'économie régénérative',

    # 🗳 DAO et gouvernance participative
    'dao', 'vote', 'gouvernance', 'multisig', 'participatif', 'vote communautaire', 'communauté active',
    'quadratic voting', 'tokenholder', 'stakeholder', 'liquid democracy', 'permissionless', 'trustless governance',

    # 💡 Concepts Web3 populaires
    'smart contract', 'dapp', 'layer2', 'zk-rollup', 'interopérabilité', 'zero knowledge',
    'metaverse', 'digital identity', 'self-sovereign identity', 'web3 social', 'token-gated', 'soulbound tokens',

    # 🌍 Inclusion & impact social
    'inclusivité', 'unbanked', 'green mining', 'climat', 'responsabilité', 'empowerment', 'fair finance', 'impact social',
    'régénératif', 'ReFi', 'public goods', 'global south', 'accès universel', 'onboarding',

    # 🪙 Crypto idéaliste
    'privacy coin', 'monero', 'zk-snark', 'anonymat', 'sans trace', 'discrétion', 'anti-koyc',
    'censorship-resistant', 'non-custodial', 'peer-to-peer', 'unstoppable', 'résistant à la confiscation',

    # 🔗 Projets emblématiques Web3
    'Ethereum', 'Polkadot', 'Filecoin', 'Arweave', 'MakerDAO', 'Helium', 'Gnosis', 'Optimism', 'Radicle', 'ENS',
    'Aave', 'Gitcoin', 'Lens Protocol', 'POAP', 'Farcaster', 'Nouns', 'Lido', 'Worldcoin', 'Uniswap', 'zkSync',

    # 💬 Jargon communautaire et utopique
    'trustless', 'permissionless', 'futureproof', 'web3 is freedom', 'power to the people',
    'freedom tech', 'decentralize everything', 'fuck central banks', 'freedom chain',
    'WAGMI', 'frens', 'gm', 'builder', 'vibes', 'based', 'bullish AF', 'democratizing finance'
],

   'crypto_critic': [
    # Arnaques & fraudes
    'arnaque', 'scam', 'rug pull', 'ponzi', 'escroquerie',
    'pump and dump', 'exit scam', 'airdrop scam', 'phishing', 'fake project',

    # Pertes & crash
    'crash', 'bear market', 'dump', 'perte',
    'crypto winter', 'market collapse', 'liquidation', 'capitulation', 'REKTED',

    # Inquiétudes & critiques
    'manipulation', 'volatilité', 'instable', 'danger', 'fud',
    'déception', 'inutile', 'toxic', 'overvalued', 'hype bubble',
    'empty promises', 'not scalable', 'tether risk', 'hyperfinancialization',

    # Jargon fort
    'shitcoin', 'rekt', 'bullshit', 'arnaqués',
    'moonbois', 'degens', 'crypto bros', 'vaporware', 'scamcoin',

    # Légalité & sécurité
    'hack', 'piratage', 'illégal', 'fraude',
    'money laundering', 'terrorist financing', 'security breach', '51% attack', 'backdoor',

    # Impact social / environnement
    'pollution', 'énergie', 'unethical',
    'carbon footprint', 'energy waste', 'e-waste', 'unsustainable', 'inequality amplifier',

    # Sentiment général
    'hate', 'sceptique', 'mistrust', 'questionnable',
    'over-hyped', 'disillusionment', 'tulip mania', 'greater fool theory', 'collective delusion',
    
    # Régulation & contrôle
    'KYC obligatoire', 'taxe crypto', 'répression', 'interdiction', 'surveillance',
    'régulation stricte', 'centralisation réelle', 'oligopole minier', 'whale manipulation'
],
 'curious_finance': [
    # Questions & éducation
    'question', 'avis', 'explication', 'guide', 'débutant', 'aide', 'conseil',
    'tutoriel', 'formation', 'apprentissage', 'cours', 'comprendre', 'FAQ',

    # Stratégies & concepts simples
    'staking', 'diversification', 'take profit', 'stop loss', 'risk management', 'DCA',
    'portfolio balance', 'hedging', 'swing trading', 'dollar cost averaging', 'yield farming',

    # Rendements & indicateurs
    'rendement', 'ROI', 'APY', 'APR', 'profit', 'drawdown',
    'impermanent loss', 'TVL', 'collateral ratio', 'earnings', 'yield',

    # Produits & ordres de base
    'market order', 'limit order', 'futures', 'options', 'liquidity pool',
    'perpetual futures', 'spot trading', 'leverage', 'margin trading', 'trailing stop',

    # Tokens & marché
    'tokenomics', 'stablecoin', 'USDT', 'DAI', 'market cap', 'liquidité',
    'supply', 'circulating supply', 'inflation', 'deflation', 'burn mechanism',

    # Plateformes & protocoles connus
    'binance', 'coinbase', 'kraken', 'uniswap', 'aave', 'DeFi',
    'metamask', 'etherscan', 'opensea', 'pancakeswap', 'curve',

    # Fiscalité & frais
    'frais', 'fiscalité', 'KYC', 'compliance',
    'capital gains', 'tax reporting', 'gas fees', 'transaction costs', 'exchange fees',

    # Confiance & transparence
    'sécurité', 'réputation', 'transparence',
    'audit', 'cold storage', 'multi-sig', 'insurance fund', 'proof of reserves',
    
    # Analyse technique & indicators
    'chart pattern', 'support', 'resistance', 'RSI', 'MACD',
    'moving average', 'fibonacci', 'bollinger bands', 'volume analysis', 'order book',
    
    # Outils & services financiers  
    'crypto bot', 'trading bot', 'portfolio tracker', 'crypto card', 'lending',
    'borrowing', 'flash loan', 'synthetic assets', 'trading view', 'automated strategy'
]

}

# Pour compatibilité avec le code existant, mappage vers les anciens noms de variables
feature_mapping = {
    'tech_developer': 'score_dev_builder',
    'web3_believer': 'score_ideological',
    'curious_finance': 'score_curious_finance',
    'crypto_critic': 'score_hater_disappointed',
}





# Nettoyage dynamique des mots-clés des profils
for profile in profile_keywords:
    profile_keywords[profile] = [
        kw for kw in profile_keywords[profile]
        if kw not in neutral_keywords
    ]


# Calculer un score pour chaque profil
for profile, keywords in profile_keywords.items():
    feature_name = feature_mapping.get(profile, f"score_{profile}")
    df_combined[feature_name] = df_combined["text"].apply(
        lambda text: sum(1 for word in keywords if word in str(text).lower()) / len(keywords)
        if isinstance(text, str) and keywords else 0
    )


# Préparer les caractéristiques pour le clustering (maintenir la compatibilité)
features = [
    'polarite',
    'sentiment_num',
    'text_length',
    'score_dev_builder',
    'score_ideological',
    'score_curious_finance',
    'score_hater_disappointed'
]




# Standardiser les caractéristiques pour le K-means
print("🔄 Standardisation des caractéristiques...")
X = df_combined[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Déterminer le nombre optimal de clusters (méthode du coude)
print("🔍 Détermination du nombre optimal de clusters...")
inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Graphique de la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Nombre de clusters (k)', fontsize=12)
plt.ylabel('Inertie', fontsize=12)
plt.title('Méthode du coude pour déterminer k optimal', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_folder, "methode_coude.png"), dpi=300, bbox_inches='tight')
plt.close()  # Fermer la figure au lieu de l'afficher avec plt.show()

# Calcul du score de Silhouette
print("🔍 Calcul des scores de Silhouette...")
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    # Le score de Silhouette nécessite au moins 2 clusters et moins d'échantillons que le total
    if k > 1 and k < len(X_scaled):
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Pour k = {k}, le score de Silhouette est : {silhouette_avg:.3f}")
    else:
        silhouette_scores.append(0)

# Graphique des scores de Silhouette
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'bo-')
plt.xlabel('Nombre de clusters (k)', fontsize=12)
plt.ylabel('Score de Silhouette', fontsize=12)
plt.title('Score de Silhouette pour déterminer k optimal', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_folder, "silhouette_scores.png"), dpi=300, bbox_inches='tight')
plt.close()

# Après le graphique de la méthode du coude
print("Graphique de la méthode du coude terminé")

# Avant d'appliquer K-means
print("Application de K-means en cours...")

# 📍 Choix manuel du nombre de clusters
n_clusters = 5 # 🛠️ Tu peux changer cette valeur librement (entre 2 et 10 par exemple)
print(f"✅ Nombre de clusters choisi manuellement : {n_clusters}")


# Appliquer K-means avec ce k optimal
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_combined["cluster"] = kmeans.fit_predict(X_scaled)
clusters = df_combined["cluster"].values  # ✅ pour les visualisations




# Réduction de dimensionnalité pour visualisation
print("🔄 Réduction de dimensionnalité avec PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_combined["pca_x"] = X_pca[:, 0]
df_combined["pca_y"] = X_pca[:, 1]

# Analyser les clusters
print("🔍 Analyse des clusters...")
cluster_stats = []

# Liste complète incluant les anciens et nouveaux profils
all_profiles = list(profile_keywords.keys())

for i in range(n_clusters):
    cluster_data = df_combined[df_combined["cluster"] == i]
    size = len(cluster_data)
    sentiment_dist = cluster_data["sentiment"].value_counts(normalize=True).to_dict()
    
    # Déterminer les caractéristiques dominantes du cluster
    mean_features = cluster_data[features].mean()
    
    # Identifier le profil dominant (en utilisant le mappage inverse)
    profile_scores = {}
    for profile in all_profiles:
        feature_name = feature_mapping.get(profile, f"score_{profile}")
        if feature_name in mean_features:
            profile_scores[profile] = mean_features[feature_name]
    
    dominant_profile = max(profile_scores, key=profile_scores.get)
    
    # Extraire les termes les plus fréquents
    cluster_texts = " ".join(cluster_data["text"].fillna(""))
    common_words = Counter(cluster_texts.split()).most_common(10)
    
    # Stocker les statistiques
    cluster_stats.append({
        'cluster': i,
        'size': size,
        'percent': size / len(df_combined) * 100,
        'sentiment': sentiment_dist,
        'dominant_profile': dominant_profile,
        'profile_scores': profile_scores,
        'common_words': common_words,
        'mean_features': mean_features
    })

# Noms clairs pour les profils retenus
profile_names = {
    'tech_developer': 'Développeurs',
    'trader_investor': 'Investisseurs',
    'web3_believer': 'Idéalistes',
    'crypto_critic': 'Sceptiques',
    'curious_tech': 'Curieux techno',
    'curious_finance': 'Curieux finance'
}



cluster_names = {}
for stat in cluster_stats:
    cluster_names[stat['cluster']] = profile_names.get(stat['dominant_profile'], f"Cluster {stat['cluster']}")

# Visualisation détaillée de Silhouette pour le nombre de clusters choisi
print(f"🔄 Visualisation de Silhouette pour {n_clusters} clusters...")

# Calcul des coefficients de Silhouette pour chaque échantillon
silhouette_vals = silhouette_samples(X_scaled, clusters)
df_combined["silhouette_val"] = silhouette_vals

# Créer un graphique de visualisation de Silhouette
plt.figure(figsize=(12, 8))
y_lower, y_upper = 0, 0

# Palette de couleurs pour les clusters
silhouette_colors = cm.nipy_spectral(np.array(range(n_clusters)) / n_clusters)

# Boucle corrigée : affichage des vrais noms de profils
for cluster_id in sorted(df_combined["cluster"].unique()):
    cluster_silhouette_vals = silhouette_vals[clusters == cluster_id]
    cluster_silhouette_vals.sort()
    
    y_upper += len(cluster_silhouette_vals)
    
    # Utiliser les vrais noms : "Curieux", "Sceptiques", etc.
    cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id + 1}")
    
    plt.barh(
        range(y_lower, y_upper),
        cluster_silhouette_vals,
        height=1.0,
        edgecolor='none',
        color=silhouette_colors[cluster_id]
    )
    
    plt.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), cluster_label, fontsize=10)
    
    y_lower += len(cluster_silhouette_vals)

# Score de Silhouette moyen global
silhouette_avg = silhouette_score(X_scaled, clusters)
plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
            label=f'Score moyen: {silhouette_avg:.3f}')

plt.title('Visualisation des coefficients de Silhouette par cluster', fontsize=16)
plt.xlabel('Coefficient de Silhouette', fontsize=12)
plt.ylabel('Cluster', fontsize=12)
plt.legend(loc='best')
plt.yticks([])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "silhouette_detail.png"), dpi=300, bbox_inches='tight')
plt.close()

# Sauvegarder les résultats
print("💾 Sauvegarde des résultats...")
df_combined["cluster_name"] = df_combined["cluster"].map(cluster_names)

# Export vers CSV
df_combined.to_csv(os.path.join(results_folder, "crypto_clustersREDIT.csv"), index=False)

# Export vers SQLite
conn_results = sqlite3.connect(os.path.join(results_folder, "crypto_clustersREDIT.db"))
df_combined.to_sql("clustering_results", conn_results, if_exists="replace", index=False)
conn_results.close()

# Visualisation des clusters (graphique de dispersion PCA)
plt.figure(figsize=(12, 8))

# Palette de couleurs pour les clusters
colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#7B68EE', '#FF7F50', '#9370DB', '#3CB371', '#CD5C5C', '#4682B4']

for i in range(n_clusters):
    cluster_data = df_combined[df_combined["cluster"] == i]
    plt.scatter(
        cluster_data["pca_x"], 
        cluster_data["pca_y"], 
        alpha=0.7,
        s=50, 
        label=f"{cluster_names[i]} ({len(cluster_data)})",
        c=colors[i % len(colors)]
    )

plt.title("Visualisation des clusters de profils crypto", fontsize=16)
plt.xlabel(f"Composante principale 1 ({pca.explained_variance_ratio_[0]:.2%} de variance)", fontsize=12)
plt.ylabel(f"Composante principale 2 ({pca.explained_variance_ratio_[1]:.2%} de variance)", fontsize=12)
plt.legend(fontsize=10, loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "clusters_pca.png"), dpi=300, bbox_inches='tight')
plt.close()

# Configuration des graphiques
plt.style.use('ggplot')  # alternative intégrée par défaut
plt.rcParams.update({
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# 1. Méthode du coude simplifiée
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
plt.title('Optimisation du nombre de clusters\n(Méthode du coude)', pad=15)
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie intra-classe')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "1_methode_coude.png"), dpi=120)
plt.close()

# 2. Score de Silhouette simplifié
plt.figure(figsize=(10, 5))
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.title('Qualité des clusters\n(Score de Silhouette)', pad=15)
plt.xlabel('Nombre de clusters')
plt.ylabel('Score moyen')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "2_silhouette_score.png"), dpi=120)
plt.close()

# 3. Visualisation 2D des clusters avec noms
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    cluster_data = df_combined[df_combined["cluster"] == i]
    plt.scatter(
        cluster_data["pca_x"], 
        cluster_data["pca_y"],
        s=40,
        alpha=0.7,
        label=f"{cluster_names[i]} ({len(cluster_data)} membres)",
        c=colors[i % len(colors)]
    )

plt.title('Répartition des profils crypto\n(Projection PCA 2D)', pad=15)
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "3_clusters_2d.png"), dpi=120, bbox_inches='tight')
plt.close()

# 4. Diagramme des caractéristiques par cluster
plt.figure(figsize=(14, 7))
cluster_features = df_combined.groupby('cluster_name')[features].mean()
cluster_features.plot(kind='bar', width=0.8, cmap='viridis')
plt.title('Caractéristiques moyennes par type de profil', pad=15)
plt.xlabel('Type de profil crypto')
plt.ylabel('Score moyen')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "4_caracteristiques_clusters.png"), dpi=120, bbox_inches='tight')
plt.close()

# 5. Nuages de mots par cluster
for i in range(n_clusters):
    plt.figure(figsize=(10, 5))
    cluster_text = " ".join(df_combined[df_combined["cluster"] == i]["text"])
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=30
    ).generate(cluster_text)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Mots clés - {cluster_names[i]}", pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"5_mots_cles_cluster_{i}.png"), dpi=120)
    plt.close()
# 🔍 Résumé qualitatif automatique des clusters
print("🧾 Génération du résumé qualitatif par cluster...")

summary_lines = []
summary_lines.append("📊 RÉSUMÉ QUALITATIF DES CLUSTERS WEB3")
summary_lines.append("==================================================\n")

for stat in cluster_stats:
    cluster_id = stat['cluster']
    cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    sentiment_counts = stat['sentiment']
    dominant_profile = profile_names.get(stat['dominant_profile'], stat['dominant_profile']).capitalize()
    common_words = ", ".join([word for word, _ in stat['common_words']])
    
    summary_lines.append(f"🔹 Cluster {cluster_id + 1} : {cluster_name}")
    summary_lines.append(f"   - Type dominant : {dominant_profile}")
    summary_lines.append(f"   - Taille : {stat['size']} personnes ({stat['percent']:.2f}%)")
    summary_lines.append("   - Répartition des sentiments :")
    for sent, pct in sentiment_counts.items():
        summary_lines.append(f"     • {sent.capitalize()} : {pct*100:.1f}%")
    summary_lines.append(f"   - Mots clés fréquents : {common_words}")
    summary_lines.append("")

# Écrire dans un fichier .txt dans le dossier des résultats
summary_path = os.path.join(results_folder, "cluster_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"📁 Résumé qualitatif sauvegardé dans : {summary_path}")
