import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================================
# ÉTAPE 1 : Préparation
# ==========================================
def preparer_documents():
    df = pd.read_csv(r"data\tmdb_5000_movies.csv")
    documents = []

    for index, row in df.iterrows():
        genres_list = json.loads(row['genres'])
        genres_text = ", ".join([g['name'] for g in genres_list])
        contenu = f"Titre: {row['title']}. Genre: {genres_text}. Synopsis: {row['overview']}"

        doc = {
            "id": f"doc_{index:03d}",
            "contenu": contenu,
            "metadata": {
                "source": "tmdb_5000_movies.csv",
                "titre": row['title'],
                "annee": str(row['release_date'])[:4] if pd.notnull(row['release_date']) else "Inconnue",
                "note": row['vote_average'],
                "genres": genres_text,
                "langue": row['original_language']
            }
        }
        documents.append(doc)
    return documents

# ==========================================
# ÉTAPE 2 : Chunking
# ==========================================
def chunker(texte: str, taille_max: int = 400, overlap: int = 50) -> list[str]:
    chunks = []
    debut = 0
    longueur = len(texte)

    while debut < longueur:
        fin = debut + taille_max
        if fin >= longueur:
            chunks.append(texte[debut:].strip())
            break

        dernier_point = texte.rfind('. ', debut, fin)
        dernier_espace = texte.rfind(' ', debut, fin)

        if dernier_point != -1 and (fin - dernier_point) < 150:
            index_coupe = dernier_point + 1
        elif dernier_espace != -1:
            index_coupe = dernier_espace
        else:
            index_coupe = fin

        chunk = texte[debut:index_coupe].strip()
        chunks.append(chunk)
        debut = index_coupe - overlap
    return chunks

def preparer_corpus_complet(documents: list, taille_max: int = 400, overlap: int = 50):
    chunks_avec_meta = []
    textes_a_embedder = []
    
    for doc in documents:
        morceaux = chunker(doc["contenu"], taille_max, overlap)
        for morceau in morceaux:
            chunks_avec_meta.append({
                "contenu": morceau,
                "metadata": doc["metadata"] 
            })
            textes_a_embedder.append(morceau)
    return chunks_avec_meta, textes_a_embedder

# ==========================================
# ÉTAPE 3 : Embeddings
# ==========================================
def embedder_chunks(chunks: list[str], modele: SentenceTransformer) -> np.ndarray:
    print(f"⏳ Encodage de {len(chunks)} chunks en cours (cela peut prendre quelques minutes)...")
    vecteurs = modele.encode(chunks, show_progress_bar=True)
    return vecteurs

# ==========================================
# ÉTAPE 4 : Création et Sauvegarde FAISS
# ==========================================
def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    vecteurs_float32 = vecteurs.astype(np.float32)
    index.add(vecteurs_float32)
    return index

def sauvegarder_index(index, chunks_avec_meta: list, chemin_base: str):
    chemin_faiss = f"{chemin_base}.faiss"
    faiss.write_index(index, chemin_faiss)
    
    chemin_meta = f"{chemin_base}.json"
    with open(chemin_meta, 'w', encoding='utf-8') as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=4)
    print(f"💾 Base sauvegardée avec succès ! ({index.ntotal} éléments)")

# ==========================================
# EXÉCUTION DU SCRIPT 
if __name__ == "__main__":
    print("🚀 DÉMARRAGE DE L'INDEXATION DES FILMS...")
    
    print("\n1️⃣ Chargement et préparation du fichier CSV...")
    docs = preparer_documents()
    
    print("\n2️⃣ Découpage des textes (Chunking)...")
    chunks_avec_meta, textes = preparer_corpus_complet(docs)
    
    print("\n3️⃣ Chargement du modèle d'Intelligence Artificielle...")
    modele = SentenceTransformer("all-mpnet-base-v2")
    
    print("\n4️⃣ Transformation des textes en vecteurs mathématiques...")
    vecteurs = embedder_chunks(textes, modele)
    
    print("\n5️⃣ Création et sauvegarde de la base de données FAISS...")
    index = creer_index_faiss(vecteurs)
    sauvegarder_index(index, chunks_avec_meta, "base_films")
    
    print("\nINDEXATION TERMINÉE !")