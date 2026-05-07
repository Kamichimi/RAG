import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq



def charger_index(chemin_base: str):
    index = faiss.read_index(f"{chemin_base}.faiss")
    with open(f"{chemin_base}.json", 'r', encoding='utf-8') as f:
        chunks_avec_meta = json.load(f)
    return index, chunks_avec_meta

def lire_fichier_contexte(chemin: str) -> str:
    try:
        with open(chemin, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ Fichier {chemin} introuvable. Utilisation d'un prompt par défaut.")
        return "Tu es un assistant utile. Réponds uniquement avec le contexte fourni."

def rechercher(question: str, modele, index, chunks_avec_meta: list, k: int = 4) -> list[dict]:
    vecteur_question = np.array(modele.encode([question]), dtype=np.float32)
    distances, indices = index.search(vecteur_question, k)
    
    resultats = []
    for i in range(k):
        idx = indices[0][i]
        chunk_trouve = chunks_avec_meta[idx]
        resultats.append(chunk_trouve)
    return resultats

def generer_reponse(client, question: str, chunks_pertinents: list, prompt_systeme: str) -> str:
    contexte_texte = ""
    for i, chunk in enumerate(chunks_pertinents):
        meta = chunk['metadata']
        contexte_texte += f"\n--- Film {i+1} ---\n"
        contexte_texte += f"Titre: {meta['titre']} ({meta['annee']}) | Note: {meta['note']}/10\n"
        contexte_texte += f"Genres: {meta['genres']}\n"
        contexte_texte += f"Synopsis: {chunk['contenu']}\n"

    message_systeme = f"{prompt_systeme}\n\nVOICI LE CONTEXTE DES FILMS :\n{contexte_texte}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": message_systeme},
            {"role": "user", "content": f"Voici ma demande : {question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Erreur : GROQ_API_KEY introuvable dans le fichier .env")
        return

    print("⏳ Chargement de la base de connaissances et des modèles...")
    
    # 2. Chargement des composants
    client = Groq(api_key=api_key)
    prompt_systeme = lire_fichier_contexte("contexte.txt")
    
    try:
        index, chunks_avec_meta = charger_index("base_films")
        modele = SentenceTransformer("all-mpnet-base-v2") 
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la base : {e}")
        print("👉 As-tu bien exécuté ton script d'indexation d'abord ?")
        return

    os.system('cls' if os.name == 'nt' else 'clear') 
    print("="*60)
    print("🍿 ASSISTANT CINÉMA RAG OPÉRATIONNEL 🍿")
    print("="*60)
    print(f"✅ {len(chunks_avec_meta)} chunks de films chargés en mémoire.")
    print("💡 Tapez 'quit', 'exit' ou 'q' pour quitter.\n")

    while True:
        question = input("👤 Vous : ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Au revoir et bons films !")
            break
        if not question:
            continue

        print("🤖 L'assistant réfléchit (Recherche FAISS + Groq)...\n")
        
        resultats = rechercher(question, modele, index, chunks_avec_meta, k=4)
        
        reponse = generer_reponse(client, question, resultats, prompt_systeme)
        
        print("-" * 60)
        print("🎬 La recommandation est :")
        print(reponse)
        print("-" * 60)
        
        print("\n🔍 Sources consultées :")
        for res in resultats:
            titre = res['metadata']['titre']
            annee = res['metadata']['annee']
            print(f"   - {titre} ({annee})")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()