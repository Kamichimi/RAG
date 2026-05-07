# Projet RAG - Assistant de Recommandation de Films

Ce projet implémente un système de RAG (Retrieval-Augmented Generation) permettant d'interroger une base de données de 5 000 films via une interface conversationnelle intelligente.

## Architecture du Système

Le projet est divisé en deux phases distinctes pour optimiser les performances et les coûts d'API.

PHASE 1 : INDEXATION 
 (à exécuter une fois) 
                                                         
  Données brutes (CSV) → Nettoyage → Chunking → Embedding 
                          ↓                              
                     Base FAISS                          

                            ↓

 PHASE 2 : INTERROGATION 
 (à chaque question)     
                                                         
  Question → Embedding → Recherche vectorielle            
                          ↓                              
                    Top-k chunks                         
                          ↓                              
                  LLM Groq + Contexte                    
                          ↓                              
                 Réponse avec sources                    


## 🛠 Outils et Technologies
- **Groq API** : Moteur d'inférence ultra-rapide (Modèle Llama 3.3 70B).
- **Sentence-Transformers** : Modèle `all-mpnet-base-v2` pour transformer le texte en vecteurs.
- **FAISS** : Bibliothèque de Facebook pour la recherche de similarité vectorielle.
- **Pandas** : Manipulation des données du dataset TMDB.

## Questions & Réponses de Conception

Voici les choix techniques effectués durant le développement :

**Q1. Comment convertir les données tabulaires (CSV) en texte cohérent ?**
**R1.** J'utilise un "template" qui fusionne le titre, le genre et le synopsis (overview) en un seul paragraphe descriptif. Cela donne plus de contexte au modèle d'embedding qu'une simple liste de mots.

**Q2. Comment extraire les genres qui sont au format JSON imbriqué ?**
**R2.** J'utilise la bibliothèque `json` de Python pour parser la chaîne, puis une compréhension de liste pour extraire uniquement les noms des genres (ex: "Action, Drama") afin de simplifier le texte.

**Q3. Comment éviter de relancer l'indexation (très lente) à chaque test ?**
**R3.** J'ai implémenté une stratégie de persistance. L'index mathématique est sauvegardé dans un fichier `.faiss` et les métadonnées textuelles dans un fichier `.json`. Le système les recharge en quelques millisecondes.

**Q4. Comment guider le LLM pour faire des recommandations pertinentes ?**
**R4.** Via un "Prompt Système" (stocké dans `contexte.txt`). On lui assigne le rôle d'expert cinéma et on lui impose des contraintes strictes : ne pas inventer de films, citer l'année et la note, et expliquer le choix.

**Q5. Que faire si l'utilisateur demande un film récent (ex: 2024) absent de la base ?**
**R5.** Le prompt système contient une consigne de transparence. Si l'information n'est pas dans le contexte, le LLM doit l'indiquer au lieu d'halluciner, tout en proposant des alternatives proches présentes dans la base.

**Q6. Pourquoi avoir choisi le modèle 'all-mpnet-base-v2' ?**
**R6.** Puisque le dataset TMDB est majoritairement en anglais, ce modèle spécialisé offre une bien meilleure précision sémantique qu'un modèle multilingue pour capturer les nuances des synopsis.

**Q7. Quelle stratégie de Chunking a été adoptée ?**
**R7.** J'ai les textes par blocs de 400 caractères avec un chevauchement (overlap) de 50 caractères. Le découpage se fait intelligemment à la fin des phrases pour préserver le sens.

## 📂 Structure des fichiers
- `indexation.py` : Script à lancer pour préparer la base.
- `rag.py` : Script principal pour lancer l'assistant (interface CLI).
- `contexte.txt` : Fichier de configuration du comportement de l'IA.
- `.env` : Contient la clé API GROQ_API_KEY.
