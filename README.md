Les données utiliser par le RAG se trouve dans le dossier data
Le dossier model contient les modèles nécessaire au RAG : embbeding et LLM
rag_data contient les données sous forme vectorielle ainsi que les chunks de data.

Le fichier embedding.py contient toute les fonctions relative à l'embbeding des data
main.py contient les fonction principale permettant de faire tourner rag

embedding.ipynb permet de télécharger le modèle nécessaire au embeding et de générer les embeding à partir de data et de les enregistrer sur le disque.
generative.ipynb permet de télécharger un LLM avec huggingface
main.ipynb permet de de lancer le modèle rag et d'effectuer des test