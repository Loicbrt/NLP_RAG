# RAG Avancé (Retrieval-Augmented Generation)

Le RAG est une technique rajoutant un système de recherche d'information à un LLM. Ce système permet à un LLM d'avoir accès à des données qui n'étaient pas présente lors de son entraînement. Pour ce faire, lorsque le modèle reçoit un prompt, il cherche d'abord dans la base de données, les données les plus pertinentes puis, à partir de celle-ci génère une réponse.
Ce système permet à un modèle de sourcer des documents et d'utiliser des données dont il n'a pas accès lors de l'entraînement, notamment des données futur.

Pour ce projet, nous avons décidé de créer une base de données sur le jeu DS Pokémon Diamant et Perle, le but étant de créer un chatbot pouvant répondre à des questions précise sur le jeu. Pour cela, nous avons recueilli un grand nombre de données variées sur internet portant sur le jeu.


Technique utilisée.

Les jeux de données sont découpés en morceaux appelés chunks. C'est chunks sont ensuite converti en vecteur par un modèle d'embedding. 
Lorsque que l'utilisateur envoie son prompt, on converti en vecteur le prompt et l'on choisit les chunks qui sont le plus proche vectoriellement du prompt. On envoie ensuite ces chunks dans le LLM.

Nous avons utilisé transformers pour télécharger des modèles, openai pour utiliser le LLM gpt-5-nano, sentence_transformers pour importer le modèle d'embedding.

Le RAG a de nombreux cas d'usage. Cela permet par exemple à une IA de générer du texte avec des sources, de chercher des données sur des databases, voir sur Internet pour les plus grands modèles. Au sein d'une entreprise, avec des données protégées, cela permet à un LLM d'avoir accès à ces données sans avoir à entraîner de modèle dessus. 

Pipeline

Les données utilisées par le RAG se trouvent dans le dossier data.
Le dossier model contient les modèles nécessaires au RAG : embedding et LLM
rag_data contient les données sous forme vectorielle ainsi que les chunks de data.

Le fichier embedding.py contient toutes les fonctions relatives à l'embedding des data.
main.py contient les fonctions principales permettant de faire tourner rag.

embedding.ipynb permet de télécharger le modèle nécessaire au embedding et de générer les embedding à partir de data et de les enregistrer sur le disque.
generative.ipynb permet de télécharger un LLM avec huggingface.
main.ipynb permet de lancer le modèle rag et d'effectuer des tests.

Nous avons implémenté la recherche hybride et une chanking stratégie pour essayer de remédier au fait que l'on n'arrivait pas à trouver les chunks de data les plus pertinent

Choix technique :

Nous utilisons un pc très peu puissant et qui ne possède pas de gpu, nous avons donc opté pour des solutions peu gourmandes.

Nous avons utilisé all-MiniLM-L6-v2, pour vectoriser nos données. L'embedding a cependant était assez rapide, un modèle plus évolué aurait pu être considéré.

Nous avons utilisé faiss pour indexer les vecteurs, car c'était une bibliothèque très connue, qui fait le travail.

Nous avons d'abord télécharger le modèle mistralai/Mistral-7B-Instruct-v0.3 en local pour ne pas payer de clé api. Cependant même avec seulement 7B de paramètre, le modèle prend 5 + min pour répondre à un prompt. Nous avons ensuite utilisé une api pour accéder à  gpt-5-nano, plus rapide et performant.


Exécution : 

Pour installer les librairies : pip install -r requirements.txt

Il faut avoir une clé api openai dans les variable locale de l'ordinateur pour utiliser openai ou télécharger le modèle mistral avec le fichier jupyter generative.ipynb

Installer 

all-MiniLM-L6-v2 et générer les vecteurs dans embedding.ipynb (le vecteurs sont déjà enregistrés dans le repos git, donc il n'y en a pas vraimet besoin sauf si vous rajouté des données)

lancer depuis le fichier main.ipynb le modèle rag, il y'a plusieurs exemple d'utilisation.


Résultat :

Le jeu de données est encore très faible, et le rag a beaucoup de mal à trouver les chunks les plus pertinents. L'ajout du rag est négligeable sur le résultat, voir détrimentaire. Nos extensions n'ont pas non plus apporté d'amélioration significative.

Un meilleur chunking et des métadonnées auraient pu améliorer les résultats. 