import numpy as np
import os
import faiss
import csv
from rank_bm25 import BM25Okapi

def get_chunks_from_dir(directory, chunk_size = 500, overlap = 50):

    chunks = []
    chunk_sources = []  

    documents = os.listdir(directory)

    for doc in documents:
        with open(directory+"/"+doc) as f:
            text = f.read()
        
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
            chunk_sources.append(doc)

    return(chunks,chunk_sources)

def search_similar_chunks(embed_model, index, chunks, chunk_sources, query, k=3):
    embedded_querry = embed_model.encode([query])
    distance , idx = index.search(np.array(embedded_querry), k)
    idx = idx[0]
    
    return [(chunks[i], chunk_sources[i]) for i in idx]

def make_rag_data(directory, embed_model,name ="",method= "classic"):
    if method == "classic":
        chunks, chunk_sources = get_chunks_from_dir("data")
    elif method == "varying":
        chunks, chunk_sources = get_chunk_from_folder("data")#jsp comment nomé les deux fonctions ;(
    else:
        print("erreur de param")
        return(0)

    index = get_index_from_chunks(embed_model,chunks)


    with open("rag_data/"+name+"chunks_data.chunks", 'w') as f:
        wr = csv.writer(f)
        wr.writerow(chunks)
        wr.writerow(chunk_sources)

    faiss.write_index(index, "rag_data/"+name+"chunks.index")

def get_index_from_chunks(embed_model,chunks):
    chunk_embeddings = embed_model.encode(chunks)

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)    
    index.add(chunk_embeddings)
    return(index)

def retrieve_data(name = ""):
    index = faiss.read_index("rag_data/"+name+"chunks.index")
    with open("rag_data/"+name+"chunks_data.chunks", 'r') as f:
        reader = csv.reader(f)
        chunks = next(reader)
        chunk_sources = next(reader)
    return chunks, chunk_sources, index


def rrf_score(rank, n = 20):
        return 1 / (n + rank)

def rank_fusion(list_1,list_2, k=3, n=20):

    scores = {}

    for rank, idx in enumerate(list_1):
        scores[idx] = scores.get(idx, 0) + rrf_score(rank)

    for rank, idx in enumerate(list_2):
        scores[idx] = scores.get(idx, 0) + rrf_score(rank)

    ranked = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    return ranked[:k]

def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

def hybrid_search(embed_model, index, chunks, chunk_sources, query, k=3, n = 20):
    embedded_querry = embed_model.encode([query])
    distance , idx1 = index.search(np.array(embedded_querry), n)
    idx1 = idx1[0]
    idx1 = [ int(x) for x in idx1 ]

    tokenized_corpus = [chunk.split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    idx2 = sort_index(bm25_scores)[:n]

    idx = rank_fusion(idx1,idx2)
    return [(chunks[i], chunk_sources[i]) for i in idx]


def split_text(file):
    with open("data/"+file, newline="") as f:
        text = f.read()
        chunks = text.split("\n\n")
    return chunks

def get_chunks_from_txt(file):
    chunks = split_text(file)
    chunk_sources = [file]*len(chunks)
    return(chunks, chunk_sources)

def get_chunks_from_csv(file,delimiter = ';'):
    chunks = []
    chunk_sources = []

    with open("data/"+file, newline="") as f:
        reader = csv.DictReader(f,delimiter = delimiter)

        for row_id, row in enumerate(reader):
            text = []
            for col, val in row.items():
                try:
                    text.append(col+": "+val)
                except:
                    print("erreur à la ligne : \n"+row)
            chunk = ", ".join(text)
            chunks.append(chunk)
            chunk_sources.append(file)
    return(chunks, chunk_sources)

def get_chunk_from_folder(folder):
    chunks = []
    chunk_sources = []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            new_chunks, new_chunk_sources = get_chunks_from_txt(file)
        elif file.endswith(".csv"):
            new_chunks, new_chunk_sources = get_chunks_from_csv(file)
        else:
            print("fichier non reconnu")
            new_chunks = []
            new_chunk_sources = []
        chunks += new_chunks
        chunk_sources += new_chunk_sources
    return(chunks, chunk_sources)
