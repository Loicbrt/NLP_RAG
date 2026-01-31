from embedding import * 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI


def rag(query,model = False, search = "classic",vector_name = ""):

    if(model == False):
        embed_model = torch.load("model/embed_model_all-MiniLM-L6-v2", weights_only=False)
    else:
        embed_model = model
    
    chunks, chunk_sources, index = retrieve_data(vector_name)

    if search == "classic":
        retrieved_chunks = search_similar_chunks(embed_model, index,chunks, chunk_sources,query)
    elif search == "hybrid":
        retrieved_chunks = hybrid_search(embed_model, index,chunks, chunk_sources,query)
    else:
        print("invalid search parameter")
        return(0)
    
    context= "\n\n".join(retrieved_chunks[0])

    prompt = f"""Réponds à la question en t'appuyant sur les documents venant de la database interne ci dessous, ci les documents ne contiennent pas d'informations pertinent précise qu'il n'existe pas de document dans la database permettant de répondre de manière pertinante:

    Documents :
    {context}

    Question :
    {query}

    Réponse :
    """ 

    return(prompt, retrieved_chunks)

def professeur_Chen(query,LLM = "mistral",model = False, search = "classic", vector_name = ""):

    prompt, retrieved_chunks = rag(query,model,search,vector_name)
 

    if LLM == "mistral":
        tokenizer = AutoTokenizer.from_pretrained("model/mistral")
        model = AutoModelForCausalLM.from_pretrained("model/mistral")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif LLM == "openai":
        client = OpenAI()
        outputs = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        ).output_text
    else:
        print("Le modèle LLM fourni n'est pas bon")
        return(0)
    return(outputs, retrieved_chunks)
