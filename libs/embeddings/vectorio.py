#!/usr/local/env python

try:
    from llama_stack_client import LlamaStackClient
except Exception as e:
    print(f"Caught fatal exception: {e}") 

# create or register a collection in the vector db
def registerVectorCollection(
    embedClient: LlamaStackClient,
    vectorDbId: str,
    embeddingModel: str,
    embeddingDim: int,
    providerId: str,
) -> None:
    # call LlamaStack
    embedClient.vector_stores.create(
        name=vectorDbId,
        extra_body={
            "embedding_model": embeddingModel,
            "embedding_dimension": embeddingDim,
            "provider_id": providerId,
        },
    )

# get vdb id by name
def getVdbIdByName(embedClient: LlamaStackClient, vdb_name: str) -> str:
    dbs: list = [v.id for v in embedClient.vector_stores.list().data if v.name == vdb_name]

    # check...
    if len(dbs) > 1:
        raise Exception(f"{vdb_name} is declared in multiple entries: Alias Error")
    else:
        return dbs[0]

# compute embeddings from a list of string inputs
def computeEmbeddings(embedClient: LlamaStackClient, inputList: list, vdb_name: str):
    if len(inputList) == 0:
        return []

    vector_db = [v for v in embedClient.vector_stores.list().data if v.name == vdb_name]
    output_list = []

    if len(vector_db) > 1:
        raise Exception(f"{vdb_name} is declared in multiple entries")
    else:
        embedding_model = vector_db[0].metadata.get("embedding_model")
        embedding_dimension = vector_db[0].metadata.get("embedding_dimension")

    # compute embeddings
    for item in inputList:
        # call embeddings api from llama stack
        chunk_embeddings = embedClient.embeddings.create(
            input=[item.get("content")],
            model=embedding_model,
            dimensions=embedding_dimension
        )
        
        # insert computed embedding into chunk object
        item["embedding"] = chunk_embeddings.data[0].embedding
        item["embedding_model"] = embedding_model
        item["embedding_dimension"] = embedding_dimension

        # fill object list
        output_list.append(item)

    # return objects
    return output_list

