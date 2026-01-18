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
    embedClient.vector_stores.register(
        vector_db_id=vectorDbId,
        vector_db_name=vectorDbId,
        embedding_model=embeddingModel,
        embedding_dimension=embeddingDim,
        provider_id=providerId,
    )

# get vdb id by name
def getVDBByName(embedClient: LlamaStackClient, vdb_name: str) -> str:
    dbs: list = [v.identifier for v in embedClient.vector_stores.list() if v.vector_db_name == vdb_name]

    # check...
    if len(dbs) > 1:
        raise Exception(f"{vdb_name} is declared in multiple entries: Alias Error")
    else:
        return dbs[0]