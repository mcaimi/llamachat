#!/usr/bin/env/python

try:
    from llama_stack_client import LlamaStackClient, RAGDocument
    from libs.shared.settings import Properties
    import mimetypes as mt
    import pdfplumber
except ImportError as e:
    print(f"Caught Exception: {e}")
            
# create or register a collection in the vector db
def registerVectorCollection(embedClient: LlamaStackClient,
                            vectorDbId: str,
                            embeddingModel: str,
                            embeddingDim: int,
                            providerId: str) -> None:
    # call LlamaStack
    embedClient.vector_dbs.register(
        vector_db_id=vectorDbId,
        embedding_model=embeddingModel,
        embedding_dimension=embeddingDim,
        provider_id=providerId,
    )

def embedDocuments(embedClient: LlamaStackClient,
                   ragDocs: list,
                   vectorDbId: str,
                   chunkSize: int,
                   timeout: int) -> None:
    embedClient.tool_runtime.rag_tool.insert(
        documents=ragDocs,
        vector_db_id=vectorDbId,
        chunk_size_in_tokens=chunkSize,
        timeout=timeout,
    )

def prepareDocuments(uploaded_files: list) -> list:
    list_of_docs = []
    for i, ufile in enumerate(uploaded_files):
        mtype = mt.guess_type(ufile.name)[0]
        match mtype:
            case "application/pdf":
                with pdfplumber.open(ufile) as pdf:
                    file_contents = ''
                    for page in pdf.pages:
                        file_contents += page.extract_text()

                metadata = {"name": f"{ufile.name}", "mimetype": {mtype}}
            case "text/plain":
                file_contents = ufile.read().decode("utf-8")
                metadata = {"name": f"{ufile.name}", "mimetype": {mtype}}
            
        list_of_docs.append(RAGDocument(
            document_id=f"rag_document_{i}",
            content=file_contents,
            mime_type=mtype,
            metadata=metadata,
        ))

    # return data
    return list_of_docs