#!/usr/bin/env python

try:
    import streamlit as st
    import mimetypes as mt
    from llama_stack_client import LlamaStackClient
    from dotenv import dotenv_values
    from libs.shared.settings import Properties
    from libs.shared.session import Session
    from libs.embeddings.embeddings import registerVectorCollection, embedDocuments, prepareDocuments
except ImportError as e:
    print(f"Caught Exception: {e}")

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# load session
stSession = Session(st.session_state)
stSession.add_to_session_state("api_base_url", appSettings.config_parameters.openai.default_local_api)

# build streamlit UI
st.set_page_config(page_title="🧠 Embeddings", initial_sidebar_state="collapsed", layout="wide")
st.html("assets/embeddings.html")

# llamastack client
embedClient = LlamaStackClient(base_url=stSession.session_state.api_base_url)

# file uploader
st.subheader("Upload Documents", divider=True)
uploaded_files = st.file_uploader(
            "Upload file(s) or directory",
            accept_multiple_files=True,
            type=["txt", "pdf", "md"],  # Add more file types as needed
)

if uploaded_files:
    st.success(f"Successfully uploaded {len(uploaded_files)} files")

    # providers
    providers = embedClient.providers.list()
    vector_io_provider = st.selectbox(label="Vector Providers", options=[p.provider_id for p in providers if p.api=="vector_io"])

    # embedding model
    embedding_model_name = st.selectbox(label="Available embedding models", options=stSession.list_models(model_type="embedding"))

    # Add memory bank name input field
    vector_db_mode = st.radio("Select Memory Bank For Embedding", ["**Existing Collection**", "**New Collection**"],
                              captions=["Embed Documents in a Pre Existing Collection", "Register a new Memory Bank in the Vector DB"])

    if vector_db_mode == "**Existing Collection**":
        vector_dbs = embedClient.vector_dbs.list() or []
        if not vector_dbs:
            st.info("No vector databases available for selection.")
        else:
            vector_dbs = [vector_db.identifier for vector_db in vector_dbs]
            vector_db_name = st.multiselect(
                label="Select Document Collections to use in RAG embeddings",
                options=vector_dbs, max_selections=1)
    else:
        # add new memory bank
        vector_db_name = st.text_input(
            "Document Collection Name",
            value="rag_vector_db",
            help="Enter a unique identifier for this document collection",
        )

    if st.button("Embed Documents...."):
        # documents to embed:
        rag_docs = []
        with st.spinner("**Loading Documents...**"):
            rag_docs = prepareDocuments(uploaded_files)

        st.markdown(f"Loaded **{len(rag_docs)}** documents into the ingestion pipeline. Ready to embed...")

        # embed documents!
        if isinstance(vector_db_name, list):
            vector_db_id = vector_db_name[0]
        else:
            vector_db_id = vector_db_name

        # create new collection if necessary
        vector_dbs = embedClient.vector_dbs.list() or []
        if len(vector_dbs) == 0 or vector_db_id not in [v.identifier for v in vector_dbs]:
            # create vector db on provider
            st.markdown(f"**Creating new Collection {vector_db_id} on the vdb...**")
            registerVectorCollection(embedClient=embedClient,
                                    vectorDbId=vector_db_id,
                                    embeddingModel=embedding_model_name,
                                    embeddingDim=appSettings.config_parameters.vectorstore.embedding_dimensions,
                                    providerId=vector_io_provider)

        if len(rag_docs) > 0:
            with st.spinner("**Embedding...**"):
                embedDocuments(embedClient=embedClient,
                            ragDocs=rag_docs,
                            vectorDbId=vector_db_id,
                            chunkSize=appSettings.config_parameters.vectorstore.chunk_size_in_tokens,
                            timeout=appSettings.config_parameters.vectorstore.embedding_timeout)
            
            st.markdown("**Embedding Done**")

# remove client after embedding
del embedClient
