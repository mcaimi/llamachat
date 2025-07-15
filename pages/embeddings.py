#!/usr/bin/env python

try:
    import streamlit as st
    import mimetypes as mt
    from llama_stack_client import RAGDocument, LlamaStackClient
    from dotenv import dotenv_values
    from libs.shared.settings import Properties
    import pdfplumber
except ImportError as e:
    print(f"Caught Exception: {e}")

# load environment
config_env: dict = dotenv_values(".env")

# load app settings
config_filename: str = config_env.get("CONFIG_FILE", "parameters.yaml")
appSettings = Properties(config_file=config_filename)

# build streamlit UI
st.set_page_config(page_title="🧠 Embeddings", initial_sidebar_state="collapsed", layout="wide")
st.html("assets/embeddings.html")

st.markdown(f"**Embedding Model**: {st.session_state.embedding_model_name}")
st.markdown(f"**Vector DB Provider**: {st.session_state.provider_name}")
st.markdown(f"**Vector Collection**: {st.session_state.collection_name}")

# file uploader
uploaded_files = st.file_uploader("Embed a document..", accept_multiple_files=True)

# documents to embed:
rag_docs = []
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
           
    rag_docs.append(RAGDocument(
        document_id="rag_document_{i}",
        content=file_contents,
        mime_type=mtype,
        metadata=metadata,
    ))

# embed documents!
if len(rag_docs) > 0:
    embedClient = LlamaStackClient(base_url=st.session_state.api_base_url)
    # create vector db on provider
    embedClient.vector_dbs.register(
        vector_db_id=st.session_state.collection_name,
        embedding_model=st.session_state.embedding_model_name,
        embedding_dimension=appSettings.config_parameters.vectorstore.embedding_dimensions,
        provider_id=st.session_state.provider_name,
    )
    st.markdown("**Embedding...**")
    embedClient.tool_runtime.rag_tool.insert(
        documents=rag_docs,
        vector_db_id=st.session_state.collection_name,
        chunk_size_in_tokens=512
    )
    st.markdown("**Embedding Done**")
    del embedClient
