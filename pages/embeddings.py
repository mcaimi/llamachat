#!/usr/bin/env python

try:
    import streamlit as st
    import mimetypes as mt
    from llama_stack_client import RAGDocument, LlamaStackClient
    from dotenv import dotenv_values
    from libs.shared.settings import Properties
    from libs.shared.session import Session
    with st.spinner("**Loading Docling Backend...**"):
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat, DocumentStream
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.chunking import HybridChunker
    import pdfplumber
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
            type=["pdf", "xlsx", "docx", "md", "html"],  # Add more file types as needed
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

    # docling conversion options
    with st.expander("PDF Document Conversion Options", expanded=False):
        do_ocr = st.checkbox("Use OCR to convert PDFs", value=False)
        do_table_structure = st.checkbox("Use Table Structure to convert PDFs", value=True)
        pdf_conversion_backend = st.selectbox(label="Select Backend", options=["PyPDFium", "Docling Pipeline v4"], index=0)
        match pdf_conversion_backend:
            case "PyPDFium":
                pdf_backend = PyPdfiumDocumentBackend
            case "Docling Pipeline v4":
                pdf_backend = DoclingParseV4DocumentBackend

    if st.button("Convert And Embed Documents...."):
        # documents to embed:
        converted_docs = []
        rag_docs = []
        
        # Instantiate the docling conversion engine
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = do_ocr
        pdf_options.do_table_structure = do_table_structure

        # Convert PDF to Docling Document
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, 
                            InputFormat.HTML,
                            InputFormat.MD,
                            InputFormat.DOCX, 
                            InputFormat.XLSX],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    backend=pdf_backend,
                )
            }
        )

        # convert documents with docling
        with st.spinner("**Converting...**"):
            for i, ufile in enumerate(uploaded_files):       
                # ufile is a bytestream...
                src = DocumentStream(name=ufile.name, stream=ufile)
                docling_doc = converter.convert(source=src)

                # get mimetype
                mimetype = mt.guess_type(ufile.name)[0]

                # Add metadata to the Docling Document
                metadata = {
                    "name": f"{ufile.name}",
                    "mimetype":f"{mimetype}",
                    "document_id": f"document_id_{i}",
                }

                # push to array & free resources
                converted_docs.append({
                    "doc": docling_doc,
                    "metadata": metadata,
                })

        st.markdown(f"Successfully converted {len(converted_docs)} documents. Ready for ingestion...")

        with st.spinner("**Chunking...**"):
            for i, ufile in enumerate(uploaded_files):
                # perform chunking on the converted documents
                chunker = HybridChunker()
                for doc in converted_docs:
                    chunks = list(chunker.chunk(dl_doc=doc["doc"].document))
                    for i, chunk in enumerate(chunks):
                        metadata = doc["metadata"]

                        # append chunk id
                        metadata["chunk_id"] = f"{metadata["document_id"]}_chunk_id_{i}"
                        rag_docs.append({
                            "content": chunker.contextualize(chunk=chunk),
                            "mime_type": metadata.get("mimetype"),
                            "metadata": metadata,
                            "chunk_metadata": metadata,
                        })

        st.markdown(f"Successfully chunked: {len(rag_docs)} chunks to embed. Ready for embedding...")

        # free converter
        del converter

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
            embedClient.vector_dbs.register(
                vector_db_id=vector_db_id,
                embedding_model=embedding_model_name,
                embedding_dimension=appSettings.config_parameters.vectorstore.embedding_dimensions,
                provider_id=vector_io_provider,
            )

        if len(rag_docs) > 0:
            # embed documents
            with st.spinner("**Embedding...**"):
                embedClient.vector_io.insert(vector_db_id=vector_db_id, chunks=rag_docs)
            
            st.markdown("**Embedding Done**")

# remove client after embedding
del embedClient
