#!/usr/local/env python

try:
    import mimetypes as mt
    from llama_stack_client import LlamaStackClient
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat, DocumentStream
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
    from docling.chunking import HybridChunker
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
    embedClient.vector_dbs.register(
        vector_db_id=vectorDbId,
        embedding_model=embeddingModel,
        embedding_dimension=embeddingDim,
        provider_id=providerId,
    )


def createDoclingConverter(
    do_ocr: bool,
    do_table_structure: bool,
    pdf_backend: PyPdfiumDocumentBackend
    | DoclingParseV4DocumentBackend = PyPdfiumDocumentBackend,
) -> DocumentConverter:
    # Instantiate the docling conversion engine
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = do_ocr
    pdf_options.do_table_structure = do_table_structure

    # Convert PDF to Docling Document
    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.HTML,
            InputFormat.MD,
            InputFormat.DOCX,
            InputFormat.XLSX,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_options,
                backend=pdf_backend,
            )
        },
    )

    # return handler
    return converter


def prepareDocuments(converter: DocumentConverter, uploaded_files: list) -> list:
    converted_docs = []

    for i, ufile in enumerate(uploaded_files):
        # ufile is a bytestream...
        src = DocumentStream(name=ufile.name, stream=ufile)
        docling_doc = converter.convert(source=src)

        # get mimetype
        mimetype = mt.guess_type(ufile.name)[0]

        # Add metadata to the Docling Document
        metadata = {
            "name": f"{ufile.name}",
            "mimetype": f"{mimetype}",
            "document_id": f"document_id_{i}",
        }

        # push to array & free resources
        converted_docs.append(
            {
                "doc": docling_doc,
                "metadata": metadata,
            }
        )

    # return documents
    return converted_docs


def chunkFiles(converted_docs: list) -> list:
    for i, ufile in enumerate(converted_docs):
        # perform chunking on the converted documents
        chunker = HybridChunker()
        docs = []

        for doc in converted_docs:
            chunks = list(chunker.chunk(dl_doc=doc["doc"].document))
            for i, chunk in enumerate(chunks):
                # contextualize chunk for content storage
                chunk_content = chunker.contextualize(chunk=chunk)
                # fill metadata
                metadata = {
                    "name": chunk.meta.origin.filename,
                    "uri": chunk.meta.origin.uri,
                    "headings": chunk.meta.headings,
                    "captions": chunk.meta.captions,
                    "mimetype": chunk.meta.origin.mimetype,
                    "document_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}",
                    "chunk_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}_chunk_{i}",
                }
                # fill chunk metadata
                chunk_metadata = {
                    "document_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}",
                    "chunk_id": f"{chunk.meta.origin.filename}_{chunk.meta.origin.binary_hash}_chunk_{i}",
                    "source": metadata.get("url") or metadata.get("name"),
                }

                # append chunk to doc list
                docs.append(
                    {
                        "content": chunk_content,
                        "metadata": metadata,
                        "chunk_metadata": chunk_metadata,
                    }
                )

    # return docs
    return docs
