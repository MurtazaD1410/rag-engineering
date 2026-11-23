# %% Cell 1
import json
from typing import List

from dotenv import load_dotenv

# LangChain components
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from unstructured.chunking.title import chunk_by_title

# %% Cell 2
# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf

load_dotenv()


# %% Cell 3
def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"📄 partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,  # Path to pdf
        strategy="hi_res",  # Using the most accurate but slower processing method for extraction
        infer_table_structure=True,  # Keep tables as structured HTML not jumbled text
        extract_image_block_types=["Image"],  # Grab images
        extract_image_block_to_payload=True,  # Convert to base64 that can be used
    )

    print(f"✅ Extracted {len(elements)} elements")
    return elements


# Testing with pdf
file_path = "./docs/attention-is-all-you-need.pdf"
elements = partition_document(file_path)

# %% Cell 4
len(elements)

# %% Cell 5
# All types of atomic elements we see from unstructured
set([str(type(el)) for el in elements])

# %% Cell 6
elements[30].to_dict()

# %% Cell 7
# Gather all image elements
images = [element for element in elements if element.category == "Image"]
print(f"found {len(images)} images")

images[0].to_dict()

# %% Cell 8
# Gather all table elements
tables = [element for element in elements if element.category == "Table"]
print(f"found {len(images)} tables")

tables[0].to_dict()

# %% Cell 9


def create_chunks_by_title(elements):
    """Create intelligent chunks based on title stratergy"""

    print("🔨 creating smart chunks")

    chunks = chunk_by_title(
        elements,  # Pass all elements
        max_characters=3000,  #
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    print(f"Created {len(chunks)} chunks")
    return chunks


chunks = create_chunks_by_title(elements)
# %% Cell 10
chunks
set([str(type(chunk)) for chunk in chunks])
# %% Cell 11
# print one chunk
chunks[2].to_dict()
# view original elements inside chunk
chunks[4].metadata.orig_elements


# %% Cell 12
def seperate_content_types(chunk):
    """Analyse what types of contents are in the chunk"""
    content_data = {"text": chunk.text, "images": [], "tables": [], "types": ["text"]}

    # check for tables and images
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # handle tables
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            # handle images
            elif (
                element_type == "Image"
                and hasattr(element, "metadata")
                and hasattr(element.metadata, "image_base64")
            ):
                content_data["types"].append("image")
                content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""

    try:
        model = ChatOllama(model="qwen3-vl:8b", temperature=0)

        prompt_text = f"""
        You are creating a searchable description for document content retrieval.

        TEXT CONTENT:
        {text}
        """

        if tables:
            prompt_text += "\nTABLES:\n"

            for i, table in enumerate(tables):
                prompt_text += f"Table {i + 1}:\n{table}\n\n"

        prompt_text += """
        YOUR TASK:
        Generate a comprehensive, searchable description covering:

        1. Key facts, numbers, and data points from text and tables.
        2. Main topics and concepts discussed.
        3. Questions this content could answer.
        4. Visual content analysis (charts, diagrams, patterns in images).
        5. Alternative search terms users might use.

        Prioritize findability over brevity.

        SEARCHABLE DESCRIPTION:
        """

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        message = HumanMessage(content=message_content)
        response = model.invoke([message])

        return response.content

    except Exception as e:
        print(f"🔴 Failed to create AI summary: {e}")

        summary = f"{text[:300]}..."

        if tables:
            summary += f" [Contains {len(tables)} table(s)]"

        if images:
            summary += f" [Contains {len(images)} image(s)]"

        return summary


def summarise_chunks(chunks):
    """Process all chunks with AI summaries"""

    print("🧠 Processing chunks with AI summaries...")

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{total_chunks}")

        content_data = seperate_content_types(chunk)

        print(f"Types found: {content_data['types']}")
        print(f"Tables: {len(content_data['tables'])}")
        print(f"Images: {len(content_data['images'])}")

        if content_data["images"] or content_data["tables"]:
            print("Creating AI summary for mixed content...")

            enhanced_content = create_ai_enhanced_summary(
                content_data["text"], content_data["tables"], content_data["images"]
            )

        else:
            print("Using raw text (no tables/images)")
            enhanced_content = content_data["text"]

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "images_base64": content_data["images"],
                        "tables_html": content_data["tables"],
                    }
                )
            },
        )

        langchain_documents.append(doc)

    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


processed_chunks = summarise_chunks(chunks=chunks)

# %% Cell 13
processed_chunks


# %% Cell 14
def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []

    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": chunk.page_content,
            "metadata": {
                "original_content": json.loads(
                    chunk.metadata.get("original_content", "{}")
                )
            },
        }
        export_data.append(chunk_data)

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ exported {len(export_data)} chunks to {filename}")
    return export_data


# export your chunks
json_data = export_chunks_to_json(processed_chunks)


# %% Cell 15
def create_vector_store(documents, persist_directory="multi_modal_db/chroma_db"):
    """Create and persist ChromaDB vecotr store"""
    print("🔮 Creating embeddings and stroing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create ChromaDB Vector Store
    print("--- Creating vecotr store---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finished creating vector store")

    print(f"✅ Vector store created and persists at {persist_directory}")
    return vectorstore


db = create_vector_store(processed_chunks)

# %% Cell 16
query = "What are the two main components of the Transformer architecture?"
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)

# export chunks to json
export_chunks_to_json(chunks, filename="rag_results.json")


# %% Cell 17
def run_complete_ingestion_pipeline(pdf_path: str):
    """Run the complete RAG ingestion pipeline"""
    print("🚀 Starting the RAG Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Partition
    elements = partition_document(pdf_path)
    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)
    # Step 3: Summerize
    summerised_chunks = summarise_chunks(chunks)
    # Step 4: Vector Store
    db = create_vector_store(summerised_chunks, "multi_modal_db_v2/chroma_db")

    print("🎉 Pipeline Completed Successfully")
    return db


# %% Cell 18
db = run_complete_ingestion_pipeline("./docs/attention-is-all-you-need.pdf")

# %% Cell 19
query = "How many attention heads does the Transformer use, and what is the dimension of each head?"
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)


def generate_final_answer(chunks, query):
    """Generating final content using multimodal content"""

    try:
        # using vision model llm for images
        model = ChatOllama(model="qwen3-vl:8b", temperature=0)

        # build the text prompt
        prompt_text = f"""Based on the following documents please answer this quesiton:{query}
        CONTENT TO ANALYSE:
        """

        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i + 1} ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                # add tables as HTML
                tables_html = original_data.get("tables_html", "")
                if tables_html:
                    prompt_text += "TABLE:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j + 1}:\n{table}\n\n"

            prompt_text += "\n"
            prompt_text += """
            Please provide a clear comprehensive answer using the text, images and tables above, if the document doesnt have sufficient imformation please respond with "I don't have enough information to answer the question"

            ANSWER:
            """

            # Build message content starting with text
            message_content = [{"type": "text", "text": prompt_text}]

            for chunk in chunks:
                if "original_content" in chunk.metadata:
                    original_data = json.loads(chunk.metadata["original_content"])
                    images_base64 = original_data.get("images_base64", "")
                    for image_base64 in images_base64:
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            }
                        )

        message = HumanMessage(content=message_content)
        response = model.invoke([message])

        return response.content

    except Exception as e:
        print(f"🔴 Failed to generate answer: {e}")
        return "Sorry, I encountered an error while generating your answer."


# usage
final_answer = generate_final_answer(chunks, query)
print(final_answer)
