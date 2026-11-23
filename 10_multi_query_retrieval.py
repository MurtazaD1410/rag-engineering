from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel

load_dotenv()

persistent_directory = "db/chroma_db"

# load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


model = ChatOllama(model="qwen3:8b", temperature=0)


# pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]


# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────
model_with_tools = model.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
Original query: {original_query}
Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = model_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, query in enumerate(query_variations, 1):
    print(f"{i}: {query}")

print("\n" + "=" * 60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)  # fetch more docs for better results
all_retrieved_results = []  # Store all results in rrf

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)
    all_retrieved_results.append(docs)

    print(f"Retrieved {len(docs)} documents:\n")

    for j, doc in enumerate(docs, 1):
        print(f"document {j}:")
        print(f"{doc.page_content[:150]}...\n")

    print("-" * 50)

print("\n" + "=" * 60)
print("Multi query retreival complete")
