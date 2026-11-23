# %% Cell 1
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

# %% Cell 2
%pip install rank_bm25

# %% Cell 3
# ──────────────────────────────────────────────────────────────────
# SETUP: Create our sample company data
# ──────────────────────────────────────────────────────────────────

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.",
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.",
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]


# %% Cell 4
documents =[Document(page_content=chunk, metadata={"source":f"chunk_{i}"}) for  i, chunk in enumerate(chunks)]

print("Sample Data:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

print("\n" + "="*80)

# %% Cell 5
print("Setting up vector store...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore= Chroma.from_documents(documents=documents, embedding=embedding_model, collection_metadata={"hnsw:space": "cosine"})

# %% Cell 6
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

# test query
query="Space exploration company" # works in vector search but wouldnt work in keyword search

print(f"Testing: '{query}'")
test_docs= retriever.invoke(query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

# %% Cell 7
print("Setting up BM25 retriever...")
bm25_retreiver = BM25Retriever.from_documents(documents)
bm25_retreiver.k=3

# %% Cell 8
test_query="Cybertruck"

test_docs=bm25_retreiver.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

# %% Cell 9
print("Setting up hybrid retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retreiver],
    weights=[0.5,0.5]
)

print("Setup complete!\n")

# %% Cell 10

# Query 1: Mixed semantic and exact terms

# Vector search understands "purchase cost" semantically
# BM25 search finds exact "7.5 billion"
# Hybrid should combine both strengths for best result
test_query = "purchase cost 7.5 billion"

retrieved_chunks = hybrid_retriever.invoke(test_query)
for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")
print()

print("Query 1 shows how hybrid finds exact financial info using both semantic understanding and keyword matching")

# %% Cell 11
# Query 2: Semantic concept + specific product name

# Vector search understands "electric vehicle manufacturing"
# BM25 search finds exact "Cybertruck"
# Hybrid gets the best of both worlds

test_query = "electric vehicle manufacturing Cybertruck"

retrieved_chunks = hybrid_retriever.invoke(test_query)

for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")
print()

print("Query 2 demonstrates combining product-specific terms with broader concepts")

# %% Cell 12
# Query 3: Where neither alone would be perfect

# "Company performance" is semantic, "Tesla" is exact keyword
# Hybrid should find the most relevant Tesla performance info

test_query = "company performance Tesla"

retrieved_chunks = hybrid_retriever.invoke(test_query)
for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")
print()

print("Query 3 shows how hybrid handles mixed semantic/keyword queries better than either approach alone")
# %% Cell 13
# Combine the query and the relevant document contents
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


combined_input = f"""Based on the following documents, please answer this question: {test_query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in retrieved_chunks])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatOpenAI model
model = ChatOllama(model="qwen3:8b")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
