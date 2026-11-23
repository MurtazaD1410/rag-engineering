from collections import defaultdict
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

# ──────────────────────────────────────────────────────────────────
# Step 3: Apply Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "=" * 60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("=" * 60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")

    # data structures for rrf calculations
    rrf_scores = defaultdict(float)  # will store: {chunk_content:rrf_score}
    all_unique_chunk = {}  # will store: {chunk_content: actual_chunk_object}

    # for verbose output - track chunkIDs
    chunk_id_map = {}
    chunk_counter = 1

    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"processing query {query_idx} results:")

        # go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):
            # use chunk content as unique identifier
            chunk_content = chunk.page_content

            # assign simple id if chunk not seen before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1

            chunk_id = chunk_id_map[chunk_content]

            # store the chunk content if not seen before
            all_unique_chunk[chunk_content] = chunk

            # calculate position score for each
            position_score = 1 / (k + position)

            # add rrf score to map
            rrf_scores[chunk_content] += position_score

            if verbose:
                print(
                    f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})"
                )
                print(f"    Preview: {chunk_content[:80]}...")

        if verbose:
            print()

    # sort chunks by RRF score
    sorted_chunks = sorted(
        [
            (all_unique_chunk[chunk_content], score)
            for chunk_content, score in rrf_scores.items()
        ],
        key=lambda x: x[1],  # sort by rrf score
        reverse=True,
    )

    if verbose:
        print(
            f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries."
        )

    return sorted_chunks


# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieved_results, k=60, verbose=True)

# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FINAL RRF RANKING")
print("=" * 60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(
    f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations."
)
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score")
print("   • Balanced fusion using k=60 for gentle position penalties")

# ──────────────────────────────────────────────────────────────────
# Optional: Quick Usage Examples
# ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("USAGE EXAMPLES")
print("=" * 60)
