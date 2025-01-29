import os
import json
from restack_ai.function import function, log
from pydantic import BaseModel
import weaviate
from weaviate.classes.init import Auth

class VectorSearchInput(BaseModel):
    query: str
    limit: int = 3  # Default limit to return top 3 results

class SearchResult(BaseModel):
    title: str
    content: str

class VectorSearchOutput(BaseModel):
    results: list[SearchResult]

def weaviate_client():
    """
    Establish a connection to the Weaviate Cloud instance.
    """
    wcd_url = "https://4zfylktqsqkmqkougroa.c0.us-east1.gcp.weaviate.cloud"
    wcd_api_key = "UGbNHq95PEfuac6caJPFLBnMNLyzoIReZSIG"  # READ ONLY API KEY

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,
        auth_credentials=Auth.api_key(wcd_api_key),
    )
    return client

@function.defn()
async def vector_similarity_search(input: VectorSearchInput) -> VectorSearchOutput:
    """
    Perform a vector-based similarity search in Weaviate using embeddings.
    """
    try:
        client = weaviate_client()
        log.info(f"Connected to Weaviate Cloud: {client}")

        # Access the collection
        collection = client.collections.get("BookVectorizedByWeaviateEmbeddings")

        # Perform a nearest-neighbor vector search
        response = collection.query.near_vector(
            vector=input.query,
            limit=input.limit
        )

        log.info(f"Vector search response: {response}")

        results = [
            SearchResult(
                title=obj.properties.get("title", "No Title"),
                content=obj.properties.get("description", "No Description")
            )
            for obj in response.objects
        ]

        client.close()
        return VectorSearchOutput(results=results)

    except Exception as e:
        log.error("Vector similarity search failed", error=e)
        raise e

# Weaviate function schema for Gemini integration
weaviate_tools.append({
    "name": "vector_similarity_search",
    "description": "Finds relevant content using vector similarity search in Weaviate.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING"},
            "limit": {"type": "INTEGER"}
        },
        "required": ["query"]
    }
})
