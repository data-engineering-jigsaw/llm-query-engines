import os

from llama_index.core import (Settings, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

api_key = ""
os.environ['OPENAI_API_KEY'] =api_key


# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
Settings.chunk_size = 512
documents = SimpleDirectoryReader("./data/").load_data()

client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(
    "ww1", client=client, enable_hybrid=True, batch_size=20
)
# by batch size, we control the number of vectors embedded at once.  
# A larger batch size can be computationally efficient, but requires more memory.

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
# by passing through the storage context, we specify that we are 
# using the qdrant vector store, pointing to our qdrant vector database

query_engine = index.as_query_engine(
    similarity_top_k=2, sparse_top_k=12, vector_store_query_mode="hybrid"
)

response = query_engine.query("Who is ArchDuke Franz Ferdinand?")


