import os

from llama_index.core import (Document, Settings, SimpleDirectoryReader,
                              VectorStoreIndex)
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

api_key = ""
os.environ["OPENAI_API_KEY"] = api_key

reader = SimpleDirectoryReader(
    input_dir="./data/",
    required_exts=[".pdf"])

documents = reader.load_data()

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="ww1",
            description="WWI causes",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools)

# response = query_engine.query("Please explain the causes of WWI. In your explanation, include information on the Balkans, overlapping alliances, and the growth of Germany.")




