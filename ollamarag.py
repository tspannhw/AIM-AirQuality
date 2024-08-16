import os
from pymilvus import MilvusClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
import sys
import io
import json
import shutil
import sys
import datetime
import subprocess
import sys
import os
import math
import base64
from time import gmtime, strftime
import random, string
import time
import psutil
import base64
import uuid
import socket
import os
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from IPython.display import display
from pymilvus import MilvusClient
from dotenv import load_dotenv
load_dotenv(verbose=True)
import json
import requests
from datetime import datetime, timedelta
from langchain_milvus import Milvus

### Based on https://zilliz.com/blog/a-beginners-guide-to-using-llama-3-with-ollama-milvus-langchain

DIMENSION = 384 
MILVUS_URL = "http://MYLOCALIP:19530" 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
OAQ3_COLLECTION_NAME = "openaqmeasurements"
TODAYS_DATE = str( datetime.today().strftime('%Y-%m-%d') ) 
YESTERDAYS_DATE = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

# os.environ["LANGCHAIN_HUB_API_URL"] = "https://api.hub.langchain.com"
# os.environ["LANGCHAIN_HUB_API_KEY"] = "KEY"
# os.environ["LANGCHAIN_API_KEY"] = "KEY"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.hub.langchain.com"

vector_store = Milvus(
    embedding_function=embeddings,
    collection_name=OAQ3_COLLECTION_NAME,
    primary_field = "id",
    text_field="details",
    connection_args={"uri": MILVUS_URL},
)

matched_docs = vector_store.similarity_search(query = "What is the Current Air Quality Reading for pm10 ?", collection=OAQ3_COLLECTION_NAME, k = 10)
print(matched_docs)

def run_query() -> None:
    llm = Ollama(
        model="llama3",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|eot_id|>"],
    )

    query = input("\nQuery: ")
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vector_store.as_retriever(collection =  OAQ3_COLLECTION_NAME, expr="parameter == 'o3'"))

    result = qa_chain({"query": query})
    print(result)

if __name__ == "__main__":
    while True:
        run_query()
