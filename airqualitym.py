import requests
import io
import json
import shutil
import sys
import datetime
import subprocess
import os
import math
import base64
from time import gmtime, strftime
import random, string
import time
import base64
import uuid
import socket
from pymilvus import connections
from pymilvus import utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import MilvusClient
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pymilvus import model
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv(verbose=True)

model = SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2',device='cpu' )

DIMENSION = 384 
MILVUS_URL = os.environ.get("SERVERLESS_VECTORDB_URI")
TOKEN = os.environ.get("ZILLIZ_TOKEN")
COLLECTION_NAME = "airquality"
AQ_URL = "https://www.airnowapi.org/aq/observation/zipCode/current/?format=application/json&distance=5000&zipCode="
AQ_KEY = "&API_KEY="

API_KEY = os.environ.get("OPENAQ_KEY")
slack_token = os.environ.get("SLACK_BOT_TOKEN")

client = WebClient(token=slack_token)

milvus_client = MilvusClient( uri=MILVUS_URL, token=TOKEN )

OAQ3_COLLECTION_NAME = "openaqmeasurements"
TODAYS_DATE = str( datetime.today().strftime('%Y-%m-%d') ) 
YESTERDAYS_DATE = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

## schema
schema = milvus_client.create_schema(
    enable_dynamic_field=False
)

schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name='locationId', datatype=DataType.INT32)
schema.add_field(field_name='location', datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name='parameter', datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="value", datatype=DataType.FLOAT)
schema.add_field(field_name='datelocal', datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="unit", datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="latitude", datatype=DataType.FLOAT)
schema.add_field(field_name="longitude", datatype=DataType.FLOAT)
schema.add_field(field_name="country", datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="city", datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="isMobile", datatype=DataType.VARCHAR, max_length=12)
schema.add_field(field_name="isAnalysis", datatype=DataType.VARCHAR, max_length=12)
schema.add_field(field_name='entity', datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name='sensorType', datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field(field_name="details",  datatype=DataType.VARCHAR, max_length=8000)
schema.add_field(field_name="fulllocation",  datatype=DataType.VARCHAR, max_length=2000)

index_params = milvus_client.prepare_index_params()

index_params.add_index(
    field_name="id",
    index_type="STL_SORT"
)

index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 100}
)

if not milvus_client.has_collection(collection_name = OAQ3_COLLECTION_NAME):
    milvus_client.create_collection(
        collection_name = OAQ3_COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    
    res = milvus_client.get_load_state(
        collection_name = OAQ3_COLLECTION_NAME
    )
    print(res)

count = 0
    
for pages in range(100):
    aqpage = int(pages) + 1
    details = ""
    fulllocation = ""
    data = []

    url =  'https://api.openaq.org/v2/measurements?country=US&date_from={0}&date_to={1}&limit=1000&page={2}&offset=0&sort=desc&radius=1000&order_by=datetime'.format(
        str(YESTERDAYS_DATE), str(TODAYS_DATE), str(aqpage) )
    headers = {"accept": "application/json", "x-api-key": str(API_KEY)}

    response = requests.get(url, headers=headers)

    if ( len(response.text) > 0 ):
        openaq2 = json.loads(response.text)
    else:
        openaq2['results'] = {}

    try:
        for jsonitems in openaq2['results']:
            count = count + 1
            fulllocation = 'Location {0}: {1}, {2}, {3} @ {4},{5}'.format(jsonitems.get('locationId'),
                                                                   jsonitems.get('location'),jsonitems.get('city'),jsonitems.get('country'), 
                                                                   jsonitems.get('coordinates')['latitude'],jsonitems.get('coordinates')['longitude'] )
            details = 'Current Air Quality Reading for {0} is {1} {2} for {3} at {4}. Is Mobile: {5} Is Analysis: {6} Entity: {7} Sensor Type: {8}'.format(jsonitems.get('parameter'), jsonitems.get('value'), jsonitems.get('unit'),fulllocation, jsonitems.get('date')['local'],
            jsonitems.get('isMobile'), jsonitems.get('isAnalysis'), jsonitems.get('entity'), jsonitems.get('sensorType'))  
        
            data.append({ "locationId": int(jsonitems.get('locationId')),
                          "location": str(jsonitems.get('location','')), 
                          "parameter": str(jsonitems.get('parameter','')), 
                          "value": float(jsonitems.get('value')), 
                          "datelocal": str(jsonitems.get('date')['local']), 
                          "unit": str(jsonitems.get('unit','')), 
                          "latitude": float(jsonitems.get('coordinates')['latitude']), 
                          "longitude": float(jsonitems.get('coordinates')['longitude']), 
                          "country": str(jsonitems.get('country','')), 
                          "city": str(jsonitems.get('city','')), 
                          "isMobile": str(jsonitems.get('isMobile','')), 
                          "isAnalysis": str(jsonitems.get('isAnalysis','')), 
                          "entity": str(jsonitems.get('entity','')), 
                          "sensorType": str(jsonitems.get('sensorType','')), 
                          "vector": model(details), "details": str(details), "fulllocation": str(fulllocation)})
        
        res = milvus_client.insert(collection_name=OAQ3_COLLECTION_NAME, data=data)
        print(count)
    except Exception as ex:
        print("Error building the airquality", ex)
    # print(res)
    # print(data)