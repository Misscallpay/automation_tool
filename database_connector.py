import pandas as pd
from pymongo import MongoClient

def connect_to_mongodb(database_name='mcp'):
    client = MongoClient('192.168.1.7', 27017)
    db = client[database_name]
    return client, db

def retrieve_data_from_collection(db, collection_name):
    collection = db[collection_name]
    
    # Retrieve data from MongoDB excluding _id field
    data = pd.DataFrame(list(collection.find({}, {'_id': 0})))

    # Data cleaning for the CDR collection
    if collection_name == 'cdr':
        data = data[data['disposition'] != 'ANSWERED']

    return data

def close_mongodb_connection(client):
    client.close()
