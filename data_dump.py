import pymongo
import pandas as pd  
import json
from mushroom.config import mongo_client

DATA_FILE_PATH = "/config/workspace/mushrooms.csv"
DATABASE_NAME = "classification"
COLLECTION_NAME = "mushroom"

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns :{df.shape}")

    #Convert dataframe to json to dump these records in mongodb
    df.reset_index(drop = True,inplace = True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    #Insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    