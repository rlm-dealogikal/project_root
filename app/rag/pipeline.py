# /app/rag/pipeline.py
import pandas as pd
import glob
import json
import os

from config.settings import CSV_PATH, JSON_PATH
from app.preprocessing.text_cleaner import clean_text

def load_documents():

    documents = []
    metadatas = []

    # CSV
    csv_files = glob.glob(os.path.join(CSV_PATH, "*.csv"))

    for file in csv_files:
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            text = row.astype(str).str.cat(sep=" . ")

            documents.append(clean_text(text))

            metadatas.append({
                "source_file": os.path.basename(file),
                "type": "csv"
            })

    # JSON
    json_files = glob.glob(os.path.join(JSON_PATH, "*.json"))

    for file in json_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)

            text_doc = f"""
            Policy Name: {data.get("policy_name","")}
            Description: {data.get("policy_description","")}
            """

            documents.append(clean_text(text_doc))

            metadatas.append({
                "source_file": os.path.basename(file),
                "type": "json_policy"
            })

        except Exception as e:
            print(e)

    return documents, metadatas