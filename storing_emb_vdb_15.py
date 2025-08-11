from sentence_transformers import SentenceTransformer
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import os

# -----------------------
# CONFIGURATION
# -----------------------
CSV_FILE = "final_contract_analysis_combined.csv"

import os 
from dotenv import load_dotenv
load_dotenv()

# Database connection parameters
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
TABLE_NAME = "contracts"

# -----------------------
# CONNECT TO POSTGRES
# -----------------------
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# -----------------------
# CREATE TABLE IF NOT EXISTS
# -----------------------
create_table_sql = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    contract_id TEXT,
    file_size_chars INTEGER,
    summary TEXT,
    word_count INTEGER,
    termination_clause TEXT,
    confidentiality_clause TEXT,
    liability_clause TEXT,
    embedding vector(384)  -- all-MiniLM-L6-v2 has 384 dimensions
);
"""
cur.execute(create_table_sql)
conn.commit()

# -----------------------
# LOAD CSV
# -----------------------
df = pd.read_csv(CSV_FILE)

# -----------------------
# LOAD EMBEDDING MODEL
# -----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------
# PREPARE DATA
# -----------------------
records = []
for _, row in df.iterrows():
    text_to_embed = row["summary"]  # You can also concatenate multiple fields
    embedding = model.encode(text_to_embed).tolist()

    records.append((
        row["contract_id"],
        int(row["file_size_chars"]),
        row["summary"],
        int(row["word_count"]),
        row["termination_clause"],
        row["confidentiality_clause"],
        row["liability_clause"],
        embedding
    ))

# -----------------------
# INSERT DATA
# -----------------------
insert_sql = f"""
INSERT INTO {TABLE_NAME} (
    contract_id,
    file_size_chars,
    summary,
    word_count,
    termination_clause,
    confidentiality_clause,
    liability_clause,
    embedding
) VALUES %s
"""
execute_values(cur, insert_sql, records)
conn.commit()

print(f"✅ Inserted {len(records)} rows into {TABLE_NAME}")

# -----------------------
# CLEANUP
# -----------------------
cur.close()
conn.close()
