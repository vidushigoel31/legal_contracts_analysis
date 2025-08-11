import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import os 
from dotenv import load_dotenv
load_dotenv()

# Database connection parameters
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

TABLE_NAME = "legal_contracts"

# --------------------
# 1. Load CSV metadata
# --------------------
csv_path = "sampled_metadata1.csv"  # your CSV file
df_meta = pd.read_csv(csv_path)

# Normalize the CSV filename for joining
df_meta["base_name"] = df_meta["Filename"].apply(
    lambda x: x.split(".pdf")[0].strip()
)

# --------------------
# 2. Load JSONL chunks
# --------------------
jsonl_path = "processed_contracts.jsonl"  # your JSONL file
chunks = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

df_chunks = pd.DataFrame(chunks)

# Extract base name from contract_id for join
df_chunks["base_name"] = df_chunks["contract_id"].apply(
    lambda x: x.replace("norm_", "").split("-EX")[0].strip()
)

# --------------------
# 3. Join chunks with metadata
# --------------------
df_joined = pd.merge(df_chunks, df_meta, on="base_name", how="left")

# Combine text + metadata into one field for embedding
def combine_for_embedding(row):
    meta_parts = [
        f"Document Name: {row['Document Name-Answer']}",
        f"Parties: {row['Parties-Answer']}",
        f"Agreement Date: {row['Agreement Date-Answer']}",
        f"Effective Date: {row['Effective Date-Answer']}",
        f"Expiration Date: {row['Expiration Date-Answer']}"
    ]
    meta_text = " | ".join([str(p) for p in meta_parts if pd.notna(p)])
    return f"{meta_text}\n\nContract Text:\n{row['text']}"

df_joined["content_for_embedding"] = df_joined.apply(combine_for_embedding, axis=1)

# --------------------
# 4. Create embeddings
# --------------------
print("Creating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df_joined["content_for_embedding"].tolist(), show_progress_bar=True)

# --------------------
# 5. Store in PostgreSQL with pgvector
# --------------------

# Helper function to adapt numpy arrays for PostgreSQL
def addapt_numpy_array(numpy_array):
    return AsIs(numpy_array.tolist())

# Register the adapter
register_adapter(np.ndarray, addapt_numpy_array)

# Connect to PostgreSQL
print("Connecting to PostgreSQL...")
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# Create table with all necessary columns
print("Creating table...")
create_table_sql = f"""
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS {TABLE_NAME};

CREATE TABLE {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    document_id TEXT UNIQUE,
    contract_id TEXT,
    chunk_id INTEGER,
    base_name TEXT,
    filename TEXT,
    document_name TEXT,
    parties TEXT,
    agreement_date TEXT,
    effective_date TEXT,
    expiration_date TEXT,
    text_content TEXT,
    content_for_embedding TEXT,
    word_count INTEGER,
    embedding vector(384),  -- all-MiniLM-L6-v2 has 384 dimensions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_contract_id ON {TABLE_NAME}(contract_id);
CREATE INDEX IF NOT EXISTS idx_base_name ON {TABLE_NAME}(base_name);
CREATE INDEX IF NOT EXISTS idx_embedding_cosine ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops);
"""

cur.execute(create_table_sql)
conn.commit()

# Insert data
print(f"Inserting {len(df_joined)} records...")

insert_sql = f"""
INSERT INTO {TABLE_NAME} (
    document_id, contract_id, chunk_id, base_name, filename,
    document_name, parties, agreement_date, effective_date, expiration_date,
    text_content, content_for_embedding, word_count, embedding
) VALUES %s
ON CONFLICT (document_id) DO NOTHING;
"""

# Prepare data for insertion
insert_data = []
for idx, (_, row) in enumerate(df_joined.iterrows()):
    document_id = f"{row['contract_id']}_chunk_{row['chunk_id']}"
    
    # Handle NaN values by converting to None
    def safe_str(val):
        return None if pd.isna(val) else str(val)
    
    insert_data.append((
        document_id,
        safe_str(row['contract_id']),
        int(row['chunk_id']) if pd.notna(row['chunk_id']) else None,
        safe_str(row['base_name']),
        safe_str(row.get('Filename')),
        safe_str(row.get('Document Name-Answer')),
        safe_str(row.get('Parties-Answer')),
        safe_str(row.get('Agreement Date-Answer')),
        safe_str(row.get('Effective Date-Answer')),
        safe_str(row.get('Expiration Date-Answer')),
        safe_str(row['text']),
        safe_str(row['content_for_embedding']),
        int(row.get('word_count', 0)) if pd.notna(row.get('word_count')) else None,
        embeddings[idx].tolist()  # Convert numpy array to list
    ))

# Batch insert using execute_values for better performance
psycopg2.extras.execute_values(
    cur, insert_sql, insert_data, template=None, page_size=100
)

conn.commit()

# Verify insertion
cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
count = cur.fetchone()[0]
print(f"✅ Successfully inserted {count} records into PostgreSQL.")

# Test if vector extension is working
print("Testing vector extension...")
try:
    cur.execute("SELECT '[1,2,3]'::vector(3);")
    print("✅ Vector extension is working")
except Exception as e:
    print(f"❌ Vector extension error: {e}")
    print("Please ensure pgvector extension is properly installed")

# Create a function for similarity search
similarity_function_sql = f"""
CREATE OR REPLACE FUNCTION search_similar_contracts(
    query_embedding vector(384),
    similarity_threshold float DEFAULT 0.5,
    max_results int DEFAULT 10
)
RETURNS TABLE(
    document_id text,
    contract_id text,
    chunk_id integer,
    document_name text,
    parties text,
    text_content text,
    similarity float
) AS $
BEGIN
    RETURN QUERY
    SELECT 
        lc.document_id,
        lc.contract_id,
        lc.chunk_id,
        lc.document_name,
        lc.parties,
        lc.text_content,
        1 - (lc.embedding <=> query_embedding) as similarity
    FROM {TABLE_NAME} lc
    WHERE 1 - (lc.embedding <=> query_embedding) > similarity_threshold
    ORDER BY lc.embedding <=> query_embedding
    LIMIT max_results;
END;
$ LANGUAGE plpgsql;
"""

try:
    cur.execute(similarity_function_sql)
    conn.commit()
    print("✅ Created similarity search function.")
except Exception as e:
    print(f"❌ Error creating similarity function: {e}")
    print("Will use direct SQL queries instead")

# Close connections
cur.close()
conn.close()

print(f"✅ PostgreSQL setup complete! Table '{TABLE_NAME}' is ready for RAG queries.")
print("\nNext steps:")
print("1. Use the RAG query script to search contracts")
print("2. The similarity search function is available as 'search_similar_contracts()'")