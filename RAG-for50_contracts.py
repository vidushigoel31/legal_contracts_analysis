import os
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import textwrap
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Database connection parameters

DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

TABLE_NAME = "legal_contracts"

# Azure OpenAI client setup
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION")
)

model = SentenceTransformer('all-MiniLM-L6-v2')

def rag(question):
    # Search
    embedding = model.encode([question])[0].tolist()
    
    conn = psycopg2.connect(
        dbname="postgres", user="postgres", password="qwerty12", 
        host="localhost", port="5432"
    )
    cur = conn.cursor()
    
    cur.execute("""
        SELECT text_content FROM legal_contracts
        ORDER BY embedding <=> %s::vector(384) LIMIT 3
    """, (embedding,))
    
    docs = [row[0][:500] for row in cur.fetchall()]
    cur.close()
    conn.close()
    
    # Answer
    context = "\n\n".join(docs)
    
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "user", "content": f"Question: {question}\n\nContext: {context}\n\nAnswer:"}
        ]
        # max_tokens=200
    )
    
    return response.choices[0].message.content

# Test
print(rag("what was NEONSYSTEMSINC distributor AGREEMENT?"))