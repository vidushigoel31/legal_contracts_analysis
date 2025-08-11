import psycopg2
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import os

from dotenv import load_dotenv
load_dotenv()
# -----------------------
# CONFIG
# -----------------------
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
TABLE_NAME = "contracts"

# Azure OpenAI config
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION")
    
)


# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------
# CONNECT TO DB
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
# FUNCTION: RETRIEVE RELEVANT CONTRACTS
# -----------------------
def retrieve_contracts(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    cur.execute(f"""
        SELECT contract_id, summary, termination_clause, confidentiality_clause, liability_clause
        FROM {TABLE_NAME}
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
    """, (query_embedding, top_k))
    return cur.fetchall()


# -----------------------
# FUNCTION: ANSWER USING PROMPT
# -----------------------
def answer_question(query):
    results = retrieve_contracts(query)

    # Build context for the LLM
    context_blocks = []
    for idx, row in enumerate(results, 1):
        context_blocks.append(
            f"Contract {idx}:\n"
            f"Summary: {row[1]}\n"
            f"Termination: {row[2]}\n"
            f"Confidentiality: {row[3]}\n"
            f"Liability: {row[4]}"
        )
    context_text = "\n\n".join(context_blocks)

    # Prompt template
    prompt = f"""
You are a legal contract assistant. 
Use only the provided contract excerpts to answer the question.
If the answer is not present, say you cannot find it in the documents.

Question:
{query}

Contract Context:
{context_text}

Answer:
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a highly skilled legal assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    print(response.choices[0].message.content.strip())

# -----------------------
# DEMO
# -----------------------
if __name__ == "__main__":
    user_query = "What is the summary of amendment no.2  manufacturing and supply agreement?"
    answer = answer_question(user_query)

    print("---- Final Answer ----")
    print(answer)
