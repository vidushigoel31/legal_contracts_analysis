# Legal Contract Analysis & Retrieval System

## 📌 Overview
This project demonstrates an **end-to-end pipeline** for analyzing legal contracts using LLM-powered clause extraction, summarization, and semantic search via embeddings stored in PostgreSQL.

The workflow includes:
- Sampling contracts from the CUAD dataset
- Text normalization and cleaning
- Clause extraction (termination, confidentiality, liability)
- Contract summarization
- Semantic search over clauses & summaries (15 contracts)
- RAG implementation for all 50 sampled contracts

---

## 📂 Dataset
- **Source:** CUAD (Contract Understanding Atticus Dataset)
- **Total documents:** 510 contracts
- **Selected subset for analysis:** 50 randomly chosen contracts

**Script:**
- `random_sampling_50_contracts.py` – Selects a random subset of 50 contracts from the CUAD dataset

---

## ⚙️ Workflow

### **1. Data Preprocessing**
- **Normalization:** Removes unwanted characters, fixes spacing, and ensures text consistency
- **Script:** `normalising_50_contracts.py`

---

### **2. LLM-Powered Information Extraction & Summarization**

#### Part A – Clause Extraction
For each contract:
- Termination conditions
- Confidentiality clauses
- Liability clauses

#### Part B – Contract Summary
A concise **100–150 word summary** covering:
- Purpose of the agreement
- Key obligations of each party
- Notable risks or penalties

**Implementation Notes:**
- Due to cost constraints with closed-source models, only **15 contracts** were processed using **Azure OpenAI**
- Results saved to `processed_contracts.json`

**Script:**
- `clause_sum_extraction_for_15contracts.py`

---

### **3. Semantic Search over Clauses & Summaries (15 Contracts)**
- **Embeddings:** SentenceTransformers
- **Vector Database:** PostgreSQL with pgvector extension
- **Scripts:**
  - `storing_emb_vdb_15.py` – Generates embeddings for clauses & summaries, stores in PostgreSQL
  - `RAG-for15_contracts.py` – Retrieval-Augmented Generation (RAG) over the 15 processed contracts

---

### **4. RAG for 50 Contracts (Full Text Search)**
- Chunked documents for better retrieval performance (`chunking_of_50_contracts.py`)
- Embedded using SentenceTransformers
- Stored in PostgreSQL for semantic search

**Scripts:**
- `storing_emb_vdb_50.py` – Stores embeddings for all chunks of the 50 contracts
- `RAG-for50_contracts.py` – RAG implementation over all 50 contracts

---

## 📁 Folder Structure

```
├── CUAD_v1/                                 # Original dataset
├── normalized_contracts1/                   # Normalized contract texts
├── sampled_contracts1/                      # Sampled 50 contracts
├── chunking_of_50_contracts.py              # Chunking script
├── clause_sum_extraction_for_15contracts.py # Clause extraction + summarization
├── final_contract_analysis_combined.csv     # Combined output file
├── normalising_50_contracts.py              # Normalization script
├── processed_contracts.json                 # Output of clause extraction & summarization
├── RAG-for15_contracts.py                   # RAG search over 15 contracts
├── RAG-for50_contracts.py                   # RAG search over 50 contracts
├── random_sampling_50_contracts.py          # Random sampling script
├── sampled_metadata1.csv                    # Metadata for sampled contracts
├── storing_emb_vdb_15.py                     # Embedding storage for 15 contracts
├── storing_emb_vdb_50.py                     # Embedding storage for 50 contracts
└── .env                                     # Environment variables
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following:

```env
# Azure OpenAI configuration
azure_api_key=YOUR_AZURE_API_KEY
azure_endpoint=YOUR_AZURE_ENDPOINT
api_version=YOUR_API_VERSION
model=YOUR_MODEL_NAME

# PostgreSQL database configuration
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=qwerty12
DB_HOST=localhost
DB_PORT=5432
```

---

## 🛠 Technology Stack
- **Programming Language:** Python
- **LLM:** Azure OpenAI
- **Embeddings:** SentenceTransformers
- **Vector DB:** PostgreSQL + pgvector
- **Dataset:** CUAD legal contracts

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/legal-contract-analysis.git
cd legal-contract-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up `.env`** with your API and DB credentials

4. **Run sampling and normalization**
```bash
python random_sampling_50_contracts.py
python normalising_50_contracts.py
```

5. **Run clause extraction & summarization for 15 contracts**
```bash
python clause_sum_extraction_for_15contracts.py
```

6. **Store embeddings & run RAG for 15 contracts**
```bash
python storing_emb_vdb_15.py
python RAG-for15_contracts.py
```

7. **Run RAG for all 50 contracts**
```bash
python chunking_of_50_contracts.py
python storing_emb_vdb_50.py
python RAG-for50_contracts.py
```

---

## 📈 Future Improvements
- Extend clause extraction & summarization to more contracts
- Explore **Graph RAG** for advanced relationship mapping
- Build a **frontend UI** for contract search & insights
