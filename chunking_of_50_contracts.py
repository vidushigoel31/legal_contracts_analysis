# split_contracts.py

import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. CONFIGURATION
INPUT_DIR    = Path("normalized_contracts1")        # folder containing your 50 .txt files
OUTPUT_FILE  = Path("processed_contracts.jsonl")    # where we’ll write the chunk metadata
CHUNK_SIZE   = 4000     # approx max chars per chunk
CHUNK_OVERLAP= 400      # chars overlapping between chunks

# 2. INITIALIZE THE SPLITTER
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],  # try big breaks first, then smaller
)

# 3. PROCESS EACH FILE
records = []
for txt_path in sorted(INPUT_DIR.glob("*.txt")):
    contract_id = txt_path.stem  # e.g. "norm_ACCURAYINC_09_01_2010-EX-10.31-D"
    full_text   = txt_path.read_text(encoding="utf-8")

    # split_text returns a list of chunks (strings)
    chunks = splitter.split_text(full_text)

    for idx, chunk in enumerate(chunks):
        records.append({
            "contract_id": contract_id,
            "chunk_id":    idx,
            "text":        chunk
        })

# 4. WRITE OUT JSONL
with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
    for rec in records:
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Split {len(records)} total chunks from {INPUT_DIR.name} into {OUTPUT_FILE}")
