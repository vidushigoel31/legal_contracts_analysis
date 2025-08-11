
import os
import random
import shutil
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Paths
TXT_DIR = "CUAD_v1/full_contract_txt"  # Directory containing all 510 TXT files
MASTER_CSV = "CUAD_v1/master_clauses.csv"  # Path to master clauses CSV
OUTPUT_DIR = "sampled_contracts1"  # Directory to store sampled contracts

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all contract filenames from TXT directory
all_txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith('.txt')]
print(f"Found {len(all_txt_files)} TXT files in {TXT_DIR}")

# Randomly sample 50 contracts
sampled_files = random.sample(all_txt_files, 50)
print(f"Sampled {len(sampled_files)} contracts")

# Copy sampled files to output directory
for filename in sampled_files:
    src_path = os.path.join(TXT_DIR, filename)
    dst_path = os.path.join(OUTPUT_DIR, filename)
    shutil.copy(src_path, dst_path)
print(f"Copied sampled contracts to {OUTPUT_DIR}")

# Create a list of contract names (without .txt extension) for reference
contract_names = [f.replace('.txt', '') for f in sampled_files]

# Load master CSV to get metadata for sampled contracts
master_df = pd.read_csv(MASTER_CSV)
print(f"Master CSV columns: {list(master_df.columns)}")

# Check if 'FilterName' column exists
if 'FilterName' in master_df.columns:
    # Try to match using FilterName column
    # First, let's see what the FilterName values look like
    print(f"Sample FilterName values: {master_df['FilterName'].head(5).tolist()}")
    
    # Create a pattern matching function to handle potential naming differences
    def match_contract_name(filter_name, contract_name):
        # Convert both to lowercase for case-insensitive comparison
        filter_lower = str(filter_name).lower()
        contract_lower = str(contract_name).lower()
        
        # Check if contract name is contained in filter name or vice versa
        return contract_lower in filter_lower or filter_lower in contract_lower
    
    # Find matches
    matched_indices = []
    for contract_name in contract_names:
        for idx, row in master_df.iterrows():
            if match_contract_name(row['FilterName'], contract_name):
                matched_indices.append(idx)
                break
    
    # Filter master CSV to only include sampled contracts
    sampled_df = master_df.iloc[matched_indices]
else:
    # If FilterName column doesn't exist, try to find a column that might contain contract names
    print("'FilterName' column not found. Trying to find a suitable column...")
    
    # Try each column to see if it contains contract names
    for col in master_df.columns:
        # Skip non-string columns
        if master_df[col].dtype == 'object':
            # Check if any contract names match values in this column
            matches = 0
            for contract_name in contract_names:
                if any(contract_name.lower() in str(val).lower() for val in master_df[col]):
                    matches += 1
            
            if matches > 0:
                print(f"Column '{col}' has {matches} matches with contract names")
                if matches >= len(contract_names) * 0.5:  # If at least 50% match
                    print(f"Using column '{col}' for matching")
                    # Filter based on this column
                    sampled_df = master_df[master_df[col].apply(
                        lambda x: any(contract_name.lower() in str(x).lower() 
                                     for contract_name in contract_names)
                    )]
                    break
    else:
        print("No suitable column found for matching. Using all contracts.")
        sampled_df = pd.DataFrame()  # Empty dataframe

# Save filtered metadata
if not sampled_df.empty:
    sampled_df.to_csv("sampled_metadata1.csv", index=False)
    print(f"Saved metadata for {len(sampled_df)} sampled contracts to sampled_metadata1.csv")
else:
    print("No matching metadata found. Creating empty CSV file.")
    pd.DataFrame().to_csv("sampled_metadata1.csv", index=False)

# Print sample of selected contracts
print("\nSample of selected contracts:")
for i, contract in enumerate(sampled_files[:5]):
    print(f"{i+1}. {contract}")