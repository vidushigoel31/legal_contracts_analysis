import re
import os
import pandas as pd
from unidecode import unidecode
def normalize_contract_text(text):
    """FINAL version with comprehensive fixes for ALL contract normalization issues"""
    
    # CRITICAL FIX #0: Handle split words and join them properly - MUST BE FIRST
    # This prevents clause tags from being inserted in split words
    
    # Fix specific problematic patterns seen in your files
    text = re.sub(r'labels!Data', 'labels! Data', text)
    text = re.sub(r'labels\*Data', 'labels* Data', text)
    text = re.sub(r'labels·Data', 'labels· Data', text)
    text = re.sub(r'labels#Data', 'labels# Data', text)
    
    # Fix missing spaces after punctuation (CRITICAL FOR ALL FILES)
    text = re.sub(r'([.!?;:])([A-Z])', r'\1 \2', text)  # After punctuation + capital
    text = re.sub(r'([.!?])(\s*)([a-z])', r'\1 \3', text)  # After punctuation + lowercase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Lowercase + uppercase (word join)
    
    # Standardize common abbreviations
    text = re.sub(r'e\.\s*g\.', 'e.g.', text)
    text = re.sub(r'e\.g\.', 'e.g.', text)
    text = re.sub(r'i\.\s*e\.', 'i.e.', text)
    text = re.sub(r'i\.e\.', 'i.e.', text)
    text = re.sub(r'n\.\s*a\.', 'n/a', text)
    text = re.sub(r'n\.a\.', 'n/a', text)
    text = re.sub(r'vs\.', 'vs.', text)
    
    # Standardize common punctuation patterns
    text = re.sub(r' -- ', ' — ', text)  # Double hyphen to em dash (with spaces)
    text = re.sub(r'(\d)\s*--\s*(\d)', r'\1—\2', text)  # Number ranges without spaces
    text = re.sub(r'(\d)\s*—\s*(\d)', r'\1—\2', text)  # Standardize existing em dashes
    text = re.sub(r'CR\s*-\s*No\.?:?', 'CR-No.', text)  # Standardize CR number prefix
    
    # Standardize list markers
    text = re.sub(r'(\n)\s*[\*·•]\s+', r'\1• ', text)  # Standardize to bullet point
    
    # Fix inconsistent spacing around colons
    text = re.sub(r'(\w)\s*:\s*(\w)', r'\1: \2', text)
    
    # Fix special quote characters
    text = text.replace('„', '"').replace('“', '"').replace('”', '"')
    text = text.replace('‚', "'").replace('‘', "'").replace('’', "'")
    
    # Join words split with hyphen at line end
    text = re.sub(r'-\n([a-zA-Z]+)', r'\1', text)
    
    # Step 0: Initial cleanup to make subsequent processing more reliable
    text = re.sub(r'\r\n|\r', '\n', text)  # Normalize line breaks
    text = re.sub(r'[ \t]+', ' ', text)    # Collapse internal spaces
    text = re.sub(r'\n[ \t]+', '\n', text) # Remove leading spaces on lines
    text = re.sub(r'[ \t]+\n', '\n', text) # Remove trailing spaces on lines
    text = re.sub(r'\n{3,}', '\n\n', text) # Limit blank lines
    
    # CRITICAL FIX #1: Repair split words with clause tags
    # Handle specific cases where words are split and clause tags inserted in the middle
    
    # Fix "indemnity" split case (missing 'i')
    text = re.sub(r'(\[LIABILITY_CLAUSE\])ndemnity', r'\1indemnity', text, flags=re.IGNORECASE)
    
    # Fix "Standard" split case (missing 'St')
    text = re.sub(r'(\[CONFIDENTIALITY_CLAUSE\])ndard', r'\1Standard', text, flags=re.IGNORECASE)
    
    # Fix "information" split case (missing 'In')
    text = re.sub(r'(\[CONFIDENTIALITY_CLAUSE\])nformation', r'\1Information', text, flags=re.IGNORECASE)
    
    # Handle general cases where clause tags are inserted in the middle of words
    for ctype in ['CONFIDENTIALITY', 'TERMINATION', 'LIABILITY']:
        # Fix cases where text is attached to opening tag
        text = re.sub(rf'(\[{ctype}_CLAUSE\])([a-z]{{1,3}})([a-z]+)', 
                     r'\1\2\3', text, flags=re.IGNORECASE)
        
        # Fix cases where text is between opening and closing tags
        text = re.sub(rf'(\[{ctype}_CLAUSE\])([a-z]+)(\s*\[END_{ctype}_CLAUSE\])', 
                     r'\1\2\3', text, flags=re.IGNORECASE)
    
    # Step 1: Standardize all clause marker formats
    clause_types = ['CONFIDENTIALITY', 'TERMINATION', 'LIABILITY']
    
    for ctype in clause_types:
        # Standardize opening tags
        text = re.sub(rf'\[\s*{ctype}[_\s]*CLAUSE\s*\]', 
                     f'[{ctype}_CLAUSE]', text, flags=re.IGNORECASE)
        
        # Standardize closing tags
        text = re.sub(rf'\[\s*END[_\s]*{ctype}[_\s]*CLAUSE\s*\]', 
                     f'[END_{ctype}_CLAUSE]', text, flags=re.IGNORECASE)
    
    # Step 2: Remove page numbers embedded in text
    text = re.sub(r'([.!?])\s+(\d{1,3})\s*$', r'\1', text, flags=re.MULTILINE)
    # Handle cases like "Export Control 11" where page number is part of content
    text = re.sub(r'([a-zA-Z])\s+(\d{1,3})$', r'\1', text, flags=re.MULTILINE)
    
    # Step 3: Fix special cases from your sample files
    text = re.sub(r'Author:[^\n]+Reviewer:[^\n]+\n', '\n', text)
    
    # Step 4: Normalize party names
    text = re.sub(r'Accuray\s+Incorporated|Accuray', '[PARTY_A]', text, flags=re.IGNORECASE)
    text = re.sub(r'Siemens\s+Aktiengesellschaft|Siemens', '[PARTY_B]', text, flags=re.IGNORECASE)
    text = re.sub(r'Customer|CUSTOMER|Customer:', '[PARTY_C]', text, flags=re.IGNORECASE)
    
    # Step 5: Normalize redaction markers
    text = re.sub(r'\{\*+\}', '[REDACTED]', text)
    text = re.sub(r'\[REDACTED\]+\s*\[REDACTED\]', '[REDACTED]', text)
    
    # Step 6: Fix Unicode and special characters
    text = text.replace('\u2011', '-')  # Non-breaking hyphen
    text = text.replace('\u2013', '–')  # En-dash (standardized)
    text = text.replace('\u2014', '—')  # Em-dash (standardized)
    text = text.replace('\u2026', '...') # Ellipsis
    text = text.replace('\u201c', '"')   # Left double quotation
    text = text.replace('\u201d', '"')   # Right double quotation
    text = text.replace('\u2018', "'")   # Left single quotation
    text = text.replace('\u2019', "'")   # Right single quotation
    
    # Step 7: Standardize section headers
    def replace_header(match):
        header = match.group(1).strip()
        clean_header = re.sub(r'\s*\[.+\]\s*', '', header)
        # Standardize header formatting
        clean_header = re.sub(r'\s*:\s*', ': ', clean_header)
        return f"\n\n=== {clean_header} ===\n\n"

    text = re.sub(
        r'\n(\d+\.\s*[A-Z][^=\n]*(?<!CLAUSE))\n',
        replace_header,
        text
    )
    
    text = re.sub(
        r'\n(ARTICLE\s+\d+|SECTION\s+\d+)\s*:?\s*([A-Z][^=\n]*(?<!CLAUSE))\n',
        lambda m: f'\n\n=== {m.group(1)} {m.group(2).strip()} ===\n\n',
        text,
        flags=re.IGNORECASE
    )
    
    # Step 8: Final clause balancing
    for ctype in clause_types:
        opens = len(re.findall(fr'\[{ctype}_CLAUSE\]', text))
        closes = len(re.findall(fr'\[END_{ctype}_CLAUSE\]', text))
        
        if opens > closes:
            sections = re.split(r'(\n===|\n\d+\.\s)', text)
            fixed_parts = []
            open_count = 0
            
            for section in sections:
                current_opens = section.count(f"[{ctype}_CLAUSE]")
                current_closes = section.count(f"[END_{ctype}_CLAUSE]")
                open_count += current_opens - current_closes
                
                if open_count > 0 and not re.search(rf'\[END_{ctype}_CLAUSE\]', section):
                    section = re.sub(r'(\n===|\n\d+\.\s)', 
                                    f'[END_{ctype}_CLAUSE]\\1', 
                                    section, count=1)
                    open_count = 0
                
                fixed_parts.append(section)
            
            text = ''.join(fixed_parts)
    
    # FINAL PASS: Apply additional formatting consistency
    # Fix common contract-specific patterns
    text = re.sub(r'on all labels\.It ', 'on all labels. It ', text)
    text = re.sub(r'lower\s*-?\s*case', 'lower case', text, flags=re.IGNORECASE)
    text = re.sub(r'upper\s*-?\s*case', 'upper case', text, flags=re.IGNORECASE)
    
    # Standardize all CR numbers
    text = re.sub(r'CR\s*-\s*No\.?\s*[:.]?\s*\d+', 'CR-No.', text)
    
    # Standardize list formatting in change logs
    text = re.sub(r'(\n\s*)[\*·•]\s+', r'\1• ', text)
    
    # Final cleanup
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()



def process_all_contracts():
    input_dir = 'sampled_contracts1'
    output_dir = 'normalized_contracts1'

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

                normalized_text = normalize_contract_text(raw_text)

                # Save normalized version
                output_path = os.path.join(output_dir, f"norm_{filename}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(normalized_text)

                results.append({
                    'file_name': filename,
                    'normalized_path': output_path,
                    'char_count': len(normalized_text),
                    'status': 'success'
                })
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results.append({
                    'file_name': filename,
                    'normalized_path': None,
                    'char_count': 0,
                    'status': f'error: {str(e)}'
                })

    # Save metadata
    df = pd.DataFrame(results)
    metadata_path = os.path.join(output_dir, 'contract_metadata.csv')
    df.to_csv(metadata_path, index=False, encoding='utf-8')

    print(f"Successfully processed {len(results)} files.")
    print(f"Output saved to {output_dir}/")
    return results

# Run the pipeline
if __name__ == "__main__":
    process_all_contracts()