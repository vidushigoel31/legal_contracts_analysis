"""
Complete Contract Analysis Pipeline - Azure OpenAI Version
Analyzes complete contract files from a folder using Azure OpenAI
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from openai import AzureOpenAI
import time
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'input_folder': Path("normalized_contracts1"),
    'output_csv': Path("final_contract_analysis_combined.csv"),
    'azure_endpoint': "https://swccoai4alaoa01.openai.azure.com/",  # Replace with your Azure endpoint
    'api_key': "2tspozJ9W7AN1LDyvt0RZyLQySYYBDrwPnUDvOxceA50ncamMD94JQQJ99BDACfhMk5XJ3w3AAABACOGXUzU",  # Replace with your Azure API key
    'api_version': "2024-02-15-preview",  # Replace with your API version
    'azure_deployment': "gpt-4",  # Your deployed model name in Azure
    'max_contracts': 14,  # Process 14 contracts (positions 2-15)
    'start_index': 0,     # Skip first contract (start from index 1 = contract 2)
    'rate_limit_delay': 1.0,  # Set to number for testing (e.g., 5)
    'rate_limit_delay': 1.0,  # seconds between API calls
}

@dataclass
class ContractResult:
    contract_id: str
    summary: str = ""
    word_count: int = 0
    termination_clause: str = ""
    confidentiality_clause: str = ""
    liability_clause: str = ""
    extraction_success: bool = False
    summary_success: bool = False
    processing_time: float = 0.0
    file_size: int = 0

class CompleteContractAnalyzer:
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, azure_deployment: str):
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint
        )
        self.azure_deployment = azure_deployment
        self.prompts = self._setup_prompts()
        
    def _setup_prompts(self) -> Dict[str, str]:
        """Setup all prompts for extraction and summarization"""
        return {
            'summary': """You are an expert legal contract analyst. Create a precise summary following this structure:

REQUIREMENTS:
- Exactly 100-150 words
- Must include: Purpose, Party Obligations, Risks/Penalties
- Use clear, professional language
- Focus on key business terms

EXAMPLE FORMAT:
"This [Agreement Type] between [Party A] and [Party B] establishes [main purpose/objective]. [Party A] must [key obligations], while [Party B] must [key obligations]. The agreement includes [payment/financial terms]. Key risks include [major risks], with penalties of [specific penalties if any]. Termination occurs under [termination conditions]. [Additional notable provisions or compliance requirements]."

SAMPLE OUTPUT:
"This Software Development Agreement between TechCorp and StartupXYZ establishes custom software development services for a mobile application. TechCorp must deliver fully functional software within 6 months, provide documentation, and ensure code quality, while StartupXYZ must pay $150,000 in milestone payments and provide timely feedback on deliverables. The agreement includes penalty clauses for late delivery ($5,000 per week) and scope changes requiring written approval. Key risks include intellectual property disputes, with TechCorp retaining code ownership unless full payment is received. Termination occurs with 30-day notice or immediately for material breach, with partial payment due for completed work."

Now create a summary for this contract:

CONTRACT TEXT:
{contract_text}

STRUCTURED SUMMARY (100-150 words):""",

            
"termination": """
You are a legal contract analyzer. Analyze the following contract text and extract ONLY termination conditions/clauses.
Look for clauses that describe:
- How and when the contract can be terminated
- Notice periods for termination
- Conditions that trigger automatic termination
- Rights of parties to terminate
- Consequences of termination
Contract Text:
{text}
Instructions:
- If termination clauses are found, extract the complete relevant sentences/paragraphs
- If no termination clauses are found, respond with "NONE"
- Be precise and include specific terms, timeframes, and conditions
- Do not summarize, extract the actual clause text
Extracted Termination Clause:""",
            "confidentiality": """
You are a legal contract analyzer. Analyze the following contract text and extract ONLY confidentiality/non-disclosure clauses.
Look for clauses that describe:
- Confidential information definitions
- Obligations to maintain confidentiality
- Non-disclosure requirements
- Exceptions to confidentiality
- Duration of confidentiality obligations
Contract Text:
{text}
Instructions:
- If confidentiality clauses are found, extract the complete relevant sentences/paragraphs
- If no confidentiality clauses are found, respond with "NONE"
- Be precise and include specific definitions and obligations
- Do not summarize, extract the actual clause text
Extracted Confidentiality Clause:""",
            "liability": """
You are a legal contract analyzer. Analyze the following contract text and extract ONLY liability clauses.
Look for clauses that describe:
- Limitation of liability
- Indemnification provisions
- Disclaimer of warranties
- Allocation of risks
- Damages and penalties
- Liability caps or exclusions
Contract Text:
{text}
Instructions:
- If liability clauses are found, extract the complete relevant sentences/paragraphs
- If no liability clauses are found, respond with "NONE"
- Be precise and include specific limitations, exclusions, and amounts
- Do not summarize, extract the actual clause text
Extracted Liability Clause:"""
        }
    
    def load_contract_file(self, file_path: Path) -> str:
        """Load complete contract text from file"""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"  Loaded {len(content)} characters from {file_path.name}")
            return content
        except Exception as e:
            logger.error(f"  Error loading {file_path}: {e}")
            return ""
    
    def make_api_call(self, prompt: str, max_tokens: int = 300) -> str:
        """Make Azure OpenAI API call with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert legal contract analyzer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return ""
    
    def extract_all_clauses(self, contract_text: str) -> Dict[str, str]:
        """Extract all clause types from contract text"""
        clauses = {}
        
        for clause_type in ['termination', 'confidentiality', 'liability']:
            logger.info(f"    Extracting {clause_type} clause...")
            
            prompt = self.prompts[clause_type].format(text=contract_text)
            result = self.make_api_call(prompt, max_tokens=400)
            
            # Clean result
            if result and result.upper() != "NONE":
                clauses[f'{clause_type}_clause'] = result
            else:
                clauses[f'{clause_type}_clause'] = ""
            
            # Rate limiting
            time.sleep(CONFIG['rate_limit_delay'])
        
        return clauses
    
    def generate_summary(self, contract_text: str) -> Dict[str, any]:
        """Generate contract summary"""
        logger.info("    Generating summary...")
        
        prompt = self.prompts['summary'].format(contract_text=contract_text)
        summary = self.make_api_call(prompt, max_tokens=250)
        
        # Validate summary
        word_count = len(summary.split()) if summary else 0
        is_valid = 100 <= word_count <= 150 if summary else False
        
        return {
            'summary': summary,
            'word_count': word_count,
            'is_valid': is_valid
        }
    
    def analyze_contract(self, contract_file: Path) -> ContractResult:
        """Analyze a single contract file completely"""
        start_time = time.time()
        
        contract_id = contract_file.stem
        logger.info(f"Analyzing contract: {contract_id}")
        
        # Load contract text
        contract_text = self.load_contract_file(contract_file)
        if not contract_text:
            return ContractResult(contract_id=contract_id)
        
        result = ContractResult(
            contract_id=contract_id,
            file_size=len(contract_text)
        )
        
        try:
            # Extract clauses
            clauses = self.extract_all_clauses(contract_text)
            result.termination_clause = clauses.get('termination_clause', '')
            result.confidentiality_clause = clauses.get('confidentiality_clause', '')
            result.liability_clause = clauses.get('liability_clause', '')
            
            # Check if any clauses were extracted
            result.extraction_success = any(clauses.values())
            
            # Generate summary
            summary_result = self.generate_summary(contract_text)
            result.summary = summary_result['summary']
            result.word_count = summary_result['word_count']
            result.summary_success = summary_result['is_valid']
            
            logger.info(f"  ✅ Success: Clauses={result.extraction_success}, "
                       f"Summary={result.summary_success} ({result.word_count} words)")
            
        except Exception as e:
            logger.error(f"  ❌ Error analyzing {contract_id}: {e}")
        
        result.processing_time = time.time() - start_time
        return result

def load_contract_files(folder_path: Path, max_contracts: Optional[int] = None, start_index: int = 0) -> List[Path]:
    """Load contract files from folder with start index support"""
    logger.info(f"Loading contract files from {folder_path}...")
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all text files in the folder
    contract_files = []
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.text']:
            contract_files.append(file_path)
    
    # Sort files for consistent processing order
    contract_files.sort()
    
    logger.info(f"Found {len(contract_files)} contract files")
    
    # Apply start index (skip first N contracts)
    if start_index > 0:
        contract_files = contract_files[start_index:]
        logger.info(f"Skipping first {start_index} contracts, {len(contract_files)} remaining")
    
    # Limit contracts if specified
    if max_contracts:
        contract_files = contract_files[:max_contracts]
        logger.info(f"Limited to {len(contract_files)} contracts for processing")
    
    return contract_files

def save_results(results: List[ContractResult], output_path: Path):
    """Save results and generate analytics"""
    
    # Convert to DataFrame
    data = []
    for result in results:
        data.append({
            'contract_id': result.contract_id,
            'file_size_chars': result.file_size,
            'summary': result.summary,
            'word_count': result.word_count,
            'termination_clause': result.termination_clause,
            'confidentiality_clause': result.confidentiality_clause,
            'liability_clause': result.liability_clause,
            'extraction_success': result.extraction_success,
            'summary_success': result.summary_success,
            'processing_time_seconds': round(result.processing_time, 2)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    # Generate analytics
    print("\n" + "="*60)
    print("FINAL PROCESSING RESULTS")
    print("="*60)
    
    total = len(results)
    successful_extractions = sum(1 for r in results if r.extraction_success)
    successful_summaries = sum(1 for r in results if r.summary_success)
    
    print(f"Total contracts processed: {total}")
    print(f"Successful clause extractions: {successful_extractions}/{total} ({successful_extractions/total*100:.1f}%)")
    print(f"Valid summaries (100-150 words): {successful_summaries}/{total} ({successful_summaries/total*100:.1f}%)")
    
    if results:
        avg_time = sum(r.processing_time for r in results) / len(results)
        total_time = sum(r.processing_time for r in results)
        avg_size = sum(r.file_size for r in results) / len(results)
        print(f"Average processing time per contract: {avg_time:.1f} seconds")
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Average file size: {avg_size:,.0f} characters")
    
    # Clause extraction breakdown
    print(f"\nClause Extraction Breakdown:")
    termination_count = sum(1 for r in results if r.termination_clause)
    confidentiality_count = sum(1 for r in results if r.confidentiality_clause)
    liability_count = sum(1 for r in results if r.liability_clause)
    
    print(f"  Termination clauses: {termination_count}/{total} ({termination_count/total*100:.1f}%)")
    print(f"  Confidentiality clauses: {confidentiality_count}/{total} ({confidentiality_count/total*100:.1f}%)")
    print(f"  Liability clauses: {liability_count}/{total} ({liability_count/total*100:.1f}%)")
    
    # Summary quality
    valid_summaries = [r for r in results if r.summary_success]
    if valid_summaries:
        avg_words = sum(r.word_count for r in valid_summaries) / len(valid_summaries)
        print(f"\nSummary Quality:")
        print(f"  Average word count: {avg_words:.1f}")
        print(f"  Word count range: {min(r.word_count for r in valid_summaries)}-{max(r.word_count for r in valid_summaries)}")
    
    print(f"\n📁 Results saved to: {output_path}")
    
    # Show sample results
    show_sample_results(results[:3])

def show_sample_results(sample_results: List[ContractResult]):
    """Display sample results for quality check"""
    print(f"\n" + "="*60)
    print("SAMPLE RESULTS")
    print("="*60)
    
    for i, result in enumerate(sample_results, 1):
        print(f"\nSample {i}: {result.contract_id}")
        print(f"File size: {result.file_size:,} characters")
        print(f"Processing time: {result.processing_time:.1f}s")
        print(f"Summary ({result.word_count} words): {result.summary[:200]}...")
        
        clauses_found = []
        if result.termination_clause:
            clauses_found.append("Termination")
        if result.confidentiality_clause:
            clauses_found.append("Confidentiality")
        if result.liability_clause:
            clauses_found.append("Liability")
        
        print(f"Clauses extracted: {', '.join(clauses_found) if clauses_found else 'None'}")
        print("-" * 40)

def estimate_cost_and_time(num_contracts: int, azure_deployment: str = "gpt-4"):
    """Estimate API costs and processing time for Azure OpenAI"""
    # Rough estimates
    api_calls_per_contract = 4  # 3 clause extractions + 1 summary
    tokens_per_call = 1500  # average input + output (higher for complete contracts)
    
    total_calls = num_contracts * api_calls_per_contract
    total_tokens = total_calls * tokens_per_call
    
    # Azure OpenAI pricing (as of 2024)
    if "gpt-4" in azure_deployment.lower():
        cost_per_1k_tokens = 0.06   # Combined input/output estimate
    else:
        cost_per_1k_tokens = 0.002  # GPT-3.5 turbo estimate
    
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    estimated_time = total_calls * (1.5 + CONFIG['rate_limit_delay'])  # seconds
    
    print(f"\n💰 Cost & Time Estimate (Azure OpenAI):")
    print(f"  Contracts: {num_contracts}")
    print(f"  API calls: {total_calls}")
    print(f"  Estimated tokens: {total_tokens:,}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print(f"  Estimated time: {estimated_time/60:.1f} minutes")

def main():
    """Main execution function"""
    print("🚀 Starting Complete Contract Analysis Pipeline (Azure OpenAI)")
    print(f"Model: {CONFIG['azure_deployment']}")
    print(f"Input folder: {CONFIG['input_folder']}")
    
    # Initialize analyzer
    analyzer = CompleteContractAnalyzer(
        CONFIG['api_key'],
        CONFIG['azure_endpoint'],
        CONFIG['api_version'],
        CONFIG['azure_deployment']
    )
    
    # Load contract files
    contract_files = load_contract_files(
        CONFIG['input_folder'], 
        CONFIG['max_contracts'], 
        CONFIG.get('start_index', 0)  # Use start_index if specified
    )
    
    if not contract_files:
        print("❌ No contract files found!")
        return
    
    # Show estimates
    estimate_cost_and_time(len(contract_files), CONFIG['azure_deployment'])
    
    # Confirm before proceeding
    if len(contract_files) > 10:
        response = input(f"\n⚠️  Process {len(contract_files)} contracts? This will take time and cost money. (y/n): ")
        if response.lower() != 'y':
            print("❌ Processing cancelled.")
            return
    
    print(f"\n📊 Processing {len(contract_files)} contracts...")
    
    # Process all contracts
    results = []
    for i, contract_file in enumerate(contract_files, 1):
        print(f"\n[{i}/{len(contract_files)}] Processing {contract_file.name}")
        
        result = analyzer.analyze_contract(contract_file)
        results.append(result)
        
        # Progress update every 10 contracts
        if i % 10 == 0:
            success_rate = sum(1 for r in results if r.extraction_success or r.summary_success) / len(results) * 100
            print(f"  Progress: {i}/{len(contract_files)} completed, {success_rate:.1f}% success rate")
    
    # Save results
    save_results(results, CONFIG['output_csv'])
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📈 Check {CONFIG['output_csv']} for complete results")

if __name__ == "__main__":
    main()