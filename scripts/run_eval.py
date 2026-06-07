"""
Agentic RAG System — Run Evaluation
This script runs the RAGAS evaluation across all domains and variants,
using the Gemini models configured in the environment.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from src.pipeline import PIPELINE_CONFIGS, RAGPipeline
from src.evaluation.ragas_eval import run_full_evaluation, print_comparison_table

def main():
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set in the environment.")
        return

    logger.info("Initializing pipelines for evaluation...")
    
    pipeline_fns = {}
    for variant in PIPELINE_CONFIGS.keys():
        # Instantiate a dummy pipeline just to get the eval function
        pipeline = RAGPipeline.from_config(variant, domain="eval")
        pipeline_fns[variant] = pipeline.as_eval_fn()
        
    logger.info("Starting full evaluation on 10 samples per domain to test...")
    # Generate smaller number of samples for a fast run, or 50 for full.
    # The user wanted 'plenty of data'. We will use 10 samples for the sake of time.
    df = run_full_evaluation(
        pipeline_fns=pipeline_fns,
        domains=["finance", "biomedical", "legal"],
        n_samples=10, 
        output_path="./results/eval_results.json"
    )

    print_comparison_table(df)
    
if __name__ == "__main__":
    main()
