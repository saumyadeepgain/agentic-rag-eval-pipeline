"""
Quick-start script: demonstrates the full pipeline end-to-end with sample documents.
Run this to verify your setup before running the full evaluation.

Usage:
    python quickstart.py
"""

import os
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        print("Copy .env.example to .env and add your key.")
        return

    print("\n" + "="*60)
    print("AGENTIC RAG SYSTEM — Quick Start")
    print("="*60)

    # Sample documents from a fictional company policy
    sample_docs = [
        """Refund Policy: Customers may request a full refund within 30 days of purchase.
        Products must be returned in original condition. Digital downloads are non-refundable
        unless the product is defective. Refunds are processed within 5-7 business days.
        Contact support@company.com to initiate a return.""",

        """Warranty Information: All hardware products come with a 1-year limited warranty
        covering manufacturing defects. The warranty does not cover physical damage, water damage,
        or unauthorized modifications. Extended warranty plans are available for purchase separately.
        Warranty claims must be submitted within the warranty period with proof of purchase.""",

        """Shipping Policy: Standard shipping takes 5-7 business days. Express shipping (2-3 days)
        is available for an additional fee. Free shipping on orders over $50. International shipping
        is available to 25 countries. Tracking information is provided via email once the order ships.""",
    ]

    sample_query = "What is the refund policy and how long does it take?"

    print(f"\nSample query: '{sample_query}'")
    print(f"Running on {len(sample_docs)} sample documents\n")

    # Test all 4 variants
    from src.pipeline import RAGPipeline

    for variant in ["naive", "hybrid", "hybrid_rerank", "agentic_full"]:
        print(f"\n{'-'*50}")
        print(f"Variant: {variant}")
        print(f"{'-'*50}")

        try:
            pipeline = RAGPipeline.from_config(variant, domain="demo")
            pipeline.ingest(sample_docs)
            result = pipeline.run(sample_query)

            print(f"Answer: {result['answer'][:300]}...")
            print(f"Citations: {len(result.get('citations', []))}")
            print(f"Context docs used: {result.get('used_context', len(result.get('contexts', [])))}")
            if result.get('agent_trace'):
                trace = result['agent_trace']
                print(f"States: {' → '.join(trace.get('states_visited', []))}")
                print(f"Latency: {trace.get('total_latency_ms', 0):.0f}ms")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Quick start complete!")
    print("Next: Run the Streamlit app with:")
    print("  streamlit run app/streamlit_app.py")
    print("\nOr run the full evaluation:")
    print("  See README.md for evaluation commands")
    print("="*60)

if __name__ == "__main__":
    main()
