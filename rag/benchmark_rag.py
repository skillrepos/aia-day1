#!/usr/bin/env python3
"""
Benchmark script for RAG system performance
"""

import time
import statistics
from rag_skeleton import KnowledgeBase

def benchmark_search(kb: KnowledgeBase, queries: list, query_type: str, iterations: int = 3):
    """Benchmark search performance for a specific query type"""
    print(f"\n{query_type}:")
    print("-" * 60)

    all_times = []

    for query in queries:
        query_times = []
        print(f"\n  Query: '{query}'")
        print(f"  Running {iterations} iterations (to measure consistency)...")

        for i in range(iterations):
            start_time = time.time()
            results = kb.search(query, max_results=3)
            end_time = time.time()

            elapsed = end_time - start_time
            query_times.append(elapsed)

            # Handle None results gracefully
            result_count = len(results) if results else 0
            print(f"    • Iteration {i+1}: {elapsed:.4f}s → {result_count} results")

        avg_time = statistics.mean(query_times)
        std_dev = statistics.stdev(query_times) if len(query_times) > 1 else 0

        print(f"  ✓ Average: {avg_time:.4f}s (±{std_dev:.4f}s)")
        all_times.extend(query_times)

    return all_times

def main():
    """Run benchmark tests on the RAG system"""
    print("="*60)
    print("RAG System Performance Benchmark")
    print("="*60)

    # Initialize and time the knowledge base loading
    print("\n[Phase 1] Knowledge Base Initialization")
    print("-"*40)

    start_time = time.time()
    kb = KnowledgeBase("../knowledge_base_pdfs")
    load_time = time.time() - start_time

    print(f"✓ Knowledge base loaded in {load_time:.2f} seconds")

    # Get statistics
    stats = kb.get_statistics()
    if stats:
        print(f"✓ Loaded {stats['total_documents']} documents")
        print(f"✓ Total size: {stats['total_characters']:,} characters")

    # Benchmark different query types
    print("\n[Phase 2] Search Performance Testing")
    print("-"*40)
    print("Testing different query types to measure:")
    print("  • Search speed and consistency")
    print("  • Semantic understanding (natural vs keyword queries)")
    print("  • Edge case handling")

    # Define query categories
    simple_queries = [
        "return policy",
        "shipping",
        "password reset",
    ]

    complex_queries = [
        "How do I return a defective product?",
        "What are my shipping options for international orders?",
        "I forgot my password and can't access my email",
    ]

    edge_cases = [
        "xyz123notfound",
        "a",
        "the quick brown fox jumps over the lazy dog"
    ]

    # Run benchmarks for each category
    search_times = []
    search_times.extend(benchmark_search(kb, simple_queries, "Simple Keyword Queries", iterations=3))
    search_times.extend(benchmark_search(kb, complex_queries, "Complex Natural Language Queries", iterations=3))
    search_times.extend(benchmark_search(kb, edge_cases, "Edge Cases (testing robustness)", iterations=3))

    # Calculate overall statistics
    print("\n" + "="*60)
    print("Overall Performance Summary")
    print("="*60)

    if search_times:
        print(f"Total searches performed: {len(search_times)}")
        print(f"Average search time: {statistics.mean(search_times):.4f}s")
        print(f"Median search time: {statistics.median(search_times):.4f}s")
        print(f"Min search time: {min(search_times):.4f}s")
        print(f"Max search time: {max(search_times):.4f}s")

        if len(search_times) > 1:
            print(f"Standard deviation: {statistics.stdev(search_times):.4f}s")

    # Test memory efficiency
    print("\n[Phase 3] Rapid Search Performance")
    print("-"*40)
    print("Testing system performance under rapid consecutive queries...")
    print("(This simulates multiple users or high-frequency searches)\n")

    # Perform rapid consecutive searches
    rapid_start = time.time()
    rapid_count = 20
    for i in range(rapid_count):
        kb.search("test query", max_results=1)
    rapid_time = time.time() - rapid_start

    print(f"✓ {rapid_count} rapid searches completed in {rapid_time:.2f}s")
    print(f"✓ Average time per rapid search: {rapid_time/rapid_count:.4f}s")
    print(f"✓ Queries per second: {rapid_count/rapid_time:.1f}")

    # Performance grade
    print("\n" + "="*60)
    print("Performance Grade")
    print("="*60)

    avg_search = statistics.mean(search_times) if search_times else 0

    if avg_search < 0.1:
        grade = "A+ (Excellent)"
    elif avg_search < 0.3:
        grade = "A (Very Good)"
    elif avg_search < 0.5:
        grade = "B (Good)"
    elif avg_search < 1.0:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Optimization)"

    print(f"Grade: {grade}")
    print(f"Based on average search time of {avg_search:.3f} seconds")

    print("\n✅ Benchmark completed successfully!")

if __name__ == "__main__":
    main()

