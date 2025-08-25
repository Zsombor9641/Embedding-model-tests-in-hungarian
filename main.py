import os
import re
import chromadb
import json
import uuid
import time
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from models import BGEEmbedder, OllamaEmbedder, OpenAIEmbedder, GeminiEmbedder, HuBERTEmbedder


def build_chroma_collection(texts, model, collection_name=None):
    """Build ChromaDB collection with embeddings - simplified version with timing"""
    start_time = time.time()

    # Use unique collection name to avoid conflicts
    if collection_name is None:
        collection_name = f"embeddings_{int(time.time())}"
    else:
        collection_name = f"{collection_name}_{int(time.time())}"

    # Create client with temporary in-memory storage
    client = chromadb.Client()

    # Create new collection (no need to delete since name is unique)
    collection = client.create_collection(name=collection_name)
    print(f"Created collection: {collection_name}")

    # Generate embeddings
    print("Generating embeddings...")
    embedding_start = time.time()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embedding_time = time.time() - embedding_start
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Embedding generation time: {embedding_time:.2f} seconds')

    # Generate IDs for documents
    ids = [f"doc_{i}" for i in range(len(texts))]

    # Add embeddings to collection
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        ids=ids
    )

    total_time = time.time() - start_time
    print(f"Added {len(texts)} documents to collection")
    print(f"Total collection build time: {total_time:.2f} seconds")

    return collection, embeddings, {
        'total_time': total_time,
        'embedding_time': embedding_time,
        'collection_time': total_time - embedding_time
    }


def parse_topics(filepath):
    """Parse topics from markdown file"""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    topics = []
    chunks = re.split(r"^##\s*", content, flags=re.MULTILINE)
    for chunk in chunks[1:]:
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        description = "\n".join(lines[1:]).strip()
        text = f"{title}: {description}"
        topics.append(text)
    return topics


def evaluate_models_cleanservice_data(model, model_name):
    """Evaluate models using Clearservice data with ChromaDB and detailed timing"""
    print(f"\n📊 Evaluating {model_name}...")
    total_start_time = time.time()

    # Load questions
    df = pd.read_csv("data/cs_qa.csv")
    print(f"Clearservice data loaded: {len(df)} items.")

    # Parse topics
    topic_chunks = parse_topics("data/topics.txt")
    print(f"Topics parsed: {len(topic_chunks)} topics.")

    # Create ChromaDB collection with timing
    collection, embeddings, build_times = build_chroma_collection(topic_chunks, model, "cleanservice_topics")

    # Evaluate retriever with the embedder model
    search_start_time = time.time()
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_3 = 0
    num_questions = len(df)

    query_times = []
    embedding_times = []

    print(f"Evaluating {num_questions} questions...")

    for idx, question in enumerate(df['question']):
        if (idx + 1) % 10 == 0:
            print(f"Processing question {idx + 1}/{num_questions}")

        topic = df['topic'][idx]

        # Time individual query embedding
        query_embedding_start = time.time()
        query_embedding = model.encode([question], normalize_embeddings=True)[0]
        query_embedding_time = time.time() - query_embedding_start
        embedding_times.append(query_embedding_time)

        # Time ChromaDB query
        query_start = time.time()
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        query_time = time.time() - query_start
        query_times.append(query_time)

        # Extract topic names from results
        top_texts = results['documents'][0]
        result = [text.split(':', 1)[0].strip() for text in top_texts]

        # Debug: print first few results
        if idx < 3:
            print(f"Question: {question}")
            print(f"Expected topic: {topic}")
            print(f"Retrieved topics: {result}")
            print(f"Query embedding time: {query_embedding_time:.4f}s")
            print(f"Search time: {query_time:.4f}s")
            print("-" * 50)

        # Recall@1
        if topic == result[0]:
            recall_at_1 += 1
        # Recall@3
        if topic in result:
            recall_at_3 += 1
            rank = result.index(topic) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    search_time = time.time() - search_start_time
    total_time = time.time() - total_start_time

    if num_questions > 0:
        recall_at_1_score = recall_at_1 / num_questions
        recall_at_3_score = recall_at_3 / num_questions
        mrr = sum(reciprocal_ranks) / num_questions

        # Calculate timing statistics
        avg_query_embedding_time = np.mean(embedding_times)
        avg_search_time = np.mean(query_times)
        total_query_embedding_time = sum(embedding_times)
        total_search_time = sum(query_times)

        print(f"\n📈 Final Results for {model_name}:")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"Recall@1: {recall_at_1_score:.4f}")
        print(f"Recall@3: {recall_at_3_score:.4f}")

        print(f"\n⏱️  Timing Results:")
        print(f"Total evaluation time: {total_time:.2f}s")
        print(f"Collection build time: {build_times['total_time']:.2f}s")
        print(f"  - Embedding generation: {build_times['embedding_time']:.2f}s")
        print(f"  - Collection setup: {build_times['collection_time']:.2f}s")
        print(f"Search phase time: {search_time:.2f}s")
        print(f"  - Total query embeddings: {total_query_embedding_time:.2f}s")
        print(f"  - Total search time: {total_search_time:.2f}s")
        print(f"Average per query:")
        print(f"  - Embedding time: {avg_query_embedding_time:.4f}s")
        print(f"  - Search time: {avg_search_time:.4f}s")
        print(f"  - Total per query: {(total_query_embedding_time + total_search_time) / num_questions:.4f}s")

        return {
            'mrr': mrr,
            'recall_at_1': recall_at_1_score,
            'recall_at_3': recall_at_3_score,
            'timing': {
                'total_time': total_time,
                'build_time': build_times['total_time'],
                'embedding_generation_time': build_times['embedding_time'],
                'collection_setup_time': build_times['collection_time'],
                'search_phase_time': search_time,
                'total_query_embedding_time': total_query_embedding_time,
                'total_search_time': total_search_time,
                'avg_query_embedding_time': avg_query_embedding_time,
                'avg_search_time': avg_search_time,
                'avg_total_per_query': (total_query_embedding_time + total_search_time) / num_questions,
                'queries_per_second': num_questions / search_time if search_time > 0 else 0
            }
        }
    else:
        print("No questions to evaluate.")
        return None


def test_single_model(model_class, model_name):
    """Test a single model with error handling and detailed timing"""
    print(f"\n{'=' * 60}")
    print(f"🧪 Testing: {model_name}")
    print(f"{'=' * 60}")

    try:
        # Initialize model
        model_init_start = time.time()
        model = model_class
        model_init_time = time.time() - model_init_start
        print(f"Model initialization time: {model_init_time:.2f}s")

        results = evaluate_models_cleanservice_data(model, model_name)

        if results:
            print(f"✅ {model_name} completed successfully!")
            print(f"🏆 Best metric - MRR: {results['mrr']:.3f}")
            print(f"⚡ Speed - {results['timing']['queries_per_second']:.1f} queries/second")
            return results
        else:
            print(f"❌ {model_name} failed to return results.")
            return None

    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_full_evaluation():
    """Run evaluation on all models with comprehensive timing"""
    evaluation_start_time = time.time()

    models = [
        ("HuBERT Hungarian", HuBERTEmbedder()),
        ("BGE-M3", BGEEmbedder()),
        ("SentenceTransformer Multilingual",
         SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')),
    ]

    # Add API-based models if keys are available
    load_dotenv()

    # Test OpenAI availability first
    if os.getenv('OPENAI_API_KEY'):
        print("🧪 Testing OpenAI API availability...")
        try:
            # Test with a small request first
            test_model = OpenAIEmbedder("text-embedding-ada-002")
            test_embedding = test_model.encode(["test"], convert_to_numpy=True, normalize_embeddings=True)
            print("✅ OpenAI API working - adding OpenAI models")
            models.extend([
                ("OpenAI Ada-002", OpenAIEmbedder("text-embedding-ada-002")),
                ("OpenAI 3-Small", OpenAIEmbedder("text-embedding-3-small"))
            ])
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                print("⚠️  OpenAI quota exceeded - skipping OpenAI models")
                print("   💡 Check your billing at: https://platform.openai.com/account/usage")
            else:
                print(f"⚠️  OpenAI API error - skipping OpenAI models: {e}")

    # Test Gemini availability
    if os.getenv('GEMINI_API_KEY'):
        print("🧪 Testing Gemini API availability...")
        try:
            test_model = GeminiEmbedder()
            test_embedding = test_model.encode(["test"], convert_to_numpy=True, normalize_embeddings=True)
            print("✅ Gemini API working - adding Gemini model")
            models.append(("Gemini", GeminiEmbedder()))
        except Exception as e:
            print(f"⚠️  Gemini API error - skipping Gemini model: {e}")

    # Test Ollama availability
    try:
        import ollama
        ollama.list()
        models.extend([
            ("Ollama Nomic", OllamaEmbedder("nomic-embed-text:latest")),
            ("Ollama MiniLM", OllamaEmbedder("all-minilm:latest"))
        ])
    except:
        print("Ollama not available, skipping Ollama models.")

    results = {}
    timing_summary = {}

    for model_name, model in models:
        result = test_single_model(model, model_name)
        if result:
            results[model_name] = {
                'mrr': result['mrr'],
                'recall_at_1': result['recall_at_1'],
                'recall_at_3': result['recall_at_3']
            }
            timing_summary[model_name] = result['timing']

    total_evaluation_time = time.time() - evaluation_start_time

    # Print summary
    print(f"\n{'=' * 80}")
    print("📊 FINAL RESULTS SUMMARY")
    print(f"{'=' * 80}")

    if results:
        print(f"{'Model':<30} {'MRR':<8} {'R@1':<8} {'R@3':<8} {'QPS':<8} {'Avg/Q':<8}")
        print("-" * 80)

        # Sort by MRR descending
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mrr'], reverse=True)

        for model_name, result in sorted_results:
            timing = timing_summary[model_name]
            print(f"{model_name:<30} "
                  f"{result['mrr']:<8.3f} "
                  f"{result['recall_at_1']:<8.3f} "
                  f"{result['recall_at_3']:<8.3f} "
                  f"{timing['queries_per_second']:<8.1f} "
                  f"{timing['avg_total_per_query']:<8.3f}")

        # Timing summary
        print(f"\n⏱️  TIMING SUMMARY")
        print("-" * 80)
        print(f"Total evaluation time: {total_evaluation_time:.1f} seconds")
        print(f"Models tested: {len(results)}")

        fastest_model = min(timing_summary.items(), key=lambda x: x[1]['avg_total_per_query'])
        slowest_model = max(timing_summary.items(), key=lambda x: x[1]['avg_total_per_query'])

        print(f"⚡ Fastest model: {fastest_model[0]} ({fastest_model[1]['avg_total_per_query']:.3f}s per query)")
        print(f"🐌 Slowest model: {slowest_model[0]} ({slowest_model[1]['avg_total_per_query']:.3f}s per query)")
        print(
            f"🏃 Speed difference: {slowest_model[1]['avg_total_per_query'] / fastest_model[1]['avg_total_per_query']:.1f}x")

        # Save results with timing
        comprehensive_results = {
            'evaluation_metadata': {
                'total_evaluation_time': total_evaluation_time,
                'models_tested': len(results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': results,
            'timing': timing_summary
        }

        with open("evaluation_results_with_timing.json", 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: evaluation_results_with_timing.json")

        # Create performance vs speed ranking
        print(f"\n🏆 PERFORMANCE vs SPEED RANKING")
        print("-" * 80)
        # Calculate composite score (higher MRR, lower time is better)
        composite_scores = {}
        for model_name in results:
            mrr = results[model_name]['mrr']
            time_per_query = timing_summary[model_name]['avg_total_per_query']
            # Normalize scores (MRR/time_per_query gives higher score for better performance and speed)
            composite_scores[model_name] = mrr / time_per_query

        sorted_composite = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, score) in enumerate(sorted_composite, 1):
            mrr = results[model_name]['mrr']
            time_per_query = timing_summary[model_name]['avg_total_per_query']
            print(f"{i:2d}. {model_name:<30} Score: {score:.2f} (MRR: {mrr:.3f}, Time: {time_per_query:.3f}s)")

    else:
        print("❌ No successful evaluations completed.")

    return results, timing_summary


if __name__ == "__main__":
    results, timing = run_full_evaluation()
