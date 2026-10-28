"""Week 02: Mini Semantic Search using ChromaDB."""

import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


def load_faqs(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} FAQs")
    return df


def clean_faqs(df):
    before = len(df)
    df = df.drop_duplicates(subset=['question'])
    df = df.dropna(subset=['question', 'answer'])
    df['question'] = df['question'].str.strip()
    df['answer'] = df['answer'].str.strip()
    print(f"Cleaned: {before} -> {len(df)}")
    return df


def chunk_faqs(df):
    df['chunk'] = df['question'] + " " + df['answer']
    print(f"Created {len(df)} chunks")
    return df['chunk'].tolist()


def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    print(f"Embeddings: {embeddings.shape}")
    return model, embeddings


def build_store(chunks, embeddings, df):
    client = chromadb.Client()
    collection = client.create_collection("faqs")
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{"question": q, "answer": a}
                   for q, a in zip(df['question'], df['answer'])]
    )
    print(f"Stored {collection.count()} documents")
    return collection


def search(query, model, collection, n=3):
    qe = model.encode([query]).tolist()
    return collection.query(query_embeddings=qe, n_results=n)


def main():
    df = load_faqs('week-02/data/faqs.csv')
    df = clean_faqs(df)
    chunks = chunk_faqs(df)
    model, embeddings = embed_chunks(chunks)
    collection = build_store(chunks, embeddings, df)

    queries = [
        "How do I track changes in my code?",
        "What tool helps apps run everywhere?",
        "How can I find patterns in data?",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = search(q, model, collection)
        for i, meta in enumerate(results['metadatas'][0]):
            print(f"  {i+1}. {meta['question']}")

    df['q_length'] = df['question'].str.len()
    lengths = df['q_length'].to_numpy()
    print(f"\nAvg length: {np.mean(lengths):.1f}")
    print(f"Median: {np.median(lengths):.1f}")
    print(f"Max: {np.max(lengths)}")
    print(f"75th pct: {np.percentile(lengths, 75):.1f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
