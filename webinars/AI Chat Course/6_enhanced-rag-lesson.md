# Enhanced RAG Techniques: Building Advanced Document Retrieval Systems

## Introduction

Imagine you're a librarian with a magical ability to not only find books by their titles but also understand their contents, recognize patterns across different books, and even find connections that aren't immediately obvious. This is what we're aiming to achieve with enhanced RAG (Retrieval-Augmented Generation) techniques. Today, we'll explore how to build such a system step by step.

## Part 1: Implementing Hybrid Search

Let's start by understanding why hybrid search is important. Traditional keyword search is like looking at a book's index, while semantic search is like understanding the meaning of the content. Hybrid search combines both approaches to get the best of both worlds.

### Building a Hybrid Search System

```python
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridSearcher:
    def __init__(self, documents: List[Dict]):
        """
        Initialize the hybrid searcher with documents.
        Each document should have 'content' and 'metadata' fields.
        """
        self.documents = documents
        
        # Initialize semantic search components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_embeddings = self._create_embeddings()
        
        # Initialize keyword search components
        tokenized_docs = [doc['content'].split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def _create_embeddings(self):
        """Create embeddings for all documents."""
        texts = [doc['content'] for doc in documents]
        return self.embedding_model.encode(texts)
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: The search query
            k: Number of results to return
            alpha: Weight between keyword (0) and semantic (1) search
        """
        # Get semantic search scores
        query_embedding = self.embedding_model.encode(query)
        semantic_scores = cosine_similarity([query_embedding], self.document_embeddings)[0]
        
        # Get keyword search scores
        keyword_scores = np.array(self.bm25.get_scores(query.split()))
        
        # Normalize scores
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
        keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min())
        
        # Combine scores
        combined_scores = (alpha * semantic_scores) + ((1 - alpha) * keyword_scores)
        
        # Get top k results
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        
        return [
            {
                'document': self.documents[idx],
                'score': combined_scores[idx],
                'semantic_score': semantic_scores[idx],
                'keyword_score': keyword_scores[idx]
            }
            for idx in top_indices
        ]
```

Let's examine how this system works with a practical example:

```python
# Example usage of hybrid search
documents = [
    {
        'content': 'The quick brown fox jumps over the lazy dog',
        'metadata': {'type': 'sentence', 'category': 'example'}
    },
    {
        'content': 'A fox that is brown and fast leaps above a dog that is lazy',
        'metadata': {'type': 'sentence', 'category': 'example'}
    }
]

searcher = HybridSearcher(documents)
results = searcher.hybrid_search("fast fox jumping", k=2)

# This query will match both documents, but with different scores:
# - First document: High keyword match for "fox", moderate semantic match
# - Second document: High semantic match for "fast", moderate keyword match
```

## Part 2: Adding Metadata Filtering

Metadata filtering allows us to narrow down our search based on document properties. Think of it as having special tags on library books that help you find exactly what you need.

```python
class MetadataEnhancedSearcher(HybridSearcher):
    def filtered_search(self, 
                       query: str, 
                       metadata_filters: Dict,
                       k: int = 5, 
                       alpha: float = 0.5):
        """
        Perform hybrid search with metadata filtering.
        
        Args:
            query: Search query
            metadata_filters: Dictionary of metadata field:value pairs
            k: Number of results
            alpha: Weight between keyword and semantic search
        """
        def matches_filters(doc_metadata):
            """Check if document matches all metadata filters."""
            for key, value in metadata_filters.items():
                if key not in doc_metadata:
                    return False
                    
                if isinstance(value, list):
                    if doc_metadata[key] not in value:
                        return False
                elif doc_metadata[key] != value:
                    return False
            return True
        
        # First, filter documents by metadata
        filtered_indices = [
            i for i, doc in enumerate(self.documents)
            if matches_filters(doc['metadata'])
        ]
        
        if not filtered_indices:
            return []
            
        # Get scores only for filtered documents
        query_embedding = self.embedding_model.encode(query)
        filtered_embeddings = self.document_embeddings[filtered_indices]
        
        semantic_scores = cosine_similarity(
            [query_embedding], 
            filtered_embeddings
        )[0]
        
        keyword_scores = np.array([
            self.bm25.get_score(query.split(), idx)
            for idx in filtered_indices
        ])
        
        # Normalize and combine scores
        combined_scores = self._combine_scores(
            semantic_scores, 
            keyword_scores, 
            alpha
        )
        
        # Get top k results from filtered set
        k = min(k, len(filtered_indices))
        top_k_indices = np.argsort(combined_scores)[-k:][::-1]
        
        return [
            {
                'document': self.documents[filtered_indices[idx]],
                'score': combined_scores[idx]
            }
            for idx in top_k_indices
        ]
```

## Part 3: Advanced Chunking Strategies

Different types of documents require different chunking strategies. Let's implement a flexible chunking system that can adapt to various document types.

```python
class AdaptiveChunker:
    def __init__(self):
        """Initialize the adaptive chunking system."""
        self.nlp = spacy.load('en_core_web_sm')
        
    def chunk_by_semantic_units(self, 
                              text: str, 
                              target_size: int = 1000,
                              overlap: int = 100):
        """
        Chunk text by semantic units (sentences and paragraphs).
        
        Args:
            text: Input text
            target_size: Target chunk size in characters
            overlap: Number of characters to overlap between chunks
        """
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in doc.sents:
            # If adding this sentence would exceed target size
            if current_size + len(sent.text) > target_size and current_chunk:
                # Store current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_tokens = current_chunk[-2:]  # Keep last 2 sentences
                current_chunk = overlap_tokens
                current_size = sum(len(s) for s in overlap_tokens)
            
            current_chunk.append(sent.text)
            current_size += len(sent.text)
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def chunk_by_section_markers(self, 
                               text: str,
                               markers: List[str] = ['##', 'Chapter', 'Section']):
        """
        Chunk text based on section markers.
        
        Args:
            text: Input text
            markers: List of strings that indicate section breaks
        """
        chunks = []
        current_chunk = []
        
        for line in text.split('\n'):
            # Check if line starts with any marker
            is_marker = any(line.strip().startswith(m) for m in markers)
            
            if is_marker and current_chunk:
                # Store current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            
            current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
```

### Putting It All Together

Now let's combine all these techniques into a complete RAG system:

```python
class EnhancedRAGSystem:
    def __init__(self):
        self.chunker = AdaptiveChunker()
        self.searcher = None
        
    def process_documents(self, 
                         documents: List[Dict],
                         chunk_strategy: str = 'semantic'):
        """
        Process documents using specified chunking strategy.
        
        Args:
            documents: List of document dictionaries
            chunk_strategy: 'semantic' or 'section'
        """
        processed_chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata']
            
            # Choose chunking strategy
            if chunk_strategy == 'semantic':
                chunks = self.chunker.chunk_by_semantic_units(content)
            else:
                chunks = self.chunker.chunk_by_section_markers(content)
            
            # Create chunk documents with metadata
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'source_id': doc.get('id')
                    }
                })
        
        # Initialize searcher with processed chunks
        self.searcher = MetadataEnhancedSearcher(processed_chunks)
        
    def query(self, 
              query: str, 
              metadata_filters: Dict = None,
              k: int = 5):
        """
        Query the RAG system.
        
        Args:
            query: User query
            metadata_filters: Optional metadata filters
            k: Number of results to return
        """
        if metadata_filters:
            results = self.searcher.filtered_search(
                query, 
                metadata_filters,
                k=k
            )
        else:
            results = self.searcher.hybrid_search(query, k=k)
            
        return results
```

## Practical Examples

Let's see how these enhanced techniques work in practice:

```python
# Initialize the RAG system
rag_system = EnhancedRAGSystem()

# Example documents
documents = [
    {
        'content': 'Chapter 1\nIntroduction to Python...',
        'metadata': {
            'type': 'tutorial',
            'subject': 'programming',
            'level': 'beginner'
        }
    },
    {
        'content': 'Advanced Python Concepts...',
        'metadata': {
            'type': 'tutorial',
            'subject': 'programming',
            'level': 'advanced'
        }
    }
]

# Process documents
rag_system.process_documents(documents, chunk_strategy='section')

# Query with metadata filtering
results = rag_system.query(
    "How do I start programming?",
    metadata_filters={'level': 'beginner'}
)
```

## Exercises for Understanding

1. Try Different Chunking Strategies:
   Experiment with different chunking approaches and observe their impact on search quality:

```python
def chunking_experiment():
    text = load_sample_document()
    chunker = AdaptiveChunker()
    
    # Try different strategies
    semantic_chunks = chunker.chunk_by_semantic_units(text)
    section_chunks = chunker.chunk_by_section_markers(text)
    
    # Compare results
    analyze_chunk_quality(semantic_chunks, section_chunks)
```

2. Optimize Hybrid Search Weights:
   Experiment with different weights between semantic and keyword search:

```python
def optimize_hybrid_weights(test_queries):
    weights = [0.2, 0.4, 0.6, 0.8]
    results = {}
    
    for alpha in weights:
        results[alpha] = [
            searcher.hybrid_search(query, alpha=alpha)
            for query in test_queries
        ]
    
    analyze_search_quality(results)
```

## Best Practices and Tips

1. Document Pre-processing
   - Clean and normalize text before chunking
   - Remove irrelevant content
   - Standardize formatting

2. Metadata Design
   - Choose meaningful metadata fields
   - Use consistent metadata schemas
   - Consider hierarchical metadata

3. Performance Optimization
   - Cache frequently accessed results
   - Implement batch processing for large documents
   - Monitor and adjust chunk sizes

4. Quality Assurance
   - Regularly evaluate search quality
   - Collect user feedback
   - Monitor system performance

## Conclusion

Enhanced RAG techniques provide powerful tools for building sophisticated document retrieval systems. Remember:

1. Hybrid search combines the strengths of keyword and semantic search
2. Metadata filtering adds precision to search results
3. Adaptive chunking strategies improve result quality
4. Regular monitoring and optimization are essential

Keep experimenting with these techniques and adjusting them to your specific use case to build the most effective RAG system for your needs.
