# Information Retrieval and Search

## Table of Contents

1. [Introduction](#introduction)
2. [Vector Space Model for IR](#vector-space-model-for-ir)
3. [TF-IDF Weighting](#tf-idf-weighting)
4. [BM25 Ranking Function](#bm25-ranking-function)
5. [Relevance Ranking](#relevance-ranking)
6. [Query Expansion](#query-expansion)
7. [Inverted Index](#inverted-index)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Advanced Retrieval Models](#advanced-retrieval-models)
10. [Key Takeaways](#key-takeaways)

## Introduction

Information Retrieval (IR) is the task of finding relevant documents from a collection given a user query. IR systems power search engines, recommendation systems, and question answering.

The IR problem: Given query $q$ and document collection $D = \{d_1, \ldots, d_N\}$, rank documents by relevance to $q$.

IR involves:
- **Indexing**: Organizing documents for efficient retrieval
- **Ranking**: Scoring documents by relevance
- **Evaluation**: Measuring system effectiveness

## Vector Space Model for IR

The Vector Space Model (VSM) represents documents and queries as vectors, enabling geometric similarity computation.

### VSM Representation

Documents and queries represented as term vectors:

$$\mathbf{d} = [w_{d,1}, w_{d,2}, \ldots, w_{d,V}]$$
$$\mathbf{q} = [w_{q,1}, w_{q,2}, \ldots, w_{q,V}]$$

where $w_{d,i}$ is the weight of term $i$ in document $d$, and $V$ is vocabulary size.

### Similarity Measures

**Cosine similarity** (most common):

$$\text{sim}(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{||\mathbf{q}|| \times ||\mathbf{d}||} = \frac{\sum_{i=1}^{V} w_{q,i} w_{d,i}}{\sqrt{\sum_{i=1}^{V} w_{q,i}^2} \sqrt{\sum_{i=1}^{V} w_{d,i}^2}}$$

**Dot product** (unnormalized):

$$\text{sim}(q, d) = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{V} w_{q,i} w_{d,i}$$

**Jaccard similarity** (for binary vectors):

$$\text{sim}(q, d) = \frac{|\mathbf{q} \cap \mathbf{d}|}{|\mathbf{q} \cup \mathbf{d}|}$$

### VSM Assumptions

VSM makes simplifying assumptions:
- **Term independence**: Terms are independent
- **Position invariance**: Word order doesn't matter
- **Geometric similarity**: Similar documents are close in space

These assumptions enable efficient computation but ignore semantic relationships.

## TF-IDF Weighting

TF-IDF weighting balances term frequency with inverse document frequency to identify distinctive terms.

### Term Frequency

Term frequency measures how often a term appears in a document:

$$tf(t, d) = \frac{\text{count}(t, d)}{\text{length}(d)}$$

Alternative formulations:
- **Raw count**: $tf(t, d) = \text{count}(t, d)$
- **Log normalization**: $tf(t, d) = 1 + \log(\text{count}(t, d))$
- **Boolean**: $tf(t, d) = 1$ if $t \in d$, else $0$

### Inverse Document Frequency

IDF measures term rarity across the collection:

$$idf(t, D) = \log \frac{|D|}{df(t)}$$

where $df(t)$ is document frequency (number of documents containing $t$).

IDF is high for rare terms and low for common terms.

### TF-IDF Weight

Combining TF and IDF:

$$w_{t,d} = tf(t, d) \times idf(t, D)$$

This gives high weight to:
- Terms frequent in the document (high TF)
- Terms rare in the collection (high IDF)

### Normalization

Normalization addresses document length variation:

**Cosine normalization**:
$$w_{t,d} = \frac{tf(t, d) \times idf(t, D)}{||\mathbf{d}||_2}$$

**Pivoted normalization**: Adjusts for document length bias:
$$w_{t,d} = \frac{tf(t, d) \times idf(t, D)}{(1 - s) + s \times \frac{|d|}{\text{avgdl}}}$$

where $s$ is a slope parameter and $\text{avgdl}$ is average document length.

## BM25 Ranking Function

BM25 (Best Matching 25) is a probabilistic ranking function that improves upon TF-IDF.

### BM25 Formula

BM25 score for document $d$ given query $q$:

$$\text{BM25}(q, d) = \sum_{t \in q} idf(t) \times \frac{tf(t, d) \times (k_1 + 1)}{tf(t, d) + k_1 \times (1 - b + b \times \frac{|d|}{\text{avgdl}})}$$

where:
- $k_1$: Term frequency saturation parameter (typically 1.2-2.0)
- $b$: Length normalization parameter (typically 0.75)
- $\text{avgdl}$: Average document length

### BM25 Components

**IDF component**: Same as TF-IDF, measures term importance
**TF component**: Saturated term frequency, prevents over-weighting frequent terms
**Length normalization**: Penalizes long documents, controlled by $b$

### Term Frequency Saturation

BM25's TF component saturates:

$$tf_{BM25} = \frac{tf \times (k_1 + 1)}{tf + k_1}$$

As $tf \to \infty$, $tf_{BM25} \to k_1 + 1$ (saturation point).

This prevents very frequent terms from dominating scores.

### Length Normalization

Length normalization addresses the bias toward long documents:

$$L = 1 - b + b \times \frac{|d|}{\text{avgdl}}$$

- $b = 0$: No length normalization
- $b = 1$: Full length normalization
- Typical $b = 0.75$: Partial normalization

### BM25 Variants

**BM25+**: Adds term-independent component
**BM25L**: Logarithmic variant
**BM25-adpt**: Adaptive parameter selection

BM25 is widely used and performs excellently in practice.

## Relevance Ranking

Relevance ranking orders documents by how well they satisfy the information need.

### Relevance vs Similarity

**Similarity**: Textual similarity between query and document
**Relevance**: How well document satisfies information need

Relevance depends on:
- **Topical relevance**: About the same topic
- **User intent**: Matches what user wants
- **Quality**: Authoritative, trustworthy
- **Freshness**: Up-to-date information

### Ranking Factors

Beyond text similarity:
- **PageRank/Authority**: Link-based importance
- **Recency**: Publication date
- **User signals**: Clicks, dwell time
- **Personalization**: User history, preferences

### Learning to Rank

Machine learning approaches learn ranking functions:

**Pointwise**: Predict relevance score for each document
**Pairwise**: Learn to compare document pairs
**Listwise**: Optimize entire ranking directly

Features include:
- Text similarity scores
- Document metadata
- User behavior signals
- Query-document match features

## Query Expansion

Query expansion adds related terms to improve retrieval effectiveness.

### Why Expand Queries

Queries are short (2-3 words on average), leading to:
- **Vocabulary mismatch**: Query terms don't match document terms
- **Polysemy**: Same word, different meanings
- **Synonymy**: Different words, same meaning

### Pseudo-Relevance Feedback

Pseudo-relevance feedback (PRF) assumes top-ranked documents are relevant:

1. Initial retrieval with original query
2. Extract terms from top-$k$ documents
3. Add top terms to query
4. Re-rank with expanded query

**Rocchio algorithm**:
$$\mathbf{q}_{new} = \alpha \mathbf{q}_{orig} + \beta \frac{1}{|R|} \sum_{d \in R} \mathbf{d} - \gamma \frac{1}{|NR|} \sum_{d \in NR} \mathbf{d}$$

where $R$ is relevant documents, $NR$ is non-relevant.

### Thesaurus-Based Expansion

Use external resources:
- **WordNet**: Synonym sets
- **Domain thesauri**: Specialized vocabularies
- **Co-occurrence**: Terms that appear together

### Query Reformulation

Alternative approaches:
- **Stemming**: Reduce to root forms
- **Spelling correction**: Fix typos
- **Query segmentation**: Identify phrases
- **Entity recognition**: Identify named entities

## Inverted Index

An inverted index maps terms to documents containing them, enabling efficient retrieval.

### Index Structure

For each term $t$, store:
- **Posting list**: List of documents containing $t$
- **Term frequency**: Count in each document
- **Positions**: Word positions (for phrase queries)

Example:
```
"machine" → [(doc1, 3, [5, 12, 45]), (doc3, 1, [2]), ...]
```

### Index Construction

**Tokenization**: Split documents into terms
**Normalization**: Lowercase, stemming
**Posting creation**: For each term-document pair, add to posting list
**Sorting**: Sort postings by document ID
**Compression**: Compress posting lists

### Index Operations

**Lookup**: Find documents containing query terms
**Intersection**: Find documents containing all terms (AND query)
**Union**: Find documents containing any term (OR query)
**Phrase queries**: Use position information

### Compression

Posting lists are compressed:

**Delta encoding**: Store differences between document IDs
**Variable-byte encoding**: Variable-length integers
**Gamma codes**: Elias gamma coding
**Golomb codes**: Optimal for geometric distributions

Compression reduces storage by 5-10x.

### Distributed Indexing

For large collections:
- **Partitioning**: Split index across machines
- **Sharding**: Partition by term or document
- **Replication**: Multiple copies for availability

## Evaluation Metrics

IR evaluation measures how well systems retrieve relevant documents.

### Precision and Recall

**Precision**: Fraction of retrieved documents that are relevant

$$P = \frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Retrieved}|}$$

**Recall**: Fraction of relevant documents that are retrieved

$$R = \frac{|\text{Relevant} \cap \text{Retrieved}|}{|\text{Relevant}|}$$

**F-measure**: Harmonic mean

$$F_1 = \frac{2PR}{P + R}$$

### Mean Average Precision (MAP)

MAP averages precision at each relevant document:

$$\text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{|R_q|} \sum_{k=1}^{|R_q|} P@k(q)$$

where $P@k(q)$ is precision at rank $k$ for query $q$, $R_q$ is relevant documents.

MAP rewards systems that rank relevant documents highly.

### Normalized Discounted Cumulative Gain (NDCG)

NDCG measures ranking quality with graded relevance:

$$\text{DCG}@k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}$$

$$\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$

where IDCG is ideal DCG (perfect ranking).

NDCG handles multiple relevance levels and position importance.

### Mean Reciprocal Rank (MRR)

MRR measures how quickly the first relevant document is found:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the rank of the first relevant document.

Useful for tasks where finding any relevant result matters.

### Evaluation Challenges

**Relevance judgments**: Expensive to create, may be subjective
**Test collections**: TREC, CLEF provide standard benchmarks
**Online evaluation**: Click-through rates, dwell time
**User studies**: Controlled experiments with real users

## Advanced Retrieval Models

Beyond VSM and BM25, advanced models capture more sophisticated patterns.

### Language Models for IR

Language modeling approach:

$$P(q | d) = \prod_{t \in q} P(t | d)$$

Document language model:

$$P(t | d) = \lambda P_{MLE}(t | d) + (1 - \lambda) P(t | C)$$

where $C$ is the collection language model (smoothing).

**Query likelihood**: Rank by $P(q | d)$
**KL divergence**: Rank by KL divergence between query and document models

### Probabilistic IR Models

**Binary Independence Model**: Probabilistic relevance
**Okapi BM25**: Probabilistic ranking function (discussed above)
**Divergence from Randomness**: Information-theoretic approach

### Learning to Rank

Machine learning for ranking:

**Features**: BM25 scores, TF-IDF, language model scores, metadata
**Algorithms**: 
- **Ranking SVM**: Pairwise approach
- **LambdaRank**: Gradient boosting for ranking
- **Neural ranking**: Deep learning models

### Neural Information Retrieval

Neural approaches:
- **Dense retrieval**: Embedding-based similarity
- **Cross-encoders**: Joint query-document encoding
- **Reranking**: Neural reranking of initial results

## Key Takeaways

1. **Vector Space Model enables geometric IR**: Representing documents and queries as vectors allows similarity computation and efficient ranking.

2. **TF-IDF balances frequency and rarity**: Weighting terms by both document frequency and collection rarity identifies distinctive, informative terms.

3. **BM25 improves upon TF-IDF**: Term frequency saturation and length normalization make BM25 more robust and effective than basic TF-IDF.

4. **Inverted indexes enable efficient retrieval**: Mapping terms to documents allows fast lookup and set operations (intersection, union) for query processing.

5. **Query expansion addresses vocabulary mismatch**: Adding related terms improves recall by bridging the gap between query and document vocabularies.

6. **Evaluation requires multiple metrics**: Precision, recall, MAP, NDCG, and MRR capture different aspects of retrieval effectiveness.

7. **Relevance is multifaceted**: Beyond text similarity, relevance depends on authority, freshness, user intent, and quality signals.

8. **Neural IR complements classical methods**: While neural approaches show promise, classical methods like BM25 remain competitive and are often used in hybrid systems.
