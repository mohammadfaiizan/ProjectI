# Text Preprocessing and Tokenization

## Table of Contents

1. [Introduction](#introduction)
2. [Tokenization Fundamentals](#tokenization-fundamentals)
3. [Word-Level Tokenization](#word-level-tokenization)
4. [Subword Tokenization](#subword-tokenization)
5. [Stemming and Lemmatization](#stemming-and-lemmatization)
6. [Text Normalization](#text-normalization)
7. [Stopword Removal](#stopword-removal)
8. [Regular Expression Patterns](#regular-expression-patterns)
9. [Preprocessing Pipelines](#preprocessing-pipelines)
10. [Key Takeaways](#key-takeaways)

## Introduction

Text preprocessing is the foundational step in natural language processing that transforms raw text into a format suitable for computational analysis. The quality of preprocessing directly impacts downstream NLP tasks, making it crucial to understand the various techniques and their trade-offs.

Raw text contains numerous challenges including inconsistent formatting, punctuation, capitalization variations, and morphological complexity. Effective preprocessing addresses these issues while preserving semantic information necessary for language understanding.

The preprocessing pipeline typically involves multiple stages: tokenization (breaking text into units), normalization (standardizing text), morphological processing (reducing word variations), and filtering (removing noise). Each stage requires careful consideration of the target language, domain, and application requirements.

## Tokenization Fundamentals

Tokenization is the process of segmenting text into discrete units called tokens. The choice of tokenization strategy depends on the linguistic properties of the text and the requirements of downstream tasks.

### Tokenization Granularity

Tokenization can occur at multiple levels:

- **Character-level**: Each character becomes a token
- **Subword-level**: Text is split into subword units (morphemes, syllables, or learned segments)
- **Word-level**: Text is split at word boundaries
- **Sentence-level**: Text is segmented into sentences

The granularity affects vocabulary size, out-of-vocabulary handling, and computational requirements. Character-level tokenization creates very large sequences but handles any vocabulary. Word-level tokenization is intuitive but struggles with unknown words. Subword tokenization balances these concerns.

### Tokenization Challenges

Tokenization faces several linguistic challenges:

**Whitespace ambiguity**: Not all languages use spaces to delimit words. Languages like Chinese, Japanese, and Thai require specialized word segmentation algorithms.

**Punctuation handling**: Punctuation can be attached to words (e.g., "don't") or separate (e.g., "word."). The decision affects token counts and semantic interpretation.

**Contractions and clitics**: Languages contain contractions ("can't", "I'm") and clitics that attach to words, requiring careful handling.

**Multi-word expressions**: Phrases like "New York" or "machine learning" may need to be treated as single tokens depending on the application.

### Tokenization Algorithms

Basic tokenization algorithms include:

**Whitespace tokenization**: Split on whitespace characters. Simple but language-dependent.

```python
def whitespace_tokenize(text):
    return text.split()
```

**Punctuation-aware tokenization**: Split on whitespace and punctuation, handling edge cases.

**Rule-based tokenization**: Use linguistic rules and dictionaries to identify word boundaries, particularly important for languages without word delimiters.

**Statistical tokenization**: Learn tokenization rules from data using statistical models, common in machine translation and modern NLP systems.

## Word-Level Tokenization

Word-level tokenization splits text into words, typically using whitespace and punctuation as delimiters. This approach is intuitive and produces interpretable tokens.

### Simple Word Tokenization

The simplest approach splits text on whitespace:

$$T = \{w_1, w_2, \ldots, w_n\}$$

where $T$ is the tokenized sequence and $w_i$ are word tokens.

However, this naive approach fails with punctuation, contractions, and hyphenated words. More sophisticated methods handle these cases.

### Punctuation Handling

Punctuation can be handled in multiple ways:

**Attached punctuation**: Keep punctuation with words ("word." → ["word."])
**Separated punctuation**: Split punctuation into separate tokens ("word." → ["word", "."])
**Normalized punctuation**: Replace punctuation with special tokens ("word." → ["word", "<PERIOD>"])

The choice depends on the application. For language modeling, attached punctuation may be preferable. For parsing, separated punctuation provides more structure.

### Tokenization Libraries

Popular tokenization libraries include:

**NLTK**: Provides multiple tokenizers including word_tokenize, which uses the Penn Treebank tokenization conventions.

**spaCy**: Industrial-strength tokenizer that handles contractions, punctuation, and multi-word expressions intelligently.

**Stanford CoreNLP**: Comprehensive tokenization with support for multiple languages and linguistic annotations.

These libraries implement sophisticated rules for handling edge cases like URLs, email addresses, and special characters.

## Subword Tokenization

Subword tokenization addresses the vocabulary limitation problem by splitting words into smaller units. This enables handling of rare words and out-of-vocabulary terms while maintaining reasonable vocabulary sizes.

### Byte Pair Encoding (BPE)

BPE is a compression algorithm adapted for tokenization. It starts with character-level tokens and iteratively merges the most frequent pairs.

**Algorithm**:
1. Initialize vocabulary with all characters
2. Count all adjacent symbol pairs
3. Merge the most frequent pair
4. Repeat until desired vocabulary size

The merge operation creates new subword units. For example, if "th" and "e" are frequent, they merge to create "the" as a subword unit.

BPE vocabulary size is controlled by the number of merge operations. Typical sizes range from 10,000 to 50,000 subwords.

### WordPiece Tokenization

WordPiece is similar to BPE but uses a different merging criterion. Instead of merging the most frequent pair, WordPiece merges the pair that maximizes the likelihood of the training data.

The likelihood is computed as:

$$L = \sum_{i=1}^{N} \log P(w_i)$$

where $P(w_i)$ is the probability of word $w_i$ under the current vocabulary.

WordPiece is used in BERT and other transformer models. It tends to create more linguistically meaningful subwords compared to BPE.

### SentencePiece

SentencePiece treats the input as a raw stream of Unicode characters, including spaces. This allows it to handle languages without word boundaries naturally.

Key features:
- **Language-agnostic**: Works for any language without modification
- **Reversible**: Can reconstruct original text from tokens
- **Subword sampling**: Supports sampling multiple segmentations for regularization

SentencePiece uses either BPE or unigram language modeling as the underlying algorithm. The unigram model is particularly effective for languages with complex morphology.

### Subword Tokenization Trade-offs

Subword tokenization offers several advantages:

**Vocabulary coverage**: Handles any word through composition of subwords
**Morphological awareness**: Captures morphemes and word structure
**Efficiency**: Smaller vocabulary than character-level, fewer tokens than word-level

However, it introduces complexity:

**Tokenization ambiguity**: Multiple valid segmentations may exist
**Sequence length**: Increases sequence length compared to word-level
**Interpretability**: Subwords are less interpretable than full words

## Stemming and Lemmatization

Stemming and lemmatization reduce words to their base forms, helping normalize morphological variations.

### Stemming

Stemming uses heuristic rules to remove suffixes, producing stems that may not be valid words. For example, "running" → "run", "flies" → "fli".

**Porter Stemmer**: The most widely used English stemmer, applies a series of rules sequentially:

```
Step 1a: SSES → SS (caresses → caress)
Step 1b: (m>0) EED → EE (agreed → agree)
...
```

**Snowball Stemmer**: Extends Porter's algorithm to multiple languages, also known as Porter2.

**Lancaster Stemmer**: More aggressive than Porter, produces shorter stems but may over-stem.

Stemming is fast and language-independent (with language-specific rules) but can produce invalid stems and may conflate semantically different words.

### Lemmatization

Lemmatization uses vocabulary and morphological analysis to return the canonical form (lemma) of a word. It requires part-of-speech information for accuracy.

For example:
- "better" (adjective) → "good"
- "better" (verb) → "better"
- "running" → "run"
- "mice" → "mouse"

Lemmatization is more accurate than stemming but computationally expensive and requires linguistic resources (dictionaries, morphological analyzers).

### Morphological Analysis

For morphologically rich languages, full morphological analysis may be necessary:

**Morpheme segmentation**: Split words into morphemes (prefixes, stems, suffixes)
**Morphological tagging**: Identify morphological features (tense, case, number)
**Paradigm generation**: Generate all inflected forms from a lemma

Languages like Arabic, Finnish, and Turkish have complex morphology requiring sophisticated analysis.

## Text Normalization

Text normalization standardizes text to a canonical form, reducing variation and improving consistency.

### Case Normalization

Case normalization handles capitalization:

**Lowercasing**: Convert all text to lowercase. Common but loses information (e.g., "US" vs "us")
**Case folding**: More aggressive than lowercasing, handles Unicode case mappings
**Case preservation**: Keep original case, may be important for named entities

The choice depends on the task. Lowercasing is standard for many applications but problematic for named entity recognition.

### Unicode Normalization

Unicode provides multiple representations for the same character:

**NFC (Canonical Composition)**: Precomposed characters (é)
**NFD (Canonical Decomposition)**: Decomposed characters (e + ́)
**NFKC (Compatibility Composition)**: Compatibility precomposed
**NFKD (Compatibility Decomposition)**: Compatibility decomposed

Normalization ensures consistent representation:

```python
import unicodedata
text = unicodedata.normalize('NFC', text)
```

### Number Normalization

Numbers can be normalized in various ways:

**Digit preservation**: Keep numbers as-is ("123")
**Word replacement**: Replace with special token ("<NUM>")
**Digit masking**: Replace digits with placeholders ("###")
**Spelling out**: Convert to words ("one hundred twenty-three")

The approach depends on whether numeric values are important or should be abstracted.

### Special Character Handling

Special characters require careful handling:

**URLs and emails**: May be replaced with special tokens or removed
**Hashtags and mentions**: Preserved for social media analysis
**Emojis**: May be removed, converted to text, or kept as Unicode
**HTML entities**: Decoded to characters

## Stopword Removal

Stopwords are common words that carry little semantic information (e.g., "the", "a", "is"). Removing them can reduce noise and computational cost.

### Stopword Lists

Stopword lists are language-specific collections of common function words:

**NLTK stopwords**: ~180 English stopwords
**spaCy stopwords**: Language-specific lists for multiple languages
**Custom lists**: Domain-specific stopwords (e.g., removing "paper" in academic contexts)

### When to Remove Stopwords

Stopword removal is beneficial when:
- Reducing vocabulary size is important
- Semantic content is in content words
- Computational efficiency matters

Stopword removal is problematic when:
- Function words carry meaning (e.g., "not" in sentiment analysis)
- Syntactic structure is important
- Working with short texts where every word matters

### Frequency-Based Filtering

Instead of predefined lists, frequency-based filtering removes words above or below frequency thresholds:

**High-frequency filtering**: Remove very common words (similar to stopwords)
**Low-frequency filtering**: Remove rare words (may be typos or noise)

The thresholds are typically determined empirically on the corpus.

## Regular Expression Patterns

Regular expressions provide powerful pattern matching for text preprocessing tasks.

### Common Patterns

**Email addresses**:
```regex
\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b
```

**URLs**:
```regex
https?://[^\s]+
```

**Phone numbers** (US format):
```regex
\b\d{3}[-.]?\d{3}[-.]?\d{4}\b
```

**Dates**:
```regex
\d{1,2}[/-]\d{1,2}[/-]\d{2,4}
```

### Pattern Application

Regular expressions are used for:

**Entity extraction**: Find dates, URLs, emails in text
**Normalization**: Replace patterns with standardized forms
**Filtering**: Remove or extract text matching patterns
**Segmentation**: Split text based on patterns

Careful pattern design is crucial to avoid false positives and negatives.

## Preprocessing Pipelines

A preprocessing pipeline combines multiple steps in sequence. The order matters and should be optimized for the specific task.

### Pipeline Design

Typical pipeline order:

1. **Text extraction**: Extract text from formats (PDF, HTML, etc.)
2. **Encoding normalization**: Ensure consistent character encoding
3. **Unicode normalization**: Standardize Unicode representation
4. **Sentence segmentation**: Split into sentences
5. **Tokenization**: Split into tokens
6. **Normalization**: Case, numbers, special characters
7. **Morphological processing**: Stemming or lemmatization
8. **Filtering**: Remove stopwords, low-frequency words
9. **Feature extraction**: Create numerical representations

### Pipeline Optimization

Pipeline optimization considerations:

**Efficiency**: Order operations to minimize redundant processing
**Modularity**: Make components interchangeable for experimentation
**Caching**: Cache expensive operations (lemmatization, POS tagging)
**Parallelization**: Process documents in parallel when possible

### Domain Adaptation

Preprocessing must adapt to domain characteristics:

**Social media**: Handle hashtags, mentions, emojis, informal language
**Scientific text**: Preserve technical terms, mathematical notation
**Multilingual**: Language-specific tokenization and normalization
**Historical text**: Handle archaic spellings and OCR errors

## Key Takeaways

1. **Tokenization is fundamental**: The choice of tokenization strategy (word, subword, character) fundamentally affects downstream performance and should match the task requirements.

2. **Subword tokenization balances trade-offs**: BPE, WordPiece, and SentencePiece enable handling of rare words while maintaining reasonable vocabulary sizes, making them essential for modern NLP.

3. **Normalization preserves information**: Careful normalization (case, Unicode, numbers) reduces variation while preserving semantic content. Over-normalization can lose important distinctions.

4. **Morphological processing is language-dependent**: Stemming and lemmatization effectiveness varies dramatically across languages. Morphologically rich languages require sophisticated analysis.

5. **Stopword removal is task-dependent**: Function words carry meaning in many contexts. Automatic stopword removal should be avoided without task-specific evaluation.

6. **Pipeline design matters**: The order of preprocessing steps affects results. Experimentation and domain knowledge guide optimal pipeline construction.

7. **Regular expressions are powerful but brittle**: Pattern matching enables efficient preprocessing but requires careful design and validation to avoid errors.

8. **Preprocessing impacts everything**: Decisions made during preprocessing propagate through the entire NLP pipeline. Invest time in understanding and optimizing preprocessing for your specific use case.
