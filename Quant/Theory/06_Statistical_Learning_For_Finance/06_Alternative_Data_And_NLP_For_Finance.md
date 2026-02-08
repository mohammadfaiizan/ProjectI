# Alternative Data and NLP for Finance

## Alternative Data Sources

Alternative data refers to non-traditional data sources used for investment decisions.

### Satellite Data

**Retail traffic:** Count cars in parking lots to predict sales
**Oil storage:** Measure shadows of oil tanks to estimate inventory
**Crop yields:** Analyze satellite imagery to forecast agricultural production
**Construction activity:** Monitor building progress

**Applications:**
- Predict earnings before announcements
- Estimate commodity supply/demand
- Track economic activity

**Challenges:**
- Image processing complexity
- Weather interference
- Privacy concerns
- Cost

### Web Scraping

**E-commerce:** Product prices, availability, reviews
**Job postings:** Hiring trends, wage information
**Social media:** Sentiment, trends
**Government websites:** Regulatory filings, economic data

**Applications:**
- Competitive intelligence
- Demand forecasting
- Sentiment analysis

**Legal/ethical considerations:**
- Terms of service
- Rate limiting
- Data ownership

### Social Media

**Twitter/X:** Real-time sentiment, news flow
**Reddit:** Discussion forums, meme stocks
**LinkedIn:** Professional networks, hiring trends

**Applications:**
- Sentiment analysis
- Early signal detection
- Viral trends

**Challenges:**
- Noise, spam
- Bot detection
- Sentiment extraction accuracy

### Credit Card Transactions

**Aggregated spending:** Category-level spending patterns
**Geographic trends:** Regional economic activity
**Merchant data:** Retail performance

**Applications:**
- Predict retail earnings
- Track consumer trends
- Economic indicators

**Privacy:** Aggregated, anonymized data only

### Other Sources

**Mobile data:** Location, app usage
**IoT sensors:** Supply chain monitoring
**Email receipts:** Purchase data
**Patent filings:** Innovation trends
**Academic papers:** Research insights

## Sentiment Analysis for Financial Text

Sentiment analysis extracts sentiment from financial text.

### Text Sources

**Earnings calls:** Management commentary, Q&A
**News articles:** Financial news, analyst reports
**SEC filings:** 10-K, 10-Q, 8-K filings
**Social media:** Twitter, Reddit, forums
**Research reports:** Analyst reports, research papers

### Approaches

### Dictionary-Based

**Sentiment dictionaries:** Lists of positive/negative words
**Scoring:** Count positive/negative words, compute score

**Financial lexicons:** Domain-specific dictionaries
- Loughran-McDonald: Financial sentiment dictionary
- Harvard IV-4: General sentiment
- Custom dictionaries: Built for specific use cases

**Limitations:**
- Context ignored
- Negation handling ("not good")
- Sarcasm detection

### Machine Learning

**Supervised:** Train classifier on labeled data
- Features: Bag of words, TF-IDF, word embeddings
- Models: Naive Bayes, SVM, logistic regression

**Deep learning:** RNN, LSTM, transformers
- Better context understanding
- Requires more data

### Fine-Tuning for Finance

**Pre-trained models:** Start with general language models (BERT, GPT)
**Domain adaptation:** Fine-tune on financial text
**Task-specific:** Fine-tune for sentiment classification

**Financial BERT:** Pre-trained on financial documents
**FinBERT:** Specifically for financial sentiment

### Applications

**Earnings call analysis:** Extract sentiment from management tone
**News impact:** Measure sentiment of news articles
**Social media:** Track sentiment on stocks
**Regulatory filings:** Analyze tone of disclosures

## Topic Modeling

Topic modeling identifies themes in document collections.

### Latent Dirichlet Allocation (LDA)

**Generative process:**
1. For each document $d$:
   - Draw topic distribution $\theta_d \sim \text{Dir}(\alpha)$
2. For each word $w$ in document $d$:
   - Draw topic $z \sim \text{Multinomial}(\theta_d)$
   - Draw word $w \sim \text{Multinomial}(\phi_z)$

**Parameters:**
- $\theta_d$: Document-topic distribution
- $\phi_z$: Topic-word distribution
- $\alpha$: Dirichlet prior for topics
- $\beta$: Dirichlet prior for words

**Inference:** Variational EM or Gibbs sampling

### Applications: Market Regime Identification

**Documents:** Time periods (e.g., monthly market summaries)
**Words:** Market events, economic indicators
**Topics:** Market regimes (bull, bear, volatile, stable)

**Use:**
- Identify current regime
- Predict regime transitions
- Understand regime characteristics

### Other Topic Models

**Dynamic Topic Models:** Topics evolve over time
**Correlated Topic Models:** Allow topic correlations
**Non-negative Matrix Factorization:** Alternative approach

## NLP Pipelines

### Tokenization

**Word tokenization:** Split text into words
**Subword tokenization:** BPE, WordPiece (for transformers)
**Sentence segmentation:** Split into sentences

**Challenges:**
- Financial jargon
- Numbers, dates
- Abbreviations

### Embeddings

**Word2Vec:** Learn word embeddings from co-occurrence
- Skip-gram: Predict context from word
- CBOW: Predict word from context

**GloVe:** Global vectors from word co-occurrence matrix

**Contextual embeddings:** BERT, ELMo (word meaning depends on context)

**Financial embeddings:** Train on financial corpus

### Fine-Tuning for Finance

**Transfer learning:**
1. Pre-train on large general corpus
2. Fine-tune on financial text
3. Task-specific fine-tuning

**Domain adaptation:**
- Continue pre-training on financial data
- Multi-task learning
- Adversarial training

### Preprocessing

**Cleaning:** Remove HTML, special characters
**Normalization:** Lowercase, expand contractions
**Stop word removal:** Remove common words (may keep financial stop words)
**Stemming/Lemmatization:** Reduce words to root form

**Financial-specific:**
- Handle numbers, percentages
- Preserve financial terms
- Handle abbreviations

## Event-Driven Strategies Using News Flow

News flow can drive trading strategies.

### Event Detection

**Named entity recognition:** Identify companies, people, locations
**Event extraction:** Identify events (mergers, earnings, etc.)
**Temporal extraction:** Identify when events occurred

**Tools:** spaCy, NLTK, financial NER models

### News Impact

**Immediate impact:** Price reaction to news
**Delayed impact:** Gradual price adjustment
**Sentiment impact:** Positive/negative news effects

**Measurement:**
- Abnormal returns around news
- Volume spikes
- Volatility changes

### Trading Strategies

**Momentum:** Trade on positive news
**Mean reversion:** Fade initial reaction
**Pairs trading:** News on one stock affects related stocks
**Sector rotation:** News affects entire sectors

### Implementation

**Real-time processing:** Process news as it arrives
**Signal generation:** Convert news to trading signals
**Risk management:** Limit exposure to news-driven trades
**Backtesting:** Test strategies on historical news

## Data Quality, Bias, and Decay

### Data Quality Issues

**Missing data:** Incomplete records
**Errors:** Incorrect values, typos
**Inconsistencies:** Different formats, units
**Duplicates:** Same data from multiple sources

**Validation:**
- Range checks
- Consistency checks
- Cross-validation with known sources

### Bias

**Selection bias:** Data not representative
**Survivorship bias:** Only surviving entities in dataset
**Look-ahead bias:** Using future information
**Survivorship bias:** Only successful strategies survive

**Mitigation:**
- Include delisted stocks
- Proper time ordering
- Out-of-sample testing

### Data Decay

**Concept drift:** Relationships change over time
**Data staleness:** Old data less relevant
**Regime changes:** Different market conditions

**Solutions:**
- Regular retraining
- Exponential weighting
- Regime detection
- Online learning

### Alternative Data Specific Issues

**Coverage:** Not all entities covered equally
**Latency:** Data arrives with delay
**Cost:** Expensive data sources
**Legal:** Privacy, terms of service

**Due diligence:**
- Understand data collection
- Verify accuracy
- Check legal compliance
- Assess value vs cost

## Practical Considerations

### Data Acquisition

**Cost-benefit:** Expensive data must provide value
**Vendor evaluation:** Assess data quality, coverage
**In-house vs vendor:** Build vs buy decision
**Legal compliance:** Ensure proper use

### Data Processing

**Storage:** Large volumes require efficient storage
**Processing:** Real-time vs batch processing
**Infrastructure:** Cloud vs on-premise
**Scalability:** Handle growing data volumes

### Integration

**Data fusion:** Combine multiple sources
**Temporal alignment:** Match timestamps
**Normalization:** Standardize formats
**Feature engineering:** Create meaningful features

### Model Development

**Feature extraction:** Convert raw data to features
**Model selection:** Choose appropriate models
**Validation:** Proper time-series validation
**Monitoring:** Track model performance

### Production

**Latency:** Real-time processing requirements
**Reliability:** Handle data outages
**Monitoring:** Track data quality, model performance
**Updates:** Retrain as new data arrives

### Regulatory Considerations

**Privacy:** GDPR, CCPA compliance
**Insider trading:** Ensure no material non-public information
**Fair use:** Respect data ownership
**Documentation:** Maintain audit trails

## Evaluation Metrics

### Predictive Performance

**Accuracy:** For classification tasks
**RMSE/MAE:** For regression
**Sharpe ratio:** Risk-adjusted returns
**Information ratio:** Active return per tracking error

### Business Metrics

**Alpha:** Excess return
**Beta:** Market exposure
**Maximum drawdown:** Worst peak-to-trough
**Win rate:** Percentage of profitable trades

### Data Quality Metrics

**Coverage:** Percentage of entities covered
**Freshness:** Time since last update
**Completeness:** Percentage of non-missing values
**Consistency:** Agreement across sources

## Future Directions

### Large Language Models

**GPT, BERT:** Pre-trained on vast text
**Financial LLMs:** Fine-tuned for finance
**Applications:** Summarization, Q&A, analysis

### Multimodal Learning

**Text + images:** Combine news and charts
**Text + audio:** Earnings calls (audio + transcript)
**Text + numerical:** Combine text with financial data

### Real-Time Processing

**Streaming:** Process data as it arrives
**Low latency:** Minimize delay
**Scalability:** Handle high volumes

### Explainability

**Interpretability:** Understand model decisions
**Attribution:** Which data sources matter
**Regulatory:** Meet explainability requirements
