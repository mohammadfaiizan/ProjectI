# Human Perceptual Dataset and Evaluation

## Table of Contents
1. [Dataset Construction Methodology](#dataset-construction-methodology)
2. [2-Alternative Forced Choice (2AFC) Protocol](#2-alternative-forced-choice-2afc-protocol)
3. [Distortion Types and Generation](#distortion-types-and-generation)
4. [Human Annotation Process](#human-annotation-process)
5. [Quality Control and Validation](#quality-control-and-validation)
6. [Statistical Analysis Framework](#statistical-analysis-framework)
7. [Inter-annotator Agreement](#inter-annotator-agreement)
8. [Data Collection Pipeline](#data-collection-pipeline)
9. [Dataset Characteristics and Properties](#dataset-characteristics-and-properties)
10. [Evaluation Metrics and Benchmarking](#evaluation-metrics-and-benchmarking)

---

## 1. Dataset Construction Methodology

### 1.1 Overall Design Philosophy

The LPIPS dataset was designed to provide a systematic evaluation framework for perceptual similarity metrics by collecting human judgments on carefully constructed image triplets.

**CORE PRINCIPLES:**
- Systematic coverage of distortion types
- Balanced representation of visual content
- Controlled experimental conditions
- Scalable annotation methodology
- Statistical rigor in data collection

**DATASET SCALE:**
- Total triplets: ~150,000 image patch triplets
- Human judgments: ~300,000+ annotations
- Coverage: Traditional and CNN-based distortions
- Patch size: 64x64 pixels for controlled evaluation

### 1.2 Image Source Selection

**BASE IMAGE COLLECTION:**
- Source datasets: Multiple standard computer vision datasets
- Content diversity: Natural images, textures, objects, scenes
- Resolution requirements: Sufficient for patch extraction
- Quality filtering: Manual and automatic quality assessment

**PATCH EXTRACTION STRATEGY:**
- Random spatial sampling from source images
- Minimum content requirements (avoid blank regions)
- Scale normalization to 64x64 pixels
- Color space standardization (RGB, 0-255 range)

**CONTENT BALANCING:**
- Texture-rich regions: High-frequency content
- Smooth regions: Low-frequency content  
- Edge regions: Boundary information
- Object-centric patches: Semantic content

### 1.3 Triplet Construction Approach

**TRIPLET STRUCTURE:**
Each triplet consists of:
- Reference patch (R): Original or minimally distorted
- Candidate patch 1 (C1): Applied distortion type A
- Candidate patch 2 (C2): Applied distortion type B

**DISTORTION PAIRING STRATEGY:**
- Same-type comparisons: Different severity levels
- Cross-type comparisons: Different distortion types
- Mixed severity: Varying distortion intensities
- Balanced distribution across distortion categories

**DIFFICULTY CALIBRATION:**
- Easy comparisons: Large perceptual differences
- Medium comparisons: Moderate perceptual differences  
- Hard comparisons: Subtle perceptual differences
- Stratified sampling across difficulty levels

---

## 2. 2-Alternative Forced Choice (2AFC) Protocol

### 2.1 Protocol Design Rationale

**ADVANTAGES OF 2AFC:**
- Binary choice reduces annotation complexity
- Eliminates scale calibration issues
- Reduces inter-annotator variability
- Enables statistical significance testing
- Provides clean training signal for learning

**COMPARISON TO ALTERNATIVES:**
- Rating scales: Subject to personal calibration differences
- Ranking: More complex for large sets
- Pairwise comparison: Exponential scaling issues
- Absolute scoring: Reference point ambiguity

### 2.2 Task Description

**ANNOTATION TASK:**
"Which of the two patches (left or right) is more similar to the reference patch (top)?"

**INTERFACE DESIGN:**
- Reference patch displayed prominently at top
- Two candidate patches side-by-side below
- Clear "Left" and "Right" selection buttons
- Optional "Equal" choice for indistinguishable cases
- Response time recording for quality analysis

**INSTRUCTION TO ANNOTATORS:**
- Focus on overall visual similarity
- Consider all aspects: color, texture, structure
- Make intuitive perceptual judgments
- No specific technical criteria provided
- Work at comfortable pace for quality

### 2.3 Response Collection and Processing

**RESPONSE ENCODING:**
- Binary choice: 0 (choose C1) or 1 (choose C2)
- Confidence optional: Not used in main analysis
- Response time: Recorded for quality filtering
- Annotator ID: For inter-rater analysis

**QUALITY FILTERING:**
- Extremely fast responses (<200ms) filtered out
- Extremely slow responses (>30s) reviewed manually
- Consistency checks within annotator
- Cross-validation with known cases

**AGGREGATION STRATEGY:**
- Multiple judgments per triplet (typically 2-5)
- Majority vote for final ground truth
- Soft labels for split decisions (e.g., 0.5 for 1-1 split)
- Confidence weighting based on agreement

---

## 3. Distortion Types and Generation

### 3.1 Traditional Distortion Categories

Traditional distortions represent classical image degradations studied in image processing and compression literature.

**GAUSSIAN NOISE:**
- Mathematical model: `I_noisy = I_clean + N(0, sigma^2)`
- Parameter range: sigma = 0.01 to 0.2 (normalized intensity)
- Characteristics: Additive, spatially uncorrelated
- Perceptual effect: Reduces image clarity, preserves structure

**IMPULSE NOISE (Salt and Pepper):**
- Mathematical model: Random pixels set to 0 or 255
- Parameter range: 1% to 20% pixel corruption
- Characteristics: Sparse, high-contrast artifacts
- Perceptual effect: Local detail destruction

**MOTION BLUR:**
- Mathematical model: Convolution with motion kernel
- Parameter range: 1-15 pixel displacement
- Kernel types: Linear, circular motion patterns
- Perceptual effect: Directional smearing, detail loss

**GAUSSIAN BLUR:**
- Mathematical model: Convolution with Gaussian kernel
- Parameter range: sigma = 0.5 to 5.0 pixels
- Characteristics: Isotropic smoothing
- Perceptual effect: Overall softening, high-frequency loss

**JPEG COMPRESSION:**
- Algorithm: DCT-based lossy compression
- Parameter range: Quality factor 10-90
- Characteristics: Block artifacts, frequency domain losses
- Perceptual effect: Compression artifacts, detail loss

**QUANTIZATION:**
- Mathematical model: Intensity level reduction
- Parameter range: 8 bits to 2 bits per channel
- Characteristics: Posterization effects
- Perceptual effect: Banding, color reduction

### 3.2 CNN-based Distortion Categories

CNN-based distortions represent artifacts from modern deep learning-based image processing methods.

**SUPER-RESOLUTION ARTIFACTS:**
- Source: Various SR algorithms (SRCNN, ESRGAN, etc.)
- Characteristics: Hallucinated details, reconstruction errors
- Parameter variation: Different SR methods and scales
- Perceptual effect: Unnatural textures, aliasing

**DENOISING ARTIFACTS:**
- Source: Deep denoising networks (DnCNN, etc.)
- Characteristics: Over-smoothing, detail loss
- Parameter variation: Different noise levels and methods
- Perceptual effect: Plastic appearance, texture elimination

**COLORIZATION ARTIFACTS:**
- Source: Automatic colorization methods
- Characteristics: Color bleeding, semantic inconsistencies
- Parameter variation: Different colorization approaches
- Perceptual effect: Unnatural color distributions

**STYLE TRANSFER ARTIFACTS:**
- Source: Neural style transfer methods
- Characteristics: Texture inconsistencies, content distortion
- Parameter variation: Different style and content weights
- Perceptual effect: Artistic but semantically inconsistent

**GENERATIVE MODEL ARTIFACTS:**
- Source: GAN and VAE outputs
- Characteristics: Mode collapse artifacts, training instabilities
- Parameter variation: Different generator architectures
- Perceptual effect: Unrealistic details, temporal inconsistencies

### 3.3 Distortion Parameter Selection

**PARAMETER SPACE SAMPLING:**
- Logarithmic spacing for perceptual parameters
- Uniform coverage across severity levels
- Focus on perceptually relevant ranges
- Calibration using pilot studies

**SEVERITY LEVEL DESIGN:**
- Subtle distortions: Barely perceptible changes
- Moderate distortions: Clearly visible but acceptable
- Strong distortions: Objectionable quality degradation
- Extreme distortions: Heavily degraded images

**CROSS-DISTORTION CALIBRATION:**
- Perceptual severity matching across distortion types
- Pilot studies to calibrate parameter ranges
- Iterative refinement based on human judgments
- Statistical validation of severity distributions

---

## 4. Human Annotation Process

### 4.1 Annotator Recruitment and Screening

**RECRUITMENT CRITERIA:**
- Normal or corrected-to-normal vision
- Age range: 18-65 years
- Computer literacy for web interface
- English language comprehension for instructions

**SCREENING PROCESS:**
- Vision screening with test images
- Practice session with known cases
- Consistency check with gold standard examples
- Performance threshold for inclusion

**ANNOTATOR DEMOGRAPHICS:**
- Age distribution: Balanced across adult age ranges
- Gender distribution: Approximately balanced
- Educational background: Diverse educational levels
- Geographic distribution: Multiple regions/countries

### 4.2 Training and Calibration

**TRAINING PROTOCOL:**
- Introduction to task and interface
- Practice session with feedback
- Explanation of similarity concept
- Examples of different distortion types

**CALIBRATION METHODOLOGY:**
- Gold standard examples with known answers
- Inter-annotator consistency training
- Feedback on initial performance
- Iterative improvement process

**QUALITY METRICS:**
- Consistency with gold standard: >80% agreement required
- Intra-annotator consistency: Repeated examples
- Response time distribution: Within normal ranges
- Overall performance trends: Learning curve analysis

### 4.3 Annotation Interface Design

**INTERFACE REQUIREMENTS:**
- Clear visual presentation of triplets
- Intuitive button layout for responses
- Progress indicators for annotator motivation
- Quality feedback mechanisms

**TECHNICAL SPECIFICATIONS:**
- Monitor calibration requirements
- Browser compatibility testing
- Response time measurement accuracy
- Image display standardization

**USER EXPERIENCE OPTIMIZATION:**
- Minimized cognitive load
- Clear instructions and examples
- Comfortable viewing conditions
- Break recommendations for fatigue management

---

## 5. Quality Control and Validation

### 5.1 Real-time Quality Monitoring

**RESPONSE TIME ANALYSIS:**
- Expected range: 2-20 seconds per triplet
- Outlier detection: <1s or >30s responses
- Pattern analysis: Consistent timing patterns
- Fatigue detection: Performance degradation over time

**CONSISTENCY CHECKS:**
- Repeated triplets: Same annotator, same response
- Reverse order: Swapped candidate order
- Transitivity: A>B, B>C implies A>C relationships
- Known cases: Pre-validated obvious comparisons

**STATISTICAL MONITORING:**
- Inter-annotator agreement rates
- Response distribution analysis
- Bias detection in individual annotators
- Overall dataset balance monitoring

### 5.2 Post-collection Validation

**EXPERT VALIDATION:**
- Expert review of subset of annotations
- Comparison with known perceptual literature
- Validation against existing benchmarks
- Sanity checks with extreme cases

**CROSS-VALIDATION:**
- Split annotator groups for validation
- Hold-out test sets for metric evaluation
- Temporal validation: Consistent over time
- Geographic validation: Consistent across regions

**OUTLIER ANALYSIS:**
- Statistical outlier detection in responses
- Manual review of flagged cases
- Root cause analysis for anomalies
- Corrective action implementation

### 5.3 Data Cleaning and Filtering

**INCLUSION CRITERIA:**
- Minimum number of judgments per triplet (≥2)
- Annotator qualification requirements
- Response time within acceptable ranges
- Consistency with quality checks

**EXCLUSION CRITERIA:**
- Corrupted or malformed image triplets
- Annotator performance below threshold
- Responses with technical issues
- Suspected random or malicious responses

**FINAL DATASET STATISTICS:**
- Total valid triplets: ~150,000
- Average judgments per triplet: 2.0
- Annotator agreement rate: 82.6%
- Quality-filtered response rate: 94.3%

---

## 6. Statistical Analysis Framework

### 6.1 Agreement Metrics

**INTER-ANNOTATOR AGREEMENT:**
- Percentage agreement: Simple agreement rate
- Cohen's Kappa: Chance-corrected agreement
- Fleiss' Kappa: Multi-rater agreement
- Intraclass correlation: Continuous similarity scores

**MATHEMATICAL FORMULATION:**
```
Percentage Agreement = (Number of agreed pairs) / (Total pairs)

Cohen's Kappa = (P_observed - P_expected) / (1 - P_expected)
where P_observed = observed agreement rate
      P_expected = agreement expected by chance
```

### 6.2 Confidence Intervals and Significance Testing

**BOOTSTRAP CONFIDENCE INTERVALS:**
- Resampling-based confidence estimation
- Non-parametric approach for non-normal distributions
- 95% confidence intervals for all reported metrics
- Stable estimates with 1000+ bootstrap samples

**STATISTICAL SIGNIFICANCE TESTING:**
- McNemar's test for paired binary comparisons
- Chi-square tests for categorical associations
- T-tests for continuous metric comparisons
- Multiple comparison corrections (Bonferroni, FDR)

**POWER ANALYSIS:**
- Sample size calculations for desired effect sizes
- Power analysis for detecting meaningful differences
- Effect size estimation using Cohen's d
- Minimum detectable difference calculations

### 6.3 Bias Analysis and Correction

**SYSTEMATIC BIAS DETECTION:**
- Order effects: Left/right presentation bias
- Learning effects: Performance change over time
- Fatigue effects: Performance degradation
- Content bias: Performance variation by image type

**BIAS CORRECTION STRATEGIES:**
- Randomization of presentation order
- Counterbalancing across annotators
- Statistical adjustment for detected biases
- Sensitivity analysis with bias-corrected data

**DEMOGRAPHIC ANALYSIS:**
- Performance variation across age groups
- Gender differences in perceptual judgments
- Cultural and geographic variations
- Educational background effects

---

## 7. Inter-annotator Agreement

### 7.1 Agreement Rate Analysis

**OVERALL AGREEMENT STATISTICS:**
- Raw agreement rate: 82.6% across all triplets
- Chance-corrected agreement (Kappa): 0.65
- Substantial agreement category (0.6-0.8 range)
- Comparable to other perceptual studies

**AGREEMENT BY DIFFICULTY:**
- Easy comparisons (large differences): 95% agreement
- Medium comparisons: 85% agreement
- Hard comparisons (subtle differences): 70% agreement
- Expected pattern: Higher agreement for clearer differences

**AGREEMENT BY DISTORTION TYPE:**
- Traditional distortions: 84% average agreement
- CNN-based distortions: 81% average agreement
- Cross-type comparisons: 80% agreement
- Slight preference consistency for familiar distortions

### 7.2 Factors Affecting Agreement

**IMAGE CONTENT FACTORS:**
- Texture complexity: Lower agreement for complex textures
- Semantic content: Higher agreement for object-centric patches
- Spatial frequency: Better agreement for high-frequency content
- Color complexity: Mixed effects depending on distortion

**DISTORTION FACTORS:**
- Severity level: Higher agreement for severe distortions
- Distortion type: Some types more subjective than others
- Artifact visibility: Clear artifacts increase agreement
- Semantic preservation: Content changes reduce agreement

**ANNOTATOR FACTORS:**
- Experience level: Trained annotators more consistent
- Fatigue level: Performance decline after extended sessions
- Individual differences: Stable individual biases
- Cultural background: Minor but detectable effects

### 7.3 Agreement Quality Assessment

**BENCHMARK COMPARISON:**
- Literature comparison: Similar studies report 75-85% agreement
- Expert agreement: Professional evaluators achieve 85-90%
- Repeated measures: Individual consistency 88-92%
- Cross-validation: Stable agreement across subsets

**RELIABILITY ANALYSIS:**
- Test-retest reliability: 0.79 correlation
- Internal consistency: Cronbach's alpha = 0.82
- Split-half reliability: 0.81 correlation
- Temporal stability: Consistent over data collection period

**VALIDITY INDICATORS:**
- Face validity: Results align with intuitive expectations
- Construct validity: Agreement patterns make theoretical sense
- Criterion validity: Correlation with expert judgments
- Convergent validity: Agreement with related measures

---

## 8. Data Collection Pipeline

### 8.1 Infrastructure and Platform

**TECHNICAL ARCHITECTURE:**
- Web-based annotation platform
- Scalable server infrastructure
- Database design for triplet storage
- Real-time quality monitoring systems

**PLATFORM FEATURES:**
- User account management and authentication
- Progress tracking and payment systems
- Quality control dashboard for administrators
- Automated data validation and cleaning

**SCALABILITY CONSIDERATIONS:**
- Distributed annotation across multiple annotators
- Load balancing for concurrent users
- Data backup and recovery systems
- Performance monitoring and optimization

### 8.2 Workflow Management

**TASK ASSIGNMENT:**
- Dynamic triplet assignment to annotators
- Load balancing across available annotators
- Priority assignment for urgent triplets
- Redundancy management for quality control

**PROGRESS MONITORING:**
- Real-time annotation rate tracking
- Individual annotator performance monitoring
- Overall dataset completion tracking
- Quality metric dashboards for administrators

**PAYMENT AND INCENTIVES:**
- Fair compensation based on time requirements
- Performance bonuses for high-quality work
- Feedback mechanisms for annotator improvement
- Long-term retention strategies

### 8.3 Data Processing Pipeline

**RAW DATA INGESTION:**
- Annotation response collection
- Timestamp and metadata recording
- Initial format validation
- Database storage with backup

**QUALITY FILTERING:**
- Automated quality checks
- Statistical outlier detection
- Manual review of flagged cases
- Final inclusion/exclusion decisions

**AGGREGATION AND CONSENSUS:**
- Multi-annotator response aggregation
- Consensus determination algorithms
- Confidence score calculation
- Final ground truth assignment

**EXPORT AND DISTRIBUTION:**
- Standardized data format creation
- Train/validation/test split generation
- Documentation and metadata preparation
- Public dataset release preparation

---

## 9. Dataset Characteristics and Properties

### 9.1 Statistical Properties

**RESPONSE DISTRIBUTION:**
- Binary choice distribution: Approximately balanced
- Soft label distribution: Concentrated near 0 and 1
- Confidence distribution: Higher confidence for easy cases
- Response time distribution: Log-normal with 8s median

**CONTENT DISTRIBUTION:**
- Distortion type balance: 50% traditional, 50% CNN-based
- Severity level distribution: Uniform across perceptual scales
- Image content diversity: Balanced across texture/object types
- Spatial frequency distribution: Full spectrum representation

**QUALITY METRICS:**
- Overall annotation quality: 94.3% pass rate
- Inter-annotator agreement: 82.6% average
- Expert validation agreement: 87.2%
- Test-retest reliability: 0.79 correlation

### 9.2 Comparative Analysis

**TRADITIONAL VS CNN-BASED DISTORTIONS:**
- Agreement rates: Traditional 84%, CNN-based 81%
- Difficulty distribution: CNN-based slightly more challenging
- Annotator preferences: Slight bias toward traditional familiarity
- Performance implications: Similar discrimination ability

**SEVERITY LEVEL ANALYSIS:**
- Low severity: 70% correct discrimination
- Medium severity: 85% correct discrimination  
- High severity: 95% correct discrimination
- Expected monotonic relationship confirmed

**CONTENT TYPE ANALYSIS:**
- Natural images: 83% agreement rate
- Texture patches: 81% agreement rate
- Object-centric patches: 85% agreement rate
- Abstract patterns: 79% agreement rate

### 9.3 Benchmark Establishment

**HUMAN PERFORMANCE CEILING:**
- Upper bound estimate: 82.6% based on agreement
- Perfect annotator simulation: 85-90% theoretical maximum
- Practical target: 75-80% for automatic metrics
- Realistic expectation: 70-75% for general-purpose metrics

**BASELINE PERFORMANCE:**
- Random choice: 50% accuracy
- Pixel-wise metrics: 59-62% accuracy
- Traditional perceptual metrics: 65-68% accuracy
- Deep feature baselines: 69-71% accuracy

**DIFFICULTY STRATIFICATION:**
- Easy subset (>90% agreement): Clear performance targets
- Medium subset (70-90% agreement): Discriminative evaluation
- Hard subset (<70% agreement): Research challenge cases
- Balanced evaluation across difficulty levels

---

## 10. Evaluation Metrics and Benchmarking

### 10.1 Primary Evaluation Metrics

**2AFC ACCURACY:**
- Definition: Percentage of human choices correctly predicted
- Formula: `Accuracy = (Correct predictions) / (Total predictions)`
- Interpretation: Higher values indicate better perceptual alignment

**STATISTICAL SIGNIFICANCE:**
- Confidence intervals: Bootstrap-based 95% CI
- Significance testing: McNemar's test for paired comparisons
- Effect size: Cohen's d for magnitude assessment
- Multiple comparisons: FDR correction for multiple metrics

**ROBUSTNESS ANALYSIS:**
- Subset performance: Agreement rate stratification
- Cross-validation: K-fold validation across triplets
- Temporal stability: Performance over time periods
- Annotator stratification: Performance across annotator groups

### 10.2 Secondary Evaluation Metrics

**CORRELATION ANALYSIS:**
- Pearson correlation: Linear relationship strength
- Spearman correlation: Monotonic relationship assessment
- Kendall's tau: Rank-based correlation measure
- Partial correlation: Controlling for confounding factors

**DISCRIMINATION METRICS:**
- Area under ROC curve: Binary classification performance
- Precision-recall curves: Performance at different thresholds
- F1 scores: Balanced precision-recall assessment
- Matthews correlation coefficient: Balanced binary metric

**CALIBRATION METRICS:**
- Reliability diagrams: Predicted vs actual agreement rates
- Brier score: Probabilistic prediction accuracy
- Expected calibration error: Systematic bias assessment
- Calibration slope: Linear calibration assessment

### 10.3 Benchmark Protocols

**STANDARD EVALUATION PROTOCOL:**
1. Load test triplets with human ground truth
2. Compute metric predictions for all triplets
3. Compare predictions with human choices
4. Calculate 2AFC accuracy with confidence intervals
5. Report statistical significance vs baselines

**COMPARATIVE EVALUATION:**
- Multiple metric comparison on same test set
- Statistical significance testing between metrics
- Performance stratification by difficulty/content
- Computational efficiency benchmarking

**REPRODUCIBILITY REQUIREMENTS:**
- Standardized test set splits
- Fixed random seeds for replicability
- Detailed implementation specifications
- Performance variance reporting across runs

**FUTURE BENCHMARK EXTENSIONS:**
- Video similarity evaluation protocols
- Multi-modal similarity assessment
- Domain-specific benchmark creation
- Real-time performance evaluation standards

---

## Summary and Conclusions

The human perceptual dataset and evaluation framework established by the LPIPS work represents a significant contribution to computer vision evaluation methodology. Key achievements include:

**METHODOLOGICAL CONTRIBUTIONS:**
- Systematic 2AFC evaluation protocol for perceptual similarity
- Large-scale human annotation dataset (150k triplets, 300k judgments)
- Comprehensive coverage of traditional and modern distortion types
- Rigorous quality control and statistical validation

**EMPIRICAL FINDINGS:**
- Human agreement rate of 82.6% establishes performance ceiling
- Traditional distortions show slightly higher agreement than CNN-based
- Difficulty stratification enables discriminative evaluation
- Cross-annotator consistency validates dataset reliability

**BENCHMARK ESTABLISHMENT:**
- Standard evaluation protocol for perceptual similarity metrics
- Performance baselines across metric categories
- Statistical frameworks for significance testing
- Foundation for future perceptual similarity research

The dataset and evaluation framework continue to serve as the gold standard for perceptual similarity evaluation, enabling systematic comparison of metrics and driving improvements in human-aligned computer vision systems.