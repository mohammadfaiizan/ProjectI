# ProjectI

A comprehensive repository covering AI/ML/Deep Learning, Data Structures & Algorithms, Agentic AI, Programming Languages, and Low-Level Design.

---

## Table of Contents

- [AI-ML-DL](#ai-ml-dl)
  - [Theory](#theory)
  - [Models](#models)
  - [Framework](#framework)
  - [LPIPS](#lpips)
  - [Interview](#ai-ml-dl-interview)
- [DSA](#dsa)
  - [DSA 450](#dsa-450)
  - [DSA 150](#dsa-150)
  - [Problem](#problem)
  - [Theory](#dsa-theory)
  - [Random](#random)
- [Agentic AI](#agentic-ai)
  - [Theory](#agentic-theory)
  - [Frameworks](#frameworks)
  - [Agent System Examples](#agent-system-examples)
  - [Interview Questions](#agentic-interview)
- [Language](#language)
  - [C++](#c)
  - [Python](#python)
- [LLD](#lld)

---

## AI-ML-DL

Comprehensive coverage of Artificial Intelligence, Machine Learning, and Deep Learning -- from mathematical foundations to production deployment.

### Theory

165 markdown files organized across 10 major domains with three-digit prefixed naming.

```
AI-ML-DL/Theory/
├── 001_Mathematics/              (15 files)
│   ├── 001_Linear_Algebra/
│   ├── 002_Calculus_And_Optimization/
│   └── 003_Statistics_And_Probability/
├── 002_ML_Fundamentals/          (20 files)
│   ├── 001_Core_Concepts/
│   ├── 002_Supervised_Learning/
│   ├── 003_Unsupervised_Learning/
│   └── 004_Evaluation_And_Selection/
├── 003_Deep_Learning/            (25 files)
│   ├── 001_NN_Fundamentals/
│   ├── 002_Architectures/
│   ├── 003_Training_Techniques/
│   └── 004_Regularization_And_Optimization/
├── 004_Computer_Vision/          (18 files)
│   ├── 001_Image_Processing/
│   ├── 002_Classical_CV/
│   └── 003_Deep_Learning_For_Vision/
├── 005_NLP/                      (22 files)
│   ├── 001_Text_Processing/
│   ├── 002_Classical_NLP/
│   ├── 003_Deep_Learning_For_NLP/
│   └── 004_Large_Language_Models/
├── 006_Reinforcement_Learning/   (15 files)
│   ├── 001_RL_Fundamentals/
│   ├── 002_Value_Based_Methods/
│   ├── 003_Policy_Based_Methods/
│   └── 004_Advanced_RL/
├── 007_Specialized_Domains/      (11 files)
│   ├── 001_Graph_Neural_Networks/
│   ├── 002_Generative_Models/
│   └── 003_Meta_Learning/
├── 008_MLOps_And_Production/     (18 files)
│   ├── 001_ML_Engineering/
│   ├── 002_Deployment/
│   └── 003_Monitoring_And_Maintenance/
├── 009_Ethics_And_Fairness/      (8 files)
│   ├── 001_AI_Ethics/
│   ├── 002_Fairness_And_Bias/
│   └── 003_Explainable_AI/
└── 010_Research_And_Frontier/    (12 files)
    ├── 001_Current_Research/
    ├── 002_Emerging_Technologies/
    └── 003_Future_Directions/
```

### Models

Implementation files for major model architectures with overview documentation.

```
AI-ML-DL/Models/
├── CNN/                 16 .py implementations + overview .md
├── Generative_AI/       18 .py implementations + overview .md
├── NLP/                 19 .py implementations + overview .md
├── Time_Series/         12 .py implementations + overview .md
└── Content/             5 blueprint/structure .md files
```

### Framework

Hands-on framework guides with theory, implementation code, and interview prep.

```
AI-ML-DL/Framework/
├── PyTorch/             ~200 .py files across 13 segments
│   ├── 001_fundamentals/
│   ├── 002_ml_dl_syntax/
│   ├── 003_autograd/
│   ├── 004_neural_networks/
│   ├── 005_data_preprocessing/
│   ├── 006_loss_optimization/
│   ├── 007_data_loading/
│   ├── 008_computer_vision/
│   ├── 009_nlp/
│   ├── 010_advanced_training/
│   ├── 011_model_analysis/
│   ├── 012_production/
│   └── 013_ecosystem/
├── NumPy/               5 theory .md + 5 code .py + 5 interview .md
│   ├── 01_Foundations_NDArray_And_DTypes
│   ├── 02_Indexing_Slicing_Reshaping_And_Manipulation
│   ├── 03_Mathematics_Statistics_And_Broadcasting
│   ├── 04_Linear_Algebra_FFT_Random_And_Polynomials
│   └── 05_Performance_IO_Interop_And_Advanced
└── Pandas/              5 theory .md + 5 code .py + 5 interview .md
    ├── 01_Foundations_Series_DataFrame_And_DTypes
    ├── 02_Indexing_Selection_Filtering_And_Manipulation
    ├── 03_Data_Cleaning_Transformation_And_Aggregation
    ├── 04_Time_Series_IO_And_Visualization
    └── 05_Performance_Advanced_And_Best_Practices
```

### LPIPS

Learned Perceptual Image Patch Similarity -- complete study including paper, theory, implementation, and interview prep.

```
AI-ML-DL/LPIPS/
├── Docs/                Paper PDFs and LaTeX source
├── Theory/              15 .md files (fundamentals to advanced)
├── Interview/           10 .md files (Q&A)
├── LPIPS/               Core implementation files
├── LPIPS_Zhang/         Full reference implementation
└── Supporting_Models/   AlexNet, SqueezeNet, VGG backbones
```

### AI-ML-DL Interview

20 markdown files covering the full AI/ML/DL spectrum -- 20 Q&A per file (400 questions total).

```
AI-ML-DL/Interview/
├── 01_Machine_Learning_Fundamentals.md
├── 02_Supervised_Learning.md
├── 03_Unsupervised_Learning.md
├── 04_Model_Evaluation_And_Metrics.md
├── 05_Bias_Variance_And_Regularization.md
├── 06_Decision_Trees_And_Ensemble_Methods.md
├── 07_Support_Vector_Machines_And_Kernels.md
├── 08_Neural_Network_Fundamentals.md
├── 09_Activation_Functions_And_Optimization.md
├── 10_Convolutional_Neural_Networks.md
├── 11_Recurrent_Neural_Networks_And_LSTMs.md
├── 12_Transformers_And_Attention.md
├── 13_Natural_Language_Processing.md
├── 14_Generative_Models.md
├── 15_Transfer_Learning_And_Fine_Tuning.md
├── 16_Reinforcement_Learning.md
├── 17_Feature_Engineering_And_Preprocessing.md
├── 18_Dimensionality_Reduction.md
├── 19_Model_Deployment_And_MLOps.md
└── 20_Large_Language_Models_And_Modern_AI.md
```

---

## DSA

Data Structures and Algorithms -- problems, theory implementations, and curated problem sets in C++ and Python.

### DSA 450

451 C++ solutions organized by topic (based on the DSA 450 sheet).

```
DSA/DSA_450/
├── 01_array/
├── 02_matrix/
├── 03_string/
├── 04_searching_sorting/
├── 05_linked_list/
├── 06_binary_tree/
├── 07_bst/
├── 08_greedy/
├── 09_backtracking/
├── 10_stacks_queues/
├── 11_heap/
├── 12_graph/
├── 13_trie/
├── 14_dynamic_programming/
└── 15_bit-manipulation/
```

### DSA 150

143 Python solutions -- a focused subset of essential problems.

### Problem

565 additional practice problems in Python organized by topic.

### DSA Theory

110 Python implementation files covering core data structure and algorithm theory.

```
DSA/Theory/
├── Array/                  18 files
├── String/                 14 files
├── SearchingSorting/        16 files
├── Matrix/                 12 files
├── LinkedList/             11 files
├── Stack/                   4 files
├── Queue/                   4 files
├── Recursion/               8 files
├── Backtracking/            8 files
├── Bit_Manipulation/       11 files
├── Tree/                   14 files
├── AdvancedTree/            8 files
├── Heap/                    4 files
├── PriorityQueue/           3 files
├── Graph/                  12 files
├── Greedy/                  4 files
├── Dynamic_Programming/    18 files
├── Trie/                    6 files
└── DSA_Definition/         C++ definitions
```

### Random

49 additional problems by difficulty.

```
DSA/Random/
├── Easy/       31 files
├── Medium/     12 files
└── Hard/        6 files
```

---

## Agentic AI

Complete guide to AI Agents -- theory, frameworks, production implementations, and interview prep.

### Agentic Theory

10 markdown files covering core agent concepts: LLMs, RAG, memory, multi-agent systems, tool use, planning, reasoning, safety, evaluation, and production deployment.

### Frameworks

6 framework deep-dives with theory and implementation code.

```
Agentic_AI/02_Frameworks/
├── LangChain_LangGraph/     1 .md + 3 .py
├── CrewAI/                  1 .md + 2 .py
├── AutoGen/                 1 .md + 2 .py
├── OpenAI_Assistants/       1 .md + 2 .py
├── LlamaIndex/              1 .md + 2 .py
└── Custom_Framework/        1 .md + 2 .py
```

### Agent System Examples

10 production-grade agent implementations, each with description, implementation, and a full LangChain-based modular version.

```
Agentic_AI/03_Agent_System_Examples/
├── 001_RAG_Chatbot/
├── 002_Research_Assistant/
├── 003_Code_Review_Agent/
├── 004_Customer_Support_Agent/
├── 005_Data_Analysis_Agent/
├── 006_Content_Generation_Pipeline/
├── 007_Multi_Agent_Task_Solver/
├── 008_Autonomous_Web_Agent/
├── 009_Document_Processing_System/
└── 010_Trading_Analysis_Agent/
    ├── Description.md
    ├── Implementation.py
    └── LangChain/
        ├── Config.py
        ├── Tools.py
        ├── Agent.py
        ├── Main.py
        └── Sample_Input.py
```

### Agentic Interview

6 markdown files with 25-30 Q&A each covering agents, RAG, multi-agent systems, tool use, safety, and production deployment.

---

## Language

Programming language deep-dives with structured modules.

### C++

15 modules covering C++ from basics to advanced topics -- 96 implementation files.

```
Language/Cpp/
├── 001_basics_cpp/
├── 002_advanced_oop/
├── 003_templates_metaprogramming/
├── 004_modern_cpp_11_14_17_20/
├── 005_memory_and_pointers/
├── 006_concurrency_multithreading/
├── 007_stl_deep_dive/
├── 008_design_patterns_cpp/
├── 009_linux_system_level_cpp/
├── 010_cmake_and_toolchain/
├── 011_testing_debugging_profiling/
├── 012_cpp_network_and_web/
├── 013_cpp_for_ai_and_gpu/
├── 014_advanced_projects/
└── 015_cpp_latest_and_future/
```

### Python

3 modules with 32 implementation files covering fundamentals, OOP, and advanced Python.

```
Language/Python/
├── Module1_Fundamentals/    10 files
├── Module2_OOP/             10 files
└── Module3_Advanced/        12 files
```

---

## LLD

Low-Level Design -- OOP, SOLID, design patterns, and system design problems in Python.

```
LLD/
├── 01_OOPS_Concepts/        15 files
├── 02_SOLID_Principles/     10 files
├── 03_Design_Patterns/      23 files (all GoF patterns)
└── 04_Design_Problems/      25 files (real-world system designs)
```

---

## Repository Stats

| Section | Files | Format |
|---------|-------|--------|
| AI-ML-DL Theory | 165 | .md |
| AI-ML-DL Models | 69 | .py + .md |
| AI-ML-DL Framework (PyTorch) | ~200 | .py |
| AI-ML-DL Framework (NumPy) | 15 | .md + .py |
| AI-ML-DL Framework (Pandas) | 15 | .md + .py |
| AI-ML-DL LPIPS | 64 | .md + .py |
| AI-ML-DL Interview | 20 | .md |
| DSA 450 | 451 | .cpp |
| DSA 150 | 143 | .py |
| DSA Problem | 565 | .py |
| DSA Theory | 110 | .py |
| Agentic AI | 106 | .md + .py |
| Language (C++) | 96 | .cpp |
| Language (Python) | 32 | .py |
| LLD | 73 | .py |

**Total: ~2,100+ files**
