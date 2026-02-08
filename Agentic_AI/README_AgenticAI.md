# Agentic AI - Complete Learning Repository

A comprehensive, implementation-focused repository covering Agentic AI from beginner to professional level. Organized into 4 main sections: Theory, Frameworks, Agent System Examples, and Interview Preparation.

---

## Repository Structure

```
Agentic_AI/
|
|-- README_AgenticAI.md
|
|-- 01_Theory/
|   |-- 001_Introduction_To_AI_Agents.md
|   |-- 002_Agent_Architectures_And_Design_Patterns.md
|   |-- 003_LLMs_Prompting_And_Function_Calling.md
|   |-- 004_RAG_And_Knowledge_Systems.md
|   |-- 005_Memory_And_State_Management.md
|   |-- 006_Multi_Agent_Systems.md
|   |-- 007_Tool_Use_And_API_Integration.md
|   |-- 008_Planning_Reasoning_And_Decision_Making.md
|   |-- 009_Safety_Guardrails_And_Evaluation.md
|   |-- 010_Production_Deployment_And_Monitoring.md
|
|-- 02_Frameworks/
|   |
|   |-- LangChain_LangGraph/
|   |   |-- 001_LangChain_LangGraph_Theory.md
|   |   |-- 002_LangChain_Agents_And_Chains.py
|   |   |-- 003_LangGraph_Stateful_Workflows.py
|   |   |-- 004_LangChain_RAG_Pipeline.py
|   |
|   |-- CrewAI/
|   |   |-- 001_CrewAI_Theory.md
|   |   |-- 002_CrewAI_Agent_Crews.py
|   |   |-- 003_CrewAI_Advanced_Workflows.py
|   |
|   |-- AutoGen/
|   |   |-- 001_AutoGen_Theory.md
|   |   |-- 002_AutoGen_Multi_Agent_Chat.py
|   |   |-- 003_AutoGen_Code_Generation.py
|   |
|   |-- OpenAI_Assistants/
|   |   |-- 001_OpenAI_Assistants_Theory.md
|   |   |-- 002_OpenAI_Function_Calling.py
|   |   |-- 003_OpenAI_Assistants_Advanced.py
|   |
|   |-- LlamaIndex/
|   |   |-- 001_LlamaIndex_Theory.md
|   |   |-- 002_LlamaIndex_RAG_Agent.py
|   |   |-- 003_LlamaIndex_Query_Engine.py
|   |
|   |-- Custom_Framework/
|       |-- 001_Building_Custom_Agent_Theory.md
|       |-- 002_Custom_Agent_From_Scratch.py
|       |-- 003_Custom_Multi_Agent_System.py
|
|-- 03_Agent_System_Examples/
|   |
|   |-- 001_RAG_Chatbot/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 002_Research_Assistant/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 003_Code_Review_Agent/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 004_Customer_Support_Agent/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 005_Data_Analysis_Agent/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 006_Content_Generation_Pipeline/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 007_Multi_Agent_Task_Solver/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 008_Autonomous_Web_Agent/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 009_Document_Processing_System/
|   |   |-- Description.md
|   |   |-- Implementation.py
|   |
|   |-- 010_Trading_Analysis_Agent/
|       |-- Description.md
|       |-- Implementation.py
|
|-- 04_Interview_Questions/
    |-- 001_Fundamentals_And_Core_Concepts.md
    |-- 002_Agent_Patterns_And_Architectures.md
    |-- 003_RAG_Knowledge_And_Memory.md
    |-- 004_Frameworks_And_Tools.md
    |-- 005_System_Design_And_Production.md
    |-- 006_Advanced_And_Scenario_Based.md
```

---

## Section Details

### 01_Theory/ (10 files, .md format)

Large, comprehensive theory files progressing from beginner to professional level. Each file contains detailed explanations, diagrams (ASCII), code snippets, real-world examples, and best practices.

| # | File | Topics Covered |
|---|------|----------------|
| 001 | Introduction_To_AI_Agents | What are agents, history, types (reactive/deliberative/hybrid), agent vs chatbot, anatomy of an agent, perception-reasoning-action loop, real-world use cases |
| 002 | Agent_Architectures_And_Design_Patterns | ReAct, Plan-and-Execute, Chain-of-Thought, Reflection, State Machine, Pipeline, single vs multi-agent, orchestration patterns, when to use what |
| 003 | LLMs_Prompting_And_Function_Calling | LLM internals for agents, prompt engineering for agents, system prompts, few-shot, function calling (OpenAI/Anthropic/Google), structured outputs, token management |
| 004 | RAG_And_Knowledge_Systems | RAG architecture, chunking strategies, embedding models, vector databases, retrieval methods (dense/sparse/hybrid), reranking, advanced RAG (corrective, self-RAG, agentic RAG), knowledge graphs |
| 005 | Memory_And_State_Management | Short-term vs long-term memory, episodic/semantic/procedural memory, conversation history, context window management, memory consolidation, state machines for agents |
| 006 | Multi_Agent_Systems | Multi-agent architectures (master-worker, peer-to-peer, hierarchical), communication protocols, task delegation, consensus, coordination, collaboration patterns |
| 007 | Tool_Use_And_API_Integration | Function calling, tool discovery, dynamic tool loading, API integration, database agents, web browsing, code execution sandboxing, tool composition, safety |
| 008 | Planning_Reasoning_And_Decision_Making | Task decomposition, hierarchical planning, tree-of-thought, graph-of-thought, self-reflection, causal reasoning, uncertainty handling, meta-cognition |
| 009 | Safety_Guardrails_And_Evaluation | Prompt injection defense, output validation, content filtering, human-in-the-loop, evaluation metrics, benchmarks, red-teaming, testing strategies |
| 010 | Production_Deployment_And_Monitoring | Containerization, scaling, cost optimization, latency management, observability, logging, tracing, A/B testing, CI/CD for agents, LLMOps |

### 02_Frameworks/ (6 subfolders, 18 .py + 6 .md = 24 files)

Each framework subfolder contains:
- 1 theory `.md` file covering the framework's architecture, concepts, pros/cons, and when to use it
- 2-3 `.py` implementation files with production-ready, runnable code

| Framework | Theory | Implementations |
|-----------|--------|-----------------|
| LangChain/LangGraph | Architecture, LCEL, agents, graph-based workflows | Agents+Chains, Stateful Workflows, RAG Pipeline |
| CrewAI | Role-based agents, crews, tasks, processes | Agent Crews, Advanced Workflows |
| AutoGen | Conversational agents, code generation, group chat | Multi-Agent Chat, Code Generation |
| OpenAI Assistants | Assistants API, threads, function calling | Function Calling, Advanced Assistants |
| LlamaIndex | Data framework, query engines, agent tools | RAG Agent, Query Engine |
| Custom Framework | Building from scratch, architecture decisions | Custom Agent, Custom Multi-Agent |

### 03_Agent_System_Examples/ (10 projects, 10 .md + 10 .py = 20 files)

Complete end-to-end agent system implementations. Each project has:
- `Description.md` -- Problem statement, architecture diagram, component breakdown, data flow, design decisions
- `Implementation.py` -- Full working Python implementation with all components

| # | Project | Complexity |
|---|---------|------------|
| 001 | RAG Chatbot | Beginner |
| 002 | Research Assistant | Beginner-Intermediate |
| 003 | Code Review Agent | Intermediate |
| 004 | Customer Support Agent | Intermediate |
| 005 | Data Analysis Agent | Intermediate |
| 006 | Content Generation Pipeline | Intermediate-Advanced |
| 007 | Multi-Agent Task Solver | Advanced |
| 008 | Autonomous Web Agent | Advanced |
| 009 | Document Processing System | Advanced |
| 010 | Trading Analysis Agent | Professional |

### 04_Interview_Questions/ (6 files, .md format)

Each file contains 25-30 questions with detailed answers, progressing from basic to advanced within each topic.

| # | File | Focus Area |
|---|------|------------|
| 001 | Fundamentals_And_Core_Concepts | Agent definition, types, architectures, LLM basics, prompting |
| 002 | Agent_Patterns_And_Architectures | ReAct, CoT, planning, reflection, state machines, multi-agent |
| 003 | RAG_Knowledge_And_Memory | RAG pipeline, embeddings, vector DBs, chunking, memory systems |
| 004 | Frameworks_And_Tools | LangChain, CrewAI, AutoGen, OpenAI, framework comparison |
| 005 | System_Design_And_Production | Scalability, deployment, monitoring, cost, security, testing |
| 006 | Advanced_And_Scenario_Based | System design scenarios, debugging, optimization, real-world problems |

---

## Total File Count

| Section | .md Files | .py Files | Total |
|---------|-----------|-----------|-------|
| 01_Theory | 10 | 0 | 10 |
| 02_Frameworks | 6 | 18 | 24 |
| 03_Agent_System_Examples | 10 | 10 | 20 |
| 04_Interview_Questions | 6 | 0 | 6 |
| **Total** | **32** | **28** | **60** |

---

## Learning Path

**Beginner (Weeks 1-3):**
- Theory files 001-003
- Framework: LangChain basics
- Projects: RAG Chatbot, Research Assistant
- Interview: File 001

**Intermediate (Weeks 4-6):**
- Theory files 004-007
- Frameworks: CrewAI, AutoGen, OpenAI
- Projects: Code Review, Customer Support, Data Analysis
- Interview: Files 002-004

**Advanced/Professional (Weeks 7-10):**
- Theory files 008-010
- Frameworks: LlamaIndex, Custom Framework
- Projects: Multi-Agent Solver, Web Agent, Document Processing, Trading Agent
- Interview: Files 005-006

---

## Prerequisites

- Python 3.9+
- Basic understanding of machine learning concepts
- Familiarity with REST APIs
- Understanding of LLMs (GPT, Claude, etc.)

## Recommended Tools

- OpenAI API / Anthropic API key
- Vector database (ChromaDB, Pinecone, or Weaviate)
- Python virtual environment (venv or conda)
- Docker (for production deployment topics)
