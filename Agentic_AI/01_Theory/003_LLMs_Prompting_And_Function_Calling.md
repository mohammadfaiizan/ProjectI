# LLMs, Prompting, and Function Calling for Agentic AI

## Table of Contents

1. [LLM Fundamentals for Agent Developers](#llm-fundamentals-for-agent-developers)
2. [Prompt Engineering for Agents](#prompt-engineering-for-agents)
3. [Function Calling / Tool Use](#function-calling--tool-use)
4. [Structured Outputs](#structured-outputs)
5. [Token Management](#token-management)
6. [Advanced Techniques](#advanced-techniques)

---

## LLM Fundamentals for Agent Developers

### Transformer Architecture (Simplified)

Large Language Models (LLMs) are built on the Transformer architecture, which revolutionized natural language processing. Understanding the basics helps agent developers make informed decisions about model selection and prompt design.

#### Core Components

```
Input Text → Tokenization → Embedding Layer
                                    ↓
                            Positional Encoding
                                    ↓
                    ┌───────────────────────────────┐
                    │   Multi-Head Self-Attention   │
                    │   (Query, Key, Value)         │
                    └───────────────────────────────┘
                                    ↓
                            Layer Normalization
                                    ↓
                    ┌───────────────────────────────┐
                    │   Feed-Forward Network        │
                    │   (MLP with activation)       │
                    └───────────────────────────────┘
                                    ↓
                            Layer Normalization
                                    ↓
                    [Repeat N times (e.g., 32-96 layers)]
                                    ↓
                            Output Projection
                                    ↓
                    Token Probability Distribution
                                    ↓
                            Sampling Strategy
                                    ↓
                            Generated Token
```

#### Key Concepts for Agent Developers

**Attention Mechanism:**
- Allows the model to focus on relevant parts of the input
- Critical for understanding context in multi-turn conversations
- Determines how well agents can track conversation history

**Layer Depth:**
- Deeper models (more layers) = better reasoning but slower inference
- Shallow models = faster but may miss complex patterns
- Balance depends on agent latency requirements

**Vocabulary Size:**
- Larger vocabularies = better token efficiency
- Smaller vocabularies = faster tokenization
- Affects token counting accuracy

### Token Prediction and Generation

LLMs work by predicting the next token in a sequence. Each token is assigned a probability, and the model samples from this distribution.

#### Token Prediction Process

```python
# Simplified token prediction example
def predict_next_token(model, context_tokens):
    """
    Simplified representation of token prediction
    
    Args:
        model: The LLM model
        context_tokens: List of token IDs representing context
    
    Returns:
        Probability distribution over vocabulary
    """
    # Forward pass through transformer
    logits = model.forward(context_tokens)
    
    # Convert logits to probabilities
    probs = softmax(logits[-1])  # Last token's logits
    
    # Example output shape: [vocab_size] = [50,000+]
    return probs

# Example usage
context = "The agent decided to"
probs = predict_next_token(model, tokenize(context))
# probs might be: {"call": 0.3, "execute": 0.25, "invoke": 0.15, ...}
```

#### Generation Strategies

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def demonstrate_sampling_strategies():
    """Demonstrate different token sampling approaches"""
    
    prompt = "The AI agent should"
    
    # Greedy decoding (always pick highest probability)
    response_greedy = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # Deterministic
        max_tokens=50
    )
    
    # Random sampling with temperature
    response_random = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # More randomness
        max_tokens=50
    )
    
    # Top-k sampling (only consider top k tokens)
    response_topk = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_k=50,  # Only sample from top 50 tokens
        max_tokens=50
    )
    
    return {
        "greedy": response_greedy.choices[0].message.content,
        "random": response_random.choices[0].message.content,
        "top_k": response_topk.choices[0].message.content
    }
```

### Context Windows and Their Impact on Agents

Context windows determine how much information an agent can consider when making decisions. This is crucial for multi-turn conversations and tool-calling scenarios.

#### Context Window Sizes by Model

| Model | Context Window | Use Case for Agents |
|-------|---------------|---------------------|
| GPT-4 Turbo | 128K tokens | Long conversations, extensive tool history |
| GPT-4 | 8K tokens | Standard agent tasks |
| GPT-3.5 Turbo | 16K tokens | Cost-effective for medium contexts |
| Claude 3 Opus | 200K tokens | Very long document analysis agents |
| Claude 3 Sonnet | 200K tokens | Multi-document RAG agents |
| Gemini Pro 1.5 | 1M tokens | Massive context agents |
| Llama 3 70B | 8K tokens | Open-source agent deployments |
| Mistral Large | 32K tokens | European-compliant agents |

#### Context Window Management for Agents

```python
from typing import List, Dict
import tiktoken

class AgentContextManager:
    """Manages context window for agent conversations"""
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 8000):
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
        self.conversation_history: List[Dict] = []
        self.system_prompt = ""
        self.tool_results: List[Dict] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def add_tool_result(self, tool_name: str, result: str):
        """Add tool execution result"""
        self.tool_results.append({
            "tool": tool_name,
            "result": result
        })
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def get_total_tokens(self) -> int:
        """Calculate total tokens in current context"""
        total = self.count_tokens(self.system_prompt)
        
        for msg in self.conversation_history:
            total += self.count_tokens(f"{msg['role']}: {msg['content']}")
        
        for tool_result in self.tool_results:
            total += self.count_tokens(
                f"Tool {tool_result['tool']}: {tool_result['result']}"
            )
        
        return total
    
    def trim_to_fit(self, reserve_tokens: int = 1000):
        """Trim oldest messages to fit within context window"""
        max_context = self.max_tokens - reserve_tokens
        
        while self.get_total_tokens() > max_context:
            if len(self.conversation_history) > 1:
                # Keep system message, remove oldest user/assistant pair
                self.conversation_history.pop(1)  # Remove first non-system
            else:
                break
    
    def get_messages(self) -> List[Dict]:
        """Get formatted messages for API call"""
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        messages.extend(self.conversation_history)
        
        # Append tool results as assistant messages
        for tool_result in self.tool_results:
            messages.append({
                "role": "assistant",
                "content": f"Tool {tool_result['tool']} returned: {tool_result['result']}"
            })
        
        return messages

# Usage example
context_manager = AgentContextManager(model="gpt-4", max_tokens=8000)
context_manager.system_prompt = "You are a helpful AI agent."
context_manager.add_message("user", "What is the weather?")
context_manager.add_message("assistant", "I'll check the weather for you.")
context_manager.add_tool_result("get_weather", "Sunny, 72°F")

print(f"Total tokens: {context_manager.get_total_tokens()}")
context_manager.trim_to_fit(reserve_tokens=1000)
messages = context_manager.get_messages()
```

### Temperature, Top-p, and Sampling Strategies

These parameters control the randomness and creativity of model outputs, directly affecting agent behavior.

#### Temperature

Temperature controls randomness:
- **0.0**: Deterministic, always picks most likely token (greedy)
- **0.7**: Balanced creativity and consistency
- **1.0**: Standard randomness
- **>1.5**: High creativity, may produce nonsensical outputs

```python
def demonstrate_temperature_effects():
    """Show how temperature affects agent responses"""
    
    prompt = "Analyze this data and provide insights: [1, 2, 3, 4, 5]"
    
    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
    results = {}
    
    for temp in temperatures:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=100
        )
        results[temp] = response.choices[0].message.content
    
    return results

# Agent-specific temperature guidelines
AGENT_TEMPERATURE_CONFIGS = {
    "reasoning_agent": 0.0,      # Deterministic reasoning
    "creative_agent": 0.9,        # Creative content generation
    "code_agent": 0.2,           # Consistent code generation
    "conversational_agent": 0.7,  # Natural conversation
    "planning_agent": 0.1,       # Consistent planning
}
```

#### Top-p (Nucleus Sampling)

Top-p samples from the smallest set of tokens whose cumulative probability exceeds p.

```python
def demonstrate_top_p():
    """Demonstrate top-p sampling"""
    
    prompt = "Generate a creative solution to reduce server costs"
    
    # Low top-p: conservative, only high-probability tokens
    response_low = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.5,  # Only consider tokens up to 50% cumulative probability
        max_tokens=150
    )
    
    # High top-p: more diverse, considers more tokens
    response_high = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.95,  # Consider tokens up to 95% cumulative probability
        max_tokens=150
    )
    
    return {
        "low_top_p": response_low.choices[0].message.content,
        "high_top_p": response_high.choices[0].message.content
    }
```

#### Sampling Strategy Selection for Agents

```python
class AgentSamplingConfig:
    """Configuration for agent sampling strategies"""
    
    @staticmethod
    def get_config(agent_type: str) -> dict:
        """Get optimal sampling config for agent type"""
        configs = {
            "tool_calling_agent": {
                "temperature": 0.0,  # Deterministic tool selection
                "top_p": 0.1,
                "top_k": 10
            },
            "reasoning_agent": {
                "temperature": 0.1,
                "top_p": 0.3,
                "top_k": 20
            },
            "creative_writing_agent": {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 50
            },
            "code_generation_agent": {
                "temperature": 0.2,
                "top_p": 0.5,
                "top_k": 30
            },
            "multi_agent_coordinator": {
                "temperature": 0.3,
                "top_p": 0.7,
                "top_k": 40
            }
        }
        return configs.get(agent_type, {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40
        })
```

### Model Selection Criteria

Choosing the right model is critical for agent performance, cost, and latency.

#### Model Comparison Matrix

| Model | Provider | Context | Speed | Cost | Best For |
|-------|----------|---------|-------|------|----------|
| GPT-4 Turbo | OpenAI | 128K | Medium | High | Complex reasoning, tool use |
| GPT-4 | OpenAI | 8K | Slow | Very High | Maximum quality |
| GPT-3.5 Turbo | OpenAI | 16K | Fast | Low | Simple agents, high volume |
| Claude 3 Opus | Anthropic | 200K | Medium | High | Long context, analysis |
| Claude 3 Sonnet | Anthropic | 200K | Fast | Medium | Balanced performance |
| Claude 3 Haiku | Anthropic | 200K | Very Fast | Low | Fast responses, simple tasks |
| Gemini Pro 1.5 | Google | 1M | Medium | Medium | Massive context |
| Llama 3 70B | Meta | 8K | Medium | Self-hosted | Privacy, customization |
| Mistral Large | Mistral | 32K | Fast | Medium | European compliance |

#### Model Selection Decision Tree

```python
class ModelSelector:
    """Helps select the right model for agent tasks"""
    
    def __init__(self):
        self.models = {
            "gpt-4-turbo": {
                "provider": "openai",
                "context": 128000,
                "cost_per_1k_tokens": 0.01,
                "latency_ms": 2000,
                "capabilities": ["reasoning", "tool_use", "code"]
            },
            "gpt-3.5-turbo": {
                "provider": "openai",
                "context": 16000,
                "cost_per_1k_tokens": 0.0015,
                "latency_ms": 500,
                "capabilities": ["tool_use", "code"]
            },
            "claude-3-opus": {
                "provider": "anthropic",
                "context": 200000,
                "cost_per_1k_tokens": 0.015,
                "latency_ms": 3000,
                "capabilities": ["reasoning", "analysis", "long_context"]
            },
            "claude-3-sonnet": {
                "provider": "anthropic",
                "context": 200000,
                "cost_per_1k_tokens": 0.003,
                "latency_ms": 1500,
                "capabilities": ["reasoning", "tool_use", "balanced"]
            },
            "claude-3-haiku": {
                "provider": "anthropic",
                "context": 200000,
                "cost_per_1k_tokens": 0.00025,
                "latency_ms": 500,
                "capabilities": ["fast", "simple_tasks"]
            }
        }
    
    def select_model(
        self,
        required_context: int,
        max_latency_ms: int,
        max_cost_per_1k: float,
        required_capabilities: list
    ) -> str:
        """Select model based on requirements"""
        
        candidates = []
        
        for model_name, specs in self.models.items():
            # Check context requirement
            if specs["context"] < required_context:
                continue
            
            # Check latency requirement
            if specs["latency_ms"] > max_latency_ms:
                continue
            
            # Check cost requirement
            if specs["cost_per_1k_tokens"] > max_cost_per_1k:
                continue
            
            # Check capabilities
            has_all_capabilities = all(
                cap in specs["capabilities"] for cap in required_capabilities
            )
            if not has_all_capabilities:
                continue
            
            candidates.append((model_name, specs))
        
        if not candidates:
            raise ValueError("No model meets all requirements")
        
        # Select cheapest that meets requirements
        candidates.sort(key=lambda x: x[1]["cost_per_1k_tokens"])
        return candidates[0][0]

# Usage
selector = ModelSelector()
selected = selector.select_model(
    required_context=50000,
    max_latency_ms=2000,
    max_cost_per_1k=0.01,
    required_capabilities=["reasoning", "tool_use"]
)
print(f"Selected model: {selected}")
```

#### Cost Calculation for Agents

```python
class AgentCostCalculator:
    """Calculate costs for agent operations"""
    
    def __init__(self):
        self.pricing = {
            "gpt-4-turbo": {
                "input": 0.01,   # per 1K tokens
                "output": 0.03   # per 1K tokens
            },
            "gpt-3.5-turbo": {
                "input": 0.0015,
                "output": 0.002
            },
            "claude-3-opus": {
                "input": 0.015,
                "output": 0.075
            },
            "claude-3-sonnet": {
                "input": 0.003,
                "output": 0.015
            }
        }
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate total cost for a request"""
        if model not in self.pricing:
            raise ValueError(f"Unknown model: {model}")
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def estimate_monthly_cost(
        self,
        model: str,
        avg_input_tokens: int,
        avg_output_tokens: int,
        requests_per_day: int
    ) -> dict:
        """Estimate monthly costs"""
        daily_cost = self.calculate_cost(
            model, avg_input_tokens, avg_output_tokens
        ) * requests_per_day
        
        monthly_cost = daily_cost * 30
        
        return {
            "per_request": self.calculate_cost(
                model, avg_input_tokens, avg_output_tokens
            ),
            "daily": daily_cost,
            "monthly": monthly_cost,
            "requests_per_day": requests_per_day
        }

# Example usage
calculator = AgentCostCalculator()
costs = calculator.estimate_monthly_cost(
    model="gpt-4-turbo",
    avg_input_tokens=2000,
    avg_output_tokens=500,
    requests_per_day=1000
)
print(f"Monthly cost: ${costs['monthly']:.2f}")
```

---

## Prompt Engineering for Agents

### System Prompts Design

System prompts define the agent's role, behavior, and constraints. Well-designed system prompts are crucial for reliable agent performance.

#### System Prompt Components

A good system prompt includes:
1. **Role Definition**: Who/what the agent is
2. **Capabilities**: What the agent can do
3. **Constraints**: What the agent cannot do
4. **Output Format**: How responses should be structured
5. **Behavior Guidelines**: How to handle edge cases

```python
# Example: Well-structured system prompt
SYSTEM_PROMPT_TEMPLATE = """
You are {agent_name}, an AI agent specialized in {domain}.

CAPABILITIES:
{capabilities}

CONSTRAINTS:
{constraints}

OUTPUT FORMAT:
{output_format}

BEHAVIOR GUIDELINES:
{guidelines}
"""

# Example: Data Analysis Agent
DATA_ANALYSIS_AGENT_PROMPT = """
You are DataAnalyzer, an AI agent specialized in data analysis and insights generation.

CAPABILITIES:
- Analyze datasets and identify patterns
- Generate statistical summaries
- Create visualizations (describe them)
- Provide actionable insights
- Answer questions about data

CONSTRAINTS:
- Never make up data that doesn't exist
- Always cite specific data points when making claims
- If data is insufficient, clearly state limitations
- Do not perform calculations you cannot verify

OUTPUT FORMAT:
1. Executive Summary (2-3 sentences)
2. Key Findings (bulleted list)
3. Statistical Summary (if applicable)
4. Insights and Recommendations
5. Limitations and Caveats

BEHAVIOR GUIDELINES:
- Start with a brief overview
- Use clear, non-technical language when possible
- Highlight the most important findings first
- Provide context for statistical measures
- Suggest next steps or follow-up analyses
"""
```

#### Role-Based System Prompts

```python
class SystemPromptBuilder:
    """Build system prompts for different agent roles"""
    
    ROLES = {
        "research_assistant": {
            "role": "Research Assistant",
            "capabilities": [
                "Search and synthesize information",
                "Cite sources accurately",
                "Identify knowledge gaps",
                "Provide balanced perspectives"
            ],
            "constraints": [
                "Only use information from provided sources",
                "Clearly distinguish facts from opinions",
                "Acknowledge uncertainty when present"
            ],
            "output_format": "Markdown with citations",
            "tone": "Academic, objective"
        },
        "code_reviewer": {
            "role": "Code Review Agent",
            "capabilities": [
                "Analyze code quality",
                "Identify bugs and security issues",
                "Suggest improvements",
                "Check adherence to best practices"
            ],
            "constraints": [
                "Focus on actionable feedback",
                "Prioritize security and performance",
                "Consider maintainability"
            ],
            "output_format": "Structured review with severity levels",
            "tone": "Constructive, technical"
        },
        "customer_support": {
            "role": "Customer Support Agent",
            "capabilities": [
                "Answer product questions",
                "Troubleshoot issues",
                "Escalate complex problems",
                "Maintain professional tone"
            ],
            "constraints": [
                "Never make promises about features",
                "Always be polite and empathetic",
                "Escalate billing issues to human agents"
            ],
            "output_format": "Conversational, helpful",
            "tone": "Friendly, professional"
        }
    }
    
    @classmethod
    def build_prompt(cls, role_name: str, custom_params: dict = None) -> str:
        """Build system prompt for a role"""
        if role_name not in cls.ROLES:
            raise ValueError(f"Unknown role: {role_name}")
        
        role_config = cls.ROLES[role_name]
        if custom_params:
            role_config = {**role_config, **custom_params}
        
        prompt = f"""You are {role_config['role']}, an AI agent specialized in this domain.

CAPABILITIES:
"""
        for cap in role_config['capabilities']:
            prompt += f"- {cap}\n"
        
        prompt += "\nCONSTRAINTS:\n"
        for constraint in role_config['constraints']:
            prompt += f"- {constraint}\n"
        
        prompt += f"\nOUTPUT FORMAT: {role_config['output_format']}\n"
        prompt += f"TONE: {role_config['tone']}\n"
        
        return prompt

# Usage
prompt = SystemPromptBuilder.build_prompt("research_assistant")
print(prompt)
```

#### Dynamic System Prompt Construction

```python
class DynamicSystemPrompt:
    """Build system prompts dynamically based on context"""
    
    def __init__(self):
        self.base_prompt = ""
        self.rules = []
        self.examples = []
        self.output_schema = None
    
    def set_base_role(self, role: str, description: str):
        """Set the base role and description"""
        self.base_prompt = f"You are {role}. {description}\n\n"
    
    def add_rule(self, rule: str, priority: int = 0):
        """Add a behavioral rule"""
        self.rules.append({"rule": rule, "priority": priority})
    
    def add_example(self, user_input: str, expected_output: str):
        """Add a few-shot example"""
        self.examples.append({
            "input": user_input,
            "output": expected_output
        })
    
    def set_output_schema(self, schema: dict):
        """Set expected output schema"""
        self.output_schema = schema
    
    def build(self) -> str:
        """Build the complete system prompt"""
        prompt = self.base_prompt
        
        # Add rules (sorted by priority)
        if self.rules:
            prompt += "RULES:\n"
            sorted_rules = sorted(self.rules, key=lambda x: x["priority"], reverse=True)
            for i, rule_item in enumerate(sorted_rules, 1):
                prompt += f"{i}. {rule_item['rule']}\n"
            prompt += "\n"
        
        # Add examples
        if self.examples:
            prompt += "EXAMPLES:\n"
            for i, example in enumerate(self.examples, 1):
                prompt += f"\nExample {i}:\n"
                prompt += f"User: {example['input']}\n"
                prompt += f"Agent: {example['output']}\n"
            prompt += "\n"
        
        # Add output schema
        if self.output_schema:
            prompt += "OUTPUT SCHEMA:\n"
            prompt += f"{self.output_schema}\n"
        
        return prompt

# Usage example
prompt_builder = DynamicSystemPrompt()
prompt_builder.set_base_role(
    "TaskExecutor",
    "An agent that executes tasks by breaking them into steps and using tools."
)
prompt_builder.add_rule("Always verify tool results before proceeding", priority=10)
prompt_builder.add_rule("If a tool fails, try alternative approaches", priority=8)
prompt_builder.add_rule("Report progress after each major step", priority=5)

prompt_builder.add_example(
    "Check the weather",
    "I'll check the weather for you. [Calls get_weather tool] The weather is sunny, 72°F."
)

system_prompt = prompt_builder.build()
print(system_prompt)
```

### Few-Shot Prompting with Examples

Few-shot prompting provides examples to guide the model's behavior. This is especially effective for agents that need to follow specific patterns.

#### Basic Few-Shot Prompting

```python
FEW_SHOT_EXAMPLES = """
Example 1:
User: What is the capital of France?
Agent: The capital of France is Paris.

Example 2:
User: What is the capital of Germany?
Agent: The capital of Germany is Berlin.

Example 3:
User: What is the capital of Italy?
Agent: The capital of Italy is Rome.
"""

def create_few_shot_prompt(examples: str, user_query: str) -> str:
    """Create a few-shot prompt"""
    return f"""{examples}

Now answer this question following the same format:
User: {user_query}
Agent:"""
```

#### Few-Shot Prompting for Tool Use

```python
TOOL_USE_FEW_SHOT = """
Example 1:
User: Get the weather in New York
Agent: I'll check the weather for New York.
[Function Call: get_weather(location="New York")]
[Function Result: Sunny, 72°F]
Agent: The weather in New York is sunny and 72°F.

Example 2:
User: Calculate 15 * 23
Agent: I'll calculate that for you.
[Function Call: calculate(expression="15 * 23")]
[Function Result: 345]
Agent: 15 multiplied by 23 equals 345.

Example 3:
User: Search for Python tutorials
Agent: I'll search for Python tutorials.
[Function Call: search_web(query="Python tutorials")]
[Function Result: Found 10 results...]
Agent: I found several Python tutorials. Here are the top results: ...
"""

class FewShotToolAgent:
    """Agent that uses few-shot examples for tool calling"""
    
    def __init__(self, examples: str):
        self.examples = examples
        self.client = OpenAI()
    
    def process_request(self, user_query: str) -> str:
        """Process user request with few-shot examples"""
        prompt = f"""{self.examples}

Now handle this request following the same pattern:
User: {user_query}
Agent:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
```

#### Dynamic Few-Shot Example Selection

```python
class AdaptiveFewShot:
    """Select few-shot examples based on query similarity"""
    
    def __init__(self):
        self.example_bank = {
            "weather": [
                {
                    "input": "What's the weather?",
                    "output": "I'll check the weather. [Calls get_weather] It's sunny, 72°F."
                }
            ],
            "calculation": [
                {
                    "input": "Calculate 10 + 5",
                    "output": "I'll calculate that. [Calls calculate] The result is 15."
                }
            ],
            "search": [
                {
                    "input": "Find information about AI",
                    "output": "I'll search for that. [Calls search] Found relevant information..."
                }
            ]
        }
    
    def select_examples(self, query: str, n: int = 3) -> list:
        """Select most relevant examples for query"""
        # Simple keyword matching (could use embeddings for better matching)
        query_lower = query.lower()
        
        scored_examples = []
        for category, examples in self.example_bank.items():
            if category in query_lower:
                scored_examples.extend(examples)
        
        # Return top n examples
        return scored_examples[:n] if scored_examples else list(self.example_bank.values())[0][:n]
    
    def build_prompt(self, query: str) -> str:
        """Build prompt with selected examples"""
        examples = self.select_examples(query)
        
        prompt = "Here are some examples:\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"User: {ex['input']}\n"
            prompt += f"Agent: {ex['output']}\n\n"
        
        prompt += f"Now handle this request:\nUser: {query}\nAgent:"
        return prompt
```

### Structured Output Prompting (JSON Mode)

Structured outputs ensure agents return data in predictable formats, essential for programmatic use.

#### JSON Mode with OpenAI

```python
def structured_output_example():
    """Example of structured JSON output"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction agent. Always return JSON."
            },
            {
                "role": "user",
                "content": """Extract information from this text:
                John Doe is 30 years old and works as a Software Engineer at Tech Corp.
                His email is john.doe@techcorp.com and he lives in San Francisco."""
            }
        ],
        response_format={"type": "json_object"},  # Enable JSON mode
        temperature=0
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    return result

# Expected output:
# {
#     "name": "John Doe",
#     "age": 30,
#     "job": "Software Engineer",
#     "company": "Tech Corp",
#     "email": "john.doe@techcorp.com",
#     "location": "San Francisco"
# }
```

#### JSON Schema Prompting

```python
JSON_SCHEMA_PROMPT = """
Extract information and return it as JSON matching this schema:
{
    "person": {
        "name": "string",
        "age": "number",
        "email": "string"
    },
    "company": {
        "name": "string",
        "role": "string"
    },
    "location": "string"
}

Text to extract from: {text}
"""

def extract_with_schema(text: str) -> dict:
    """Extract structured data using schema prompt"""
    prompt = JSON_SCHEMA_PROMPT.format(text=text)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    import json
    return json.loads(response.choices[0].message.content)
```

#### Multi-Step Structured Extraction

```python
class StructuredExtractionAgent:
    """Agent that performs structured data extraction"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def extract(self, text: str, schema: dict) -> dict:
        """Extract data matching schema"""
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""Extract information from the following text and return it as JSON matching this exact schema:

{schema_str}

Text:
{text}

Return only valid JSON matching the schema."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        return json.loads(response.choices[0].message.content)
    
    def extract_multiple(self, texts: list, schema: dict) -> list:
        """Extract from multiple texts"""
        results = []
        for text in texts:
            result = self.extract(text, schema)
            results.append(result)
        return results

# Usage
agent = StructuredExtractionAgent()
schema = {
    "summary": "string",
    "key_points": ["string"],
    "sentiment": "string"
}

result = agent.extract(
    "The product launch was successful. Sales increased by 20%. Customers loved the new features.",
    schema
)
```

### Chain-of-Thought Prompting Techniques

Chain-of-thought (CoT) prompting helps agents reason through problems step by step.

#### Basic Chain-of-Thought

```python
COT_PROMPT = """
Let's solve this problem step by step.

Problem: {problem}

Think through this step by step:
1. First, understand what is being asked
2. Identify the key information
3. Break down the problem into smaller parts
4. Solve each part
5. Combine the solutions
6. Verify the answer

Solution:"""

def solve_with_cot(problem: str) -> str:
    """Solve problem using chain-of-thought"""
    prompt = COT_PROMPT.format(problem=problem)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content
```

#### Self-Consistency Chain-of-Thought

```python
class SelfConsistentCoT:
    """Use multiple reasoning paths and select best answer"""
    
    def __init__(self, n_paths: int = 5):
        self.n_paths = n_paths
        self.client = OpenAI()
    
    def solve(self, problem: str) -> dict:
        """Solve with multiple reasoning paths"""
        prompt = f"""Solve this problem step by step. Show your reasoning.

Problem: {problem}

Solution:"""
        
        solutions = []
        for _ in range(self.n_paths):
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Higher temperature for diversity
                max_tokens=500
            )
            solutions.append(response.choices[0].message.content)
        
        # Select most common answer (simplified)
        return {
            "solutions": solutions,
            "selected": solutions[0]  # In practice, would analyze and select best
        }
```

#### Tree-of-Thought for Agents

```python
class TreeOfThoughtAgent:
    """Agent that explores multiple reasoning paths"""
    
    def __init__(self):
        self.client = OpenAI()
        self.exploration_depth = 3
    
    def explore_thoughts(self, problem: str, current_thoughts: list = None) -> list:
        """Explore multiple reasoning paths"""
        if current_thoughts is None:
            current_thoughts = [""]
        
        if len(current_thoughts[0].split("\n")) > self.exploration_depth:
            return current_thoughts
        
        new_thoughts = []
        for thought in current_thoughts:
            prompt = f"""Problem: {problem}

Current reasoning:
{thought}

Generate 3 different ways to continue reasoning from this point. Number each approach."""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=300
            )
            
            # Parse multiple approaches (simplified)
            approaches = response.choices[0].message.content.split("\n\n")
            for approach in approaches[:3]:
                new_thoughts.append(thought + "\n" + approach)
        
        return self.explore_thoughts(problem, new_thoughts)
```

### Persona-Based Prompting

Personas help agents maintain consistent behavior and style.

#### Persona Definition

```python
PERSONAS = {
    "technical_expert": {
        "role": "Senior Technical Expert",
        "traits": [
            "Precise and detailed",
            "Uses technical terminology",
            "Provides code examples",
            "Explains trade-offs"
        ],
        "tone": "Professional, technical",
        "style": "Structured, with examples"
    },
    "friendly_assistant": {
        "role": "Friendly Assistant",
        "traits": [
            "Warm and approachable",
            "Uses simple language",
            "Encouraging and supportive",
            "Asks clarifying questions"
        ],
        "tone": "Casual, friendly",
        "style": "Conversational"
    },
    "analytical_consultant": {
        "role": "Analytical Consultant",
        "traits": [
            "Data-driven",
            "Provides pros/cons",
            "Uses frameworks",
            "Asks strategic questions"
        ],
        "tone": "Professional, analytical",
        "style": "Structured analysis"
    }
}

class PersonaAgent:
    """Agent with configurable persona"""
    
    def __init__(self, persona_name: str):
        if persona_name not in PERSONAS:
            raise ValueError(f"Unknown persona: {persona_name}")
        
        self.persona = PERSONAS[persona_name]
        self.client = OpenAI()
        self.system_prompt = self._build_persona_prompt()
    
    def _build_persona_prompt(self) -> str:
        """Build system prompt from persona"""
        prompt = f"""You are {self.persona['role']}.

Your characteristics:
"""
        for trait in self.persona['traits']:
            prompt += f"- {trait}\n"
        
        prompt += f"\nTone: {self.persona['tone']}\n"
        prompt += f"Style: {self.persona['style']}\n"
        
        return prompt
    
    def respond(self, user_message: str) -> str:
        """Respond using persona"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

# Usage
technical_agent = PersonaAgent("technical_expert")
friendly_agent = PersonaAgent("friendly_assistant")

response1 = technical_agent.respond("How does a database index work?")
response2 = friendly_agent.respond("How does a database index work?")
```

### Prompt Templates and Variables

Templates make prompts reusable and maintainable.

#### Template System

```python
from string import Template

class PromptTemplate:
    """Reusable prompt templates"""
    
    def __init__(self, template_str: str):
        self.template = Template(template_str)
    
    def render(self, **kwargs) -> str:
        """Render template with variables"""
        return self.template.substitute(**kwargs)
    
    def safe_render(self, **kwargs) -> str:
        """Render template with safe substitution (handles missing vars)"""
        return self.template.safe_substitute(**kwargs)

# Example templates
ANALYSIS_TEMPLATE = PromptTemplate("""
Analyze the following ${data_type}:

${data}

Provide:
1. Summary
2. Key insights
3. Recommendations

Format: ${output_format}
""")

EXTRACTION_TEMPLATE = PromptTemplate("""
Extract ${fields} from this text:

${text}

Return as ${format}.
""")

# Usage
analysis_prompt = ANALYSIS_TEMPLATE.render(
    data_type="sales data",
    data="Sales increased 20% this quarter",
    output_format="bullet points"
)

extraction_prompt = EXTRACTION_TEMPLATE.render(
    fields="name, email, phone",
    text="Contact John at john@example.com or 555-1234",
    format="JSON"
)
```

#### Nested Templates

```python
class TemplateLibrary:
    """Library of reusable prompt templates"""
    
    def __init__(self):
        self.templates = {
            "analysis": """
Analyze ${data_type}:
${data}

Provide: ${sections}
""",
            "extraction": """
Extract ${fields} from:
${text}

Format: ${format}
""",
            "summarization": """
Summarize this ${content_type}:
${content}

Length: ${length}
Style: ${style}
"""
        }
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a template by name"""
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}")
        return PromptTemplate(self.templates[name])
    
    def compose(self, template_names: list, **kwargs) -> str:
        """Compose multiple templates"""
        result = ""
        for name in template_names:
            template = self.get_template(name)
            result += template.render(**kwargs) + "\n\n"
        return result

# Usage
library = TemplateLibrary()
composed = library.compose(
    ["analysis", "extraction"],
    data_type="financial report",
    data="Revenue: $1M, Expenses: $800K",
    sections="summary, insights",
    fields="revenue, expenses",
    text="Revenue: $1M, Expenses: $800K",
    format="JSON"
)
```

### Dynamic Prompt Construction

Dynamic prompts adapt based on context, user history, or system state.

#### Context-Aware Prompt Building

```python
class DynamicPromptBuilder:
    """Build prompts dynamically based on context"""
    
    def __init__(self):
        self.context = {}
        self.history = []
    
    def set_context(self, key: str, value: any):
        """Set context variable"""
        self.context[key] = value
    
    def add_history(self, role: str, content: str):
        """Add to conversation history"""
        self.history.append({"role": role, "content": content})
    
    def build_prompt(self, base_template: str, **override_vars) -> str:
        """Build prompt with context and overrides"""
        # Merge context with overrides
        vars = {**self.context, **override_vars}
        
        # Add history context if available
        if self.history:
            recent_history = "\n".join([
                f"{h['role']}: {h['content']}" 
                for h in self.history[-3:]  # Last 3 messages
            ])
            vars['recent_history'] = recent_history
        
        template = Template(base_template)
        return template.safe_substitute(**vars)

# Usage
builder = DynamicPromptBuilder()
builder.set_context("user_name", "Alice")
builder.set_context("user_role", "developer")
builder.add_history("user", "I need help with Python")
builder.add_history("assistant", "I can help with Python!")

template = """
Hello ${user_name}, you are a ${user_role}.

Recent conversation:
${recent_history}

How can I help you today?
"""

prompt = builder.build_prompt(template)
```

#### Adaptive Prompt Selection

```python
class AdaptivePromptSelector:
    """Select prompts based on query characteristics"""
    
    def __init__(self):
        self.prompt_templates = {
            "simple": "Answer this question: ${query}",
            "complex": """
Break down this complex question step by step:
${query}

Provide a detailed analysis.
""",
            "technical": """
As a technical expert, answer:
${query}

Include code examples if relevant.
""",
            "creative": """
Think creatively about:
${query}

Provide innovative ideas.
"""
        }
    
    def select_template(self, query: str) -> str:
        """Select template based on query"""
        query_lower = query.lower()
        
        # Simple heuristics (could use ML for better selection)
        if len(query.split()) < 5:
            return "simple"
        elif any(word in query_lower for word in ["how", "why", "explain", "analyze"]):
            return "complex"
        elif any(word in query_lower for word in ["code", "function", "algorithm", "api"]):
            return "technical"
        elif any(word in query_lower for word in ["idea", "creative", "design", "innovative"]):
            return "creative"
        else:
            return "simple"
    
    def build_prompt(self, query: str) -> str:
        """Build prompt using selected template"""
        template_name = self.select_template(query)
        template = Template(self.prompt_templates[template_name])
        return template.substitute(query=query)
```

### Prompt Versioning and Management

Version control for prompts ensures reproducibility and allows A/B testing.

#### Prompt Version Manager

```python
import json
from datetime import datetime
from typing import Dict, List

class PromptVersionManager:
    """Manage versions of prompts"""
    
    def __init__(self):
        self.versions: Dict[str, List[Dict]] = {}
    
    def save_version(
        self,
        prompt_name: str,
        prompt_content: str,
        metadata: dict = None
    ) -> str:
        """Save a new version of a prompt"""
        if prompt_name not in self.versions:
            self.versions[prompt_name] = []
        
        version = {
            "version": len(self.versions[prompt_name]) + 1,
            "content": prompt_content,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.versions[prompt_name].append(version)
        return version["version"]
    
    def get_version(self, prompt_name: str, version: int = None) -> dict:
        """Get a specific version of a prompt"""
        if prompt_name not in self.versions:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        if version is None:
            # Return latest
            return self.versions[prompt_name][-1]
        
        # Return specific version
        for v in self.versions[prompt_name]:
            if v["version"] == version:
                return v
        
        raise ValueError(f"Version {version} not found for {prompt_name}")
    
    def list_versions(self, prompt_name: str) -> List[dict]:
        """List all versions of a prompt"""
        if prompt_name not in self.versions:
            return []
        return self.versions[prompt_name]
    
    def compare_versions(self, prompt_name: str, v1: int, v2: int) -> dict:
        """Compare two versions"""
        version1 = self.get_version(prompt_name, v1)
        version2 = self.get_version(prompt_name, v2)
        
        return {
            "version1": version1["content"],
            "version2": version2["content"],
            "diff_length": len(version2["content"]) - len(version1["content"]),
            "created_at_diff": (
                datetime.fromisoformat(version2["created_at"]) - 
                datetime.fromisoformat(version1["created_at"])
            ).total_seconds()
        }
    
    def export(self, filepath: str):
        """Export all versions to file"""
        with open(filepath, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def import_versions(self, filepath: str):
        """Import versions from file"""
        with open(filepath, 'r') as f:
            self.versions = json.load(f)

# Usage
manager = PromptVersionManager()

# Save versions
v1 = manager.save_version(
    "analysis_prompt",
    "Analyze this data: ${data}",
    {"author": "Alice", "purpose": "initial"}
)

v2 = manager.save_version(
    "analysis_prompt",
    "Analyze this data: ${data}\n\nProvide insights and recommendations.",
    {"author": "Bob", "purpose": "enhanced"}
)

# Retrieve
latest = manager.get_version("analysis_prompt")
specific = manager.get_version("analysis_prompt", version=1)

# Compare
diff = manager.compare_versions("analysis_prompt", 1, 2)
```

#### A/B Testing Framework for Prompts

```python
import random
from collections import defaultdict

class PromptABTester:
    """A/B test different prompt versions"""
    
    def __init__(self):
        self.variants = {}
        self.results = defaultdict(list)
    
    def register_variant(self, name: str, prompt: str, weight: float = 1.0):
        """Register a prompt variant"""
        self.variants[name] = {
            "prompt": prompt,
            "weight": weight,
            "uses": 0,
            "successes": 0
        }
    
    def select_variant(self) -> tuple:
        """Select a variant based on weights"""
        total_weight = sum(v["weight"] for v in self.variants.values())
        rand = random.uniform(0, total_weight)
        
        current = 0
        for name, variant in self.variants.items():
            current += variant["weight"]
            if rand <= current:
                variant["uses"] += 1
                return name, variant["prompt"]
    
    def record_result(self, variant_name: str, success: bool, metrics: dict = None):
        """Record test result"""
        if variant_name not in self.variants:
            raise ValueError(f"Unknown variant: {variant_name}")
        
        self.variants[variant_name]["successes"] += (1 if success else 0)
        self.results[variant_name].append({
            "success": success,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def get_statistics(self) -> dict:
        """Get A/B test statistics"""
        stats = {}
        for name, variant in self.variants.items():
            success_rate = (
                variant["successes"] / variant["uses"] 
                if variant["uses"] > 0 else 0
            )
            stats[name] = {
                "uses": variant["uses"],
                "successes": variant["successes"],
                "success_rate": success_rate
            }
        return stats

# Usage
tester = PromptABTester()
tester.register_variant("A", "Answer: ${query}", weight=1.0)
tester.register_variant("B", "Please provide a detailed answer: ${query}", weight=1.0)

# Simulate testing
for _ in range(100):
    variant_name, prompt = tester.select_variant()
    # Use prompt, measure success
    success = random.random() > 0.3  # Simulated
    tester.record_result(variant_name, success)

stats = tester.get_statistics()
print(stats)
```

---

## Function Calling / Tool Use

Function calling (also called tool use) allows LLMs to interact with external systems, APIs, and tools. This is fundamental to building agents that can take actions beyond text generation.

### OpenAI Function Calling (Complete API Examples)

OpenAI's function calling API allows models to request function execution with structured parameters.

#### Basic Function Calling Setup

```python
from openai import OpenAI
import json

client = OpenAI()

# Define available functions
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Implement the functions
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Mock weather function"""
    return f"Weather in {location}: Sunny, 72°{unit[0].upper()}"

def calculate(expression: str) -> float:
    """Mock calculator function"""
    try:
        return eval(expression)
    except:
        return "Error: Invalid expression"

# Function mapping
available_functions = {
    "get_weather": get_weather,
    "calculate": calculate
}

def run_agent_with_functions(user_message: str) -> str:
    """Run agent with function calling"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use functions when appropriate."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    
    # First API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=functions,
        tool_choice="auto"  # Let model decide
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    # Check if function was called
    if message.tool_calls:
        # Execute functions
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the function
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            # Add function result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_response)
            })
        
        # Second API call with function results
        second_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return second_response.choices[0].message.content
    
    return message.content

# Usage
result = run_agent_with_functions("What's the weather in New York?")
print(result)
```

#### Parallel Function Calling

OpenAI supports calling multiple functions in parallel, improving efficiency.

```python
def run_agent_parallel_functions(user_message: str) -> str:
    """Run agent with parallel function calling"""
    messages = [{"role": "user", "content": user_message}]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=functions,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    if message.tool_calls:
        # Execute all function calls in parallel
        import concurrent.futures
        
        def execute_tool_call(tool_call):
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_to_call = available_functions[function_name]
            result = function_to_call(**function_args)
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(result)
            }
        
        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tool_responses = list(executor.map(
                execute_tool_call, 
                message.tool_calls
            ))
        
        messages.extend(tool_responses)
        
        # Get final response
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    return message.content

# Usage
result = run_agent_parallel_functions(
    "What's the weather in New York and calculate 15 * 23"
)
print(result)
```

#### Function Calling with Error Handling

```python
class RobustFunctionAgent:
    """Agent with robust function calling and error handling"""
    
    def __init__(self):
        self.client = OpenAI()
        self.functions = functions
        self.available_functions = available_functions
        self.max_iterations = 5
    
    def execute_function(self, function_name: str, arguments: dict) -> dict:
        """Execute function with error handling"""
        try:
            if function_name not in self.available_functions:
                return {
                    "success": False,
                    "error": f"Unknown function: {function_name}"
                }
            
            function_to_call = self.available_functions[function_name]
            result = function_to_call(**arguments)
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run(self, user_message: str) -> str:
        """Run agent with error handling"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Handle errors gracefully."
            },
            {"role": "user", "content": user_message}
        ]
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=self.functions,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                messages.append(message)
                
                if not message.tool_calls:
                    return message.content
                
                # Execute function calls
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    execution_result = self.execute_function(
                        function_name, 
                        function_args
                    )
                    
                    if execution_result["success"]:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": str(execution_result["result"])
                        })
                    else:
                        # Report error to model
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": f"Error: {execution_result['error']}"
                        })
                
            except Exception as e:
                messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}. Let me try a different approach."
                })
        
        return "I've reached the maximum number of iterations. Please try rephrasing your request."

# Usage
agent = RobustFunctionAgent()
result = agent.run("Get weather for invalid location and calculate")
print(result)
```

### Anthropic Tool Use (Complete API Examples)

Anthropic's Claude models use a different tool use format with structured tool definitions.

#### Basic Anthropic Tool Use

```python
from anthropic import Anthropic

anthropic_client = Anthropic()

# Define tools for Anthropic
anthropic_tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "fahrenheit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression"
                }
            },
            "required": ["expression"]
        }
    }
]

def run_anthropic_agent(user_message: str) -> str:
    """Run Anthropic agent with tool use"""
    message = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        tools=anthropic_tools,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    # Check if tool use was requested
    if message.stop_reason == "tool_use":
        # Execute tools and create follow-up message
        tool_results = []
        
        for content_block in message.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                
                # Execute function
                function_to_call = available_functions[tool_name]
                result = function_to_call(**tool_input)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": str(result)
                })
        
        # Send tool results back
        follow_up = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_message},
                message,  # Include original message with tool_use
                {"role": "user", "content": tool_results}
            ]
        )
        
        return follow_up.content[0].text
    
    return message.content[0].text

# Usage
result = run_anthropic_agent("What's the weather in San Francisco?")
print(result)
```

#### Multi-Turn Tool Use with Anthropic

```python
class AnthropicToolAgent:
    """Multi-turn agent using Anthropic's tool use"""
    
    def __init__(self):
        self.client = Anthropic()
        self.tools = anthropic_tools
        self.available_functions = available_functions
        self.conversation_history = []
    
    def add_message(self, role: str, content: any):
        """Add message to conversation"""
        self.conversation_history.append({"role": role, "content": content})
    
    def run_turn(self, user_message: str) -> str:
        """Run a single turn of conversation"""
        self.add_message("user", user_message)
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                tools=self.tools,
                messages=self.conversation_history
            )
            
            self.add_message("assistant", response.content)
            
            # Check if tool use is needed
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name = content_block.name
                    tool_input = content_block.input
                    
                    # Execute function
                    function_to_call = self.available_functions[tool_name]
                    result = function_to_call(**tool_input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": str(result)
                    })
            
            if not tool_results:
                # No more tool calls needed
                return response.content[0].text
            
            # Add tool results
            self.add_message("user", tool_results)
        
        return "Maximum iterations reached"

# Usage
agent = AnthropicToolAgent()
result = agent.run_turn("Get weather for NYC and calculate 10 * 5")
print(result)
```

### Google Gemini Function Calling

Gemini uses a similar approach to OpenAI with function definitions.

#### Basic Gemini Function Calling

```python
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="YOUR_API_KEY")

# Define tools
gemini_tools = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    }
]

def run_gemini_agent(user_message: str) -> str:
    """Run Gemini agent with function calling"""
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        tools=gemini_tools
    )
    
    chat = model.start_chat()
    response = chat.send_message(user_message)
    
    # Check for function calls
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        function_name = function_call.name
        function_args = dict(function_call.args)
        
        # Execute function
        function_to_call = available_functions[function_name]
        result = function_to_call(**function_args)
        
        # Send result back
        follow_up = chat.send_message({
            "function_response": {
                "name": function_name,
                "response": {"result": result}
            }
        })
        
        return follow_up.text
    
    return response.text

# Usage
result = run_gemini_agent("What's the weather in Boston?")
print(result)
```

### Function Schema Definition (JSON Schema)

Proper schema definition is crucial for reliable function calling.

#### Comprehensive Schema Examples

```python
# Weather function with detailed schema
WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather conditions for a specified location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, optionally with state/country",
                    "examples": ["New York", "San Francisco, CA", "London, UK"]
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                    "description": "Temperature unit",
                    "default": "fahrenheit"
                },
                "include_forecast": {
                    "type": "boolean",
                    "description": "Whether to include 5-day forecast",
                    "default": False
                }
            },
            "required": ["location"],
            "additionalProperties": False
        }
    }
}

# Database query function schema
DATABASE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a SQL query on the database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query (read-only)",
                    "pattern": "^SELECT .*"
                },
                "table": {
                    "type": "string",
                    "description": "Table name to query",
                    "enum": ["users", "orders", "products"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                }
            },
            "required": ["query", "table"]
        }
    }
}

# Email sending function schema
EMAIL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to recipients",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "email"
                    },
                    "description": "Recipient email addresses",
                    "minItems": 1,
                    "maxItems": 10
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                    "maxLength": 200
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "default": "normal"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
}
```

#### Schema Validation

```python
from jsonschema import validate, ValidationError

class SchemaValidator:
    """Validate function call arguments against schema"""
    
    def __init__(self):
        self.schemas = {
            "get_weather": WEATHER_SCHEMA["function"]["parameters"],
            "query_database": DATABASE_SCHEMA["function"]["parameters"],
            "send_email": EMAIL_SCHEMA["function"]["parameters"]
        }
    
    def validate_call(self, function_name: str, arguments: dict) -> dict:
        """Validate function call arguments"""
        if function_name not in self.schemas:
            return {
                "valid": False,
                "error": f"Unknown function: {function_name}"
            }
        
        try:
            validate(instance=arguments, schema=self.schemas[function_name])
            return {"valid": True}
        except ValidationError as e:
            return {
                "valid": False,
                "error": str(e),
                "path": list(e.path)
            }

# Usage
validator = SchemaValidator()
result = validator.validate_call(
    "get_weather",
    {"location": "New York", "unit": "celsius"}
)
print(result)
```

### Nested Function Calls

Agents can call functions that themselves trigger other function calls.

#### Nested Function Execution

```python
class NestedFunctionAgent:
    """Agent that handles nested function calls"""
    
    def __init__(self):
        self.client = OpenAI()
        self.call_stack = []
        self.max_depth = 3
    
    def execute_with_nesting(self, function_name: str, arguments: dict, depth: int = 0) -> any:
        """Execute function, handling nested calls"""
        if depth > self.max_depth:
            return {"error": "Maximum nesting depth exceeded"}
        
        self.call_stack.append({
            "function": function_name,
            "arguments": arguments,
            "depth": depth
        })
        
        # Execute function
        function_to_call = available_functions[function_name]
        result = function_to_call(**arguments)
        
        # Check if result requires another function call
        if isinstance(result, dict) and result.get("requires_function_call"):
            nested_function = result["function"]
            nested_args = result["arguments"]
            nested_result = self.execute_with_nesting(
                nested_function, 
                nested_args, 
                depth + 1
            )
            return nested_result
        
        return result
    
    def run(self, user_message: str) -> str:
        """Run agent with nested function support"""
        messages = [{"role": "user", "content": user_message}]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=functions
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                result = self.execute_with_nesting(function_name, function_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(result)
                })
            
            final_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        return message.content
```

### Dynamic Tool Registration

Agents can dynamically register and use new tools at runtime.

#### Dynamic Tool Management

```python
class DynamicToolAgent:
    """Agent with dynamic tool registration"""
    
    def __init__(self):
        self.client = OpenAI()
        self.registered_tools = []
        self.tool_functions = {}
    
    def register_tool(self, tool_schema: dict, tool_function: callable):
        """Register a new tool dynamically"""
        self.registered_tools.append(tool_schema)
        tool_name = tool_schema["function"]["name"]
        self.tool_functions[tool_name] = tool_function
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool"""
        self.registered_tools = [
            t for t in self.registered_tools 
            if t["function"]["name"] != tool_name
        ]
        if tool_name in self.tool_functions:
            del self.tool_functions[tool_name]
    
    def run(self, user_message: str) -> str:
        """Run agent with current tool set"""
        messages = [{"role": "user", "content": user_message}]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.registered_tools if self.registered_tools else None
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in self.tool_functions:
                    result = self.tool_functions[function_name](**function_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(result)
                    })
            
            final_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            
            return final_response.choices[0].message.content
        
        return message.content

# Usage
agent = DynamicToolAgent()

# Register tools dynamically
agent.register_tool(
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    lambda a, b: a * b
)

result = agent.run("Multiply 5 and 7")
print(result)
```

---

## Structured Outputs

Structured outputs ensure agents return data in predictable, parseable formats.

### JSON Mode (OpenAI, Anthropic)

#### OpenAI JSON Mode

```python
def structured_json_output_openai(query: str, schema: dict) -> dict:
    """Get structured JSON output from OpenAI"""
    
    schema_description = json.dumps(schema, indent=2)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"Always respond with valid JSON matching this schema:\n{schema_description}"
            },
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content)

# Usage
schema = {
    "summary": "string",
    "key_points": ["string"],
    "sentiment": "string"
}

result = structured_json_output_openai(
    "Analyze: The product launch was successful",
    schema
)
```

#### Anthropic Structured Outputs

```python
def structured_output_anthropic(query: str, schema: dict) -> dict:
    """Get structured output from Anthropic"""
    
    response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True
            }
        }
    )
    
    return json.loads(response.content[0].text)

# Usage
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["summary", "key_points"]
}

result = structured_output_anthropic(
    "Summarize the benefits of AI agents",
    schema
)
```

### Pydantic Models for Validation

Pydantic provides runtime validation for structured outputs.

#### Pydantic Integration

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class AnalysisResult(BaseModel):
    """Structured analysis result"""
    summary: str = Field(description="Brief summary")
    key_points: List[str] = Field(description="Key findings")
    sentiment: str = Field(description="Overall sentiment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        allowed = ['positive', 'negative', 'neutral']
        if v.lower() not in allowed:
            raise ValueError(f'Sentiment must be one of {allowed}')
        return v.lower()

class StructuredOutputAgent:
    """Agent using Pydantic for validation"""
    
    def __init__(self, output_model: BaseModel):
        self.client = OpenAI()
        self.output_model = output_model
        self.schema = output_model.schema()
    
    def generate(self, query: str) -> BaseModel:
        """Generate structured output"""
        schema_json = json.dumps(self.schema, indent=2)
        
        prompt = f"""Extract information and return as JSON matching this schema:

{schema_json}

Query: {query}

Return only valid JSON."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result_dict = json.loads(response.choices[0].message.content)
        
        # Validate with Pydantic
        return self.output_model(**result_dict)

# Usage
agent = StructuredOutputAgent(AnalysisResult)
result = agent.generate("Analyze: The new feature received positive feedback")
print(result.summary)
print(result.key_points)
```

### Output Parsers (LangChain Style)

```python
class OutputParser:
    """Parse and validate LLM outputs"""
    
    def __init__(self, expected_format: str = "json"):
        self.expected_format = expected_format
    
    def parse(self, output: str) -> dict:
        """Parse output based on format"""
        if self.expected_format == "json":
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError("No valid JSON found in output")
        elif self.expected_format == "yaml":
            import yaml
            return yaml.safe_load(output)
        else:
            return {"raw": output}
    
    def parse_with_retry(self, output: str, max_retries: int = 3) -> dict:
        """Parse with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.parse(output)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Could request regeneration here
                continue

# Usage
parser = OutputParser("json")
result = parser.parse('{"key": "value"}')
```

### Handling Malformed Outputs

```python
class RobustOutputHandler:
    """Handle malformed outputs gracefully"""
    
    def __init__(self):
        self.client = OpenAI()
        self.retry_prompt = """The previous output was malformed. Please try again with valid JSON."""
    
    def get_structured_output(
        self, 
        query: str, 
        schema: dict, 
        max_retries: int = 3
    ) -> dict:
        """Get structured output with retry on malformed responses"""
        schema_str = json.dumps(schema, indent=2)
        
        messages = [
            {
                "role": "system",
                "content": f"Always return valid JSON matching:\n{schema_str}"
            },
            {"role": "user", "content": query}
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Validate against schema
                validate(instance=result, schema=schema)
                
                return result
                
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt < max_retries - 1:
                    messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    })
                    messages.append({
                        "role": "user",
                        "content": f"{self.retry_prompt} Error: {str(e)}"
                    })
                else:
                    raise ValueError(f"Failed to get valid output after {max_retries} attempts")

# Usage
handler = RobustOutputHandler()
result = handler.get_structured_output(
    "Extract: Name: John, Age: 30",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
)
```

---

## Token Management

Effective token management is critical for cost control and context window optimization.

### Token Counting and Budgeting

```python
import tiktoken

class TokenBudgetManager:
    """Manage token budgets for agent operations"""
    
    def __init__(self, model: str = "gpt-4", budget: int = 8000):
        self.model = model
        self.budget = budget
        self.encoding = tiktoken.encoding_for_model(model)
        self.used_tokens = 0
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: list) -> int:
        """Count tokens in message list"""
        total = 0
        for message in messages:
            # Approximate: role + content + overhead
            content = f"{message['role']}: {message['content']}"
            total += self.count_tokens(content)
        # Add overhead for message formatting
        total += len(messages) * 4
        return total
    
    def check_budget(self, estimated_tokens: int) -> bool:
        """Check if operation fits within budget"""
        return (self.used_tokens + estimated_tokens) <= self.budget
    
    def reserve_tokens(self, tokens: int):
        """Reserve tokens for an operation"""
        if not self.check_budget(tokens):
            raise ValueError(f"Insufficient token budget. Need {tokens}, have {self.budget - self.used_tokens}")
        self.used_tokens += tokens
    
    def get_remaining(self) -> int:
        """Get remaining token budget"""
        return self.budget - self.used_tokens

# Usage
manager = TokenBudgetManager(budget=8000)
messages = [
    {"role": "user", "content": "Hello, how are you?"}
]
token_count = manager.count_messages_tokens(messages)
print(f"Tokens used: {token_count}, Remaining: {manager.get_remaining()}")
```

### Context Window Optimization

```python
class ContextOptimizer:
    """Optimize context window usage"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def summarize_old_messages(self, messages: list, keep_recent: int = 5) -> list:
        """Summarize old messages, keep recent ones"""
        if len(messages) <= keep_recent:
            return messages
        
        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]
        
        # Create summary of old messages
        summary_prompt = f"""Summarize this conversation history:

{self._format_messages(old_messages)}

Provide a concise summary preserving key information."""
        
        # In practice, would call LLM here
        summary = "Previous conversation summarized..."
        
        return [
            {"role": "system", "content": f"Previous conversation: {summary}"}
        ] + recent_messages
    
    def _format_messages(self, messages: list) -> str:
        """Format messages for summarization"""
        return "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in messages
        ])
    
    def trim_to_fit(self, messages: list, reserve_output: int = 1000) -> list:
        """Trim messages to fit in context window"""
        max_input = self.max_tokens - reserve_output
        
        while self._count_tokens(messages) > max_input:
            if len(messages) <= 2:
                break
            # Remove oldest non-system message
            for i, msg in enumerate(messages):
                if msg['role'] != 'system':
                    messages.pop(i)
                    break
        
        return messages
    
    def _count_tokens(self, messages: list) -> int:
        """Count tokens in messages"""
        total = 0
        for msg in messages:
            total += len(self.encoding.encode(str(msg)))
        return total

# Usage
optimizer = ContextOptimizer(max_tokens=8000)
messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Response 1"},
    # ... many more messages
]
optimized = optimizer.trim_to_fit(messages)
```

### Truncation Strategies

```python
class SmartTruncator:
    """Intelligent text truncation"""
    
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def truncate_by_tokens(self, text: str, max_tokens: int, method: str = "end") -> str:
        """Truncate text to fit token limit"""
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        if method == "end":
            # Truncate from end
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        
        elif method == "start":
            # Truncate from start
            truncated_tokens = tokens[-max_tokens:]
            return self.encoding.decode(truncated_tokens)
        
        elif method == "middle":
            # Keep start and end, remove middle
            start_tokens = max_tokens // 2
            end_tokens = max_tokens - start_tokens
            truncated = tokens[:start_tokens] + tokens[-end_tokens:]
            return self.encoding.decode(truncated)
        
        elif method == "sentences":
            # Truncate by sentences, keeping complete sentences
            sentences = text.split('. ')
            result = []
            token_count = 0
            
            for sentence in sentences:
                sentence_tokens = len(self.encoding.encode(sentence))
                if token_count + sentence_tokens <= max_tokens:
                    result.append(sentence)
                    token_count += sentence_tokens
                else:
                    break
            
            return '. '.join(result) + '.'
        
        return text

# Usage
truncator = SmartTruncator()
long_text = "This is a very long text. " * 100
truncated = truncator.truncate_by_tokens(long_text, max_tokens=50, method="sentences")
```

### Sliding Window Approaches

```python
class SlidingWindowContext:
    """Maintain context using sliding window"""
    
    def __init__(self, window_size: int = 4000, overlap: int = 500):
        self.window_size = window_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.messages = []
    
    def add_message(self, role: str, content: str):
        """Add message to context"""
        self.messages.append({"role": role, "content": content})
    
    def get_active_window(self) -> list:
        """Get current active window of messages"""
        if not self.messages:
            return []
        
        # Always include system message if present
        system_msgs = [m for m in self.messages if m['role'] == 'system']
        other_msgs = [m for m in self.messages if m['role'] != 'system']
        
        if not other_msgs:
            return system_msgs
        
        # Calculate window
        current_tokens = 0
        window_messages = system_msgs.copy()
        
        # Add messages from end until window is full
        for msg in reversed(other_msgs):
            msg_tokens = len(self.encoding.encode(str(msg)))
            if current_tokens + msg_tokens <= self.window_size - self.overlap:
                window_messages.insert(len(system_msgs), msg)
                current_tokens += msg_tokens
            else:
                break
        
        return window_messages

# Usage
window = SlidingWindowContext(window_size=4000)
window.add_message("system", "You are helpful")
for i in range(100):
    window.add_message("user", f"Message {i}")
    window.add_message("assistant", f"Response {i}")

active = window.get_active_window()
print(f"Active window has {len(active)} messages")
```

### Cost Optimization

```python
class CostOptimizer:
    """Optimize costs through smart model selection and caching"""
    
    def __init__(self):
        self.pricing = {
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }
        self.cache = {}
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for request"""
        if model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[model]
        return (input_tokens / 1000 * pricing["input"] + 
                output_tokens / 1000 * pricing["output"])
    
    def select_cost_effective_model(
        self, 
        required_capabilities: list,
        input_tokens: int,
        estimated_output_tokens: int
    ) -> str:
        """Select most cost-effective model meeting requirements"""
        candidates = []
        
        # Simple capability matching
        if "complex_reasoning" in required_capabilities:
            candidates = ["gpt-4-turbo"]
        elif "tool_use" in required_capabilities:
            candidates = ["gpt-4-turbo", "gpt-3.5-turbo"]
        else:
            candidates = ["gpt-3.5-turbo", "claude-3-haiku"]
        
        # Select cheapest
        costs = {
            model: self.estimate_cost(model, input_tokens, estimated_output_tokens)
            for model in candidates
        }
        
        return min(costs, key=costs.get)
    
    def cache_key(self, messages: list) -> str:
        """Generate cache key from messages"""
        import hashlib
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached(self, messages: list) -> str:
        """Get cached response if available"""
        key = self.cache_key(messages)
        return self.cache.get(key)
    
    def cache_response(self, messages: list, response: str):
        """Cache response"""
        key = self.cache_key(messages)
        self.cache[key] = response

# Usage
optimizer = CostOptimizer()
model = optimizer.select_cost_effective_model(
    ["tool_use"],
    input_tokens=1000,
    estimated_output_tokens=500
)
print(f"Selected model: {model}")
```

---

## Advanced Techniques

### Multi-Turn Conversations

```python
class MultiTurnAgent:
    """Agent managing multi-turn conversations"""
    
    def __init__(self):
        self.client = OpenAI()
        self.conversation_history = []
        self.max_history = 20
    
    def add_system_message(self, content: str):
        """Add system message"""
        self.conversation_history = [
            {"role": "system", "content": content}
        ] + [m for m in self.conversation_history if m['role'] != 'system']
    
    def chat(self, user_message: str) -> str:
        """Process user message in conversation"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            # Keep system message and recent messages
            system_msg = [m for m in self.conversation_history if m['role'] == 'system']
            recent_msgs = self.conversation_history[-self.max_history+1:]
            self.conversation_history = system_msg + recent_msgs
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_history
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset(self):
        """Reset conversation"""
        self.conversation_history = [
            m for m in self.conversation_history 
            if m['role'] == 'system'
        ]

# Usage
agent = MultiTurnAgent()
agent.add_system_message("You are a helpful assistant.")
response1 = agent.chat("Hello!")
response2 = agent.chat("What did I just say?")
```

### Streaming Responses

```python
def stream_agent_response(user_message: str):
    """Stream agent response token by token"""
    messages = [{"role": "user", "content": user_message}]
    
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Usage
for token in stream_agent_response("Tell me a story"):
    print(token, end='', flush=True)
```

### Retry Strategies with Exponential Backoff

```python
import time
import random

class RetryHandler:
    """Handle retries with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.base_delay * (2 ** attempt)
                    jitter = random.uniform(0, 0.1 * delay)
                    time.sleep(delay + jitter)
                else:
                    raise last_exception
        
        raise last_exception

def api_call_with_retry():
    """Example API call"""
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )

# Usage
handler = RetryHandler(max_retries=3)
try:
    response = handler.execute_with_retry(api_call_with_retry)
except Exception as e:
    print(f"Failed after retries: {e}")
```

### Rate Limiting and Throttling

```python
from collections import deque
import time

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove old calls outside time window
        while self.call_times and self.call_times[0] < now - self.time_window:
            self.call_times.popleft()
        
        # Check if we're at limit
        if len(self.call_times) >= self.max_calls:
            sleep_time = self.time_window - (now - self.call_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                # Clean up again after sleep
                while self.call_times and self.call_times[0] < time.time() - self.time_window:
                    self.call_times.popleft()
        
        # Record this call
        self.call_times.append(time.time())
    
    def __call__(self, func):
        """Decorator for rate limiting"""
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper

# Usage
limiter = RateLimiter(max_calls=60, time_window=60.0)  # 60 calls per minute

@limiter
def make_api_call():
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
```

### Model Fallback Chains

```python
class FallbackChain:
    """Chain of models with fallback"""
    
    def __init__(self, models: list):
        self.models = models
        self.client = OpenAI()
    
    def call_with_fallback(self, messages: list, **kwargs) -> dict:
        """Call models in order until one succeeds"""
        last_error = None
        
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return {
                    "model": model,
                    "response": response
                }
            except Exception as e:
                last_error = e
                continue
        
        raise Exception(f"All models failed. Last error: {last_error}")

# Usage
chain = FallbackChain(["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"])
result = chain.call_with_fallback([
    {"role": "user", "content": "Hello"}
])
print(f"Used model: {result['model']}")
```

### Fine-Tuning Considerations for Agents

```python
class FineTuningPreparer:
    """Prepare data for agent fine-tuning"""
    
    def __init__(self):
        self.training_examples = []
    
    def add_example(
        self,
        user_message: str,
        assistant_message: str,
        tool_calls: list = None
    ):
        """Add training example"""
        example = {
            "messages": [
                {"role": "user", "content": user_message}
            ]
        }
        
        if tool_calls:
            example["messages"].append({
                "role": "assistant",
                "tool_calls": tool_calls
            })
        else:
            example["messages"].append({
                "role": "assistant",
                "content": assistant_message
            })
        
        self.training_examples.append(example)
    
    def export_training_data(self, filepath: str):
        """Export training data in OpenAI format"""
        with open(filepath, 'w') as f:
            for example in self.training_examples:
                f.write(json.dumps(example) + '\n')
    
    def validate_examples(self) -> dict:
        """Validate training examples"""
        issues = []
        
        for i, example in enumerate(self.training_examples):
            if len(example["messages"]) < 2:
                issues.append(f"Example {i}: Too few messages")
            
            if example["messages"][0]["role"] != "user":
                issues.append(f"Example {i}: First message not from user")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_examples": len(self.training_examples)
        }

# Usage
preparer = FineTuningPreparer()
preparer.add_example(
    "What's the weather?",
    "I'll check the weather for you.",
    [{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}'
        }
    }]
)

validation = preparer.validate_examples()
print(validation)
```

---

## Real-World Scenarios

### Scenario 1: E-Commerce Agent

```python
class ECommerceAgent:
    """Agent for e-commerce customer service"""
    
    def __init__(self):
        self.client = OpenAI()
        self.functions = [
            {
                "type": "function",
                "function": {
                    "name": "search_products",
                    "description": "Search for products",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "category": {"type": "string"},
                            "max_price": {"type": "number"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_order_status",
                    "description": "Get order status",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"}
                        },
                        "required": ["order_id"]
                    }
                }
            }
        ]
    
    def handle_customer_query(self, query: str) -> str:
        """Handle customer query"""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful e-commerce assistant. 
                Be friendly, professional, and helpful. Always confirm 
                order details before providing status updates."""
            },
            {"role": "user", "content": query}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.functions,
            temperature=0.7
        )
        
        message = response.choices[0].message
        # Handle tool calls...
        
        return message.content
```

### Scenario 2: Data Analysis Agent

```python
class DataAnalysisAgent:
    """Agent for data analysis tasks"""
    
    def __init__(self):
        self.client = OpenAI()
        self.context_manager = AgentContextManager()
    
    def analyze_dataset(self, data_description: str, questions: list) -> dict:
        """Analyze dataset and answer questions"""
        system_prompt = """You are a data analyst. Analyze data carefully 
        and provide insights. Always cite specific numbers."""
        
        self.context_manager.system_prompt = system_prompt
        
        results = {}
        for question in questions:
            prompt = f"""Data: {data_description}
            
Question: {question}

Provide analysis."""
            
            self.context_manager.add_message("user", prompt)
            messages = self.context_manager.get_messages()
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            self.context_manager.add_message("assistant", answer)
            results[question] = answer
        
        return results
```

---

## Conclusion

This guide covered the fundamentals of LLMs, prompting, and function calling for agentic AI development. Key takeaways:

1. **Model Selection**: Choose models based on context needs, latency requirements, and cost constraints
2. **Prompt Engineering**: Well-structured prompts with clear roles, constraints, and examples improve agent reliability
3. **Function Calling**: Proper schema definition and error handling enable robust tool use
4. **Token Management**: Effective token budgeting and context optimization reduce costs
5. **Advanced Techniques**: Streaming, retries, rate limiting, and fallbacks improve production reliability

Continue experimenting with these techniques and adapt them to your specific use cases.
```
