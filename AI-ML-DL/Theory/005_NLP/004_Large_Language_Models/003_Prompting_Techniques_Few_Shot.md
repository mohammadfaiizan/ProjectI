# Prompting Techniques and Few-Shot Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Zero-Shot Prompting](#zero-shot-prompting)
3. [Few-Shot Prompting](#few-shot-prompting)
4. [Chain-of-Thought Prompting](#chain-of-thought-prompting)
5. [Self-Consistency](#self-consistency)
6. [Tree-of-Thoughts](#tree-of-thoughts)
7. [Prompt Engineering Best Practices](#prompt-engineering-best-practices)
8. [In-Context Learning](#in-context-learning)
9. [Prompt Templates and Patterns](#prompt-templates-and-patterns)
10. [Key Takeaways](#key-takeaways)

## Introduction

Prompting techniques enable large language models to perform tasks without task-specific training by providing instructions and examples in the input. Effective prompting is crucial for leveraging pre-trained models and achieving high performance with minimal fine-tuning.

Prompting approaches:
- **Zero-shot**: No examples, just instructions
- **Few-shot**: Include examples in prompt
- **Chain-of-thought**: Include reasoning steps
- **Advanced**: Self-consistency, tree-of-thoughts

Understanding prompting enables effective use of large language models and reveals their capabilities and limitations.

## Zero-Shot Prompting

Zero-shot prompting uses only task instructions without examples.

### Basic Zero-Shot

**Format**: Task description + input

**Example**:
```
Translate English to French:
Hello, how are you?
```

**Model**: Generates translation based on instruction alone

### Zero-Shot Capabilities

**Strengths**:
- Simple and direct
- No need for examples
- Works for many tasks

**Limitations**:
- May not follow format exactly
- Performance lower than few-shot
- Sensitive to instruction wording

### Instruction Design

**Clear instructions**: Be explicit about desired output
**Format specification**: Specify output format if needed
**Task framing**: Frame task appropriately

**Example**: "Classify the sentiment of the following text as positive, negative, or neutral:"

## Few-Shot Prompting

Few-shot prompting includes examples in the prompt to demonstrate desired behavior.

### Basic Few-Shot

**Format**: Task description + examples + input

**Example**:
```
Translate English to French:
English: Hello → French: Bonjour
English: Goodbye → French: Au revoir
English: Thank you → French: ?
```

**Model**: Learns pattern from examples and applies to new input

### Few-Shot Benefits

**Better performance**: Often outperforms zero-shot
**Format learning**: Learns output format from examples
**Pattern recognition**: Recognizes patterns in examples

### Example Selection

**Diversity**: Include diverse examples
**Relevance**: Examples should be relevant to task
**Quality**: Use high-quality examples
**Number**: Typically 2-10 examples (more doesn't always help)

### Few-Shot Limitations

**Context length**: Limited by model context window
**Example quality**: Poor examples hurt performance
**Overfitting**: May overfit to example patterns

## Chain-of-Thought Prompting

Chain-of-thought (CoT) prompting includes reasoning steps, enabling complex reasoning.

### Basic Chain-of-Thought

**Format**: Include reasoning steps in examples

**Example**:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?

A: Roger started with 5 balls. He bought 2 cans × 3 balls = 6 balls. Total: 5 + 6 = 11 balls.

Q: The cafeteria had 23 apples. They used 20 for lunch and bought 6 more. How many apples do they have?
```

**Model**: Learns to show reasoning steps

### Why Chain-of-Thought Works

**Decomposition**: Breaks complex problems into steps
**Error detection**: Easier to identify errors in reasoning
**Transparency**: Makes reasoning process visible

### CoT Variants

**Manual CoT**: Human-written reasoning steps
**Auto CoT**: Model generates reasoning automatically
**Zero-shot CoT**: Add "Let's think step by step" without examples

## Self-Consistency

Self-consistency generates multiple reasoning paths and selects most consistent answer.

### Method

1. **Generate multiple paths**: Sample multiple reasoning paths
2. **Extract answers**: Extract final answers from each path
3. **Majority voting**: Select most common answer

### Benefits

**Robustness**: Reduces impact of reasoning errors
**Accuracy**: Often improves over single-path reasoning
**Reliability**: More reliable than single generation

### Implementation

**Sampling**: Use temperature > 0 for diversity
**Number of paths**: Typically 5-20 paths
**Voting**: Simple majority or weighted voting

## Tree-of-Thoughts

Tree-of-Thoughts (ToT) explores multiple reasoning paths systematically.

### Concept

**Tree structure**: Explore reasoning tree
**Search**: Systematically search promising paths
**Backtracking**: Abandon unpromising paths

### Process

1. **Generate candidates**: Generate multiple reasoning steps
2. **Evaluate**: Score each candidate
3. **Expand**: Expand promising candidates
4. **Select**: Choose best path

### Advantages

**Systematic exploration**: More thorough than single path
**Better solutions**: Finds better solutions for complex problems
**Controllable**: Can guide search process

### Limitations

**Computational cost**: More expensive than single generation
**Complexity**: More complex to implement
**Evaluation**: Need to evaluate intermediate steps

## Prompt Engineering Best Practices

Effective prompt engineering requires careful design and iteration.

### Clarity

**Be explicit**: Clearly state what you want
**Avoid ambiguity**: Remove ambiguous instructions
**Specify format**: Specify output format if important

### Structure

**Organize information**: Use clear structure (instructions, examples, input)
**Separate components**: Use separators or formatting
**Consistent formatting**: Use consistent format across examples

### Examples

**Quality over quantity**: Few high-quality examples better than many poor ones
**Diversity**: Include diverse examples
**Relevance**: Examples should match task

### Iteration

**Test and refine**: Iterate on prompts
**A/B testing**: Compare different prompt formulations
**Error analysis**: Analyze failures to improve prompts

### Common Patterns

**Role-playing**: "You are an expert translator..."
**Step-by-step**: "First, ... Then, ... Finally, ..."
**Format specification**: "Format your answer as: ..."

## In-Context Learning

In-context learning refers to models' ability to learn from examples in the prompt.

### Mechanism

**Pattern recognition**: Models recognize patterns in examples
**Adaptation**: Adapt behavior based on examples
**No parameter updates**: Learning happens in forward pass

### Why It Works

**Large capacity**: Models have capacity to store patterns
**Attention mechanism**: Attention enables focusing on examples
**Pre-training**: Pre-training on diverse data enables pattern recognition

### Limitations

**Context window**: Limited by model context length
**Example quality**: Sensitive to example quality
**Generalization**: May not generalize beyond examples

## Prompt Templates and Patterns

Standard templates and patterns enable consistent and effective prompting.

### Classification Template

```
Classify the following text into one of these categories: [categories]

Examples:
Text: [example1] → Category: [category1]
Text: [example2] → Category: [category2]

Text: [input] → Category:
```

### Generation Template

```
[Task description]

Examples:
Input: [input1] → Output: [output1]
Input: [input2] → Output: [output2]

Input: [input] → Output:
```

### Reasoning Template

```
[Problem description]

Let's solve this step by step:

Example:
Problem: [problem1]
Step 1: [step1]
Step 2: [step2]
Answer: [answer1]

Problem: [problem2]
```

### Customization

**Adapt templates**: Modify for specific tasks
**Add constraints**: Include constraints or requirements
**Specify style**: Specify output style if needed

## Key Takeaways

1. **Zero-shot prompting uses instructions alone**: Providing clear task instructions enables models to perform tasks without examples, though performance may be limited.

2. **Few-shot prompting improves performance**: Including examples in prompts demonstrates desired behavior and often improves performance over zero-shot approaches.

3. **Chain-of-thought enables complex reasoning**: Including reasoning steps in prompts helps models solve complex problems by decomposing them into manageable steps.

4. **Self-consistency improves reliability**: Generating multiple reasoning paths and selecting the most consistent answer reduces errors and improves accuracy.

5. **Tree-of-thoughts explores systematically**: Systematically exploring multiple reasoning paths enables finding better solutions for complex problems.

6. **Prompt engineering requires iteration**: Effective prompts require careful design, testing, and refinement based on performance and error analysis.

7. **In-context learning enables adaptation**: Models can learn from examples in prompts without parameter updates, enabling flexible task adaptation.

8. **Templates provide starting points**: Standard prompt templates and patterns provide foundations that can be customized for specific tasks and requirements.
