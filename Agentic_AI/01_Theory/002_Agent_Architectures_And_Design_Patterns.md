# Agent Architectures and Design Patterns

## Table of Contents

1. [Overview of Agent Architectures](#overview-of-agent-architectures)
2. [Single-Agent Architectures](#single-agent-architectures)
   - [ReAct Pattern](#react-pattern)
   - [Plan-and-Execute Pattern](#plan-and-execute-pattern)
   - [Chain-of-Thought (CoT)](#chain-of-thought-cot)
   - [Reflection/Self-Critique Pattern](#reflectionself-critique-pattern)
   - [Tool-Augmented Agents](#tool-augmented-agents)
   - [State Machine Agents](#state-machine-agents)
   - [Pipeline Agents](#pipeline-agents)
   - [Iterative Refinement](#iterative-refinement)
3. [Multi-Agent Architectures](#multi-agent-architectures)
   - [Master-Worker Pattern](#master-worker-pattern)
   - [Peer-to-Peer](#peer-to-peer)
   - [Hierarchical Teams](#hierarchical-teams)
   - [Debate/Adversarial](#debateadversarial)
   - [Ensemble](#ensemble)
4. [Orchestration Patterns](#orchestration-patterns)
5. [Choosing the Right Architecture](#choosing-the-right-architecture)
6. [Anti-Patterns and Common Mistakes](#anti-patterns-and-common-mistakes)
7. [Architecture Comparison Table](#architecture-comparison-table)
8. [Case Studies: Architecture in Production](#case-studies-architecture-in-production)

---

## Overview of Agent Architectures

### Why Architecture Matters

Agent architecture defines the fundamental structure and behavior of an AI agent system. It determines how an agent perceives its environment, processes information, makes decisions, and takes actions. The choice of architecture has profound implications for:

- **Performance**: Response latency, throughput, and resource utilization
- **Reliability**: Error handling, fault tolerance, and system stability
- **Scalability**: Ability to handle increasing workloads and complexity
- **Maintainability**: Code organization, debugging, and extensibility
- **Cost**: Computational resources and API calls required
- **Capabilities**: Types of tasks the agent can effectively handle

### Evolution of Agent Architectures

The field of agent architectures has evolved significantly:

**First Generation (Rule-Based)**
- Hard-coded rules and decision trees
- Limited adaptability
- Brittle in novel situations

**Second Generation (Reactive)**
- Stimulus-response patterns
- Fast but shallow reasoning
- Limited planning capabilities

**Third Generation (Deliberative)**
- Internal world models
- Planning and goal-oriented behavior
- Slower but more capable

**Fourth Generation (Hybrid)**
- Combines reactive and deliberative approaches
- Multi-layered architectures
- Context-aware decision making

**Fifth Generation (LLM-Based)**
- Large language models as reasoning engines
- Natural language understanding and generation
- Tool use and external integration
- Emergent capabilities through scaling

### Architectural Components

Every agent architecture consists of several key components:

1. **Perception Module**: Processes inputs from the environment
2. **Reasoning Engine**: Makes decisions based on current state
3. **Memory System**: Stores and retrieves information
4. **Action Module**: Executes decisions in the environment
5. **Learning Component**: Adapts behavior based on experience (optional)

---

## Single-Agent Architectures

### ReAct Pattern

The ReAct (Reasoning + Acting) pattern combines reasoning and acting in an interleaved manner. The agent alternates between thinking about what to do and taking actions, allowing it to dynamically adapt its plan based on observations.

#### Core Concept

ReAct agents maintain an internal reasoning trace while interacting with external tools. Each cycle consists of:
1. **Thought**: Reasoning about the current situation
2. **Action**: Selecting and executing a tool
3. **Observation**: Processing the tool's output
4. **Repeat**: Continue until the goal is achieved

#### Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Initialize Agent     │
         │  (State, Memory)      │
         └───────────┬───────────┘
                     │
                     ▼
    ┌────────────────────────────────┐
    │      THOUGHT Phase             │
    │  - Analyze current state       │
    │  - Determine next action       │
    │  - Consider available tools    │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │      ACTION Phase              │
    │  - Select appropriate tool     │
    │  - Format tool call            │
    │  - Execute tool                │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │    OBSERVATION Phase           │
    │  - Receive tool output         │
    │  - Update internal state       │
    │  - Store in memory             │
    └────────────┬───────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Goal Achieved?│
         └───────┬───────┘
                 │
        ┌────────┴────────┐
        │ NO              │ YES
        │                 │
        ▼                 ▼
    [Continue Loop]  [Return Result]
```

#### Implementation Example

```python
class ReActAgent:
    """
    ReAct Pattern Implementation
    
    Combines reasoning and acting in an interleaved loop.
    """
    
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.memory = []
        
    def execute(self, query):
        """
        Main execution loop following ReAct pattern.
        """
        state = {
            "query": query,
            "iteration": 0,
            "thoughts": [],
            "actions": [],
            "observations": []
        }
        
        while state["iteration"] < self.max_iterations:
            # THOUGHT: Generate reasoning about next step
            thought = self._think(state)
            state["thoughts"].append(thought)
            
            # Check if we've reached a conclusion
            if self._is_final_answer(thought):
                return self._extract_answer(thought)
            
            # ACTION: Select and execute tool
            action = self._select_action(thought, state)
            if action is None:
                return self._handle_no_action(state)
            
            state["actions"].append(action)
            tool_name = action["tool"]
            tool_args = action["arguments"]
            
            # OBSERVATION: Execute tool and observe results
            observation = self._execute_tool(tool_name, tool_args)
            state["observations"].append(observation)
            
            # Update state with observation
            state["iteration"] += 1
            
        return self._handle_max_iterations(state)
    
    def _think(self, state):
        """
        Generate reasoning about current situation and next steps.
        """
        context = self._build_context(state)
        
        prompt = f"""
You are a ReAct agent solving: {state['query']}

Previous thoughts:
{self._format_list(state['thoughts'])}

Previous actions:
{self._format_list(state['actions'])}

Previous observations:
{self._format_list(state['observations'])}

Available tools: {', '.join(self.tools.keys())}

Think step by step about what to do next. Format your response as:
Thought: [your reasoning]
Action: [tool_name] OR Final Answer: [answer]
"""
        
        response = self.llm.generate(prompt)
        return response
    
    def _select_action(self, thought, state):
        """
        Parse thought to extract action or determine if final answer.
        """
        if "Final Answer:" in thought:
            return None
        
        # Extract tool name and arguments from thought
        action_match = re.search(r"Action:\s*(\w+)\s*(.*)", thought)
        if action_match:
            tool_name = action_match.group(1)
            tool_args_str = action_match.group(2)
            
            if tool_name in self.tools:
                # Parse arguments (simplified - in practice use JSON schema)
                args = self._parse_arguments(tool_args_str)
                return {
                    "tool": tool_name,
                    "arguments": args
                }
        
        return None
    
    def _execute_tool(self, tool_name, arguments):
        """
        Execute the selected tool with given arguments.
        """
        tool = self.tools[tool_name]
        try:
            result = tool.execute(**arguments)
            return {
                "tool": tool_name,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "result": str(e),
                "success": False
            }
    
    def _is_final_answer(self, thought):
        """Check if thought contains final answer."""
        return "Final Answer:" in thought
    
    def _extract_answer(self, thought):
        """Extract final answer from thought."""
        match = re.search(r"Final Answer:\s*(.+)", thought)
        return match.group(1) if match else thought
    
    def _build_context(self, state):
        """Build context string from state history."""
        context_parts = []
        for i in range(len(state["thoughts"])):
            context_parts.append(f"Thought {i+1}: {state['thoughts'][i]}")
            if i < len(state["actions"]):
                context_parts.append(f"Action {i+1}: {state['actions'][i]}")
            if i < len(state["observations"]):
                context_parts.append(f"Observation {i+1}: {state['observations'][i]}")
        return "\n".join(context_parts)
    
    def _format_list(self, items):
        """Format list of items for prompt."""
        if not items:
            return "None"
        return "\n".join([f"- {item}" for item in items])
    
    def _parse_arguments(self, args_str):
        """Parse tool arguments from string (simplified)."""
        # In practice, use JSON schema validation
        return json.loads(args_str) if args_str.strip() else {}
    
    def _handle_no_action(self, state):
        """Handle case where no valid action is selected."""
        return "Unable to determine next action."
    
    def _handle_max_iterations(self, state):
        """Handle case where max iterations reached."""
        return "Maximum iterations reached. Unable to complete task."


# Example Usage
class CalculatorTool:
    def __init__(self):
        self.name = "calculator"
    
    def execute(self, expression):
        """Evaluate mathematical expression."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Error: Invalid expression"


class WebSearchTool:
    def __init__(self):
        self.name = "web_search"
    
    def execute(self, query):
        """Search the web (mock implementation)."""
        # In practice, call actual search API
        return f"Search results for: {query}"


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()  # Your LLM client
    tools = [CalculatorTool(), WebSearchTool()]
    agent = ReActAgent(llm, tools)
    
    result = agent.execute("What is 25 * 4, and then search for information about that number?")
    print(result)
```

#### When to Use ReAct

**Best For:**
- Tasks requiring dynamic tool use
- Situations where the plan needs to adapt based on observations
- Interactive problem-solving
- Exploratory tasks with uncertain outcomes

**Not Ideal For:**
- Simple, deterministic tasks
- Tasks requiring extensive pre-planning
- High-latency scenarios (many tool calls)
- Cost-sensitive applications (many LLM calls)

#### Real-World Scenario: Customer Support Agent

A customer support agent using ReAct might:
1. **Thought**: "Customer is asking about order status. I need to check the order database."
2. **Action**: Call `get_order_status(order_id)`
3. **Observation**: "Order is shipped, tracking number XYZ"
4. **Thought**: "Order is shipped. Customer might want tracking info. Let me get shipping details."
5. **Action**: Call `get_tracking_info(tracking_number)`
6. **Observation**: "Package is in transit, expected delivery tomorrow"
7. **Thought**: "I have all the information needed."
8. **Final Answer**: "Your order has been shipped! Tracking number XYZ. Expected delivery: tomorrow."

---

### Plan-and-Execute Pattern

The Plan-and-Execute pattern separates planning from execution. The agent first creates a detailed plan, then executes it step-by-step, with optional replanning if execution deviates from expectations.

#### Core Concept

This pattern follows a two-phase approach:

1. **Planning Phase**: Create a comprehensive plan breaking down the task into subtasks
2. **Execution Phase**: Execute each subtask sequentially, monitoring progress

Optional:
3. **Replanning Phase**: If execution fails or deviates, create a new plan

#### Flow Diagram

```
┌──────────────────────────────────────────────┐
│           User Query/Task                    │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │     PLANNING PHASE           │
    │  - Analyze task              │
    │  - Break into subtasks       │
    │  - Order dependencies        │
    │  - Estimate resources        │
    └──────────────┬───────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │   Plan Created   │
         │  [Task 1, 2, 3]  │
         └─────────┬────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │    EXECUTION PHASE            │
    │                               │
    │  ┌────────────────────────┐  │
    │  │  Execute Task 1        │  │
    │  │  - Monitor progress    │  │
    │  │  - Check success       │  │
    │  └───────────┬────────────┘  │
    │              │                │
    │              ▼                │
    │      ┌───────────────┐        │
    │      │ Task Success? │        │
    │      └───────┬───────┘        │
    │              │                │
    │      ┌───────┴────────┐       │
    │      │ YES            │ NO    │
    │      │                │       │
    │      ▼                ▼       │
    │  [Next Task]    [Replan]      │
    │      │                │       │
    │      └────────┬───────┘       │
    │               │               │
    │               ▼               │
    │      ┌───────────────┐        │
    │      │ All Tasks Done?│        │
    │      └───────┬───────┘        │
    │              │                │
    │      ┌───────┴────────┐       │
    │      │ YES            │ NO    │
    │      │                │       │
    │      ▼                ▼       │
    │  [Return Result]  [Continue]   │
    └───────────────────────────────┘
```

#### Implementation Example

```python
class PlanAndExecuteAgent:
    """
    Plan-and-Execute Pattern Implementation
    
    Separates planning from execution for better control and reliability.
    """
    
    def __init__(self, llm, executor, max_replans=3):
        self.llm = llm
        self.executor = executor
        self.max_replans = max_replans
        
    def execute(self, task):
        """
        Main execution method following plan-and-execute pattern.
        """
        # PHASE 1: Planning
        plan = self._create_plan(task)
        
        execution_history = []
        replan_count = 0
        
        # PHASE 2: Execution with optional replanning
        while replan_count <= self.max_replans:
            execution_result = self._execute_plan(plan, execution_history)
            
            if execution_result["success"]:
                return execution_result["result"]
            
            # PHASE 3: Replanning if execution failed
            if replan_count < self.max_replans:
                plan = self._replan(task, plan, execution_history, execution_result)
                replan_count += 1
            else:
                return self._handle_failure(execution_result)
        
        return "Failed after maximum replanning attempts."
    
    def _create_plan(self, task):
        """
        Create a detailed plan for the given task.
        """
        prompt = f"""
You are a planning agent. Break down the following task into a detailed plan.

Task: {task}

Create a plan with the following structure:
1. Each step should be specific and actionable
2. Steps should be ordered logically
3. Include dependencies between steps
4. Estimate complexity for each step

Format your response as:
PLAN:
1. [Step description] (Dependencies: [step numbers], Complexity: [low/medium/high])
2. [Step description] (Dependencies: [step numbers], Complexity: [low/medium/high])
...
"""
        
        response = self.llm.generate(prompt)
        plan = self._parse_plan(response)
        return plan
    
    def _parse_plan(self, plan_text):
        """
        Parse plan text into structured format.
        """
        steps = []
        lines = plan_text.split('\n')
        
        for line in lines:
            if re.match(r'^\d+\.', line):
                # Extract step number, description, dependencies, complexity
                match = re.match(
                    r'(\d+)\.\s*(.+?)\s*\(Dependencies:\s*([^)]+),\s*Complexity:\s*(\w+)\)',
                    line
                )
                if match:
                    step_num = int(match.group(1))
                    description = match.group(2).strip()
                    deps_str = match.group(3).strip()
                    complexity = match.group(4).strip()
                    
                    # Parse dependencies
                    dependencies = []
                    if deps_str and deps_str.lower() != 'none':
                        deps = re.findall(r'\d+', deps_str)
                        dependencies = [int(d) for d in deps]
                    
                    steps.append({
                        "number": step_num,
                        "description": description,
                        "dependencies": dependencies,
                        "complexity": complexity,
                        "status": "pending",
                        "result": None
                    })
        
        return {
            "steps": steps,
            "total_steps": len(steps)
        }
    
    def _execute_plan(self, plan, execution_history):
        """
        Execute the plan step by step.
        """
        completed_steps = set()
        results = {}
        
        while len(completed_steps) < plan["total_steps"]:
            # Find next executable steps (dependencies satisfied)
            ready_steps = self._get_ready_steps(plan, completed_steps)
            
            if not ready_steps:
                # Deadlock or circular dependency
                return {
                    "success": False,
                    "error": "Cannot proceed: circular dependencies or missing steps",
                    "completed": completed_steps,
                    "results": results
                }
            
            # Execute ready steps (can be parallelized)
            for step in ready_steps:
                try:
                    result = self._execute_step(step, execution_history, results)
                    step["status"] = "completed"
                    step["result"] = result
                    completed_steps.add(step["number"])
                    results[step["number"]] = result
                    execution_history.append({
                        "step": step["number"],
                        "action": "execute",
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    step["status"] = "failed"
                    step["error"] = str(e)
                    execution_history.append({
                        "step": step["number"],
                        "action": "execute",
                        "error": str(e),
                        "success": False
                    })
                    return {
                        "success": False,
                        "error": f"Step {step['number']} failed: {str(e)}",
                        "failed_step": step["number"],
                        "completed": completed_steps,
                        "results": results
                    }
        
        return {
            "success": True,
            "result": self._aggregate_results(results),
            "completed": completed_steps,
            "results": results
        }
    
    def _get_ready_steps(self, plan, completed_steps):
        """
        Get steps that are ready to execute (dependencies satisfied).
        """
        ready = []
        for step in plan["steps"]:
            if step["status"] == "pending":
                deps_satisfied = all(
                    dep in completed_steps 
                    for dep in step["dependencies"]
                )
                if deps_satisfied:
                    ready.append(step)
        return ready
    
    def _execute_step(self, step, execution_history, previous_results):
        """
        Execute a single plan step.
        """
        # Build context from previous results
        context = self._build_execution_context(previous_results, execution_history)
        
        prompt = f"""
You are executing step {step['number']} of a plan.

Step Description: {step['description']}

Previous Step Results:
{context}

Execute this step and provide the result. Be specific and actionable.
"""
        
        result = self.executor.execute(step["description"], context)
        return result
    
    def _build_execution_context(self, results, history):
        """
        Build context string from previous execution results.
        """
        context_parts = []
        for step_num, result in sorted(results.items()):
            context_parts.append(f"Step {step_num} result: {result}")
        return "\n".join(context_parts) if context_parts else "No previous results."
    
    def _replan(self, original_task, old_plan, execution_history, failure_info):
        """
        Create a new plan based on execution failures.
        """
        prompt = f"""
Original Task: {original_task}

Previous Plan:
{self._format_plan(old_plan)}

Execution History:
{self._format_history(execution_history)}

Failure Information:
{failure_info['error']}
Failed at step: {failure_info.get('failed_step', 'unknown')}

Create a revised plan that addresses the failure. Consider:
1. What went wrong in the previous plan
2. How to avoid the same mistake
3. Alternative approaches
4. Breaking down complex steps further

Format your response as a new plan following the same structure.
"""
        
        response = self.llm.generate(prompt)
        new_plan = self._parse_plan(response)
        return new_plan
    
    def _format_plan(self, plan):
        """Format plan for display."""
        parts = []
        for step in plan["steps"]:
            deps = ', '.join(map(str, step["dependencies"])) if step["dependencies"] else "none"
            parts.append(
                f"{step['number']}. {step['description']} "
                f"(Dependencies: {deps}, Complexity: {step['complexity']}, "
                f"Status: {step['status']})"
            )
        return "\n".join(parts)
    
    def _format_history(self, history):
        """Format execution history for display."""
        parts = []
        for entry in history:
            if entry["success"]:
                parts.append(f"Step {entry['step']}: SUCCESS - {entry.get('result', 'N/A')}")
            else:
                parts.append(f"Step {entry['step']}: FAILED - {entry.get('error', 'N/A')}")
        return "\n".join(parts)
    
    def _aggregate_results(self, results):
        """
        Aggregate results from all steps into final output.
        """
        # Simple aggregation - can be customized
        aggregated = []
        for step_num in sorted(results.keys()):
            aggregated.append(f"Step {step_num}: {results[step_num]}")
        return "\n".join(aggregated)
    
    def _handle_failure(self, execution_result):
        """Handle final failure after replanning attempts."""
        return f"Task failed: {execution_result['error']}"


# Example Executor
class TaskExecutor:
    """
    Executes individual plan steps.
    """
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def execute(self, step_description, context):
        """
        Execute a single step using available tools.
        """
        # Determine which tools are needed
        required_tools = self._identify_tools(step_description)
        
        # Execute using appropriate tools
        if required_tools:
            return self._execute_with_tools(step_description, context, required_tools)
        else:
            return self._execute_with_llm(step_description, context)
    
    def _identify_tools(self, description):
        """Identify which tools are needed for this step."""
        required = []
        description_lower = description.lower()
        
        for tool_name, tool in self.tools.items():
            if tool.matches(description_lower):
                required.append(tool)
        
        return required
    
    def _execute_with_tools(self, description, context, tools):
        """Execute step using tools."""
        # Simplified - in practice, use tool calling framework
        results = []
        for tool in tools:
            result = tool.execute(description, context)
            results.append(result)
        return "; ".join(results)
    
    def _execute_with_llm(self, description, context):
        """Execute step using LLM reasoning."""
        prompt = f"""
Execute the following step:
{description}

Context: {context}

Provide a clear, actionable result.
"""
        return self.llm.generate(prompt)


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    tools = {}  # Your tools
    executor = TaskExecutor(llm, tools)
    agent = PlanAndExecuteAgent(llm, executor)
    
    task = """
    Research the top 3 programming languages in 2024, compare their 
    performance characteristics, and create a summary report.
    """
    
    result = agent.execute(task)
    print(result)
```

#### Task Decomposition Strategies

**Hierarchical Decomposition**
- Break task into major phases
- Each phase into sub-phases
- Continue until atomic actions

**Dependency-Based Decomposition**
- Identify task dependencies first
- Create steps that respect dependencies
- Parallelize independent steps

**Goal-Oriented Decomposition**
- Identify sub-goals
- Create steps to achieve each sub-goal
- Sequence sub-goals logically

#### Replanning Strategies

**Failure-Driven Replanning**
- Trigger replanning when step fails
- Analyze failure cause
- Modify plan to avoid failure

**Deviation-Based Replanning**
- Monitor execution progress
- Replan if deviating from expected outcomes
- Adjust plan based on new information

**Periodic Replanning**
- Replan at fixed intervals
- Incorporate new information
- Optimize remaining steps

#### When to Use Plan-and-Execute

**Best For:**
- Complex, multi-step tasks
- Tasks with clear dependencies
- Scenarios requiring reliability
- Tasks where planning cost is justified

**Not Ideal For:**
- Simple, single-step tasks
- Highly dynamic environments
- Tasks requiring immediate response
- Exploratory tasks with uncertain structure

---

### Chain-of-Thought (CoT)

Chain-of-Thought prompting encourages models to break down complex problems into intermediate reasoning steps. This pattern improves reasoning quality by making the thought process explicit and traceable.

#### Core Concept

CoT agents explicitly show their reasoning process:
1. **Problem Analysis**: Break down the problem
2. **Step-by-Step Reasoning**: Show intermediate steps
3. **Conclusion**: Arrive at final answer based on reasoning

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Problem/Question            │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Problem Analysis    │
    │  - Identify key      │
    │    components        │
    │  - Extract facts     │
    │  - Note constraints  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Step 1: Reasoning   │
    │  - Show calculation  │
    │  - State assumption  │
    │  - Apply rule        │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Step 2: Reasoning   │
    │  - Build on Step 1   │
    │  - Show next logic   │
    └──────────┬───────────┘
               │
               ▼
         ┌──────────┐
         │  ...     │
         │  More    │
         │  Steps   │
         └────┬─────┘
              │
              ▼
    ┌──────────────────────┐
    │  Final Step:         │
    │  - Synthesize        │
    │  - Verify logic      │
    │  - State conclusion  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │    Final Answer      │
    └──────────────────────┘
```

#### Implementation Example

```python
class ChainOfThoughtAgent:
    """
    Chain-of-Thought Pattern Implementation
    
    Encourages step-by-step reasoning for complex problems.
    """
    
    def __init__(self, llm, max_steps=10):
        self.llm = llm
        self.max_steps = max_steps
    
    def solve(self, problem):
        """
        Solve problem using chain-of-thought reasoning.
        """
        reasoning_steps = []
        
        # Initial problem analysis
        analysis = self._analyze_problem(problem)
        reasoning_steps.append({
            "step": 0,
            "type": "analysis",
            "content": analysis
        })
        
        # Iterative reasoning steps
        current_state = analysis
        for step_num in range(1, self.max_steps + 1):
            next_step = self._reason_step(current_state, problem, reasoning_steps)
            
            if self._is_final_answer(next_step):
                reasoning_steps.append({
                    "step": step_num,
                    "type": "conclusion",
                    "content": next_step
                })
                break
            
            reasoning_steps.append({
                "step": step_num,
                "type": "reasoning",
                "content": next_step
            })
            
            current_state = next_step
        
        return self._format_response(reasoning_steps)
    
    def _analyze_problem(self, problem):
        """
        Analyze the problem to identify key components.
        """
        prompt = f"""
Analyze the following problem step by step.

Problem: {problem}

Break down the problem:
1. What are the key facts?
2. What are we trying to find or determine?
3. What constraints or conditions apply?
4. What information might be missing?

Format your response clearly.
"""
        return self.llm.generate(prompt)
    
    def _reason_step(self, current_state, original_problem, previous_steps):
        """
        Generate next reasoning step.
        """
        context = self._build_reasoning_context(previous_steps)
        
        prompt = f"""
Original Problem: {original_problem}

Previous Reasoning Steps:
{context}

Current State: {current_state}

Continue the reasoning process. Think step by step:
1. What can we conclude from the current state?
2. What is the next logical step?
3. Show your work clearly.

If you've reached a conclusion, format it as:
FINAL ANSWER: [your answer]

Otherwise, continue reasoning.
"""
        
        return self.llm.generate(prompt)
    
    def _build_reasoning_context(self, steps):
        """
        Build context from previous reasoning steps.
        """
        context_parts = []
        for step in steps:
            step_type = step["type"].upper()
            content = step["content"]
            context_parts.append(f"{step_type} Step {step['step']}: {content}")
        return "\n\n".join(context_parts)
    
    def _is_final_answer(self, step_content):
        """
        Check if step contains final answer.
        """
        return "FINAL ANSWER:" in step_content.upper()
    
    def _format_response(self, steps):
        """
        Format the complete reasoning chain.
        """
        formatted = []
        for step in steps:
            formatted.append(f"Step {step['step']} ({step['type']}):")
            formatted.append(step['content'])
            formatted.append("")
        
        # Extract final answer if present
        final_answer = None
        for step in reversed(steps):
            if "FINAL ANSWER:" in step['content'].upper():
                match = re.search(r'FINAL ANSWER:\s*(.+)', step['content'], re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip()
                break
        
        result = {
            "reasoning_chain": "\n".join(formatted),
            "steps": steps,
            "final_answer": final_answer
        }
        
        return result


# Self-Consistency Variant
class SelfConsistentCoTAgent:
    """
    Chain-of-Thought with Self-Consistency
    
    Generates multiple reasoning paths and selects most consistent answer.
    """
    
    def __init__(self, llm, num_paths=5):
        self.llm = llm
        self.num_paths = num_paths
        self.base_cot = ChainOfThoughtAgent(llm)
    
    def solve(self, problem):
        """
        Solve using multiple reasoning paths and select consensus.
        """
        paths = []
        
        # Generate multiple reasoning paths
        for i in range(self.num_paths):
            path = self.base_cot.solve(problem)
            paths.append(path)
        
        # Extract answers from all paths
        answers = [p["final_answer"] for p in paths if p["final_answer"]]
        
        if not answers:
            return {
                "consensus_answer": None,
                "confidence": 0.0,
                "paths": paths,
                "error": "No valid answers generated"
            }
        
        # Find consensus answer
        consensus_answer, confidence = self._find_consensus(answers)
        
        return {
            "consensus_answer": consensus_answer,
            "confidence": confidence,
            "paths": paths,
            "answer_distribution": self._count_answers(answers)
        }
    
    def _find_consensus(self, answers):
        """
        Find the most common answer (consensus).
        """
        # Normalize answers for comparison
        normalized = [self._normalize_answer(a) for a in answers]
        
        # Count occurrences
        counts = {}
        for ans in normalized:
            counts[ans] = counts.get(ans, 0) + 1
        
        # Find most common
        if not counts:
            return None, 0.0
        
        most_common = max(counts.items(), key=lambda x: x[1])
        consensus = most_common[0]
        confidence = most_common[1] / len(answers)
        
        return consensus, confidence
    
    def _normalize_answer(self, answer):
        """
        Normalize answer for comparison (remove whitespace, lowercase, etc.).
        """
        return answer.strip().lower()
    
    def _count_answers(self, answers):
        """
        Count distribution of answers.
        """
        normalized = [self._normalize_answer(a) for a in answers]
        counts = {}
        for ans in normalized:
            counts[ans] = counts.get(ans, 0) + 1
        return counts


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Standard CoT
    cot_agent = ChainOfThoughtAgent(llm)
    result = cot_agent.solve(
        "If a train travels 120 miles in 2 hours, and another train travels "
        "180 miles in 3 hours, which train is faster?"
    )
    print(result["reasoning_chain"])
    print(f"\nFinal Answer: {result['final_answer']}")
    
    # Self-Consistent CoT
    sc_cot_agent = SelfConsistentCoTAgent(llm, num_paths=5)
    result = sc_cot_agent.solve(
        "A store has 50 apples. They sell 20 on Monday and 15 on Tuesday. "
        "How many apples are left?"
    )
    print(f"\nConsensus Answer: {result['consensus_answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

#### CoT Prompting Techniques

**Zero-Shot CoT**
- Add "Let's think step by step" to prompt
- Model generates reasoning automatically
- No examples needed

**Few-Shot CoT**
- Provide examples with reasoning steps
- Model learns reasoning pattern
- Better for complex domains

**Self-Consistency**
- Generate multiple reasoning paths
- Select most common answer
- Improves accuracy

#### When to Use CoT

**Best For:**
- Mathematical problems
- Logical reasoning tasks
- Multi-step calculations
- Problems requiring explicit reasoning

**Not Ideal For:**
- Simple lookups
- Creative tasks
- Tasks requiring tool use
- Real-time interactions

---

### Reflection/Self-Critique Pattern

The Reflection pattern involves agents reviewing and critiquing their own outputs, then iteratively improving them. This pattern is inspired by the Reflexion framework and human self-reflection.

#### Core Concept

Reflection agents follow this cycle:
1. **Generate**: Create initial output
2. **Reflect**: Critically evaluate the output
3. **Refine**: Improve based on critique
4. **Repeat**: Continue until satisfactory

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Task/Request                │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  GENERATE Phase      │
    │  - Create initial    │
    │    output            │
    │  - Apply knowledge   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  REFLECT Phase       │
    │  - Review output     │
    │  - Identify issues   │
    │  - Check quality     │
    │  - Verify facts      │
    └──────────┬───────────┘
               │
               ▼
         ┌──────────┐
         │ Quality  │
         │ Check    │
         └────┬─────┘
              │
      ┌───────┴────────┐
      │ Acceptable?    │
      └───────┬────────┘
              │
      ┌───────┴────────┐
      │ YES            │ NO
      │                │
      ▼                ▼
  [Return Output]  ┌──────────────────────┐
                   │  REFINE Phase        │
                   │  - Address issues    │
                   │  - Improve quality   │
                   │  - Fix errors        │
                   └──────────┬───────────┘
                              │
                              ▼
                         [Loop Back]
```

#### Implementation Example

```python
class ReflectionAgent:
    """
    Reflection/Self-Critique Pattern Implementation
    
    Agents review and improve their own outputs iteratively.
    """
    
    def __init__(self, llm, generator, critic, max_iterations=5):
        self.llm = llm
        self.generator = generator
        self.critic = critic
        self.max_iterations = max_iterations
    
    def execute(self, task):
        """
        Execute task with reflection and refinement.
        """
        iteration = 0
        current_output = None
        reflection_history = []
        
        while iteration < self.max_iterations:
            # GENERATE: Create or refine output
            if iteration == 0:
                current_output = self.generator.generate(task)
            else:
                # Use previous reflection to refine
                last_reflection = reflection_history[-1]
                current_output = self.generator.refine(
                    task, 
                    current_output, 
                    last_reflection
                )
            
            # REFLECT: Critically evaluate output
            reflection = self.critic.reflect(task, current_output, reflection_history)
            reflection_history.append(reflection)
            
            # Check if output is acceptable
            if reflection["acceptable"]:
                return {
                    "output": current_output,
                    "iterations": iteration + 1,
                    "reflection_history": reflection_history,
                    "final_reflection": reflection
                }
            
            iteration += 1
        
        # Return best output even if not perfect
        return {
            "output": current_output,
            "iterations": iteration,
            "reflection_history": reflection_history,
            "final_reflection": reflection_history[-1],
            "note": "Maximum iterations reached"
        }


class OutputGenerator:
    """
    Generates initial outputs and refinements.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate(self, task):
        """
        Generate initial output for task.
        """
        prompt = f"""
Complete the following task:

Task: {task}

Provide a complete, high-quality response.
"""
        return self.llm.generate(prompt)
    
    def refine(self, task, current_output, reflection):
        """
        Refine output based on reflection feedback.
        """
        prompt = f"""
Original Task: {task}

Current Output:
{current_output}

Reflection and Critique:
{reflection['critique']}

Issues Identified:
{self._format_issues(reflection['issues'])}

Refine the output to address the identified issues. Maintain what's good, 
fix what's wrong, and improve overall quality.
"""
        return self.llm.generate(prompt)
    
    def _format_issues(self, issues):
        """Format issues list for prompt."""
        if not issues:
            return "None"
        return "\n".join([f"- {issue}" for issue in issues])


class OutputCritic:
    """
    Critically evaluates outputs and provides feedback.
    """
    
    def __init__(self, llm, quality_criteria):
        self.llm = llm
        self.quality_criteria = quality_criteria
    
    def reflect(self, task, output, previous_reflections):
        """
        Reflect on output quality and identify improvements.
        """
        context = self._build_reflection_context(previous_reflections)
        
        prompt = f"""
Task: {task}

Output to Evaluate:
{output}

Quality Criteria:
{self._format_criteria()}

Previous Reflections:
{context}

Critically evaluate this output:
1. Does it fully address the task?
2. Are there any factual errors?
3. Is the reasoning sound?
4. Is the structure clear?
5. Are there areas for improvement?

Provide:
- Overall assessment (acceptable/needs improvement)
- Specific issues found
- Detailed critique
- Suggestions for improvement
"""
        
        response = self.llm.generate(prompt)
        return self._parse_reflection(response)
    
    def _parse_reflection(self, response):
        """
        Parse reflection response into structured format.
        """
        # Extract acceptability
        acceptable = "acceptable" in response.lower() and "needs improvement" not in response.lower()
        
        # Extract issues (simplified parsing)
        issues = []
        if "issues:" in response.lower():
            issues_section = response.split("issues:")[1].split("\n\n")[0]
            issue_lines = [line.strip() for line in issues_section.split("\n") if line.strip() and line.strip().startswith("-")]
            issues = [line[1:].strip() for line in issue_lines]
        
        return {
            "acceptable": acceptable,
            "critique": response,
            "issues": issues if issues else ["General quality improvements needed"]
        }
    
    def _format_criteria(self):
        """Format quality criteria for prompt."""
        criteria_list = []
        for criterion, description in self.quality_criteria.items():
            criteria_list.append(f"- {criterion}: {description}")
        return "\n".join(criteria_list)
    
    def _build_reflection_context(self, previous_reflections):
        """Build context from previous reflections."""
        if not previous_reflections:
            return "None"
        
        context_parts = []
        for i, reflection in enumerate(previous_reflections, 1):
            context_parts.append(f"Iteration {i}:")
            context_parts.append(f"Issues: {', '.join(reflection['issues'])}")
            context_parts.append(f"Acceptable: {reflection['acceptable']}")
        
        return "\n".join(context_parts)


# Reflexion Framework Variant
class ReflexionAgent:
    """
    Reflexion Framework Implementation
    
    Agents reflect on task execution outcomes and update strategy.
    """
    
    def __init__(self, llm, executor, memory):
        self.llm = llm
        self.executor = executor
        self.memory = memory  # Stores reflection history
    
    def execute_with_reflection(self, task):
        """
        Execute task with reflection on outcomes.
        """
        # Attempt execution
        result = self.executor.execute(task)
        
        # Reflect on outcome
        reflection = self._reflect_on_outcome(task, result)
        
        # Store reflection in memory
        self.memory.store_reflection(task, result, reflection)
        
        # If failed, generate improved strategy
        if not result["success"]:
            improved_strategy = self._generate_strategy(task, reflection)
            # Retry with improved strategy
            result = self.executor.execute(task, strategy=improved_strategy)
        
        return {
            "result": result,
            "reflection": reflection,
            "strategy_used": result.get("strategy")
        }
    
    def _reflect_on_outcome(self, task, result):
        """
        Reflect on task execution outcome.
        """
        prompt = f"""
Task: {task}

Execution Result:
Success: {result['success']}
Output: {result.get('output', 'N/A')}
Error: {result.get('error', 'None')}

Reflect on this outcome:
1. What went well?
2. What went wrong?
3. Why did it fail/succeed?
4. What could be improved?
5. What should be remembered for future similar tasks?

Provide detailed reflection.
"""
        return self.llm.generate(prompt)
    
    def _generate_strategy(self, task, reflection):
        """
        Generate improved execution strategy based on reflection.
        """
        previous_reflections = self.memory.get_relevant_reflections(task)
        
        prompt = f"""
Task: {task}

Previous Reflection:
{reflection}

Similar Past Reflections:
{self._format_reflections(previous_reflections)}

Generate an improved strategy for executing this task. Consider:
1. What failed before and why
2. What approaches worked in similar situations
3. How to avoid previous mistakes
4. Best practices for this type of task

Provide a clear, actionable strategy.
"""
        return self.llm.generate(prompt)
    
    def _format_reflections(self, reflections):
        """Format reflections for prompt."""
        if not reflections:
            return "None"
        return "\n".join([f"- {r}" for r in reflections])


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Standard Reflection Agent
    quality_criteria = {
        "Accuracy": "Information must be factually correct",
        "Completeness": "All aspects of task must be addressed",
        "Clarity": "Output must be clear and well-structured",
        "Relevance": "Content must be relevant to the task"
    }
    
    generator = OutputGenerator(llm)
    critic = OutputCritic(llm, quality_criteria)
    agent = ReflectionAgent(llm, generator, critic)
    
    result = agent.execute("Write a comprehensive guide to Python decorators")
    print(f"Iterations: {result['iterations']}")
    print(f"Output: {result['output']}")
```

#### Reflection Strategies

**Output Quality Reflection**
- Review completeness
- Check accuracy
- Assess clarity
- Verify requirements

**Execution Reflection**
- Analyze what worked
- Identify failures
- Understand causes
- Learn for future

**Strategy Reflection**
- Evaluate approach
- Compare alternatives
- Optimize methods
- Adapt to context

#### When to Use Reflection

**Best For:**
- High-quality output requirements
- Complex creative tasks
- Error-sensitive applications
- Learning from mistakes

**Not Ideal For:**
- Real-time responses
- Simple tasks
- Cost-sensitive applications
- Deterministic outputs

---

### Tool-Augmented Agents

Tool-augmented agents extend their capabilities by calling external functions, APIs, and services. This pattern enables agents to interact with the real world beyond text generation.

#### Core Concept

Tool-augmented agents:
1. **Identify Need**: Determine when tools are needed
2. **Select Tool**: Choose appropriate tool(s)
3. **Call Tool**: Execute with proper parameters
4. **Process Result**: Integrate tool output into reasoning
5. **Compose Tools**: Chain multiple tools when needed

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         User Query                  │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Analyze Query       │
    │  - Determine intent   │
    │  - Identify needs     │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Tool Selection      │
    │  - Match query to    │
    │    available tools   │
    │  - Rank by relevance │
    └──────────┬───────────┘
               │
               ▼
         ┌──────────┐
         │ Tools    │
         │ Needed?  │
         └────┬─────┘
              │
      ┌───────┴────────┐
      │ YES            │ NO
      │                │
      ▼                ▼
┌──────────────┐  ┌──────────────┐
│ Extract      │  │ Generate     │
│ Parameters   │  │ Direct       │
│               │  │ Response     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 │
┌──────────────┐         │
│ Execute Tool │         │
└──────┬───────┘         │
       │                 │
       ▼                 │
┌──────────────┐         │
│ Process      │         │
│ Result       │         │
└──────┬───────┘         │
       │                 │
       └────────┬────────┘
                │
                ▼
    ┌──────────────────────┐
    │  Compose Response    │
    │  - Integrate tool    │
    │    results           │
    │  - Provide answer    │
    └──────────────────────┘
```

#### Implementation Example

```python
class ToolAugmentedAgent:
    """
    Tool-Augmented Agent Implementation
    
    Agents that can use external tools to extend capabilities.
    """
    
    def __init__(self, llm, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry  # Dictionary of available tools
    
    def execute(self, query):
        """
        Execute query using tools as needed.
        """
        # Analyze query to determine tool needs
        tool_needs = self._analyze_tool_needs(query)
        
        if not tool_needs:
            # No tools needed, direct response
            return self._generate_direct_response(query)
        
        # Select and execute tools
        tool_results = []
        for tool_need in tool_needs:
            tool_name = tool_need["tool"]
            parameters = tool_need["parameters"]
            
            # Execute tool
            result = self._execute_tool(tool_name, parameters)
            tool_results.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result
            })
        
        # Generate response incorporating tool results
        response = self._generate_response_with_tools(query, tool_results)
        
        return {
            "response": response,
            "tools_used": [tr["tool"] for tr in tool_results],
            "tool_results": tool_results
        }
    
    def _analyze_tool_needs(self, query):
        """
        Analyze query to determine which tools are needed.
        """
        # Get available tools description
        tools_description = self._describe_tools()
        
        prompt = f"""
Query: {query}

Available Tools:
{tools_description}

Determine which tools (if any) are needed to answer this query.

For each tool needed, provide:
- Tool name
- Parameters required

Format as JSON:
{{
    "tools": [
        {{
            "tool": "tool_name",
            "parameters": {{"param1": "value1"}}
        }}
    ]
}}

If no tools are needed, return: {{"tools": []}}
"""
        
        response = self.llm.generate(prompt)
        tool_needs = self._parse_tool_needs(response)
        return tool_needs
    
    def _describe_tools(self):
        """
        Generate description of available tools for LLM.
        """
        descriptions = []
        for tool_name, tool in self.tool_registry.items():
            descriptions.append(
                f"- {tool_name}: {tool.description}\n"
                f"  Parameters: {tool.parameter_schema}\n"
                f"  Returns: {tool.return_description}"
            )
        return "\n".join(descriptions)
    
    def _parse_tool_needs(self, response):
        """
        Parse tool needs from LLM response.
        """
        try:
            parsed = json.loads(response)
            return parsed.get("tools", [])
        except:
            # Fallback: try to extract from text
            return self._extract_tools_from_text(response)
    
    def _extract_tools_from_text(self, text):
        """
        Extract tool information from unstructured text (fallback).
        """
        # Simplified extraction - in practice use more robust parsing
        tools = []
        # Look for tool mentions and parameters
        # This is a simplified version
        return tools
    
    def _execute_tool(self, tool_name, parameters):
        """
        Execute a tool with given parameters.
        """
        if tool_name not in self.tool_registry:
            return {"error": f"Tool {tool_name} not found"}
        
        tool = self.tool_registry[tool_name]
        
        # Validate parameters
        validation_result = tool.validate_parameters(parameters)
        if not validation_result["valid"]:
            return {"error": f"Invalid parameters: {validation_result['errors']}"}
        
        # Execute tool
        try:
            result = tool.execute(**parameters)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_direct_response(self, query):
        """
        Generate response without using tools.
        """
        prompt = f"""
Answer the following query:

{query}

Provide a clear, comprehensive answer.
"""
        response = self.llm.generate(prompt)
        return {
            "response": response,
            "tools_used": [],
            "tool_results": []
        }
    
    def _generate_response_with_tools(self, query, tool_results):
        """
        Generate response incorporating tool results.
        """
        tool_results_text = self._format_tool_results(tool_results)
        
        prompt = f"""
Query: {query}

Tool Results:
{tool_results_text}

Generate a comprehensive answer to the query using the tool results. 
Integrate the information naturally and cite which tools provided which information.
"""
        return self.llm.generate(prompt)
    
    def _format_tool_results(self, tool_results):
        """
        Format tool results for prompt.
        """
        formatted = []
        for tr in tool_results:
            formatted.append(f"Tool: {tr['tool']}")
            formatted.append(f"Parameters: {tr['parameters']}")
            formatted.append(f"Result: {tr['result']}")
            formatted.append("")
        return "\n".join(formatted)


# Tool Base Class
class Tool:
    """
    Base class for tools that agents can use.
    """
    
    def __init__(self, name, description, parameter_schema, return_description):
        self.name = name
        self.description = description
        self.parameter_schema = parameter_schema  # JSON schema
        self.return_description = return_description
    
    def validate_parameters(self, parameters):
        """
        Validate parameters against schema.
        """
        # Simplified validation - in practice use JSON schema validator
        errors = []
        
        # Check required parameters
        required = self.parameter_schema.get("required", [])
        for param in required:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Check parameter types
        properties = self.parameter_schema.get("properties", {})
        for param_name, param_value in parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type and not self._check_type(param_value, expected_type):
                    errors.append(f"Parameter {param_name} has wrong type")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _check_type(self, value, expected_type):
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        return True
    
    def execute(self, **kwargs):
        """
        Execute the tool. Must be implemented by subclasses.
        """
        raise NotImplementedError


# Example Tools
class CalculatorTool(Tool):
    """
    Calculator tool for mathematical operations.
    """
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations",
            parameter_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            return_description="Numeric result of the calculation"
        )
    
    def execute(self, expression):
        """Execute calculation."""
        try:
            # In practice, use safe evaluation
            result = eval(expression)
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")


class WebSearchTool(Tool):
    """
    Web search tool.
    """
    
    def __init__(self, search_api):
        super().__init__(
            name="web_search",
            description="Searches the web for information",
            parameter_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            return_description="List of search results with titles and snippets"
        )
        self.search_api = search_api
    
    def execute(self, query, max_results=5):
        """Execute web search."""
        return self.search_api.search(query, max_results=max_results)


class DatabaseQueryTool(Tool):
    """
    Database query tool.
    """
    
    def __init__(self, db_connection):
        super().__init__(
            name="database_query",
            description="Queries a database",
            parameter_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query"
                    }
                },
                "required": ["query"]
            },
            return_description="Query results as list of dictionaries"
        )
        self.db = db_connection
    
    def execute(self, query):
        """Execute database query."""
        # In practice, add SQL injection protection
        cursor = self.db.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results


# Tool Composition Example
class ToolCompositionAgent(ToolAugmentedAgent):
    """
    Agent that can compose multiple tools in sequence or parallel.
    """
    
    def execute_with_composition(self, query):
        """
        Execute query using tool composition.
        """
        # Analyze for tool composition needs
        composition_plan = self._plan_tool_composition(query)
        
        results = {}
        for step in composition_plan["steps"]:
            step_type = step["type"]  # "sequential" or "parallel"
            
            if step_type == "sequential":
                # Execute tools in sequence, passing results forward
                for tool_call in step["tools"]:
                    tool_name = tool_call["tool"]
                    # Resolve parameters that depend on previous results
                    parameters = self._resolve_parameters(
                        tool_call["parameters"], 
                        results
                    )
                    result = self._execute_tool(tool_name, parameters)
                    results[tool_call["id"]] = result
            
            elif step_type == "parallel":
                # Execute tools in parallel
                parallel_results = {}
                for tool_call in step["tools"]:
                    tool_name = tool_call["tool"]
                    parameters = self._resolve_parameters(
                        tool_call["parameters"],
                        results
                    )
                    result = self._execute_tool(tool_name, parameters)
                    parallel_results[tool_call["id"]] = result
                # Merge parallel results
                results.update(parallel_results)
        
        # Generate final response
        response = self._generate_response_with_tools(query, [
            {"tool": step_id, "result": result}
            for step_id, result in results.items()
        ])
        
        return response
    
    def _plan_tool_composition(self, query):
        """
        Plan how to compose tools for the query.
        """
        # Simplified planning - in practice use more sophisticated planning
        prompt = f"""
Query: {query}

Available Tools: {', '.join(self.tool_registry.keys())}

Create a plan for using tools to answer this query. Consider:
- Which tools are needed
- Order of execution (some tools may depend on others)
- Which tools can run in parallel

Format as JSON with steps that can be sequential or parallel.
"""
        response = self.llm.generate(prompt)
        # Parse and return composition plan
        return json.loads(response)
    
    def _resolve_parameters(self, parameters, previous_results):
        """
        Resolve parameters that reference previous tool results.
        """
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$result."):
                # Reference to previous result
                result_id = value.replace("$result.", "")
                if result_id in previous_results:
                    resolved[key] = previous_results[result_id]
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Register tools
    tools = {
        "calculator": CalculatorTool(),
        "web_search": WebSearchTool(search_api=MockSearchAPI()),
        "database_query": DatabaseQueryTool(db=MockDB())
    }
    
    # Create agent
    agent = ToolAugmentedAgent(llm, tools)
    
    # Execute query
    result = agent.execute(
        "What is the square root of 144, and then search for information "
        "about that number?"
    )
    
    print(f"Response: {result['response']}")
    print(f"Tools Used: {result['tools_used']}")
```

#### Tool Selection Strategies

**Semantic Matching**
- Match query intent to tool descriptions
- Use embeddings for similarity
- Rank by relevance score

**Rule-Based Selection**
- Predefined rules for tool selection
- Fast and predictable
- Less flexible

**LLM-Based Selection**
- Use LLM to choose tools
- More flexible
- Higher latency

#### Tool Composition Patterns

**Sequential Composition**
- Tools execute in order
- Output of one feeds into next
- For dependent operations

**Parallel Composition**
- Tools execute simultaneously
- Results combined afterward
- For independent operations

**Conditional Composition**
- Tool selection based on conditions
- Branching execution paths
- For dynamic workflows

#### When to Use Tool-Augmented Agents

**Best For:**
- Tasks requiring external data
- Real-time information needs
- Complex computations
- Integration with existing systems

**Not Ideal For:**
- Pure text generation
- Tasks without external dependencies
- Cost-sensitive applications
- Simple Q&A without tools

---

### State Machine Agents

State Machine agents use finite state machines (FSM) to model agent behavior. The agent transitions between well-defined states based on inputs and conditions, providing predictable and controllable behavior.

#### Core Concept

State Machine agents consist of:
1. **States**: Distinct modes of operation
2. **Transitions**: Rules for moving between states
3. **Actions**: Operations performed in each state
4. **Conditions**: Guards that enable/disable transitions

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Initial State               │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │   State: IDLE        │
    │   - Wait for input   │
    │   - No actions       │
    └──────────┬───────────┘
               │
               │ [Input Received]
               ▼
    ┌──────────────────────┐
    │   State: PROCESSING  │
    │   - Analyze input    │
    │   - Execute logic    │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        │             │
        │ [Success]   │ [Error]
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────┐
│ State: DONE  │  │ State: ERROR│
│ - Return     │  │ - Handle    │
│   result     │  │   error     │
└──────┬───────┘  └──────┬───────┘
       │                │
       │ [Reset]        │ [Retry]
       │                │
       └────────┬───────┘
                │
                ▼
         [Back to IDLE]
```

#### Implementation Example

```python
from enum import Enum
from typing import Dict, Callable, Optional

class AgentState(Enum):
    """Agent states enumeration."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_INPUT = "waiting_input"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ERROR = "error"


class StateMachineAgent:
    """
    State Machine Agent Implementation
    
    Uses finite state machine to control agent behavior.
    """
    
    def __init__(self, llm, initial_state=AgentState.IDLE):
        self.llm = llm
        self.current_state = initial_state
        self.state_handlers: Dict[AgentState, Callable] = {}
        self.transitions: Dict[AgentState, Dict[str, AgentState]] = {}
        self.state_data: Dict[AgentState, dict] = {}
        self.context = {}
        
        self._setup_state_machine()
    
    def _setup_state_machine(self):
        """
        Setup state handlers and transitions.
        """
        # Define state handlers
        self.state_handlers = {
            AgentState.IDLE: self._handle_idle,
            AgentState.PROCESSING: self._handle_processing,
            AgentState.WAITING_INPUT: self._handle_waiting_input,
            AgentState.EXECUTING: self._handle_executing,
            AgentState.VALIDATING: self._handle_validating,
            AgentState.COMPLETED: self._handle_completed,
            AgentState.ERROR: self._handle_error
        }
        
        # Define state transitions
        self.transitions = {
            AgentState.IDLE: {
                "input_received": AgentState.PROCESSING
            },
            AgentState.PROCESSING: {
                "need_input": AgentState.WAITING_INPUT,
                "ready_execute": AgentState.EXECUTING,
                "error": AgentState.ERROR
            },
            AgentState.WAITING_INPUT: {
                "input_provided": AgentState.PROCESSING,
                "timeout": AgentState.ERROR
            },
            AgentState.EXECUTING: {
                "execution_done": AgentState.VALIDATING,
                "execution_error": AgentState.ERROR
            },
            AgentState.VALIDATING: {
                "validation_pass": AgentState.COMPLETED,
                "validation_fail": AgentState.PROCESSING,
                "error": AgentState.ERROR
            },
            AgentState.COMPLETED: {
                "reset": AgentState.IDLE
            },
            AgentState.ERROR: {
                "retry": AgentState.PROCESSING,
                "reset": AgentState.IDLE
            }
        }
    
    def execute(self, input_data):
        """
        Execute agent with input, following state machine.
        """
        self.context = {"input": input_data, "output": None, "error": None}
        self.current_state = AgentState.IDLE
        
        # Trigger initial transition
        self._transition("input_received", input_data)
        
        # Run state machine until terminal state
        max_iterations = 100
        iteration = 0
        
        while self.current_state not in [AgentState.COMPLETED, AgentState.ERROR]:
            if iteration >= max_iterations:
                self.current_state = AgentState.ERROR
                self.context["error"] = "Maximum iterations reached"
                break
            
            # Execute current state handler
            next_event = self.state_handlers[self.current_state]()
            
            if next_event:
                self._transition(next_event["event"], next_event.get("data"))
            
            iteration += 1
        
        return {
            "state": self.current_state.value,
            "output": self.context.get("output"),
            "error": self.context.get("error"),
            "context": self.context
        }
    
    def _transition(self, event: str, data=None):
        """
        Transition to next state based on event.
        """
        if self.current_state not in self.transitions:
            return False
        
        state_transitions = self.transitions[self.current_state]
        
        if event not in state_transitions:
            return False
        
        # Check transition conditions
        if not self._check_transition_condition(event, data):
            return False
        
        # Perform transition
        previous_state = self.current_state
        self.current_state = state_transitions[event]
        
        # Store transition data
        if self.current_state not in self.state_data:
            self.state_data[self.current_state] = {}
        self.state_data[self.current_state]["transition_data"] = data
        self.state_data[self.current_state]["previous_state"] = previous_state
        
        return True
    
    def _check_transition_condition(self, event: str, data) -> bool:
        """
        Check if transition condition is met.
        Override in subclasses for custom conditions.
        """
        return True
    
    # State Handlers
    def _handle_idle(self):
        """Handle IDLE state."""
        # Wait for input (already received in execute)
        return None
    
    def _handle_processing(self):
        """Handle PROCESSING state."""
        input_data = self.context["input"]
        
        # Process input using LLM
        prompt = f"""
Process the following input:
{input_data}

Analyze and determine next action.
"""
        result = self.llm.generate(prompt)
        self.context["processing_result"] = result
        
        # Determine next event
        if "need more information" in result.lower():
            return {"event": "need_input", "data": result}
        elif "ready to execute" in result.lower():
            return {"event": "ready_execute", "data": result}
        else:
            return {"event": "ready_execute", "data": result}
    
    def _handle_waiting_input(self):
        """Handle WAITING_INPUT state."""
        # In practice, this would wait for user input
        # For this example, assume input is provided
        if "user_input" in self.context:
            return {"event": "input_provided", "data": self.context["user_input"]}
        else:
            # Simulate timeout
            return {"event": "timeout", "data": None}
    
    def _handle_executing(self):
        """Handle EXECUTING state."""
        processing_result = self.context.get("processing_result", "")
        
        # Execute the determined action
        try:
            execution_result = self._perform_execution(processing_result)
            self.context["execution_result"] = execution_result
            return {"event": "execution_done", "data": execution_result}
        except Exception as e:
            self.context["error"] = str(e)
            return {"event": "execution_error", "data": str(e)}
    
    def _handle_validating(self):
        """Handle VALIDATING state."""
        execution_result = self.context.get("execution_result")
        
        # Validate execution result
        validation_prompt = f"""
Validate the following execution result:
{execution_result}

Is this result correct and complete? (yes/no)
"""
        validation_response = self.llm.generate(validation_prompt)
        
        if "yes" in validation_response.lower():
            self.context["output"] = execution_result
            return {"event": "validation_pass", "data": execution_result}
        else:
            return {"event": "validation_fail", "data": validation_response}
    
    def _handle_completed(self):
        """Handle COMPLETED state."""
        return None  # Terminal state
    
    def _handle_error(self):
        """Handle ERROR state."""
        # Could implement retry logic here
        return None  # Terminal state
    
    def _perform_execution(self, processing_result):
        """
        Perform the actual execution based on processing result.
        Override in subclasses for specific execution logic.
        """
        return f"Executed: {processing_result}"
    
    def reset(self):
        """Reset agent to initial state."""
        self.current_state = AgentState.IDLE
        self.context = {}
        self.state_data = {}


# Hierarchical State Machine Example
class HierarchicalStateMachineAgent(StateMachineAgent):
    """
    Agent with hierarchical states (states within states).
    """
    
    def __init__(self, llm):
        super().__init__(llm)
        self.state_stack = []  # Stack for nested states
    
    def push_state(self, state: AgentState):
        """Push state onto stack (enter substate)."""
        if self.current_state:
            self.state_stack.append(self.current_state)
        self.current_state = state
    
    def pop_state(self):
        """Pop state from stack (exit substate)."""
        if self.state_stack:
            self.current_state = self.state_stack.pop()
        else:
            self.current_state = AgentState.IDLE


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    agent = StateMachineAgent(llm)
    
    result = agent.execute("Analyze this data and generate a report")
    print(f"Final State: {result['state']}")
    print(f"Output: {result['output']}")
```

#### State Machine Patterns

**Simple FSM**
- Linear state progression
- Clear state transitions
- Easy to understand and debug

**Hierarchical FSM**
- States contain substates
- Nested state management
- For complex behaviors

**Concurrent FSM**
- Multiple parallel state machines
- Independent execution
- Coordination through events

#### When to Use State Machine Agents

**Best For:**
- Well-defined workflows
- Predictable behavior requirements
- Complex multi-step processes
- Systems requiring state tracking

**Not Ideal For:**
- Highly dynamic environments
- Unpredictable task structures
- Simple single-step tasks
- Tasks requiring extensive planning

---

### Pipeline Agents

Pipeline agents process tasks through a series of sequential stages. Each stage transforms the input, passing results to the next stage. This pattern is common in data processing and transformation workflows.

#### Core Concept

Pipeline agents:
1. **Stage 1**: Initial processing
2. **Stage 2**: Transformation
3. **Stage 3**: Further processing
4. **Stage N**: Final output generation

Each stage:
- Receives input from previous stage
- Performs specific transformation
- Passes output to next stage

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Input Data                   │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Stage 1: Parse     │
    │   - Extract data     │
    │   - Validate format  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Stage 2: Analyze   │
    │   - Process content  │
    │   - Extract insights │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Stage 3: Transform │
    │   - Format data      │
    │   - Apply rules      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Stage 4: Generate   │
    │   - Create output     │
    │   - Format response   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │      Final Output     │
    └──────────────────────┘
```

#### Implementation Example

```python
from abc import ABC, abstractmethod
from typing import Any, List, Optional

class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process(self, input_data: Any, context: dict) -> Any:
        """
        Process input data and return output.
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data. Override for custom validation.
        """
        return True
    
    def validate_output(self, output_data: Any) -> bool:
        """
        Validate output data. Override for custom validation.
        """
        return True


class PipelineAgent:
    """
    Pipeline Agent Implementation
    
    Processes data through sequential stages.
    """
    
    def __init__(self, stages: List[PipelineStage], llm=None):
        self.stages = stages
        self.llm = llm
        self.execution_history = []
    
    def execute(self, input_data):
        """
        Execute pipeline with input data.
        """
        context = {
            "original_input": input_data,
            "stage_outputs": {},
            "errors": []
        }
        
        current_data = input_data
        
        for i, stage in enumerate(self.stages):
            try:
                # Validate input
                if not stage.validate_input(current_data):
                    error = f"Stage {i+1} ({stage.name}): Invalid input"
                    context["errors"].append(error)
                    return self._handle_pipeline_error(context, stage, error)
                
                # Process stage
                stage_output = stage.process(current_data, context)
                
                # Validate output
                if not stage.validate_output(stage_output):
                    error = f"Stage {i+1} ({stage.name}): Invalid output"
                    context["errors"].append(error)
                    return self._handle_pipeline_error(context, stage, error)
                
                # Store output
                context["stage_outputs"][stage.name] = stage_output
                current_data = stage_output
                
                # Record execution
                self.execution_history.append({
                    "stage": stage.name,
                    "stage_number": i + 1,
                    "input": current_data if i == 0 else context["stage_outputs"][self.stages[i-1].name],
                    "output": stage_output,
                    "success": True
                })
                
            except Exception as e:
                error = f"Stage {i+1} ({stage.name}): {str(e)}"
                context["errors"].append(error)
                return self._handle_pipeline_error(context, stage, error)
        
        return {
            "success": True,
            "output": current_data,
            "context": context,
            "execution_history": self.execution_history
        }
    
    def _handle_pipeline_error(self, context, failed_stage, error):
        """
        Handle pipeline execution error.
        """
        return {
            "success": False,
            "error": error,
            "failed_stage": failed_stage.name,
            "context": context,
            "execution_history": self.execution_history
        }


# Example Pipeline Stages
class ParsingStage(PipelineStage):
    """
    Stage 1: Parse and extract input data.
    """
    
    def __init__(self, llm):
        super().__init__("parsing")
        self.llm = llm
    
    def process(self, input_data, context):
        """Parse input data."""
        if isinstance(input_data, str):
            # Parse text input
            prompt = f"""
Parse the following input and extract structured information:
{input_data}

Return structured data in JSON format.
"""
            parsed = self.llm.generate(prompt)
            # In practice, parse JSON from response
            return {"parsed_data": parsed, "original": input_data}
        return {"parsed_data": input_data}


class AnalysisStage(PipelineStage):
    """
    Stage 2: Analyze parsed data.
    """
    
    def __init__(self, llm):
        super().__init__("analysis")
        self.llm = llm
    
    def process(self, input_data, context):
        """Analyze parsed data."""
        parsed_data = input_data.get("parsed_data", input_data)
        
        prompt = f"""
Analyze the following data:
{parsed_data}

Extract key insights, patterns, and important information.
"""
        analysis = self.llm.generate(prompt)
        
        return {
            **input_data,
            "analysis": analysis,
            "insights": self._extract_insights(analysis)
        }
    
    def _extract_insights(self, analysis):
        """Extract structured insights from analysis."""
        # Simplified - in practice use more sophisticated extraction
        return {"summary": analysis}


class TransformationStage(PipelineStage):
    """
    Stage 3: Transform analyzed data.
    """
    
    def __init__(self, llm, transformation_rules=None):
        super().__init__("transformation")
        self.llm = llm
        self.transformation_rules = transformation_rules or {}
    
    def process(self, input_data, context):
        """Transform analyzed data."""
        analysis = input_data.get("analysis", "")
        insights = input_data.get("insights", {})
        
        prompt = f"""
Transform the following analyzed data according to the requirements:
Analysis: {analysis}
Insights: {insights}

Apply transformations and format the data appropriately.
"""
        transformed = self.llm.generate(prompt)
        
        return {
            **input_data,
            "transformed": transformed
        }


class GenerationStage(PipelineStage):
    """
    Stage 4: Generate final output.
    """
    
    def __init__(self, llm, output_format="text"):
        super().__init__("generation")
        self.llm = llm
        self.output_format = output_format
    
    def process(self, input_data, context):
        """Generate final output."""
        transformed = input_data.get("transformed", "")
        analysis = input_data.get("analysis", "")
        
        prompt = f"""
Generate the final output based on:
Transformed Data: {transformed}
Analysis: {analysis}

Format: {self.output_format}

Create a comprehensive, well-formatted response.
"""
        final_output = self.llm.generate(prompt)
        
        return {
            "final_output": final_output,
            "metadata": {
                "format": self.output_format,
                "stages_completed": len(context["stage_outputs"]) + 1
            }
        }


# Conditional Pipeline Stage
class ConditionalStage(PipelineStage):
    """
    Stage that routes to different next stages based on condition.
    """
    
    def __init__(self, name, condition_func, true_stage, false_stage):
        super().__init__(name)
        self.condition_func = condition_func
        self.true_stage = true_stage
        self.false_stage = false_stage
    
    def process(self, input_data, context):
        """Process and determine routing."""
        condition_result = self.condition_func(input_data, context)
        context["routing_decision"] = condition_result
        
        if condition_result:
            context["next_stage"] = self.true_stage
        else:
            context["next_stage"] = self.false_stage
        
        return input_data  # Pass through


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Create pipeline stages
    stages = [
        ParsingStage(llm),
        AnalysisStage(llm),
        TransformationStage(llm),
        GenerationStage(llm, output_format="markdown")
    ]
    
    # Create pipeline agent
    agent = PipelineAgent(stages, llm)
    
    # Execute pipeline
    result = agent.execute(
        "Analyze customer feedback: 'The product is great but delivery was slow'"
    )
    
    if result["success"]:
        print("Pipeline Output:")
        print(result["output"]["final_output"])
        print("\nExecution History:")
        for entry in result["execution_history"]:
            print(f"  {entry['stage']}: {'Success' if entry['success'] else 'Failed'}")
    else:
        print(f"Pipeline Error: {result['error']}")
```

#### Pipeline Patterns

**Linear Pipeline**
- Sequential stages
- One-way data flow
- Simple and predictable

**Branching Pipeline**
- Conditional routing
- Multiple paths
- Dynamic execution

**Parallel Pipeline**
- Multiple stages run simultaneously
- Results merged
- For independent processing

**Feedback Pipeline**
- Output feeds back to earlier stages
- Iterative refinement
- For improvement loops

#### When to Use Pipeline Agents

**Best For:**
- Data transformation workflows
- Multi-stage processing
- Clear stage boundaries
- Batch processing tasks

**Not Ideal For:**
- Interactive tasks
- Tasks requiring dynamic planning
- Simple single-step operations
- Real-time streaming

---

### Iterative Refinement

Iterative Refinement agents improve outputs through multiple refinement cycles. They generate drafts, critique them, and refine iteratively until quality thresholds are met.

#### Core Concept

Iterative Refinement follows this cycle:
1. **Draft**: Generate initial output
2. **Critique**: Evaluate draft quality
3. **Refine**: Improve based on critique
4. **Repeat**: Continue until satisfactory

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Task/Request                │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  DRAFT Phase         │
    │  - Generate initial  │
    │    output            │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  CRITIQUE Phase      │
    │  - Evaluate quality  │
    │  - Identify issues  │
    │  - Score output      │
    └──────────┬───────────┘
               │
               ▼
         ┌──────────┐
         │ Quality  │
         │ Score    │
         └────┬─────┘
              │
      ┌───────┴────────┐
      │ Above         │
      │ Threshold?    │
      └───────┬────────┘
              │
      ┌───────┴────────┐
      │ YES            │ NO
      │                │
      ▼                ▼
┌──────────────┐  ┌──────────────┐
│ Return Final │  │ REFINE Phase │
│ Output       │  │ - Address    │
│              │  │   issues     │
│              │  │ - Improve    │
│              │  │   quality    │
└──────────────┘  └──────┬───────┘
                         │
                         │ [Loop Back]
                         ▼
```

#### Implementation Example

```python
class IterativeRefinementAgent:
    """
    Iterative Refinement Agent Implementation
    
    Improves outputs through draft-critique-refine cycles.
    """
    
    def __init__(self, llm, drafter, critic, refiner, 
                 quality_threshold=0.8, max_iterations=5):
        self.llm = llm
        self.drafter = drafter
        self.critic = critic
        self.refiner = refiner
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
    
    def execute(self, task):
        """
        Execute task with iterative refinement.
        """
        iteration = 0
        current_draft = None
        refinement_history = []
        
        while iteration < self.max_iterations:
            # DRAFT: Generate or refine output
            if iteration == 0:
                current_draft = self.drafter.generate(task)
            else:
                last_critique = refinement_history[-1]["critique"]
                current_draft = self.refiner.refine(
                    task, 
                    current_draft, 
                    last_critique
                )
            
            # CRITIQUE: Evaluate draft quality
            critique = self.critic.evaluate(task, current_draft, refinement_history)
            quality_score = critique["quality_score"]
            
            refinement_history.append({
                "iteration": iteration + 1,
                "draft": current_draft,
                "critique": critique,
                "quality_score": quality_score
            })
            
            # Check if quality threshold met
            if quality_score >= self.quality_threshold:
                return {
                    "output": current_draft,
                    "quality_score": quality_score,
                    "iterations": iteration + 1,
                    "refinement_history": refinement_history,
                    "converged": True
                }
            
            iteration += 1
        
        # Return best draft even if threshold not met
        best_iteration = max(
            refinement_history, 
            key=lambda x: x["quality_score"]
        )
        
        return {
            "output": best_iteration["draft"],
            "quality_score": best_iteration["quality_score"],
            "iterations": iteration,
            "refinement_history": refinement_history,
            "converged": False,
            "note": "Maximum iterations reached"
        }


class DraftGenerator:
    """
    Generates initial drafts.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate(self, task):
        """Generate initial draft."""
        prompt = f"""
Generate an initial draft for the following task:

Task: {task}

Create a comprehensive draft. It doesn't need to be perfect - 
we'll refine it iteratively.
"""
        return self.llm.generate(prompt)


class DraftCritic:
    """
    Critically evaluates drafts and provides quality scores.
    """
    
    def __init__(self, llm, evaluation_criteria):
        self.llm = llm
        self.evaluation_criteria = evaluation_criteria
    
    def evaluate(self, task, draft, previous_iterations):
        """
        Evaluate draft and provide critique with quality score.
        """
        context = self._build_evaluation_context(previous_iterations)
        
        prompt = f"""
Task: {task}

Draft to Evaluate:
{draft}

Evaluation Criteria:
{self._format_criteria()}

Previous Iterations:
{context}

Evaluate this draft:
1. Rate quality on scale 0.0 to 1.0
2. Identify specific issues
3. Provide detailed critique
4. Suggest improvements

Format response as:
QUALITY_SCORE: [0.0-1.0]
ISSUES:
- [issue 1]
- [issue 2]
CRITIQUE: [detailed critique]
IMPROVEMENTS: [suggestions]
"""
        
        response = self.llm.generate(prompt)
        return self._parse_evaluation(response)
    
    def _parse_evaluation(self, response):
        """Parse evaluation response."""
        # Extract quality score
        score_match = re.search(r'QUALITY_SCORE:\s*([\d.]+)', response)
        quality_score = float(score_match.group(1)) if score_match else 0.5
        
        # Extract issues
        issues = []
        if "ISSUES:" in response:
            issues_section = response.split("ISSUES:")[1].split("CRITIQUE:")[0]
            issue_lines = [
                line.strip()[2:] 
                for line in issues_section.split("\n") 
                if line.strip().startswith("-")
            ]
            issues = issue_lines
        
        # Extract critique
        critique = ""
        if "CRITIQUE:" in response:
            critique_section = response.split("CRITIQUE:")[1].split("IMPROVEMENTS:")[0]
            critique = critique_section.strip()
        
        # Extract improvements
        improvements = ""
        if "IMPROVEMENTS:" in response:
            improvements = response.split("IMPROVEMENTS:")[1].strip()
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "critique": critique,
            "improvements": improvements,
            "full_response": response
        }
    
    def _format_criteria(self):
        """Format evaluation criteria."""
        return "\n".join([
            f"- {criterion}: {description}"
            for criterion, description in self.evaluation_criteria.items()
        ])
    
    def _build_evaluation_context(self, previous_iterations):
        """Build context from previous iterations."""
        if not previous_iterations:
            return "None"
        
        context_parts = []
        for iter_data in previous_iterations[-3:]:  # Last 3 iterations
            context_parts.append(
                f"Iteration {iter_data['iteration']}: "
                f"Quality Score: {iter_data['quality_score']:.2f}, "
                f"Issues: {len(iter_data['critique']['issues'])}"
            )
        return "\n".join(context_parts)


class DraftRefiner:
    """
    Refines drafts based on critique.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def refine(self, task, current_draft, critique):
        """Refine draft based on critique."""
        prompt = f"""
Original Task: {task}

Current Draft:
{current_draft}

Critique:
Quality Score: {critique['quality_score']:.2f}
Issues Identified:
{self._format_issues(critique['issues'])}
Detailed Critique: {critique['critique']}
Improvement Suggestions: {critique['improvements']}

Refine the draft to address all identified issues and improve quality. 
Maintain what's good, fix what's wrong, and incorporate improvements.
"""
        return self.llm.generate(prompt)
    
    def _format_issues(self, issues):
        """Format issues list."""
        if not issues:
            return "None"
        return "\n".join([f"- {issue}" for issue in issues])


# Adaptive Refinement Variant
class AdaptiveRefinementAgent(IterativeRefinementAgent):
    """
    Agent that adapts refinement strategy based on progress.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_threshold = 0.1  # Minimum improvement per iteration
    
    def execute(self, task):
        """Execute with adaptive refinement."""
        result = super().execute(task)
        
        # Analyze refinement trajectory
        if len(result["refinement_history"]) > 1:
            improvements = self._calculate_improvements(result["refinement_history"])
            
            # Adapt strategy if not improving
            if not improvements["improving"]:
                result["adaptation_applied"] = True
                result["note"] = "Adaptive strategy applied due to slow convergence"
        
        return result
    
    def _calculate_improvements(self, history):
        """Calculate if quality is improving."""
        scores = [h["quality_score"] for h in history]
        
        if len(scores) < 2:
            return {"improving": True, "rate": 0.0}
        
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        avg_improvement = sum(improvements) / len(improvements)
        
        return {
            "improving": avg_improvement >= self.adaptation_threshold,
            "rate": avg_improvement
        }


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    evaluation_criteria = {
        "Accuracy": "Information must be factually correct",
        "Completeness": "All aspects must be covered",
        "Clarity": "Writing must be clear and understandable",
        "Structure": "Content must be well-organized",
        "Relevance": "Content must be relevant to task"
    }
    
    drafter = DraftGenerator(llm)
    critic = DraftCritic(llm, evaluation_criteria)
    refiner = DraftRefiner(llm)
    
    agent = IterativeRefinementAgent(
        llm, drafter, critic, refiner,
        quality_threshold=0.85,
        max_iterations=5
    )
    
    result = agent.execute(
        "Write a comprehensive guide to machine learning for beginners"
    )
    
    print(f"Quality Score: {result['quality_score']:.2f}")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
    print(f"\nFinal Output:\n{result['output']}")
```

#### Refinement Strategies

**Quality-Based Refinement**
- Refine until quality threshold met
- Focus on overall improvement
- For general quality goals

**Issue-Based Refinement**
- Address specific issues each iteration
- Prioritize critical issues
- For targeted improvements

**Adaptive Refinement**
- Adjust strategy based on progress
- Change approach if stuck
- For complex refinements

#### When to Use Iterative Refinement

**Best For:**
- High-quality output requirements
- Creative writing tasks
- Complex problem-solving
- Tasks benefiting from multiple passes

**Not Ideal For:**
- Real-time responses
- Simple, deterministic tasks
- Cost-sensitive applications
- Tasks with clear single solutions

---

## Multi-Agent Architectures

Multi-agent systems involve multiple agents working together to solve problems. These architectures enable specialization, parallelization, and complex coordination.

### Master-Worker Pattern

The Master-Worker pattern uses a coordinator agent (master) that distributes tasks to specialized worker agents and aggregates their results.

#### Core Concept

Master-Worker systems:
1. **Master**: Receives task, decomposes it, assigns to workers
2. **Workers**: Execute assigned subtasks independently
3. **Aggregation**: Master collects and combines worker results
4. **Coordination**: Master manages worker lifecycle and dependencies

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Task Received               │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │   MASTER Agent       │
    │   - Decompose task   │
    │   - Plan execution   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Task Decomposition │
    │   [Task 1, 2, 3, 4] │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────┐
│ WORKER 1     │  │ WORKER 2     │
│ - Execute    │  │ - Execute    │
│   Task 1     │  │   Task 2     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ WORKER 3     │  │ WORKER 4     │
│ - Execute    │  │ - Execute    │
│   Task 3     │  │   Task 4     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
    ┌──────────────────────┐
    │   MASTER Agent       │
    │   - Collect results  │
    │   - Aggregate        │
    │   - Generate final   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │      Final Output     │
    └──────────────────────┘
```

#### Implementation Example

```python
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

class WorkerAgent:
    """
    Worker agent that executes assigned tasks.
    """
    
    def __init__(self, worker_id: str, llm, capabilities: List[str]):
        self.worker_id = worker_id
        self.llm = llm
        self.capabilities = capabilities
        self.status = "idle"
    
    def execute_task(self, task_description: str, context: dict = None):
        """
        Execute assigned task.
        """
        self.status = "working"
        
        try:
            prompt = f"""
You are Worker {self.worker_id} with capabilities: {', '.join(self.capabilities)}

Task: {task_description}

Context: {context or 'None'}

Execute this task and provide a detailed result.
"""
            result = self.llm.generate(prompt)
            
            self.status = "idle"
            return {
                "worker_id": self.worker_id,
                "success": True,
                "result": result,
                "task": task_description
            }
        except Exception as e:
            self.status = "idle"
            return {
                "worker_id": self.worker_id,
                "success": False,
                "error": str(e),
                "task": task_description
            }
    
    def can_handle(self, task_type: str) -> bool:
        """Check if worker can handle task type."""
        return task_type in self.capabilities


class MasterAgent:
    """
    Master agent that coordinates worker agents.
    """
    
    def __init__(self, llm, workers: List[WorkerAgent]):
        self.llm = llm
        self.workers = workers
        self.task_queue = []
        self.results = {}
    
    def execute(self, task: str):
        """
        Execute task using worker agents.
        """
        # Decompose task
        subtasks = self._decompose_task(task)
        
        # Assign tasks to workers
        assignments = self._assign_tasks(subtasks)
        
        # Execute tasks (can be parallelized)
        worker_results = self._execute_assignments(assignments)
        
        # Aggregate results
        final_result = self._aggregate_results(task, subtasks, worker_results)
        
        return {
            "task": task,
            "subtasks": subtasks,
            "assignments": assignments,
            "worker_results": worker_results,
            "final_result": final_result
        }
    
    def _decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """
        Decompose main task into subtasks.
        """
        prompt = f"""
Decompose the following task into independent subtasks:

Task: {task}

Available Worker Capabilities:
{self._format_worker_capabilities()}

Create subtasks that can be executed in parallel by different workers.
Each subtask should be:
1. Specific and actionable
2. Independent (can run in parallel)
3. Assigned to appropriate worker type

Format as JSON:
{{
    "subtasks": [
        {{
            "id": 1,
            "description": "subtask description",
            "worker_type": "capability needed",
            "dependencies": []
        }}
    ]
}}
"""
        response = self.llm.generate(prompt)
        parsed = json.loads(response)
        return parsed.get("subtasks", [])
    
    def _format_worker_capabilities(self):
        """Format worker capabilities for prompt."""
        capabilities = set()
        for worker in self.workers:
            capabilities.update(worker.capabilities)
        return ", ".join(capabilities)
    
    def _assign_tasks(self, subtasks: List[Dict]) -> List[Dict]:
        """
        Assign subtasks to available workers.
        """
        assignments = []
        
        for subtask in subtasks:
            worker_type_needed = subtask.get("worker_type", "")
            
            # Find available worker with matching capability
            assigned_worker = None
            for worker in self.workers:
                if worker.can_handle(worker_type_needed) and worker.status == "idle":
                    assigned_worker = worker
                    break
            
            if assigned_worker:
                assignments.append({
                    "subtask": subtask,
                    "worker": assigned_worker,
                    "status": "assigned"
                })
            else:
                # No available worker - assign to first capable worker
                for worker in self.workers:
                    if worker.can_handle(worker_type_needed):
                        assignments.append({
                            "subtask": subtask,
                            "worker": worker,
                            "status": "queued"
                        })
                        break
        
        return assignments
    
    def _execute_assignments(self, assignments: List[Dict]) -> List[Dict]:
        """
        Execute assigned tasks (can be parallelized).
        """
        results = []
        
        # Execute in parallel using thread pool
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = []
            for assignment in assignments:
                worker = assignment["worker"]
                subtask = assignment["subtask"]
                future = executor.submit(
                    worker.execute_task,
                    subtask["description"],
                    {"subtask_id": subtask["id"]}
                )
                futures.append((assignment, future))
            
            # Collect results
            for assignment, future in futures:
                result = future.result()
                results.append(result)
        
        return results
    
    def _aggregate_results(self, original_task: str, subtasks: List[Dict], 
                          worker_results: List[Dict]) -> str:
        """
        Aggregate worker results into final output.
        """
        # Build context from results
        results_context = []
        for i, result in enumerate(worker_results):
            subtask = subtasks[i] if i < len(subtasks) else {}
            results_context.append(
                f"Subtask {subtask.get('id', i+1)} ({subtask.get('description', 'N/A')}):\n"
                f"Result: {result.get('result', 'N/A')}"
            )
        
        prompt = f"""
Original Task: {original_task}

Subtask Results:
{chr(10).join(results_context)}

Synthesize these results into a comprehensive final answer that addresses 
the original task. Integrate all subtask results cohesively.
"""
        
        final_result = self.llm.generate(prompt)
        return final_result


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Create specialized workers
    workers = [
        WorkerAgent("worker1", llm, ["research", "analysis"]),
        WorkerAgent("worker2", llm, ["writing", "editing"]),
        WorkerAgent("worker3", llm, ["data_processing", "calculation"]),
        WorkerAgent("worker4", llm, ["research", "summarization"])
    ]
    
    # Create master agent
    master = MasterAgent(llm, workers)
    
    # Execute complex task
    result = master.execute(
        "Research the top 3 programming languages in 2024, analyze their "
        "performance characteristics, and create a comprehensive comparison report."
    )
    
    print("Final Result:")
    print(result["final_result"])
    print("\nWorker Results:")
    for wr in result["worker_results"]:
        print(f"  {wr['worker_id']}: {'Success' if wr['success'] else 'Failed'}")
```

#### Task Distribution Strategies

**Round-Robin**
- Distribute tasks evenly
- Simple and fair
- May not optimize for capability

**Capability-Based**
- Match tasks to worker capabilities
- Better quality
- Requires capability matching

**Load-Based**
- Consider worker load
- Better utilization
- More complex

#### Result Aggregation Strategies

**Simple Concatenation**
- Combine results sequentially
- Fast but may lack coherence

**LLM-Based Synthesis**
- Use LLM to synthesize results
- Better coherence
- Higher cost

**Structured Merging**
- Merge structured results
- For structured outputs
- Requires schema

#### When to Use Master-Worker

**Best For:**
- Parallelizable tasks
- Specialized worker capabilities
- Large-scale task processing
- Independent subtasks

**Not Ideal For:**
- Sequential dependencies
- Small, simple tasks
- Tight coupling requirements
- Real-time single queries

---

### Peer-to-Peer

Peer-to-Peer architectures involve agents that collaborate as equals, without a central coordinator. Agents communicate directly with each other to solve problems collaboratively.

#### Core Concept

Peer-to-Peer systems:
1. **No Hierarchy**: All agents are equal peers
2. **Direct Communication**: Agents communicate directly
3. **Collaborative Problem-Solving**: Work together to solve problems
4. **Consensus Building**: Reach agreement through negotiation

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Task/Problem                │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────┐
│  PEER Agent 1│  │  PEER Agent 2│
│  - Analyze   │  │  - Analyze   │
│  - Propose   │  │  - Propose   │
└──────┬───────┘  └──────┬───────┘
       │                 │
       │  [Communicate]   │
       │◄───────────────►│
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  PEER Agent 3│  │  PEER Agent 4│
│  - Contribute│  │  - Contribute│
└──────┬───────┘  └──────┬───────┘
       │                 │
       │  [Collaborate]  │
       │◄───────────────►│
       │                 │
       └────────┬────────┘
                │
                ▼
    ┌──────────────────────┐
    │   Consensus/Result    │
    └──────────────────────┘
```

#### Implementation Example

```python
class PeerAgent:
    """
    Peer agent in a peer-to-peer system.
    """
    
    def __init__(self, agent_id: str, llm, expertise: str):
        self.agent_id = agent_id
        self.llm = llm
        self.expertise = expertise
        self.peers = []
        self.message_queue = []
    
    def add_peer(self, peer):
        """Add a peer agent for communication."""
        if peer not in self.peers:
            self.peers.append(peer)
            peer.peers.append(self)
    
    def collaborate(self, task: str):
        """
        Collaborate with peers to solve task.
        """
        # Initial analysis
        my_contribution = self._analyze_task(task)
        
        # Share with peers
        messages = self._broadcast_contribution(task, my_contribution)
        
        # Process peer contributions
        peer_contributions = self._receive_contributions()
        
        # Synthesize collaborative solution
        solution = self._synthesize_solution(task, my_contribution, peer_contributions)
        
        return {
            "agent_id": self.agent_id,
            "my_contribution": my_contribution,
            "peer_contributions": peer_contributions,
            "solution": solution
        }
    
    def _analyze_task(self, task: str) -> str:
        """Analyze task from this agent's perspective."""
        prompt = f"""
You are Agent {self.agent_id} with expertise in {self.expertise}.

Task: {task}

Analyze this task from your expertise perspective and provide your contribution.
"""
        return self.llm.generate(prompt)
    
    def _broadcast_contribution(self, task: str, contribution: str):
        """Broadcast contribution to all peers."""
        messages = []
        for peer in self.peers:
            message = {
                "from": self.agent_id,
                "task": task,
                "contribution": contribution,
                "expertise": self.expertise
            }
            peer.message_queue.append(message)
            messages.append(message)
        return messages
    
    def _receive_contributions(self):
        """Receive contributions from peers."""
        contributions = []
        while self.message_queue:
            message = self.message_queue.pop(0)
            contributions.append(message)
        return contributions
    
    def _synthesize_solution(self, task: str, my_contribution: str, 
                            peer_contributions: list) -> str:
        """Synthesize solution from all contributions."""
        contributions_text = "\n\n".join([
            f"Agent {c['from']} ({c['expertise']}): {c['contribution']}"
            for c in peer_contributions
        ])
        
        prompt = f"""
Task: {task}

My Contribution ({self.expertise}):
{my_contribution}

Peer Contributions:
{contributions_text}

Synthesize a comprehensive solution that integrates all contributions.
"""
        return self.llm.generate(prompt)


class PeerToPeerSystem:
    """
    Peer-to-peer multi-agent system.
    """
    
    def __init__(self, agents: List[PeerAgent]):
        self.agents = agents
        self._establish_connections()
    
    def _establish_connections(self):
        """Establish peer connections between agents."""
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                agent1.add_peer(agent2)
    
    def solve(self, task: str):
        """
        Solve task through peer collaboration.
        """
        # Each agent collaborates
        results = []
        for agent in self.agents:
            result = agent.collaborate(task)
            results.append(result)
        
        # Aggregate solutions (could use consensus mechanism)
        final_solution = self._reach_consensus(results)
        
        return {
            "task": task,
            "agent_results": results,
            "consensus_solution": final_solution
        }
    
    def _reach_consensus(self, results: List[Dict]) -> str:
        """Reach consensus from agent solutions."""
        # Simple consensus: use LLM to synthesize
        solutions_text = "\n\n".join([
            f"Agent {r['agent_id']}: {r['solution']}"
            for r in results
        ])
        
        # Use first agent's LLM for consensus
        if self.agents:
            prompt = f"""
Multiple agents have proposed solutions. Reach a consensus:

{solutions_text}

Synthesize the best solution that incorporates the strongest aspects 
of each proposal.
"""
            return self.agents[0].llm.generate(prompt)
        return "No consensus reached"


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Create peer agents with different expertise
    agents = [
        PeerAgent("agent1", llm, "data_analysis"),
        PeerAgent("agent2", llm, "user_experience"),
        PeerAgent("agent3", llm, "technical_architecture"),
        PeerAgent("agent4", llm, "business_strategy")
    ]
    
    # Create peer-to-peer system
    system = PeerToPeerSystem(agents)
    
    # Solve task collaboratively
    result = system.solve(
        "Design a new feature for our mobile app that improves user engagement"
    )
    
    print("Consensus Solution:")
    print(result["consensus_solution"])
```

#### Communication Patterns

**Broadcast**
- Send message to all peers
- Simple but inefficient
- For announcements

**Point-to-Point**
- Direct communication between peers
- More efficient
- For specific exchanges

**Gossip Protocol**
- Peers share information with subset
- Scalable
- For large networks

#### Consensus Mechanisms

**Voting**
- Agents vote on solutions
- Majority wins
- Simple and fair

**Weighted Consensus**
- Weight votes by expertise
- Better quality
- Requires trust

**Iterative Refinement**
- Refine solution through rounds
- Higher quality
- More time-consuming

#### When to Use Peer-to-Peer

**Best For:**
- Collaborative problem-solving
- Distributed expertise
- No central authority needed
- Equal agent capabilities

**Not Ideal For:**
- Tasks requiring coordination
- Clear hierarchy benefits
- Single point of control needed
- Simple, linear tasks

---

### Hierarchical Teams

Hierarchical Teams organize agents in a tree structure with managers and specialists. Managers coordinate specialists, who execute specific tasks.

#### Core Concept

Hierarchical Teams:
1. **Managers**: Coordinate and delegate
2. **Specialists**: Execute specialized tasks
3. **Reporting**: Specialists report to managers
4. **Delegation**: Managers delegate to appropriate specialists

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Task Received               │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │   MANAGER Agent      │
    │   - Analyze task     │
    │   - Plan execution   │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────┐
│ SPECIALIST 1 │  │ SPECIALIST 2 │
│ (Expertise A)│  │ (Expertise B)│
│ - Execute    │  │ - Execute    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       │ [Report Back]   │
       └────────┬────────┘
                │
                ▼
    ┌──────────────────────┐
    │   MANAGER Agent      │
    │   - Synthesize       │
    │   - Generate final   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │      Final Output     │
    └──────────────────────┘
```

#### Implementation Example

```python
class SpecialistAgent:
    """
    Specialist agent with specific expertise.
    """
    
    def __init__(self, specialist_id: str, llm, expertise: str):
        self.specialist_id = specialist_id
        self.llm = llm
        self.expertise = expertise
        self.manager = None
    
    def set_manager(self, manager):
        """Set manager agent."""
        self.manager = manager
    
    def execute_task(self, task_description: str):
        """Execute assigned specialized task."""
        prompt = f"""
You are Specialist {self.specialist_id} with expertise in {self.expertise}.

Task: {task_description}

Execute this task using your specialized knowledge.
"""
        result = self.llm.generate(prompt)
        
        # Report to manager
        if self.manager:
            self.manager.receive_report(self.specialist_id, result)
        
        return {
            "specialist_id": self.specialist_id,
            "expertise": self.expertise,
            "result": result
        }


class ManagerAgent:
    """
    Manager agent that coordinates specialists.
    """
    
    def __init__(self, manager_id: str, llm):
        self.manager_id = manager_id
        self.llm = llm
        self.specialists = {}
        self.reports = {}
    
    def add_specialist(self, specialist: SpecialistAgent):
        """Add a specialist under this manager."""
        self.specialists[specialist.expertise] = specialist
        specialist.set_manager(self)
    
    def execute(self, task: str):
        """
        Execute task by delegating to specialists.
        """
        # Analyze task and determine needed specialists
        needed_expertise = self._determine_expertise_needed(task)
        
        # Delegate to specialists
        specialist_tasks = self._delegate_tasks(task, needed_expertise)
        
        # Collect results
        results = {}
        for expertise, specialist_task in specialist_tasks.items():
            specialist = self.specialists.get(expertise)
            if specialist:
                result = specialist.execute_task(specialist_task)
                results[expertise] = result
        
        # Synthesize final result
        final_result = self._synthesize_results(task, results)
        
        return {
            "task": task,
            "specialist_results": results,
            "final_result": final_result
        }
    
    def _determine_expertise_needed(self, task: str) -> List[str]:
        """Determine which expertise areas are needed."""
        available_expertise = list(self.specialists.keys())
        
        prompt = f"""
Task: {task}

Available Specialist Expertise: {', '.join(available_expertise)}

Determine which expertise areas are needed to complete this task.
Return as JSON list: ["expertise1", "expertise2", ...]
"""
        response = self.llm.generate(prompt)
        try:
            return json.loads(response)
        except:
            # Fallback: return all if parsing fails
            return available_expertise
    
    def _delegate_tasks(self, main_task: str, expertise_needed: List[str]) -> Dict[str, str]:
        """Delegate specific tasks to specialists."""
        tasks = {}
        
        for expertise in expertise_needed:
            prompt = f"""
Main Task: {main_task}

Create a specific subtask for a specialist with expertise in {expertise}.
The subtask should be focused and actionable.
"""
            subtask = self.llm.generate(prompt)
            tasks[expertise] = subtask
        
        return tasks
    
    def receive_report(self, specialist_id: str, result: str):
        """Receive report from specialist."""
        self.reports[specialist_id] = result
    
    def _synthesize_results(self, original_task: str, results: Dict[str, Dict]) -> str:
        """Synthesize specialist results into final output."""
        results_text = "\n\n".join([
            f"{expertise} Specialist: {r['result']}"
            for expertise, r in results.items()
        ])
        
        prompt = f"""
Original Task: {original_task}

Specialist Results:
{results_text}

Synthesize these specialist contributions into a comprehensive final solution.
"""
        return self.llm.generate(prompt)


# Multi-Level Hierarchy Example
class HierarchicalTeam:
    """
    Multi-level hierarchical team structure.
    """
    
    def __init__(self, top_manager: ManagerAgent):
        self.top_manager = top_manager
        self.levels = {}
    
    def add_level(self, level: int, managers: List[ManagerAgent]):
        """Add a level of managers."""
        self.levels[level] = managers
    
    def execute(self, task: str):
        """Execute task through hierarchy."""
        return self.top_manager.execute(task)


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Create manager
    manager = ManagerAgent("manager1", llm)
    
    # Create specialists
    specialists = [
        SpecialistAgent("spec1", llm, "frontend_development"),
        SpecialistAgent("spec2", llm, "backend_development"),
        SpecialistAgent("spec3", llm, "database_design"),
        SpecialistAgent("spec4", llm, "security")
    ]
    
    # Add specialists to manager
    for spec in specialists:
        manager.add_specialist(spec)
    
    # Execute task
    result = manager.execute(
        "Design and implement a secure web application with user authentication"
    )
    
    print("Final Result:")
    print(result["final_result"])
```

#### Hierarchy Patterns

**Flat Hierarchy**
- One manager, multiple specialists
- Simple structure
- Good for small teams

**Multi-Level Hierarchy**
- Managers have sub-managers
- Scalable
- For large organizations

**Matrix Structure**
- Specialists report to multiple managers
- Flexible
- More complex coordination

#### When to Use Hierarchical Teams

**Best For:**
- Clear specialization needs
- Coordinated multi-expertise tasks
- Scalable team structures
- Tasks requiring management

**Not Ideal For:**
- Simple, single-expertise tasks
- Peer collaboration benefits
- Flat organizational preference
- Real-time collaborative work

---

### Debate/Adversarial

Debate/Adversarial systems use competing agents with different perspectives to improve solution quality through argumentation and critique.

#### Core Concept

Debate systems:
1. **Proponent**: Argues for a solution
2. **Opponent**: Critiques and challenges
3. **Refinement**: Solutions improve through debate
4. **Judgment**: Final evaluation selects best solution

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Problem/Task                │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────┐
│ PROPONENT    │  │  OPPONENT    │
│ Agent        │  │  Agent        │
│ - Propose    │  │  - Critique  │
│   solution   │  │  - Challenge  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       │ [Argument]      │
       │◄───────────────►│
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ PROPONENT    │  │  OPPONENT    │
│ - Refine     │  │  - Counter   │
│   solution   │  │  - Point out │
│              │  │    flaws     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       │ [Continue]      │
       │◄───────────────►│
       │                 │
       └────────┬────────┘
                │
                ▼
    ┌──────────────────────┐
    │   JUDGE Agent        │
    │   - Evaluate         │
    │   - Select best      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Final Solution     │
    └──────────────────────┘
```

#### Implementation Example

```python
class ProponentAgent:
    """
    Agent that proposes and defends solutions.
    """
    
    def __init__(self, agent_id: str, llm, perspective: str):
        self.agent_id = agent_id
        self.llm = llm
        self.perspective = perspective
        self.solution_history = []
    
    def propose_solution(self, problem: str):
        """Propose initial solution."""
        prompt = f"""
You are a Proponent Agent with perspective: {self.perspective}

Problem: {problem}

Propose a solution from your perspective. Be thorough and well-reasoned.
"""
        solution = self.llm.generate(prompt)
        self.solution_history.append({
            "round": 1,
            "solution": solution,
            "type": "initial"
        })
        return solution
    
    def defend_solution(self, problem: str, current_solution: str, critique: str):
        """Defend and refine solution based on critique."""
        prompt = f"""
Problem: {problem}

Your Current Solution:
{current_solution}

Critique Received:
{critique}

Defend your solution and refine it to address valid criticisms while 
maintaining your core perspective.
"""
        refined_solution = self.llm.generate(prompt)
        self.solution_history.append({
            "round": len(self.solution_history) + 1,
            "solution": refined_solution,
            "type": "refined",
            "critique": critique
        })
        return refined_solution


class OpponentAgent:
    """
    Agent that critiques and challenges solutions.
    """
    
    def __init__(self, agent_id: str, llm, perspective: str):
        self.agent_id = agent_id
        self.llm = llm
        self.perspective = perspective
        self.critique_history = []
    
    def critique_solution(self, problem: str, solution: str):
        """Critique a proposed solution."""
        prompt = f"""
You are an Opponent Agent with perspective: {self.perspective}

Problem: {problem}

Proposed Solution:
{solution}

Critique this solution from your perspective. Identify weaknesses, 
flaws, and areas for improvement. Be constructive but thorough.
"""
        critique = self.llm.generate(prompt)
        self.critique_history.append({
            "solution": solution,
            "critique": critique
        })
        return critique


class JudgeAgent:
    """
    Agent that evaluates solutions and selects the best.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_solutions(self, problem: str, solutions: List[Dict[str, str]]):
        """Evaluate multiple solutions and select the best."""
        solutions_text = "\n\n".join([
            f"Solution {i+1} (from {s['agent']}):\n{s['solution']}"
            for i, s in enumerate(solutions)
        ])
        
        prompt = f"""
Problem: {problem}

Proposed Solutions:
{solutions_text}

Evaluate each solution and select the best one. Consider:
1. How well it addresses the problem
2. Feasibility and practicality
3. Completeness and thoroughness
4. Quality of reasoning

Provide:
1. Evaluation of each solution
2. Ranking of solutions
3. Selected best solution with justification
"""
        evaluation = self.llm.generate(prompt)
        return evaluation


class DebateSystem:
    """
    Debate/Adversarial multi-agent system.
    """
    
    def __init__(self, proponents: List[ProponentAgent], 
                 opponents: List[OpponentAgent], judge: JudgeAgent,
                 max_rounds: int = 3):
        self.proponents = proponents
        self.opponents = opponents
        self.judge = judge
        self.max_rounds = max_rounds
    
    def debate(self, problem: str):
        """
        Conduct debate to find best solution.
        """
        # Round 1: Initial proposals
        solutions = []
        for proponent in self.proponents:
            solution = proponent.propose_solution(problem)
            solutions.append({
                "agent": proponent.agent_id,
                "solution": solution,
                "round": 1
            })
        
        # Debate rounds
        for round_num in range(2, self.max_rounds + 1):
            # Opponents critique all solutions
            critiques = {}
            for opponent in self.opponents:
                for i, solution_data in enumerate(solutions):
                    critique = opponent.critique_solution(problem, solution_data["solution"])
                    if i not in critiques:
                        critiques[i] = []
                    critiques[i].append(critique)
            
            # Proponents defend and refine
            refined_solutions = []
            for i, solution_data in enumerate(solutions):
                proponent = self.proponents[i]
                combined_critique = "\n\n".join(critiques.get(i, []))
                refined = proponent.defend_solution(
                    problem,
                    solution_data["solution"],
                    combined_critique
                )
                refined_solutions.append({
                    "agent": proponent.agent_id,
                    "solution": refined,
                    "round": round_num
                })
            solutions = refined_solutions
        
        # Judge evaluates final solutions
        final_evaluation = self.judge.evaluate_solutions(problem, solutions)
        
        return {
            "problem": problem,
            "final_solutions": solutions,
            "judge_evaluation": final_evaluation,
            "rounds": self.max_rounds
        }


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Create proponents with different perspectives
    proponents = [
        ProponentAgent("prop1", llm, "user-centric design"),
        ProponentAgent("prop2", llm, "technical efficiency"),
        ProponentAgent("prop3", llm, "cost-effectiveness")
    ]
    
    # Create opponents
    opponents = [
        OpponentAgent("opp1", llm, "critical analysis"),
        OpponentAgent("opp2", llm, "practical feasibility")
    ]
    
    # Create judge
    judge = JudgeAgent(llm)
    
    # Create debate system
    debate = DebateSystem(proponents, opponents, judge, max_rounds=3)
    
    # Conduct debate
    result = debate.debate(
        "Design a new feature for improving customer retention"
    )
    
    print("Judge Evaluation:")
    print(result["judge_evaluation"])
```

#### Debate Strategies

**Structured Debate**
- Fixed rounds and format
- Predictable
- Good for formal evaluation

**Free-Form Debate**
- Unstructured argumentation
- More natural
- Harder to control

**Multi-Perspective Debate**
- Multiple proponents
- Diverse viewpoints
- Richer solutions

#### When to Use Debate/Adversarial

**Best For:**
- High-stakes decisions
- Complex problems
- Quality-critical solutions
- Multiple valid approaches

**Not Ideal For:**
- Simple, clear-cut problems
- Time-sensitive tasks
- Cost-sensitive applications
- Single obvious solution

---

### Ensemble

Ensemble systems use multiple agents to solve the same problem independently, then combine their results through voting, averaging, or other aggregation methods.

#### Core Concept

Ensemble systems:
1. **Multiple Agents**: Each solves problem independently
2. **Diverse Approaches**: Different methods/perspectives
3. **Aggregation**: Combine results through consensus
4. **Quality Improvement**: Ensemble often outperforms individuals

#### Flow Diagram

```
┌─────────────────────────────────────┐
│         Problem/Task                │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────┐
│  Agent 1     │  │  Agent 2     │
│  (Method A)  │  │  (Method B)  │
│  - Solve     │  │  - Solve     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  Agent 3     │  │  Agent 4     │
│  (Method C)  │  │  (Method D)  │
│  - Solve     │  │  - Solve     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                │
                ▼
    ┌──────────────────────┐
    │   Aggregation        │
    │   - Voting           │
    │   - Averaging        │
    │   - Weighted         │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Ensemble Result    │
    └──────────────────────┘
```

#### Implementation Example

```python
class EnsembleAgent:
    """
    Ensemble system with multiple agents.
    """
    
    def __init__(self, agents: List[Any], aggregation_method: str = "voting"):
        self.agents = agents
        self.aggregation_method = aggregation_method
    
    def solve(self, problem: str):
        """
        Solve problem using ensemble of agents.
        """
        # Each agent solves independently
        individual_results = []
        for agent in self.agents:
            result = agent.solve(problem)
            individual_results.append({
                "agent_id": getattr(agent, 'agent_id', 'unknown'),
                "result": result,
                "confidence": getattr(agent, 'confidence', 1.0)
            })
        
        # Aggregate results
        ensemble_result = self._aggregate_results(problem, individual_results)
        
        return {
            "problem": problem,
            "individual_results": individual_results,
            "ensemble_result": ensemble_result,
            "aggregation_method": self.aggregation_method
        }
    
    def _aggregate_results(self, problem: str, results: List[Dict]) -> str:
        """Aggregate individual results based on method."""
        if self.aggregation_method == "voting":
            return self._voting_aggregation(results)
        elif self.aggregation_method == "averaging":
            return self._averaging_aggregation(results)
        elif self.aggregation_method == "weighted":
            return self._weighted_aggregation(results)
        elif self.aggregation_method == "llm_synthesis":
            return self._llm_synthesis_aggregation(problem, results)
        else:
            return self._simple_concatenation(results)
    
    def _voting_aggregation(self, results: List[Dict]) -> str:
        """Aggregate by voting (for discrete answers)."""
        # Count votes for each unique answer
        votes = {}
        for result_data in results:
            answer = str(result_data["result"])
            votes[answer] = votes.get(answer, 0) + result_data.get("confidence", 1.0)
        
        # Return most voted answer
        if votes:
            return max(votes.items(), key=lambda x: x[1])[0]
        return "No consensus"
    
    def _averaging_aggregation(self, results: List[Dict]) -> str:
        """Aggregate by averaging (for numerical results)."""
        try:
            values = [float(r["result"]) for r in results]
            avg = sum(values) / len(values)
            return str(avg)
        except:
            return self._simple_concatenation(results)
    
    def _weighted_aggregation(self, results: List[Dict]) -> str:
        """Aggregate using weighted combination."""
        # Weight by confidence
        total_weight = sum(r.get("confidence", 1.0) for r in results)
        
        if total_weight == 0:
            return self._simple_concatenation(results)
        
        # For text results, use LLM synthesis with weights
        # For numerical, use weighted average
        weighted_results = []
        for r in results:
            weight = r.get("confidence", 1.0) / total_weight
            weighted_results.append(f"[Weight: {weight:.2f}] {r['result']}")
        
        return "\n".join(weighted_results)
    
    def _llm_synthesis_aggregation(self, problem: str, results: List[Dict]) -> str:
        """Aggregate using LLM synthesis."""
        results_text = "\n\n".join([
            f"Agent {r['agent_id']} (Confidence: {r.get('confidence', 1.0):.2f}):\n{r['result']}"
            for r in results
        ])
        
        # Use first agent's LLM for synthesis
        if self.agents:
            prompt = f"""
Problem: {problem}

Individual Agent Results:
{results_text}

Synthesize these results into a comprehensive ensemble solution that 
incorporates the best aspects of each individual result.
"""
            return self.agents[0].llm.generate(prompt)
        return self._simple_concatenation(results)
    
    def _simple_concatenation(self, results: List[Dict]) -> str:
        """Simple concatenation of results."""
        return "\n\n".join([
            f"Agent {r['agent_id']}: {r['result']}"
            for r in results
        ])


# Usage Example
if __name__ == "__main__":
    llm = LLMClient()
    
    # Create diverse agents (simplified - in practice use different architectures)
    agents = [
        ReActAgent(llm, []),  # ReAct approach
        ChainOfThoughtAgent(llm),  # CoT approach
        PlanAndExecuteAgent(llm, MockExecutor())  # Plan-and-Execute approach
    ]
    
    # Create ensemble
    ensemble = EnsembleAgent(agents, aggregation_method="llm_synthesis")
    
    # Solve problem
    result = ensemble.solve(
        "What are the main challenges in deploying AI systems in production?"
    )
    
    print("Ensemble Result:")
    print(result["ensemble_result"])
```

#### Aggregation Methods

**Majority Voting**
- Most common answer wins
- Simple and fast
- For discrete answers

**Weighted Voting**
- Weight votes by confidence
- Better quality
- Requires confidence scores

**Averaging**
- Average numerical results
- For continuous values
- Reduces variance

**LLM Synthesis**
- Use LLM to combine results
- Best coherence
- Higher cost

#### When to Use Ensemble

**Best For:**
- High-accuracy requirements
- Uncertain or complex problems
- Diverse solution approaches available
- Quality over speed

**Not Ideal For:**
- Simple, clear problems
- Real-time requirements
- Cost-sensitive applications
- Single best method known

---

## Orchestration Patterns

Orchestration patterns define how agent workflows are structured and executed. They determine the flow of control and data through agent systems.

### Sequential Orchestration

Sequential orchestration executes agents one after another in a fixed order, with each agent's output feeding into the next.

#### Implementation Example

```python
class SequentialOrchestrator:
    """
    Orchestrates agents in sequential order.
    """
    
    def __init__(self, agents: List[Any]):
        self.agents = agents
    
    def execute(self, initial_input):
        """Execute agents sequentially."""
        current_data = initial_input
        execution_log = []
        
        for i, agent in enumerate(self.agents):
            result = agent.execute(current_data)
            execution_log.append({
                "step": i + 1,
                "agent": agent.__class__.__name__,
                "input": current_data,
                "output": result
            })
            current_data = result
        
        return {
            "final_output": current_data,
            "execution_log": execution_log
        }
```

### Parallel Orchestration

Parallel orchestration executes multiple agents simultaneously and combines their results.

#### Implementation Example

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelOrchestrator:
    """
    Orchestrates agents in parallel.
    """
    
    def __init__(self, agents: List[Any], merge_strategy: str = "combine"):
        self.agents = agents
        self.merge_strategy = merge_strategy
    
    def execute(self, input_data):
        """Execute agents in parallel."""
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(agent.execute, input_data): agent
                for agent in self.agents
            }
            
            results = []
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    result = future.result()
                    results.append({
                        "agent": agent.__class__.__name__,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "agent": agent.__class__.__name__,
                        "error": str(e)
                    })
        
        # Merge results
        merged_result = self._merge_results(results)
        
        return {
            "merged_output": merged_result,
            "individual_results": results
        }
    
    def _merge_results(self, results: List[Dict]) -> str:
        """Merge parallel results."""
        if self.merge_strategy == "combine":
            return "\n\n".join([
                f"{r['agent']}: {r.get('result', r.get('error', 'N/A'))}"
                for r in results
            ])
        # Add other merge strategies as needed
        return str(results)
```

### Conditional Routing

Conditional routing selects which agents to execute based on conditions or input characteristics.

#### Implementation Example

```python
class ConditionalOrchestrator:
    """
    Orchestrates agents with conditional routing.
    """
    
    def __init__(self, routes: List[Dict]):
        """
        routes: List of {condition: callable, agents: List[Agent]}
        """
        self.routes = routes
    
    def execute(self, input_data):
        """Execute agents based on conditions."""
        for route in self.routes:
            condition = route["condition"]
            if condition(input_data):
                agents = route["agents"]
                # Execute agents in this route
                current_data = input_data
                for agent in agents:
                    current_data = agent.execute(current_data)
                return {"output": current_data, "route_taken": route.get("name", "unknown")}
        
        return {"output": None, "error": "No matching route"}
```

### Human-in-the-Loop

Human-in-the-loop orchestration incorporates human feedback at key decision points.

#### Implementation Example

```python
class HumanInTheLoopOrchestrator:
    """
    Orchestrates with human feedback points.
    """
    
    def __init__(self, agents: List[Any], feedback_points: List[int]):
        self.agents = agents
        self.feedback_points = feedback_points  # Indices where feedback is needed
    
    def execute(self, input_data, human_feedback_provider):
        """Execute with human feedback at specified points."""
        current_data = input_data
        execution_log = []
        
        for i, agent in enumerate(self.agents):
            # Execute agent
            result = agent.execute(current_data)
            
            # Check if human feedback needed
            if i in self.feedback_points:
                feedback = human_feedback_provider.get_feedback(result)
                result = self._incorporate_feedback(result, feedback)
            
            execution_log.append({
                "step": i + 1,
                "agent": agent.__class__.__name__,
                "result": result,
                "feedback_received": i in self.feedback_points
            })
            current_data = result
        
        return {
            "final_output": current_data,
            "execution_log": execution_log
        }
    
    def _incorporate_feedback(self, result, feedback):
        """Incorporate human feedback into result."""
        # Simplified - in practice, use LLM to incorporate feedback
        return f"{result}\n\n[Incorporated Feedback: {feedback}]"
```

### Graph-Based Workflows

Graph-based workflows model agent execution as a directed graph, enabling complex control flows.

#### Implementation Example

```python
from collections import defaultdict, deque

class GraphOrchestrator:
    """
    Orchestrates agents using graph-based workflow (LangGraph style).
    """
    
    def __init__(self):
        self.nodes = {}  # node_id -> agent
        self.edges = defaultdict(list)  # node_id -> [next_node_ids]
        self.entry_node = None
    
    def add_node(self, node_id: str, agent: Any):
        """Add a node (agent) to the graph."""
        self.nodes[node_id] = agent
        if self.entry_node is None:
            self.entry_node = node_id
    
    def add_edge(self, from_node: str, to_node: str, condition=None):
        """Add an edge between nodes."""
        self.edges[from_node].append({
            "to": to_node,
            "condition": condition
        })
    
    def execute(self, initial_input):
        """Execute graph-based workflow."""
        if self.entry_node is None:
            return {"error": "No entry node defined"}
        
        state = {
            "data": initial_input,
            "visited_nodes": [],
            "current_node": self.entry_node
        }
        
        # BFS traversal
        queue = deque([self.entry_node])
        
        while queue:
            node_id = queue.popleft()
            
            if node_id not in self.nodes:
                continue
            
            # Execute node
            agent = self.nodes[node_id]
            result = agent.execute(state["data"])
            
            state["data"] = result
            state["visited_nodes"].append(node_id)
            
            # Determine next nodes
            next_nodes = self._get_next_nodes(node_id, state)
            for next_node in next_nodes:
                if next_node not in state["visited_nodes"]:
                    queue.append(next_node)
        
        return {
            "final_output": state["data"],
            "execution_path": state["visited_nodes"]
        }
    
    def _get_next_nodes(self, node_id: str, state: dict) -> List[str]:
        """Get next nodes based on edges and conditions."""
        next_nodes = []
        for edge in self.edges[node_id]:
            condition = edge["condition"]
            if condition is None or condition(state):
                next_nodes.append(edge["to"])
        return next_nodes
```

---

## Choosing the Right Architecture

Selecting the appropriate agent architecture depends on multiple factors. Use this decision framework:

### Decision Matrix

| Factor | ReAct | Plan-Execute | CoT | Reflection | Tool-Augmented | State Machine | Pipeline | Master-Worker | Peer-to-Peer | Debate | Ensemble |
|--------|-------|--------------|-----|------------|----------------|--------------|----------|---------------|--------------|--------|----------|
| **Task Complexity** | Medium | High | Low-Med | Medium | Medium-High | Low-Med | Medium | High | Medium-High | High | Any |
| **Latency Requirement** | Medium | High | Low | High | Medium | Low | Medium | Medium | Medium | High | High |
| **Cost Sensitivity** | Medium | High | Low | High | Medium | Low | Medium | High | Medium | High | Very High |
| **Tool Usage** | High | Medium | Low | Low | Very High | Low | Low | Medium | Low | Low | Low |
| **Parallelization** | Low | Medium | Low | Low | Low | Low | Low | High | High | Low | High |
| **Reliability** | Medium | High | Medium | High | Medium | High | Medium | High | Medium | High | Very High |
| **Maintainability** | Medium | High | Low | Medium | Medium | High | High | Medium | Medium | Medium | Medium |

### Trade-offs Analysis

**Single vs Multi-Agent**
- **Single Agent**: Simpler, lower cost, faster for simple tasks
- **Multi-Agent**: Better for complex tasks, specialization, parallelization

**Planning vs Reactive**
- **Planning**: Better for complex, structured tasks
- **Reactive**: Better for dynamic, exploratory tasks

**Iterative vs Single-Pass**
- **Iterative**: Higher quality, higher cost
- **Single-Pass**: Faster, lower cost, may sacrifice quality

### Selection Guidelines

1. **Simple Q&A**: Use CoT or direct LLM
2. **Tool-Heavy Tasks**: Use ReAct or Tool-Augmented
3. **Complex Multi-Step**: Use Plan-and-Execute or Master-Worker
4. **High Quality Needed**: Use Reflection or Iterative Refinement
5. **Parallelizable**: Use Master-Worker or Ensemble
6. **Structured Workflow**: Use State Machine or Pipeline
7. **Multiple Perspectives**: Use Debate or Peer-to-Peer
8. **Maximum Accuracy**: Use Ensemble

---

## Anti-Patterns and Common Mistakes

### Over-Engineering

**Problem**: Using complex architecture for simple tasks.

**Example**: Using Master-Worker for a single-step question answering.

**Solution**: Start simple, add complexity only when needed.

### Under-Engineering

**Problem**: Using simple architecture for complex tasks.

**Example**: Using direct LLM calls for multi-step tool orchestration.

**Solution**: Match architecture complexity to task complexity.

### Ignoring Error Handling

**Problem**: Not handling failures gracefully.

**Example**: No retry logic, no fallbacks.

**Solution**: Implement comprehensive error handling and recovery.

### Poor State Management

**Problem**: Losing context between steps.

**Example**: Not passing state through pipeline stages.

**Solution**: Maintain state explicitly and pass through workflow.

### Tool Selection Issues

**Problem**: Selecting wrong tools or too many tools.

**Example**: Calling unnecessary tools, increasing latency and cost.

**Solution**: Implement intelligent tool selection with caching.

### Infinite Loops

**Problem**: Agents getting stuck in loops.

**Example**: Reflection agent never converging.

**Solution**: Set maximum iterations and convergence criteria.

### Cost Explosion

**Problem**: Too many LLM calls.

**Example**: Ensemble with 10 agents for simple task.

**Solution**: Monitor costs, optimize call frequency, use caching.

### Lack of Monitoring

**Problem**: No visibility into agent behavior.

**Example**: Can't debug why agent failed.

**Solution**: Implement logging, tracing, and monitoring.

---

## Architecture Comparison Table

| Architecture | Latency | Cost | Complexity | Reliability | Scalability | Best Use Case |
|-------------|---------|------|------------|-------------|-------------|---------------|
| **ReAct** | Medium | Medium | Medium | Medium | Medium | Dynamic tool use |
| **Plan-Execute** | High | High | High | High | Medium | Complex planning |
| **CoT** | Low | Low | Low | Medium | High | Reasoning tasks |
| **Reflection** | High | High | Medium | High | Medium | Quality-critical |
| **Tool-Augmented** | Medium | Medium | Medium | Medium | Medium | External integration |
| **State Machine** | Low | Low | Medium | High | High | Structured workflows |
| **Pipeline** | Medium | Medium | Medium | Medium | High | Data processing |
| **Iterative Refinement** | High | High | Medium | High | Medium | Quality improvement |
| **Master-Worker** | Medium | High | High | High | High | Parallelizable tasks |
| **Peer-to-Peer** | Medium | Medium | High | Medium | High | Collaborative work |
| **Hierarchical Teams** | Medium | High | High | High | High | Specialized coordination |
| **Debate** | Very High | Very High | High | Very High | Medium | Critical decisions |
| **Ensemble** | High | Very High | Medium | Very High | Medium | Maximum accuracy |

---

## Case Studies: Architecture in Production

### Case Study 1: Customer Support Chatbot

**Requirements**: Handle customer queries, check order status, process returns.

**Architecture Chosen**: ReAct Pattern

**Rationale**: 
- Dynamic tool use needed (database, order system)
- Unpredictable query types
- Real-time response needed

**Implementation**:
- ReAct agent with tools: order_lookup, return_processor, knowledge_base
- Handles diverse queries dynamically
- Average response time: 2-3 seconds

**Results**: 85% query resolution rate, 2.1s average latency

### Case Study 2: Research Report Generator

**Requirements**: Research topic, analyze data, generate comprehensive report.

**Architecture Chosen**: Master-Worker Pattern

**Rationale**:
- Multiple specialized tasks (research, analysis, writing)
- Can parallelize research
- Need coordination

**Implementation**:
- Master agent coordinates
- Workers: researcher, analyst, writer, editor
- Parallel research, sequential analysis/writing

**Results**: 40% faster than sequential, higher quality reports

### Case Study 3: Code Review Assistant

**Requirements**: Review code, identify issues, suggest improvements.

**Architecture Chosen**: Reflection Pattern

**Rationale**:
- High quality critical
- Multiple passes improve results
- Can iterate on critiques

**Implementation**:
- Initial review generation
- Self-critique for completeness
- Iterative refinement of suggestions

**Results**: 30% more issues found vs single-pass, higher suggestion quality

### Case Study 4: Multi-Domain Q&A System

**Requirements**: Answer questions across multiple domains accurately.

**Architecture Chosen**: Ensemble Pattern

**Rationale**:
- Maximum accuracy needed
- Different agents for different domains
- Can combine strengths

**Implementation**:
- 5 specialized agents (science, business, tech, etc.)
- LLM synthesis aggregation
- Confidence-weighted combination

**Results**: 15% accuracy improvement over single agent

### Case Study 5: Workflow Automation

**Requirements**: Automate multi-step business processes.

**Architecture Chosen**: Graph-Based Orchestration (LangGraph style)

**Rationale**:
- Complex conditional flows
- Need to model dependencies
- Human approval points

**Implementation**:
- Graph with conditional edges
- Human-in-the-loop nodes
- State management across nodes

**Results**: 60% process automation, 3x faster execution

---

## Conclusion

Agent architectures provide different approaches to structuring AI agent systems. The choice of architecture significantly impacts performance, cost, reliability, and capabilities. Key takeaways:

1. **Match architecture to task**: Simple tasks don't need complex architectures
2. **Consider trade-offs**: Every architecture has strengths and weaknesses
3. **Start simple**: Begin with simplest architecture that works
4. **Iterate and refine**: Add complexity only when needed
5. **Monitor and optimize**: Track performance and costs continuously

Understanding these architectures and their trade-offs enables building effective, efficient agent systems that meet specific requirements and constraints.

---

*End of Document*



