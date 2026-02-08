# Planning, Reasoning, and Decision Making

## Table of Contents

1. Why Planning Matters for Agents
2. Task Decomposition
3. Planning Strategies
4. Reasoning Patterns
5. Decision Making Under Uncertainty
6. Self-Reflection and Critique
7. Meta-Cognition
8. Advanced Reasoning Techniques
9. Evaluation of Planning Quality
10. Production Planning Systems

---

## 1. Why Planning Matters for Agents

### The Planning Gap

Without planning, an agent operates reactively, responding to each input without considering the bigger picture. With planning, an agent can:

| Without Planning | With Planning |
|-----------------|--------------|
| Responds to immediate input only | Considers the full task scope |
| May miss dependencies between steps | Identifies step ordering |
| Cannot estimate effort or cost | Provides time/cost estimates |
| Cannot recover from dead ends | Can backtrack and re-plan |
| No progress tracking | Tracks completed vs remaining steps |
| One-shot attempt | Iterative refinement |

### Planning in the Agent Loop

```
+--------+     +----------+     +----------+     +---------+
| INPUT  | --> | PLAN     | --> | EXECUTE  | --> | EVALUATE|
| (Goal) |     | (Strategy|     | (Actions)|     | (Check) |
|        |     |  & Steps)|     |          |     |         |
+--------+     +----+-----+     +----+-----+     +----+----+
                    ^                |                  |
                    |                v                  |
                    |          +----------+             |
                    +----------| RE-PLAN  |<------------+
                               | (Adjust) |  (if needed)
                               +----------+
```

---

## 2. Task Decomposition

### What Is Task Decomposition?

Breaking a complex goal into smaller, manageable subtasks that can be individually planned and executed.

### Decomposition Strategies

**Top-Down Decomposition**: Start from the goal, recursively break into sub-goals.

```python
class Top_Down_Decomposer:
    def __init__(self, llm):
        self.llm = llm

    def Decompose(self, goal, depth=0, max_depth=3):
        if depth >= max_depth:
            return [{"task": goal, "leaf": True}]

        subtasks_json = self.llm.generate(f"""
        Break this task into 2-5 subtasks:
        Task: {goal}

        Rules:
        - Each subtask should be specific and actionable
        - Subtasks should collectively achieve the original task
        - Identify dependencies between subtasks
        - Mark subtasks that are simple enough to execute directly

        Return JSON:
        [
          {{"task": "...", "depends_on": [], "is_simple": true/false}}
        ]
        """)

        subtasks = json.loads(subtasks_json)
        result = []

        for st in subtasks:
            if st["is_simple"]:
                result.append({"task": st["task"], "leaf": True, "depends_on": st["depends_on"]})
            else:
                children = self.Decompose(st["task"], depth + 1, max_depth)
                result.append({
                    "task": st["task"],
                    "leaf": False,
                    "depends_on": st["depends_on"],
                    "children": children,
                })

        return result
```

**Bottom-Up Decomposition**: Identify available actions first, then compose them into a plan.

```python
class Bottom_Up_Planner:
    def __init__(self, llm, available_tools):
        self.llm = llm
        self.tools = available_tools

    def Plan(self, goal):
        tool_descriptions = [
            f"- {t.name}: {t.description}" for t in self.tools
        ]

        plan_json = self.llm.generate(f"""
        Goal: {goal}

        Available tools/actions:
        {chr(10).join(tool_descriptions)}

        Create a plan using ONLY the available tools.
        Order steps by dependencies.

        Return JSON:
        [
          {{"step": 1, "action": "tool_name", "params": {{...}}, "purpose": "..."}}
        ]
        """)

        return json.loads(plan_json)
```

### Dependency Graph

```python
class Dependency_Graph:
    def __init__(self):
        self.tasks = {}
        self.dependencies = {}

    def Add_Task(self, task_id, task_data):
        self.tasks[task_id] = task_data
        self.dependencies[task_id] = set()

    def Add_Dependency(self, task_id, depends_on):
        self.dependencies[task_id].add(depends_on)

    def Get_Execution_Order(self):
        """Topological sort to get valid execution order."""
        visited = set()
        order = []
        temp_mark = set()

        def Visit(node):
            if node in temp_mark:
                raise ValueError(f"Circular dependency detected at {node}")
            if node in visited:
                return

            temp_mark.add(node)
            for dep in self.dependencies.get(node, set()):
                Visit(dep)
            temp_mark.discard(node)
            visited.add(node)
            order.append(node)

        for task_id in self.tasks:
            if task_id not in visited:
                Visit(task_id)

        return order

    def Get_Parallel_Groups(self):
        """Group tasks that can run in parallel."""
        order = self.Get_Execution_Order()
        groups = []
        completed = set()

        while len(completed) < len(order):
            # Find all tasks whose dependencies are all completed
            ready = [
                t for t in order
                if t not in completed
                and self.dependencies[t].issubset(completed)
            ]
            if not ready:
                break
            groups.append(ready)
            completed.update(ready)

        return groups

# Usage
graph = Dependency_Graph()
graph.Add_Task("research", {"description": "Research the topic"})
graph.Add_Task("outline", {"description": "Create outline"})
graph.Add_Task("write", {"description": "Write content"})
graph.Add_Task("images", {"description": "Generate images"})
graph.Add_Task("review", {"description": "Review final output"})

graph.Add_Dependency("outline", "research")
graph.Add_Dependency("write", "outline")
graph.Add_Dependency("images", "outline")       # Can run parallel with write
graph.Add_Dependency("review", "write")
graph.Add_Dependency("review", "images")

print(graph.Get_Parallel_Groups())
# [['research'], ['outline'], ['write', 'images'], ['review']]
```

---

## 3. Planning Strategies

### 3.1 One-Shot Planning

Generate the entire plan before execution. Fast but inflexible.

```python
class One_Shot_Planner:
    def __init__(self, llm):
        self.llm = llm

    def Plan(self, goal, context=None):
        plan = self.llm.generate(f"""
        Goal: {goal}
        Context: {context or 'None'}

        Create a detailed, step-by-step plan.
        For each step specify:
        1. What to do
        2. What tool to use (if any)
        3. Expected output
        4. How to verify success

        Return the complete plan as a numbered list.
        """)
        return self.Parse_Plan(plan)

    def Parse_Plan(self, plan_text):
        steps = []
        for line in plan_text.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                steps.append({"step": len(steps) + 1, "description": line})
        return steps
```

### 3.2 Iterative Planning (Plan-Execute-Replan)

Plan one step at a time, adjusting based on results.

```python
class Iterative_Planner:
    def __init__(self, llm, tools, max_steps=15):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def Execute(self, goal):
        history = []
        step_count = 0

        while step_count < self.max_steps:
            # Plan next step
            next_action = self.Plan_Next_Step(goal, history)

            if next_action["type"] == "finish":
                return {
                    "success": True,
                    "result": next_action["result"],
                    "steps": history,
                }

            # Execute the step
            result = self.Execute_Step(next_action)

            history.append({
                "step": step_count + 1,
                "action": next_action,
                "result": result,
            })

            step_count += 1

        return {"success": False, "reason": "Max steps reached", "steps": history}

    def Plan_Next_Step(self, goal, history):
        history_text = "\n".join(
            f"Step {h['step']}: {h['action']['description']} -> {h['result']}"
            for h in history
        )

        response = self.llm.generate(f"""
        Goal: {goal}

        Steps completed so far:
        {history_text if history else 'None yet'}

        Available tools: {[t.name for t in self.tools]}

        What should the next step be?
        If the goal is achieved, respond with: {{"type": "finish", "result": "..."}}
        Otherwise: {{"type": "action", "tool": "...", "params": {{...}}, "description": "..."}}
        """)

        return json.loads(response)

    def Execute_Step(self, action):
        if action["type"] == "action" and "tool" in action:
            tool = next((t for t in self.tools if t.name == action["tool"]), None)
            if tool:
                return tool.Execute(**action.get("params", {}))
        return {"error": "Unknown action"}
```

### 3.3 Hierarchical Planning

Multi-level planning: high-level strategy first, then detailed tactical plans.

```python
class Hierarchical_Planner:
    def __init__(self, llm):
        self.llm = llm

    def Plan(self, goal):
        # Level 1: Strategic plan (high-level phases)
        strategic = self.Strategic_Plan(goal)

        # Level 2: Tactical plan (detailed steps for each phase)
        tactical = {}
        for phase in strategic:
            tactical[phase["name"]] = self.Tactical_Plan(phase, goal)

        return {"strategic": strategic, "tactical": tactical}

    def Strategic_Plan(self, goal):
        response = self.llm.generate(f"""
        Goal: {goal}

        Create a high-level strategic plan with 3-5 major phases.
        Each phase should represent a major milestone.

        Return JSON:
        [
          {{"name": "Phase 1: ...", "objective": "...", "success_criteria": "..."}}
        ]
        """)
        return json.loads(response)

    def Tactical_Plan(self, phase, overall_goal):
        response = self.llm.generate(f"""
        Overall goal: {overall_goal}
        Current phase: {phase['name']}
        Phase objective: {phase['objective']}

        Create detailed tactical steps for this phase.
        Each step should be immediately actionable.

        Return JSON:
        [
          {{"step": 1, "action": "...", "tool": "...", "params": {{}}, "estimated_time": "..."}}
        ]
        """)
        return json.loads(response)
```

### Planning Strategy Comparison

| Strategy | Planning Time | Flexibility | Token Cost | Best For |
|----------|-------------|-------------|------------|----------|
| One-Shot | Low | Low | Low | Simple, predictable tasks |
| Iterative | Medium | High | Medium | Exploratory tasks |
| Hierarchical | High | Medium | High | Complex multi-phase projects |
| Hybrid | Medium | High | Medium | Production systems |

---

## 4. Reasoning Patterns

### 4.1 Chain-of-Thought (CoT)

Step-by-step reasoning before arriving at an answer.

```python
class CoT_Reasoner:
    def __init__(self, llm):
        self.llm = llm

    def Reason(self, question, context=None):
        prompt = f"""
        Question: {question}
        {'Context: ' + context if context else ''}

        Let's think step by step:
        1. First, identify what we know...
        2. Then, consider what we need to find...
        3. Apply relevant logic...
        4. Arrive at the conclusion...

        Think through this carefully, showing your reasoning at each step.
        """
        return self.llm.generate(prompt)
```

### 4.2 Self-Consistency

Generate multiple reasoning paths and take the majority answer.

```python
class Self_Consistency_Reasoner:
    def __init__(self, llm, num_samples=5):
        self.llm = llm
        self.num_samples = num_samples

    def Reason(self, question):
        answers = []

        for i in range(self.num_samples):
            response = self.llm.generate(
                f"Think step by step and answer: {question}",
                temperature=0.7  # Higher temp for diversity
            )
            answer = self.Extract_Final_Answer(response)
            answers.append({"reasoning": response, "answer": answer})

        # Majority vote
        from collections import Counter
        answer_counts = Counter(a["answer"] for a in answers)
        best_answer = answer_counts.most_common(1)[0]

        return {
            "answer": best_answer[0],
            "confidence": best_answer[1] / self.num_samples,
            "all_answers": answers,
        }

    def Extract_Final_Answer(self, response):
        lines = response.strip().split("\n")
        return lines[-1].strip()
```

### 4.3 Tree of Thoughts (ToT)

Explore multiple reasoning branches, evaluate each, and select the best.

```python
class Tree_Of_Thoughts:
    def __init__(self, llm, branching_factor=3, max_depth=3):
        self.llm = llm
        self.branching_factor = branching_factor
        self.max_depth = max_depth

    def Solve(self, problem):
        root = {"thought": "Start", "children": [], "score": 0}
        self.Expand(root, problem, depth=0)
        best_path = self.Find_Best_Path(root)
        return best_path

    def Expand(self, node, problem, depth):
        if depth >= self.max_depth:
            return

        # Generate multiple next thoughts
        thoughts = self.Generate_Thoughts(problem, node["thought"], self.branching_factor)

        for thought in thoughts:
            # Evaluate each thought
            score = self.Evaluate_Thought(problem, thought)

            child = {"thought": thought, "children": [], "score": score}
            node["children"].append(child)

            # Only expand promising branches
            if score > 0.5:
                self.Expand(child, problem, depth + 1)

    def Generate_Thoughts(self, problem, current_thought, n):
        response = self.llm.generate(f"""
        Problem: {problem}
        Current thinking: {current_thought}

        Generate {n} different next steps in reasoning.
        Each should be a distinct approach or perspective.
        Return as a JSON list of strings.
        """)
        return json.loads(response)

    def Evaluate_Thought(self, problem, thought):
        response = self.llm.generate(f"""
        Problem: {problem}
        Proposed thought: {thought}

        Rate this thought on a scale of 0.0 to 1.0:
        - Is it logically sound?
        - Does it make progress toward solving the problem?
        - Is it a promising direction?

        Return only a number.
        """)
        return float(response.strip())

    def Find_Best_Path(self, node, path=None):
        if path is None:
            path = []

        path.append(node["thought"])

        if not node["children"]:
            return path

        best_child = max(node["children"], key=lambda c: c["score"])
        return self.Find_Best_Path(best_child, path)
```

### 4.4 Graph of Thoughts (GoT)

Unlike ToT which is strictly hierarchical, GoT allows merging and looping between thoughts.

```python
class Graph_Of_Thoughts:
    def __init__(self, llm):
        self.llm = llm
        self.nodes = {}  # id -> thought
        self.edges = {}  # id -> [connected_ids]
        self.scores = {}

    def Add_Thought(self, thought_id, content, parent_ids=None):
        self.nodes[thought_id] = content
        self.edges[thought_id] = []

        if parent_ids:
            for pid in parent_ids:
                if pid in self.edges:
                    self.edges[pid].append(thought_id)

    def Merge_Thoughts(self, thought_ids, new_id):
        thoughts = [self.nodes[tid] for tid in thought_ids]
        merged = self.llm.generate(f"""
        Merge these thoughts into a unified, stronger thought:
        {json.dumps(thoughts)}

        Combine the best elements from each.
        """)
        self.Add_Thought(new_id, merged, thought_ids)
        return merged

    def Refine_Thought(self, thought_id):
        original = self.nodes[thought_id]
        refined = self.llm.generate(f"""
        Original thought: {original}

        Refine and improve this thought.
        Fix any logical errors, add missing considerations.
        """)
        new_id = f"{thought_id}_refined"
        self.Add_Thought(new_id, refined, [thought_id])
        return new_id

    def Get_Best_Path(self):
        # Score all terminal nodes
        terminals = [
            nid for nid in self.nodes
            if not self.edges.get(nid)
        ]

        best_id = None
        best_score = -1
        for tid in terminals:
            score = self.Score_Thought(tid)
            if score > best_score:
                best_score = score
                best_id = tid

        # Trace back path
        return self.Trace_Path(best_id)

    def Score_Thought(self, thought_id):
        if thought_id in self.scores:
            return self.scores[thought_id]

        score = float(self.llm.generate(f"""
        Rate this thought 0.0-1.0: {self.nodes[thought_id]}
        """).strip())

        self.scores[thought_id] = score
        return score

    def Trace_Path(self, node_id):
        # Find all paths leading to this node (reverse traversal)
        path = [self.nodes[node_id]]
        current = node_id

        for parent_id, children in self.edges.items():
            if current in children:
                path.insert(0, self.nodes[parent_id])
                current = parent_id

        return path
```

### Reasoning Pattern Comparison

| Pattern | Depth | Breadth | Cost | Quality | Best For |
|---------|-------|---------|------|---------|----------|
| Direct/Zero-shot | 1 | 1 | Low | Low | Simple questions |
| Chain-of-Thought | Deep | 1 | Low | Medium | Step-by-step problems |
| Self-Consistency | 1 | Wide | Medium | Medium-High | Verification |
| Tree of Thoughts | Deep | Wide | High | High | Complex problem solving |
| Graph of Thoughts | Deep | Wide | Very High | Very High | Research-level problems |

---

## 5. Decision Making Under Uncertainty

### Confidence Estimation

```python
class Confidence_Estimator:
    def __init__(self, llm):
        self.llm = llm

    def Estimate(self, question, answer):
        response = self.llm.generate(f"""
        Question: {question}
        Proposed Answer: {answer}

        Rate your confidence in this answer:
        - How certain are you? (0.0 to 1.0)
        - What are the main sources of uncertainty?
        - What additional information would increase confidence?

        Return JSON:
        {{
          "confidence": 0.0-1.0,
          "uncertainties": ["..."],
          "needed_info": ["..."]
        }}
        """)
        return json.loads(response)
```

### Decision Trees

```python
class Decision_Node:
    def __init__(self, question, options):
        self.question = question
        self.options = options  # {answer: next_node_or_action}

class Agent_Decision_Tree:
    def __init__(self, llm, root_node):
        self.llm = llm
        self.root = root_node

    def Navigate(self, context):
        current = self.root
        path = []

        while isinstance(current, Decision_Node):
            answer = self.llm.generate(f"""
            Context: {context}
            Question: {current.question}
            Options: {list(current.options.keys())}

            Choose the best option. Return only the option text.
            """).strip()

            path.append({"question": current.question, "answer": answer})

            # Find matching option
            matched = None
            for opt_key in current.options:
                if opt_key.lower() in answer.lower():
                    matched = opt_key
                    break

            if matched:
                current = current.options[matched]
            else:
                current = list(current.options.values())[0]  # default

        return {"action": current, "path": path}
```

### Risk Assessment

```python
class Risk_Assessor:
    def __init__(self, llm):
        self.llm = llm

    def Assess(self, action, context):
        response = self.llm.generate(f"""
        Proposed action: {action}
        Context: {context}

        Assess the risks:

        Return JSON:
        {{
          "risk_level": "low|medium|high|critical",
          "risks": [
            {{"description": "...", "probability": 0.0-1.0, "impact": "low|medium|high"}},
          ],
          "mitigations": ["..."],
          "recommendation": "proceed|proceed_with_caution|require_approval|abort"
        }}
        """)
        return json.loads(response)
```

---

## 6. Self-Reflection and Critique

### Reflection Pattern

```python
class Self_Reflector:
    def __init__(self, llm):
        self.llm = llm

    def Reflect_On_Output(self, task, output):
        reflection = self.llm.generate(f"""
        Task: {task}
        Output produced: {output}

        Critically evaluate this output:
        1. Does it fully address the task?
        2. Are there any factual errors?
        3. Is anything missing or incomplete?
        4. Could the quality be improved?
        5. Are there any logical inconsistencies?

        Return JSON:
        {{
          "quality_score": 1-10,
          "issues": ["..."],
          "improvements": ["..."],
          "needs_revision": true/false
        }}
        """)
        return json.loads(reflection)

    def Reflect_And_Revise(self, task, output, max_revisions=3):
        current_output = output

        for revision in range(max_revisions):
            reflection = self.Reflect_On_Output(task, current_output)

            if not reflection["needs_revision"] or reflection["quality_score"] >= 8:
                return {
                    "final_output": current_output,
                    "revisions": revision,
                    "final_score": reflection["quality_score"],
                }

            # Revise based on feedback
            current_output = self.llm.generate(f"""
            Original task: {task}
            Current output: {current_output}

            Issues found:
            {json.dumps(reflection['issues'])}

            Suggested improvements:
            {json.dumps(reflection['improvements'])}

            Produce an improved version addressing all issues.
            """)

        return {
            "final_output": current_output,
            "revisions": max_revisions,
            "note": "Max revisions reached",
        }
```

### Reflexion Framework

Agents that learn from past mistakes by maintaining a reflection memory.

```python
class Reflexion_Agent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.reflections = []  # Memory of past reflections

    def Solve(self, task, max_trials=3):
        for trial in range(max_trials):
            # Include past reflections in context
            reflection_context = "\n".join(
                f"Trial {r['trial']}: {r['reflection']}" for r in self.reflections
            )

            # Attempt solution
            plan = self.llm.generate(f"""
            Task: {task}

            {'Past reflections (learn from these):' + reflection_context if self.reflections else ''}

            Create a plan and execute it step by step.
            """)

            result = self.Execute_Plan(plan)

            # Evaluate
            evaluation = self.Evaluate(task, result)

            if evaluation["success"]:
                return {"success": True, "result": result, "trials": trial + 1}

            # Reflect on failure
            reflection = self.llm.generate(f"""
            Task: {task}
            Plan: {plan}
            Result: {result}
            Evaluation: {evaluation}

            Reflect on what went wrong and how to improve:
            - What specific mistakes were made?
            - What should be done differently next time?
            - What new information was learned?
            """)

            self.reflections.append({
                "trial": trial + 1,
                "plan": plan,
                "result": result,
                "reflection": reflection,
            })

        return {"success": False, "trials": max_trials, "reflections": self.reflections}

    def Execute_Plan(self, plan):
        # Implementation of plan execution
        pass

    def Evaluate(self, task, result):
        eval_response = self.llm.generate(f"""
        Task: {task}
        Result: {result}

        Was the task completed successfully? Return JSON:
        {{"success": true/false, "reason": "..."}}
        """)
        return json.loads(eval_response)
```

---

## 7. Meta-Cognition

Meta-cognition is "thinking about thinking" -- the agent's ability to monitor and regulate its own cognitive processes.

### Self-Monitoring

```python
class Meta_Cognitive_Agent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.cognitive_state = {
            "confidence": 1.0,
            "stuck_count": 0,
            "strategy": "default",
            "knowledge_gaps": [],
        }

    def Process(self, task):
        # Monitor cognitive state before acting
        self.Pre_Action_Check(task)

        # Choose strategy based on self-assessment
        strategy = self.Select_Strategy(task)

        # Execute with selected strategy
        result = self.Execute_With_Strategy(task, strategy)

        # Post-action reflection
        self.Post_Action_Update(task, result)

        return result

    def Pre_Action_Check(self, task):
        assessment = self.llm.generate(f"""
        Task: {task}
        My current state: {json.dumps(self.cognitive_state)}

        Self-assessment:
        1. Do I understand this task fully? (confidence 0-1)
        2. Do I have the right tools for this?
        3. What knowledge gaps exist?
        4. Am I making progress or stuck?

        Return JSON with updated cognitive state.
        """)

        update = json.loads(assessment)
        self.cognitive_state.update(update)

    def Select_Strategy(self, task):
        confidence = self.cognitive_state["confidence"]
        stuck = self.cognitive_state["stuck_count"]

        if confidence > 0.8 and stuck == 0:
            return "direct_execution"
        elif confidence > 0.5:
            return "step_by_step_with_verification"
        elif stuck > 2:
            return "seek_help_or_decompose"
        else:
            return "research_first"

    def Execute_With_Strategy(self, task, strategy):
        strategies = {
            "direct_execution": self.Direct_Execute,
            "step_by_step_with_verification": self.Careful_Execute,
            "seek_help_or_decompose": self.Decompose_And_Execute,
            "research_first": self.Research_Then_Execute,
        }
        return strategies[strategy](task)

    def Post_Action_Update(self, task, result):
        if result.get("success"):
            self.cognitive_state["stuck_count"] = 0
            self.cognitive_state["confidence"] = min(1.0, self.cognitive_state["confidence"] + 0.1)
        else:
            self.cognitive_state["stuck_count"] += 1
            self.cognitive_state["confidence"] = max(0.1, self.cognitive_state["confidence"] - 0.2)

    def Direct_Execute(self, task):
        return {"success": True, "result": self.llm.generate(f"Complete: {task}")}

    def Careful_Execute(self, task):
        steps = self.llm.generate(f"Break into steps: {task}")
        # Execute and verify each step
        return {"success": True, "result": steps}

    def Decompose_And_Execute(self, task):
        subtasks = self.llm.generate(f"Simplify into subtasks: {task}")
        return {"success": True, "result": subtasks}

    def Research_Then_Execute(self, task):
        research = self.llm.generate(f"What do I need to know to solve: {task}")
        solution = self.llm.generate(f"Given: {research}\nSolve: {task}")
        return {"success": True, "result": solution}
```

---

## 8. Advanced Reasoning Techniques

### 8.1 Analogical Reasoning

```python
class Analogical_Reasoner:
    def __init__(self, llm, case_memory):
        self.llm = llm
        self.case_memory = case_memory

    def Reason_By_Analogy(self, new_problem):
        # Find similar past problems
        similar_cases = self.case_memory.Search(new_problem, top_k=3)

        response = self.llm.generate(f"""
        New problem: {new_problem}

        Similar past cases:
        {json.dumps([{"problem": c["problem"], "solution": c["solution"]} for c in similar_cases])}

        Apply analogical reasoning:
        1. Identify structural similarities between past cases and the new problem
        2. Map the solution strategies from past cases to the new problem
        3. Adapt the solution to account for differences
        4. Propose a solution for the new problem

        Provide your reasoning and final solution.
        """)
        return response
```

### 8.2 Counterfactual Reasoning

```python
class Counterfactual_Reasoner:
    def __init__(self, llm):
        self.llm = llm

    def Analyze(self, situation, decision, outcome):
        response = self.llm.generate(f"""
        Situation: {situation}
        Decision made: {decision}
        Outcome: {outcome}

        Counterfactual analysis:
        1. What would have happened if a different decision was made?
        2. List 3 alternative decisions and their likely outcomes.
        3. Was the original decision optimal? Why or why not?
        4. What can be learned for future similar situations?

        Return JSON:
        {{
          "alternatives": [
            {{"decision": "...", "likely_outcome": "...", "probability": 0.0-1.0}}
          ],
          "original_was_optimal": true/false,
          "lessons": ["..."]
        }}
        """)
        return json.loads(response)
```

### 8.3 Causal Reasoning

```python
class Causal_Reasoner:
    def __init__(self, llm):
        self.llm = llm

    def Identify_Causes(self, observation):
        response = self.llm.generate(f"""
        Observation: {observation}

        Perform causal analysis:
        1. What are the possible causes? (generate hypotheses)
        2. For each cause, what evidence would support or refute it?
        3. What is the most likely causal chain?
        4. Are there confounding factors?

        Return JSON:
        {{
          "hypotheses": [
            {{
              "cause": "...",
              "mechanism": "...",
              "supporting_evidence": ["..."],
              "contradicting_evidence": ["..."],
              "probability": 0.0-1.0
            }}
          ],
          "most_likely_chain": "A -> B -> C -> observation",
          "confounders": ["..."]
        }}
        """)
        return json.loads(response)
```

---

## 9. Evaluation of Planning Quality

### Plan Quality Metrics

| Metric | Description | How to Measure |
|--------|-------------|---------------|
| Completeness | Does the plan cover all requirements? | Checklist against task requirements |
| Feasibility | Can each step actually be executed? | Verify tool availability and permissions |
| Efficiency | Is it the shortest reasonable path? | Compare step count to baseline |
| Correctness | Will it achieve the goal? | Simulate or verify logical chain |
| Robustness | Does it handle edge cases? | Test with variations |
| Cost | What are the resource requirements? | Estimate tokens, API calls, time |

### Plan Evaluator

```python
class Plan_Evaluator:
    def __init__(self, llm, available_tools):
        self.llm = llm
        self.tools = available_tools

    def Evaluate(self, plan, goal):
        evaluation = {
            "completeness": self.Check_Completeness(plan, goal),
            "feasibility": self.Check_Feasibility(plan),
            "efficiency": self.Check_Efficiency(plan),
            "risks": self.Assess_Risks(plan),
        }

        overall_score = (
            0.3 * evaluation["completeness"]["score"] +
            0.3 * evaluation["feasibility"]["score"] +
            0.2 * evaluation["efficiency"]["score"] +
            0.2 * (1.0 - evaluation["risks"]["risk_score"])
        )

        evaluation["overall_score"] = overall_score
        evaluation["recommendation"] = (
            "execute" if overall_score > 0.7
            else "revise" if overall_score > 0.4
            else "reject"
        )

        return evaluation

    def Check_Completeness(self, plan, goal):
        response = self.llm.generate(f"""
        Goal: {goal}
        Plan: {json.dumps(plan)}

        Does this plan address all aspects of the goal?
        Return JSON: {{"score": 0.0-1.0, "missing": ["..."]}}
        """)
        return json.loads(response)

    def Check_Feasibility(self, plan):
        tool_names = {t.name for t in self.tools}
        infeasible = []

        for step in plan:
            if "tool" in step and step["tool"] not in tool_names:
                infeasible.append(f"Tool '{step['tool']}' not available")

        score = 1.0 - (len(infeasible) / max(len(plan), 1))
        return {"score": score, "infeasible_steps": infeasible}

    def Check_Efficiency(self, plan):
        num_steps = len(plan)
        score = max(0.0, 1.0 - (num_steps - 5) * 0.1)  # Penalize plans > 5 steps
        return {"score": score, "step_count": num_steps}

    def Assess_Risks(self, plan):
        response = self.llm.generate(f"""
        Plan: {json.dumps(plan)}
        Identify risks. Return JSON:
        {{"risk_score": 0.0-1.0, "risks": ["..."]}}
        """)
        return json.loads(response)
```

---

## 10. Production Planning Systems

### Architecture

```
+------------------------------------------------------------------+
|                  PRODUCTION PLANNING SYSTEM                       |
|                                                                   |
|  +------------+     +-----------+     +------------+              |
|  | Task Input | --> | Planner   | --> | Executor   |              |
|  |            |     | (LLM +   |     | (Tool      |              |
|  |            |     |  Strategy)|     |  Runtime)  |              |
|  +------------+     +-----+-----+     +------+-----+              |
|                           |                  |                    |
|                     +-----v-----+      +-----v-----+             |
|                     | Plan      |      | Execution |             |
|                     | Store     |      | Monitor   |             |
|                     | (History) |      | (Metrics) |             |
|                     +-----------+      +-----------+             |
|                                                                   |
|  +-----------+     +------------+     +------------+              |
|  | Evaluator | <-- | Re-Planner | <-- | Error      |              |
|  | (Quality  |     | (Adaptive  |     | Handler    |              |
|  |  Check)   |     |  Strategy) |     |            |              |
|  +-----------+     +------------+     +------------+              |
+------------------------------------------------------------------+
```

### Best Practices

1. **Start with simple plans**: Use one-shot planning for straightforward tasks; escalate to iterative planning only when needed

2. **Set plan budgets**: Limit planning time, steps, and token usage to prevent over-planning

3. **Cache plans**: Store successful plans for reuse with similar future tasks

4. **Monitor plan quality**: Track completion rates, step counts, and failure points

5. **Implement fallbacks**: Have default actions when planning fails or produces invalid plans

6. **Balance planning vs execution**: Avoid "analysis paralysis" by setting planning time limits

7. **Learn from execution**: Feed execution results back into the planner to improve future plans

8. **Human-in-the-loop for critical plans**: Require approval for plans that involve high-risk actions

9. **Version your planning prompts**: Track which prompt versions produce the best plans

10. **Test with adversarial inputs**: Verify plans handle edge cases, ambiguous goals, and conflicting constraints
