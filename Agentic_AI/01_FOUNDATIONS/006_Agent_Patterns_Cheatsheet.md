# Agent Patterns Cheatsheet: Quick Reference Guide

## Table of Contents
1. [Single Agent Patterns](#single-agent-patterns)
2. [Multi-Agent Patterns](#multi-agent-patterns)
3. [Communication Patterns](#communication-patterns)
4. [Execution Patterns](#execution-patterns)
5. [Learning Patterns](#learning-patterns)
6. [Integration Patterns](#integration-patterns)

---

## Single Agent Patterns

### 1. **ReAct Pattern**
**Use Case**: Reasoning and Acting in interleaved steps
**When to Use**: Complex tasks requiring thought and action cycles

```python
class ReActAgent:
    def process(self, query):
        thought = self.think(query)
        action = self.act_on_thought(thought)
        observation = self.observe_result(action)
        return self.decide_next_step(thought, action, observation)
```

**Example**: Research assistant that thinks → searches → analyzes → acts

### 2. **Plan-and-Execute Pattern**
**Use Case**: Strategic planning followed by execution
**When to Use**: Multi-step tasks with clear objectives

```python
class PlanExecuteAgent:
    def solve_problem(self, problem):
        plan = self.create_plan(problem)
        for step in plan:
            result = self.execute_step(step)
            if not result.success:
                plan = self.replan(plan, step, result)
        return self.get_final_result()
```

**Example**: Project management agent planning and tracking tasks

### 3. **Chain-of-Thought Pattern**
**Use Case**: Step-by-step reasoning for complex problems
**When to Use**: Problems requiring logical progression

```python
class ChainOfThoughtAgent:
    def reason(self, problem):
        steps = []
        current_problem = problem
        while not self.is_solved(current_problem):
            step = self.generate_reasoning_step(current_problem)
            steps.append(step)
            current_problem = self.apply_step(current_problem, step)
        return self.construct_solution(steps)
```

**Example**: Mathematical problem solver, legal analysis

### 4. **Tool-Using Pattern**
**Use Case**: Leveraging external tools and APIs
**When to Use**: Tasks requiring specialized capabilities

```python
class ToolUsingAgent:
    def __init__(self):
        self.tools = ToolRegistry()
    
    def solve_with_tools(self, task):
        required_tools = self.identify_tools_needed(task)
        for tool in required_tools:
            result = tool.execute(task.get_tool_input(tool))
            task.incorporate_result(tool, result)
        return task.get_final_output()
```

**Example**: Data analysis agent using calculators, databases, APIs

### 5. **Self-Reflection Pattern**
**Use Case**: Evaluating and improving own performance
**When to Use**: Quality-critical applications

```python
class SelfReflectiveAgent:
    def process_with_reflection(self, input_data):
        initial_result = self.process(input_data)
        reflection = self.reflect_on_result(initial_result, input_data)
        if reflection.needs_improvement:
            improved_result = self.improve_result(initial_result, reflection)
            return improved_result
        return initial_result
```

**Example**: Writing assistant that reviews and improves its output

---

## Multi-Agent Patterns

### 1. **Master-Worker Pattern**
**Use Case**: Distributing work among multiple agents
**When to Use**: Parallelizable tasks, load distribution

```python
class MasterAgent:
    def distribute_work(self, large_task):
        subtasks = self.divide_task(large_task)
        workers = self.get_available_workers()
        results = []
        for subtask in subtasks:
            worker = self.assign_worker(subtask, workers)
            results.append(worker.execute(subtask))
        return self.combine_results(results)
```

**Example**: Data processing pipeline, content generation system

### 2. **Peer-to-Peer Collaboration**
**Use Case**: Agents working as equals on shared goals
**When to Use**: Collaborative problem solving

```python
class CollaborativeAgent:
    def collaborate(self, shared_goal, peer_agents):
        contribution = self.determine_contribution(shared_goal)
        self.coordinate_with_peers(peer_agents, contribution)
        return self.execute_collaborative_work(contribution)
```

**Example**: Research team agents, creative writing collaboration

### 3. **Hierarchical Teams**
**Use Case**: Organized structure with different responsibility levels
**When to Use**: Complex organizations, clear command structures

```python
class HierarchicalAgent:
    def __init__(self, level, subordinates=None, supervisor=None):
        self.level = level
        self.subordinates = subordinates or []
        self.supervisor = supervisor
    
    def handle_directive(self, directive):
        if self.can_handle_directly(directive):
            return self.execute_directive(directive)
        else:
            return self.delegate_to_subordinates(directive)
```

**Example**: Corporate automation, military command systems

### 4. **Specialized Roles Pattern**
**Use Case**: Agents with specific expertise areas
**When to Use**: Domain-specific tasks requiring specialized knowledge

```python
class SpecializedAgent:
    def __init__(self, specialty, capabilities):
        self.specialty = specialty
        self.capabilities = capabilities
    
    def contribute_expertise(self, problem):
        if self.is_relevant_to_specialty(problem):
            return self.apply_specialized_knowledge(problem)
        return None
```

**Example**: Medical diagnosis team (cardiologist, radiologist, etc.)

### 5. **Consensus Building Pattern**
**Use Case**: Reaching agreement among multiple agents
**When to Use**: Democratic decision making, conflict resolution

```python
class ConsensusAgent:
    def participate_in_consensus(self, proposal, other_agents):
        my_position = self.evaluate_proposal(proposal)
        negotiation_result = self.negotiate_with_others(my_position, other_agents)
        return self.commit_to_consensus(negotiation_result)
```

**Example**: Investment committee, policy making systems

---

## Communication Patterns

### 1. **Message Passing**
**Use Case**: Direct communication between agents
**When to Use**: Simple, direct coordination needs

```python
class MessagePassingAgent:
    def send_message(self, recipient, message):
        recipient.receive_message(self.agent_id, message)
    
    def receive_message(self, sender_id, message):
        response = self.process_message(message)
        if response:
            self.send_message(sender_id, response)
```

### 2. **Publish-Subscribe**
**Use Case**: Event-driven communication
**When to Use**: Loose coupling, event notifications

```python
class PubSubAgent:
    def __init__(self):
        self.subscriptions = set()
    
    def subscribe(self, topic):
        self.subscriptions.add(topic)
        message_bus.subscribe(self, topic)
    
    def publish(self, topic, data):
        message_bus.publish(topic, data)
```

### 3. **Broadcast Communication**
**Use Case**: One-to-many communication
**When to Use**: Announcements, coordination messages

```python
class BroadcastAgent:
    def broadcast(self, message, recipient_group):
        for agent in recipient_group:
            agent.receive_broadcast(self.agent_id, message)
```

### 4. **Request-Response Protocol**
**Use Case**: Service-oriented communication
**When to Use**: Client-server interactions, service requests

```python
class ServiceAgent:
    def handle_request(self, request):
        if self.can_handle(request):
            result = self.process_request(request)
            return self.create_response(result)
        return self.create_error_response("Cannot handle request")
```

---

## Execution Patterns

### 1. **Pipeline Pattern**
**Use Case**: Sequential processing stages
**When to Use**: Data transformation, assembly lines

```python
class PipelineAgent:
    def __init__(self, stages):
        self.stages = stages
    
    def process_pipeline(self, input_data):
        current_data = input_data
        for stage in self.stages:
            current_data = stage.process(current_data)
        return current_data
```

### 2. **State Machine Pattern**
**Use Case**: Behavior changes based on state
**When to Use**: Complex state-dependent behavior

```python
class StateMachineAgent:
    def __init__(self):
        self.current_state = "initial"
        self.state_transitions = {}
    
    def process_event(self, event):
        new_state = self.state_transitions.get((self.current_state, event))
        if new_state:
            self.transition_to_state(new_state)
            return self.execute_state_behavior(new_state)
```

### 3. **Feedback Loop Pattern**
**Use Case**: Continuous improvement based on results
**When to Use**: Adaptive systems, optimization

```python
class FeedbackLoopAgent:
    def execute_with_feedback(self, task):
        result = self.execute_task(task)
        feedback = self.collect_feedback(result)
        self.adjust_behavior(feedback)
        return result
```

### 4. **Error Recovery Pattern**
**Use Case**: Graceful handling of failures
**When to Use**: Robust systems requiring high availability

```python
class ErrorRecoveryAgent:
    def execute_with_recovery(self, task):
        try:
            return self.execute_task(task)
        except Exception as e:
            recovery_strategy = self.select_recovery_strategy(e, task)
            return recovery_strategy.recover(task, e)
```

---

## Learning Patterns

### 1. **Experience Replay Pattern**
**Use Case**: Learning from past experiences
**When to Use**: Reinforcement learning, skill improvement

```python
class ExperienceReplayAgent:
    def __init__(self):
        self.experience_buffer = ExperienceBuffer()
    
    def learn_from_experience(self):
        batch = self.experience_buffer.sample_batch()
        self.update_model(batch)
```

### 2. **Meta-Learning Pattern**
**Use Case**: Learning how to learn better
**When to Use**: Rapid adaptation to new tasks

```python
class MetaLearningAgent:
    def adapt_to_new_task(self, new_task):
        adaptation_strategy = self.meta_learner.generate_strategy(new_task)
        self.apply_adaptation_strategy(adaptation_strategy)
```

### 3. **Transfer Learning Pattern**
**Use Case**: Applying knowledge from one domain to another
**When to Use**: Limited data in target domain

```python
class TransferLearningAgent:
    def transfer_knowledge(self, source_domain_knowledge, target_domain):
        transferable_features = self.identify_transferable_features(
            source_domain_knowledge, target_domain
        )
        self.adapt_features_to_target(transferable_features, target_domain)
```

### 4. **Curriculum Learning Pattern**
**Use Case**: Learning in progressive difficulty
**When to Use**: Complex skill acquisition

```python
class CurriculumLearningAgent:
    def learn_with_curriculum(self, curriculum):
        for difficulty_level in curriculum:
            tasks = curriculum.get_tasks(difficulty_level)
            self.master_tasks(tasks)
            if self.evaluate_mastery(tasks):
                continue
            else:
                self.repeat_difficulty_level(difficulty_level)
```

---

## Integration Patterns

### 1. **Adapter Pattern**
**Use Case**: Integrating with external systems
**When to Use**: Legacy system integration

```python
class SystemAdapter:
    def __init__(self, external_system):
        self.external_system = external_system
    
    def adapt_request(self, agent_request):
        external_format = self.convert_to_external_format(agent_request)
        external_response = self.external_system.process(external_format)
        return self.convert_to_agent_format(external_response)
```

### 2. **Facade Pattern**
**Use Case**: Simplifying complex subsystems
**When to Use**: Complex system interactions

```python
class AgentFacade:
    def __init__(self):
        self.subsystem_agents = [AgentA(), AgentB(), AgentC()]
    
    def simplified_operation(self, request):
        # Coordinates multiple agents behind simple interface
        return self.orchestrate_subsystems(request)
```

### 3. **Observer Pattern**
**Use Case**: Monitoring and event handling
**When to Use**: Event-driven architectures

```python
class ObservableAgent:
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify_observers(self, event):
        for observer in self.observers:
            observer.handle_event(event)
```

### 4. **Proxy Pattern**
**Use Case**: Controlling access to agents
**When to Use**: Security, caching, remote access

```python
class AgentProxy:
    def __init__(self, real_agent):
        self.real_agent = real_agent
        self.access_control = AccessControl()
    
    def process_request(self, request, user):
        if self.access_control.has_permission(user, request):
            return self.real_agent.process_request(request)
        else:
            raise PermissionDeniedError()
```

---

## Pattern Selection Guide

| **Scenario** | **Recommended Pattern** | **Key Benefits** |
|-------------|------------------------|------------------|
| Complex reasoning task | ReAct, Chain-of-Thought | Structured thinking |
| Multi-step project | Plan-and-Execute | Clear planning and tracking |
| External tool usage | Tool-Using | Leverage specialized capabilities |
| Quality improvement | Self-Reflection | Enhanced output quality |
| Parallel processing | Master-Worker | Scalability and efficiency |
| Team collaboration | Peer-to-Peer, Hierarchical | Coordinated group work |
| Specialized expertise | Specialized Roles | Domain-specific knowledge |
| Decision making | Consensus Building | Democratic decisions |
| Event handling | Publish-Subscribe | Loose coupling |
| State-dependent behavior | State Machine | Complex behavior management |
| Continuous improvement | Feedback Loop | Adaptive performance |
| System integration | Adapter, Facade | External system connectivity |

---

## Quick Pattern Comparison

### **Single Agent vs Multi-Agent**
- **Single Agent**: Simpler, faster, better for focused tasks
- **Multi-Agent**: More scalable, robust, better for complex problems

### **Synchronous vs Asynchronous**
- **Synchronous**: Immediate response, simpler coordination
- **Asynchronous**: Better scalability, non-blocking operations

### **Centralized vs Distributed**
- **Centralized**: Easier coordination, single point of control
- **Distributed**: Better fault tolerance, scalability

### **Reactive vs Proactive**
- **Reactive**: Responds to events, simpler implementation
- **Proactive**: Anticipates needs, more intelligent behavior

This cheatsheet provides quick reference for selecting and implementing appropriate agent patterns based on your specific use case and requirements.
