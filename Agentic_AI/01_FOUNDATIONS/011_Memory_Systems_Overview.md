# Memory Systems Overview: Agent Memory Architectures

## Memory Types Comparison

| Memory Type | Duration | Capacity | Access Speed | Use Case |
|-------------|----------|----------|--------------|----------|
| **Working Memory** | Seconds-Minutes | 7±2 items | Very Fast | Active processing |
| **Short-term Memory** | Minutes-Hours | Limited | Fast | Recent context |
| **Long-term Memory** | Days-Years | Unlimited | Medium | Knowledge storage |
| **Episodic Memory** | Variable | Large | Medium | Experience records |
| **Semantic Memory** | Permanent | Large | Fast | Factual knowledge |
| **Procedural Memory** | Permanent | Medium | Fast | Skills & procedures |

---

## Working Memory Implementation

### **Limited Capacity Working Memory**
```python
from collections import deque
import time

class WorkingMemory:
    def __init__(self, capacity=7):
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
        self.attention_weights = {}
    
    def add_item(self, item, attention_weight=1.0):
        """Add item to working memory with attention weight"""
        item_id = self.generate_item_id(item)
        
        # Remove if at capacity and new item has higher attention
        if len(self.items) >= self.capacity:
            self.evict_least_attended()
        
        self.items.append({
            'id': item_id,
            'content': item,
            'timestamp': time.time(),
            'access_count': 0
        })
        self.attention_weights[item_id] = attention_weight
    
    def get_active_context(self):
        """Get currently active context from working memory"""
        sorted_items = sorted(
            self.items,
            key=lambda x: self.attention_weights[x['id']],
            reverse=True
        )
        return [item['content'] for item in sorted_items]
    
    def evict_least_attended(self):
        """Remove item with lowest attention weight"""
        if not self.items:
            return
        
        least_attended = min(
            self.items,
            key=lambda x: self.attention_weights[x['id']]
        )
        self.items.remove(least_attended)
        del self.attention_weights[least_attended['id']]
```

---

## Episodic Memory System

### **Experience-Based Memory**
```python
class EpisodicMemory:
    def __init__(self):
        self.episodes = []
        self.index = EpisodeIndex()
        self.similarity_threshold = 0.7
    
    def store_episode(self, experience):
        """Store new experience as episode"""
        episode = {
            'id': self.generate_episode_id(),
            'experience': experience,
            'context': experience.get('context', {}),
            'timestamp': time.time(),
            'emotional_valence': experience.get('emotion', 0),
            'importance': self.calculate_importance(experience),
            'participants': experience.get('participants', []),
            'outcome': experience.get('outcome'),
            'learned_skills': experience.get('skills_learned', [])
        }
        
        self.episodes.append(episode)
        self.index.add_episode(episode)
        
        # Trigger consolidation if needed
        if len(self.episodes) % 100 == 0:
            self.consolidate_memories()
    
    def retrieve_similar_episodes(self, current_context, k=5):
        """Retrieve episodes similar to current context"""
        similar_episodes = []
        
        for episode in self.episodes:
            similarity = self.calculate_context_similarity(
                current_context, episode['context']
            )
            
            if similarity > self.similarity_threshold:
                similar_episodes.append({
                    'episode': episode,
                    'similarity': similarity
                })
        
        # Sort by similarity and recency
        similar_episodes.sort(
            key=lambda x: (x['similarity'], -x['episode']['timestamp']),
            reverse=True
        )
        
        return similar_episodes[:k]
    
    def consolidate_memories(self):
        """Consolidate and strengthen important memories"""
        # Strengthen frequently accessed memories
        self.strengthen_frequent_memories()
        
        # Merge similar episodes
        self.merge_similar_episodes()
        
        # Archive old, unimportant memories
        self.archive_old_memories()

class PersonalAssistantMemory:
    def __init__(self):
        self.episodic_memory = EpisodicMemory()
        self.user_preferences = {}
        self.interaction_patterns = {}
    
    def remember_interaction(self, user_query, agent_response, user_feedback):
        """Remember user interaction for future reference"""
        interaction_episode = {
            'type': 'user_interaction',
            'context': {
                'query_type': self.classify_query(user_query),
                'time_of_day': time.strftime('%H:%M'),
                'day_of_week': time.strftime('%A'),
                'user_mood': self.detect_user_mood(user_query)
            },
            'experience': {
                'user_query': user_query,
                'agent_response': agent_response,
                'user_satisfaction': user_feedback.get('satisfaction', 0),
                'response_time': user_feedback.get('response_time'),
                'successful': user_feedback.get('successful', False)
            },
            'participants': ['user', 'assistant'],
            'outcome': 'successful' if user_feedback.get('successful') else 'needs_improvement'
        }
        
        self.episodic_memory.store_episode(interaction_episode)
        self.update_user_preferences(user_query, user_feedback)
```

---

## Semantic Memory System

### **Knowledge Graph Memory**
```python
class SemanticMemory:
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.concept_embeddings = {}
        self.fact_store = FactStore()
    
    def add_knowledge(self, fact, context=None):
        """Add new factual knowledge"""
        # Extract entities and relationships
        entities = self.extract_entities(fact)
        relationships = self.extract_relationships(fact)
        
        # Add to knowledge graph
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
        
        for rel in relationships:
            self.knowledge_graph.add_relationship(
                rel['subject'], rel['predicate'], rel['object']
            )
        
        # Store fact with metadata
        fact_record = {
            'content': fact,
            'context': context,
            'confidence': self.assess_fact_confidence(fact),
            'source': context.get('source') if context else 'unknown',
            'timestamp': time.time()
        }
        
        self.fact_store.store(fact_record)
    
    def query_knowledge(self, query):
        """Query semantic knowledge"""
        # Parse query to identify concepts
        query_concepts = self.extract_concepts(query)
        
        # Find related facts
        related_facts = []
        for concept in query_concepts:
            facts = self.fact_store.find_facts_about(concept)
            related_facts.extend(facts)
        
        # Use knowledge graph for inference
        inferred_facts = self.knowledge_graph.infer_facts(query_concepts)
        
        return {
            'direct_facts': related_facts,
            'inferred_facts': inferred_facts,
            'confidence_scores': self.calculate_confidence_scores(
                related_facts + inferred_facts
            )
        }
    
    def update_belief(self, fact, new_evidence):
        """Update belief in a fact based on new evidence"""
        existing_fact = self.fact_store.find_fact(fact)
        
        if existing_fact:
            # Update confidence based on new evidence
            old_confidence = existing_fact['confidence']
            evidence_weight = self.assess_evidence_weight(new_evidence)
            
            new_confidence = self.bayesian_update(
                old_confidence, evidence_weight
            )
            
            existing_fact['confidence'] = new_confidence
            existing_fact['last_updated'] = time.time()
        else:
            # Add as new fact
            self.add_knowledge(fact, {'source': 'evidence_update'})
```

---

## Procedural Memory System

### **Skill and Procedure Storage**
```python
class ProceduralMemory:
    def __init__(self):
        self.procedures = {}
        self.skill_hierarchy = SkillHierarchy()
        self.execution_history = []
    
    def learn_procedure(self, procedure_name, steps, conditions=None):
        """Learn new procedure"""
        procedure = {
            'name': procedure_name,
            'steps': steps,
            'conditions': conditions or {},
            'success_rate': 0.0,
            'execution_count': 0,
            'average_execution_time': 0.0,
            'learned_date': time.time(),
            'last_used': None
        }
        
        self.procedures[procedure_name] = procedure
        self.skill_hierarchy.add_skill(procedure_name, steps)
    
    def execute_procedure(self, procedure_name, context):
        """Execute learned procedure"""
        if procedure_name not in self.procedures:
            return None
        
        procedure = self.procedures[procedure_name]
        
        # Check if conditions are met
        if not self.check_conditions(procedure['conditions'], context):
            return {'status': 'conditions_not_met'}
        
        # Execute steps
        start_time = time.time()
        execution_result = self.execute_steps(procedure['steps'], context)
        execution_time = time.time() - start_time
        
        # Update procedure statistics
        self.update_procedure_stats(
            procedure_name, execution_result, execution_time
        )
        
        # Store execution history
        self.execution_history.append({
            'procedure': procedure_name,
            'context': context,
            'result': execution_result,
            'execution_time': execution_time,
            'timestamp': time.time()
        })
        
        return execution_result
    
    def improve_procedure(self, procedure_name, feedback):
        """Improve procedure based on execution feedback"""
        if procedure_name not in self.procedures:
            return
        
        procedure = self.procedures[procedure_name]
        
        # Analyze execution failures
        failures = [h for h in self.execution_history 
                   if h['procedure'] == procedure_name and not h['result']['success']]
        
        if failures:
            # Identify common failure patterns
            failure_patterns = self.analyze_failure_patterns(failures)
            
            # Suggest procedure modifications
            modifications = self.suggest_procedure_modifications(
                procedure, failure_patterns, feedback
            )
            
            # Apply approved modifications
            self.apply_procedure_modifications(procedure_name, modifications)

class SkillLearningAgent:
    def __init__(self):
        self.procedural_memory = ProceduralMemory()
        self.skill_detector = SkillDetector()
        self.learning_optimizer = LearningOptimizer()
    
    def learn_from_demonstration(self, demonstration):
        """Learn procedure from demonstration"""
        # Extract steps from demonstration
        steps = self.skill_detector.extract_steps(demonstration)
        
        # Identify skill name and conditions
        skill_name = self.skill_detector.identify_skill(demonstration)
        conditions = self.skill_detector.extract_conditions(demonstration)
        
        # Learn the procedure
        self.procedural_memory.learn_procedure(skill_name, steps, conditions)
        
        # Optimize learning based on past experience
        self.learning_optimizer.optimize_procedure(skill_name, demonstration)
```

---

## Memory Integration and Management

### **Unified Memory Manager**
```python
class UnifiedMemoryManager:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        self.memory_scheduler = MemoryScheduler()
    
    def process_new_information(self, information, context):
        """Process new information across memory systems"""
        
        # Add to working memory immediately
        self.working_memory.add_item(information)
        
        # Determine information type and route accordingly
        info_type = self.classify_information_type(information)
        
        if info_type == 'experience':
            self.episodic_memory.store_episode({
                'experience': information,
                'context': context
            })
        
        elif info_type == 'fact':
            self.semantic_memory.add_knowledge(information, context)
        
        elif info_type == 'procedure':
            procedure_info = self.extract_procedure_info(information)
            self.procedural_memory.learn_procedure(
                procedure_info['name'],
                procedure_info['steps'],
                procedure_info['conditions']
            )
        
        # Schedule memory consolidation
        self.memory_scheduler.schedule_consolidation()
    
    def retrieve_relevant_memories(self, query, context):
        """Retrieve relevant memories from all systems"""
        
        # Get current working memory context
        working_context = self.working_memory.get_active_context()
        
        # Retrieve from each memory system
        episodic_results = self.episodic_memory.retrieve_similar_episodes(
            context, k=3
        )
        
        semantic_results = self.semantic_memory.query_knowledge(query)
        
        relevant_procedures = self.procedural_memory.find_applicable_procedures(
            query, context
        )
        
        # Integrate and rank results
        integrated_memories = self.integrate_memory_results(
            working_context, episodic_results, semantic_results, relevant_procedures
        )
        
        return integrated_memories
    
    def consolidate_memories(self):
        """Periodic memory consolidation across systems"""
        
        # Episodic to semantic consolidation
        recent_episodes = self.episodic_memory.get_recent_episodes(
            hours=24
        )
        
        for episode in recent_episodes:
            # Extract generalizable knowledge
            general_knowledge = self.extract_general_knowledge(episode)
            if general_knowledge:
                self.semantic_memory.add_knowledge(
                    general_knowledge, 
                    {'source': 'episodic_consolidation'}
                )
        
        # Strengthen frequently accessed memories
        self.strengthen_frequent_memories()
        
        # Archive old, unused memories
        self.archive_unused_memories()

class MemoryScheduler:
    def __init__(self):
        self.consolidation_queue = []
        self.last_consolidation = time.time()
        self.consolidation_interval = 3600  # 1 hour
    
    def schedule_consolidation(self):
        """Schedule memory consolidation if needed"""
        current_time = time.time()
        
        if (current_time - self.last_consolidation) > self.consolidation_interval:
            self.consolidation_queue.append({
                'type': 'periodic_consolidation',
                'scheduled_time': current_time,
                'priority': 'normal'
            })
    
    async def run_consolidation_loop(self):
        """Run memory consolidation background loop"""
        while True:
            if self.consolidation_queue:
                consolidation_task = self.consolidation_queue.pop(0)
                await self.execute_consolidation(consolidation_task)
            
            await asyncio.sleep(60)  # Check every minute
```

---

## Memory Performance Optimization

### **Memory Caching and Indexing**
```python
class MemoryCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.keys(), 
                         key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()

class MemoryIndex:
    def __init__(self):
        self.concept_index = {}
        self.temporal_index = {}
        self.similarity_index = SimilarityIndex()
    
    def index_memory(self, memory_item):
        """Create indices for efficient memory retrieval"""
        
        # Concept-based indexing
        concepts = self.extract_concepts(memory_item)
        for concept in concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(memory_item['id'])
        
        # Temporal indexing
        timestamp = memory_item['timestamp']
        time_bucket = self.get_time_bucket(timestamp)
        if time_bucket not in self.temporal_index:
            self.temporal_index[time_bucket] = []
        self.temporal_index[time_bucket].append(memory_item['id'])
        
        # Similarity indexing
        self.similarity_index.add_item(memory_item)
```

---

## Memory Pattern Selection Guide

### **Choose Memory Type Based on:**

**Working Memory - When:**
- Need to track current active context
- Processing immediate information
- Managing attention and focus

**Episodic Memory - When:**
- Learning from experience
- Need to recall specific events
- Building personal/interaction history

**Semantic Memory - When:**
- Storing factual knowledge
- Building knowledge graphs
- Need fast fact retrieval

**Procedural Memory - When:**
- Learning skills and procedures
- Automating repeated tasks
- Improving execution efficiency

### **Memory System Design Patterns:**

1. **Hierarchical Memory**: Working → Short-term → Long-term
2. **Distributed Memory**: Specialized memory systems
3. **Associative Memory**: Content-based retrieval
4. **Adaptive Memory**: Learning and forgetting mechanisms

This overview provides the foundation for implementing sophisticated memory systems that enable agents to learn, remember, and improve over time.
