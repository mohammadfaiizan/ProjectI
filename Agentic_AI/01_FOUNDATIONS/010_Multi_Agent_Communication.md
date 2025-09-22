# Multi-Agent Communication: Coordination and Messaging Patterns

## Communication Pattern Overview

| Pattern | Latency | Scalability | Reliability | Use Case |
|---------|---------|-------------|-------------|----------|
| **Direct Messaging** | Low | Low | Medium | Simple coordination |
| **Message Queue** | Medium | High | High | Async processing |
| **Publish-Subscribe** | Medium | High | Medium | Event broadcasting |
| **Request-Response** | Low | Medium | High | Service calls |
| **Gossip Protocol** | High | Very High | High | Distributed consensus |
| **Shared Memory** | Very Low | Low | Low | High-speed coordination |

---

## Direct Communication Patterns

### **1. Point-to-Point Messaging**
```python
class DirectMessageAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.connections = {}
        self.message_queue = asyncio.Queue()
    
    async def send_message(self, recipient_id, message):
        if recipient_id in self.connections:
            await self.connections[recipient_id].receive_message(
                self.agent_id, message
            )
    
    async def receive_message(self, sender_id, message):
        await self.message_queue.put({
            'sender': sender_id,
            'message': message,
            'timestamp': time.time()
        })
        await self.process_message(sender_id, message)
```

### **2. Broadcast Communication**
```python
class BroadcastAgent:
    def __init__(self, agent_id, agent_group):
        self.agent_id = agent_id
        self.agent_group = agent_group
    
    async def broadcast(self, message, exclude_self=True):
        tasks = []
        for agent in self.agent_group:
            if exclude_self and agent.agent_id == self.agent_id:
                continue
            tasks.append(agent.receive_broadcast(self.agent_id, message))
        
        await asyncio.gather(*tasks)
    
    async def receive_broadcast(self, sender_id, message):
        # Handle broadcast message
        await self.handle_broadcast_message(sender_id, message)
```

---

## Message Queue Patterns

### **3. Asynchronous Message Queue**
```python
class QueuedCommunicationAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.message_broker = MessageBroker()
        self.subscribed_queues = set()
    
    async def send_to_queue(self, queue_name, message):
        await self.message_broker.enqueue(queue_name, {
            'sender': self.agent_id,
            'message': message,
            'timestamp': time.time()
        })
    
    async def subscribe_to_queue(self, queue_name):
        self.subscribed_queues.add(queue_name)
        asyncio.create_task(self.process_queue(queue_name))
    
    async def process_queue(self, queue_name):
        while queue_name in self.subscribed_queues:
            message = await self.message_broker.dequeue(queue_name)
            if message:
                await self.handle_queued_message(message)
            await asyncio.sleep(0.1)
```

### **4. Priority Message Queue**
```python
import heapq

class PriorityQueueAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.priority_queue = []
        self.queue_lock = asyncio.Lock()
    
    async def send_priority_message(self, recipient, message, priority):
        async with self.queue_lock:
            heapq.heappush(self.priority_queue, (priority, {
                'recipient': recipient,
                'message': message,
                'sender': self.agent_id,
                'timestamp': time.time()
            }))
    
    async def process_priority_messages(self):
        while True:
            async with self.queue_lock:
                if self.priority_queue:
                    priority, message_data = heapq.heappop(self.priority_queue)
                    await self.handle_priority_message(message_data)
            await asyncio.sleep(0.1)
```

---

## Publish-Subscribe Patterns

### **5. Topic-Based Pub-Sub**
```python
class PubSubAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.event_bus = EventBus()
        self.subscriptions = {}
    
    async def publish(self, topic, data):
        event = {
            'topic': topic,
            'data': data,
            'publisher': self.agent_id,
            'timestamp': time.time()
        }
        await self.event_bus.publish(topic, event)
    
    async def subscribe(self, topic, handler=None):
        if handler is None:
            handler = self.default_event_handler
        
        self.subscriptions[topic] = handler
        await self.event_bus.subscribe(topic, self.agent_id, handler)
    
    async def default_event_handler(self, event):
        # Default processing for subscribed events
        await self.process_event(event)

class EventBus:
    def __init__(self):
        self.subscribers = {}
    
    async def publish(self, topic, event):
        if topic in self.subscribers:
            tasks = []
            for agent_id, handler in self.subscribers[topic].items():
                tasks.append(handler(event))
            await asyncio.gather(*tasks)
    
    async def subscribe(self, topic, agent_id, handler):
        if topic not in self.subscribers:
            self.subscribers[topic] = {}
        self.subscribers[topic][agent_id] = handler
```

---

## Request-Response Patterns

### **6. Synchronous Request-Response**
```python
class RequestResponseAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.pending_requests = {}
        self.service_handlers = {}
    
    async def request(self, target_agent, service, data, timeout=30):
        request_id = self.generate_request_id()
        
        # Send request
        request_message = {
            'request_id': request_id,
            'service': service,
            'data': data,
            'sender': self.agent_id
        }
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_requests[request_id] = response_future
        
        await target_agent.receive_request(request_message)
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(response_future, timeout)
            return response
        except asyncio.TimeoutError:
            del self.pending_requests[request_id]
            raise TimeoutError(f"Request {request_id} timed out")
    
    async def receive_request(self, request_message):
        service = request_message['service']
        if service in self.service_handlers:
            response_data = await self.service_handlers[service](
                request_message['data']
            )
            
            # Send response back
            response = {
                'request_id': request_message['request_id'],
                'data': response_data,
                'status': 'success'
            }
        else:
            response = {
                'request_id': request_message['request_id'],
                'error': f"Service {service} not found",
                'status': 'error'
            }
        
        await self.send_response(request_message['sender'], response)
    
    def register_service(self, service_name, handler):
        self.service_handlers[service_name] = handler
```

---

## Distributed Communication Patterns

### **7. Gossip Protocol**
```python
class GossipAgent:
    def __init__(self, agent_id, peer_agents):
        self.agent_id = agent_id
        self.peer_agents = peer_agents
        self.gossip_data = {}
        self.gossip_interval = 1.0
    
    async def start_gossip(self):
        while True:
            await self.gossip_round()
            await asyncio.sleep(self.gossip_interval)
    
    async def gossip_round(self):
        # Select random subset of peers
        import random
        selected_peers = random.sample(
            self.peer_agents, 
            min(3, len(self.peer_agents))
        )
        
        # Share data with selected peers
        for peer in selected_peers:
            await self.exchange_gossip(peer)
    
    async def exchange_gossip(self, peer):
        # Send our data
        my_data = self.prepare_gossip_data()
        peer_data = await peer.receive_gossip(self.agent_id, my_data)
        
        # Merge received data
        self.merge_gossip_data(peer_data)
    
    async def receive_gossip(self, sender_id, gossip_data):
        # Merge received data
        self.merge_gossip_data(gossip_data)
        
        # Return our data
        return self.prepare_gossip_data()
```

### **8. Consensus Communication**
```python
class ConsensusAgent:
    def __init__(self, agent_id, peer_agents):
        self.agent_id = agent_id
        self.peer_agents = peer_agents
        self.consensus_rounds = {}
    
    async def propose_value(self, proposal_id, value):
        # Phase 1: Prepare
        prepare_responses = await self.send_prepare(proposal_id)
        
        if self.majority_agree(prepare_responses):
            # Phase 2: Accept
            accept_responses = await self.send_accept(proposal_id, value)
            
            if self.majority_agree(accept_responses):
                # Phase 3: Commit
                await self.send_commit(proposal_id, value)
                return True
        
        return False
    
    async def send_prepare(self, proposal_id):
        tasks = []
        for peer in self.peer_agents:
            tasks.append(peer.handle_prepare(self.agent_id, proposal_id))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in responses if not isinstance(r, Exception)]
    
    async def handle_prepare(self, proposer_id, proposal_id):
        # Consensus algorithm logic
        if self.can_accept_proposal(proposal_id):
            return {'status': 'promise', 'agent_id': self.agent_id}
        else:
            return {'status': 'reject', 'agent_id': self.agent_id}
```

---

## Communication Middleware

### **9. Message Router**
```python
class MessageRouter:
    def __init__(self):
        self.routes = {}
        self.filters = []
        self.transformers = []
    
    def add_route(self, pattern, destination):
        self.routes[pattern] = destination
    
    def add_filter(self, filter_func):
        self.filters.append(filter_func)
    
    def add_transformer(self, transformer_func):
        self.transformers.append(transformer_func)
    
    async def route_message(self, message):
        # Apply filters
        for filter_func in self.filters:
            if not await filter_func(message):
                return False  # Message filtered out
        
        # Apply transformations
        for transformer in self.transformers:
            message = await transformer(message)
        
        # Find destination
        destination = self.find_destination(message)
        if destination:
            await destination.receive_message(message)
            return True
        
        return False
    
    def find_destination(self, message):
        for pattern, destination in self.routes.items():
            if self.pattern_matches(pattern, message):
                return destination
        return None
```

### **10. Communication Protocol Stack**
```python
class ProtocolStack:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    async def send(self, message, destination):
        # Process through layers (top to bottom)
        processed_message = message
        for layer in reversed(self.layers):
            processed_message = await layer.process_outgoing(processed_message)
        
        # Send to destination
        await destination.receive_raw(processed_message)
    
    async def receive(self, raw_message):
        # Process through layers (bottom to top)
        processed_message = raw_message
        for layer in self.layers:
            processed_message = await layer.process_incoming(processed_message)
        
        return processed_message

class ReliabilityLayer:
    def __init__(self):
        self.pending_acks = {}
        self.sequence_numbers = {}
    
    async def process_outgoing(self, message):
        # Add sequence number and reliability headers
        seq_num = self.get_next_sequence_number(message['destination'])
        message['sequence_number'] = seq_num
        message['requires_ack'] = True
        
        # Store for potential retransmission
        self.pending_acks[seq_num] = message
        
        return message
    
    async def process_incoming(self, message):
        # Send acknowledgment if required
        if message.get('requires_ack'):
            await self.send_ack(message)
        
        return message
```

---

## Communication Security

### **11. Secure Communication**
```python
class SecureCommunicationAgent:
    def __init__(self, agent_id, private_key, public_keys):
        self.agent_id = agent_id
        self.private_key = private_key
        self.public_keys = public_keys  # Dict of agent_id -> public_key
        self.encryptor = MessageEncryptor()
    
    async def send_secure_message(self, recipient_id, message):
        if recipient_id not in self.public_keys:
            raise ValueError(f"No public key for {recipient_id}")
        
        # Encrypt message
        encrypted_message = self.encryptor.encrypt(
            message, 
            self.public_keys[recipient_id]
        )
        
        # Sign message
        signature = self.encryptor.sign(encrypted_message, self.private_key)
        
        secure_envelope = {
            'encrypted_message': encrypted_message,
            'signature': signature,
            'sender': self.agent_id
        }
        
        await self.send_message(recipient_id, secure_envelope)
    
    async def receive_secure_message(self, sender_id, secure_envelope):
        # Verify signature
        if not self.encryptor.verify_signature(
            secure_envelope['encrypted_message'],
            secure_envelope['signature'],
            self.public_keys[sender_id]
        ):
            raise SecurityError("Invalid message signature")
        
        # Decrypt message
        decrypted_message = self.encryptor.decrypt(
            secure_envelope['encrypted_message'],
            self.private_key
        )
        
        return decrypted_message
```

---

## Communication Performance Optimization

### **12. Message Batching**
```python
class BatchingAgent:
    def __init__(self, agent_id, batch_size=10, batch_timeout=1.0):
        self.agent_id = agent_id
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.message_batches = {}
    
    async def send_with_batching(self, recipient_id, message):
        if recipient_id not in self.message_batches:
            self.message_batches[recipient_id] = {
                'messages': [],
                'timer': None
            }
        
        batch = self.message_batches[recipient_id]
        batch['messages'].append(message)
        
        # Start timer if this is the first message in batch
        if len(batch['messages']) == 1:
            batch['timer'] = asyncio.create_task(
                self.batch_timeout_handler(recipient_id)
            )
        
        # Send immediately if batch is full
        if len(batch['messages']) >= self.batch_size:
            await self.flush_batch(recipient_id)
    
    async def flush_batch(self, recipient_id):
        if recipient_id in self.message_batches:
            batch = self.message_batches[recipient_id]
            
            # Cancel timer
            if batch['timer']:
                batch['timer'].cancel()
            
            # Send batched messages
            if batch['messages']:
                await self.send_batch(recipient_id, batch['messages'])
            
            # Clear batch
            del self.message_batches[recipient_id]
    
    async def batch_timeout_handler(self, recipient_id):
        await asyncio.sleep(self.batch_timeout)
        await self.flush_batch(recipient_id)
```

---

## Communication Monitoring

### **13. Communication Analytics**
```python
class CommunicationMonitor:
    def __init__(self):
        self.message_stats = {}
        self.performance_metrics = {}
        self.error_tracking = {}
    
    def track_message(self, sender_id, recipient_id, message_type, size, latency):
        # Update message statistics
        key = (sender_id, recipient_id, message_type)
        if key not in self.message_stats:
            self.message_stats[key] = {
                'count': 0,
                'total_size': 0,
                'total_latency': 0,
                'errors': 0
            }
        
        stats = self.message_stats[key]
        stats['count'] += 1
        stats['total_size'] += size
        stats['total_latency'] += latency
    
    def track_error(self, sender_id, recipient_id, error_type, error_message):
        key = (sender_id, recipient_id)
        if key not in self.error_tracking:
            self.error_tracking[key] = {}
        
        if error_type not in self.error_tracking[key]:
            self.error_tracking[key][error_type] = []
        
        self.error_tracking[key][error_type].append({
            'message': error_message,
            'timestamp': time.time()
        })
    
    def get_communication_report(self):
        return {
            'message_statistics': self.calculate_message_statistics(),
            'performance_metrics': self.calculate_performance_metrics(),
            'error_analysis': self.analyze_errors(),
            'network_topology': self.analyze_communication_patterns()
        }
```

---

## Quick Pattern Selection Guide

### **Choose Based on Requirements:**

**Low Latency Needed:**
- Direct Messaging
- Shared Memory
- Request-Response

**High Scalability Needed:**
- Message Queue
- Publish-Subscribe
- Gossip Protocol

**Reliability Critical:**
- Message Queue with persistence
- Request-Response with retries
- Consensus protocols

**Security Required:**
- Encrypted communication
- Authenticated messaging
- Secure multicast

**Event-Driven Architecture:**
- Publish-Subscribe
- Event-driven messaging
- Reactive patterns

This guide provides comprehensive patterns for implementing robust multi-agent communication systems suitable for various scales and requirements.
