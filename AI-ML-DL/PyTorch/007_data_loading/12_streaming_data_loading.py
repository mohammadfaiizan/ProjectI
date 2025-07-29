"""
PyTorch Streaming Data Loading - Real-time Data Processing
Comprehensive guide to streaming data loading for online learning and real-time applications
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import time
import threading
import queue
from collections import deque
import asyncio
import random
from typing import Iterator, Any, Optional
import json

print("=== STREAMING DATA LOADING ===")

# 1. STREAMING DATASET FUNDAMENTALS
print("\n1. STREAMING DATASET FUNDAMENTALS")

class StreamingDataset(IterableDataset):
    """Basic streaming dataset for continuous data streams"""
    
    def __init__(self, stream_generator, buffer_size=1000):
        self.stream_generator = stream_generator
        self.buffer_size = buffer_size
        
    def __iter__(self):
        return iter(self.stream_generator())

# Example stream generator
def synthetic_stream():
    """Generate synthetic data stream"""
    while True:
        x = torch.randn(10)
        y = torch.randint(0, 2, (1,)).float()
        yield x, y

streaming_ds = StreamingDataset(synthetic_stream)
streaming_dl = DataLoader(streaming_ds, batch_size=32, num_workers=0)

print("Basic streaming dataset created")

# 2. BUFFERED STREAMING
print("\n2. BUFFERED STREAMING")

class BufferedStreamingDataset(IterableDataset):
    """Streaming dataset with buffering for smooth data flow"""
    
    def __init__(self, stream_source, buffer_size=10000, batch_size=32):
        self.stream_source = stream_source
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.is_filling = False
        
    def _fill_buffer(self):
        """Fill buffer with streaming data"""
        self.is_filling = True
        for data in self.stream_source():
            if len(self.buffer) >= self.buffer_size:
                # Remove oldest items if buffer full
                for _ in range(self.batch_size):
                    if self.buffer:
                        self.buffer.popleft()
            self.buffer.append(data)
            
    def __iter__(self):
        # Start filling buffer in background
        if not self.is_filling:
            fill_thread = threading.Thread(target=self._fill_buffer)
            fill_thread.daemon = True
            fill_thread.start()
            
        # Wait for initial buffer fill
        while len(self.buffer) < min(1000, self.buffer_size):
            time.sleep(0.01)
            
        # Yield data from buffer
        while True:
            if len(self.buffer) >= self.batch_size:
                batch_data = []
                for _ in range(self.batch_size):
                    if self.buffer:
                        batch_data.append(self.buffer.popleft())
                
                if batch_data:
                    # Convert to tensors
                    xs, ys = zip(*batch_data)
                    yield torch.stack(xs), torch.stack(ys)
            else:
                time.sleep(0.001)  # Small delay if buffer low

buffered_ds = BufferedStreamingDataset(synthetic_stream, buffer_size=5000)
print("Buffered streaming dataset created")

# 3. THREADED STREAMING
print("\n3. THREADED STREAMING")

class ThreadedStreamingDataset(IterableDataset):
    """Multi-threaded streaming dataset for concurrent data processing"""
    
    def __init__(self, stream_source, num_threads=2, queue_size=1000):
        self.stream_source = stream_source
        self.num_threads = num_threads
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        
    def _producer_worker(self, worker_id):
        """Producer thread worker"""
        stream_iter = self.stream_source()
        while not self.stop_event.is_set():
            try:
                data = next(stream_iter)
                self.data_queue.put((worker_id, data), timeout=1.0)
            except StopIteration:
                break
            except queue.Full:
                continue
                
    def start_producers(self):
        """Start producer threads"""
        self.threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=self._producer_worker, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            
    def __iter__(self):
        self.start_producers()
        
        while True:
            try:
                worker_id, data = self.data_queue.get(timeout=1.0)
                yield data
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

threaded_ds = ThreadedStreamingDataset(synthetic_stream, num_threads=3)
print("Threaded streaming dataset created")

# 4. ONLINE LEARNING DATASET
print("\n4. ONLINE LEARNING DATASET")

class OnlineLearningDataset(IterableDataset):
    """Dataset for online learning with concept drift simulation"""
    
    def __init__(self, base_distribution, drift_rate=0.001, noise_level=0.1):
        self.base_distribution = base_distribution
        self.drift_rate = drift_rate
        self.noise_level = noise_level
        self.time_step = 0
        
    def _generate_sample(self):
        """Generate sample with potential concept drift"""
        # Simulate concept drift
        drift_factor = np.sin(self.time_step * self.drift_rate) * 0.5
        noise = np.random.normal(0, self.noise_level)
        
        # Generate base sample
        x = torch.randn(5) + drift_factor + noise
        
        # Label with drift
        label_prob = torch.sigmoid(x.sum() + drift_factor)
        y = torch.bernoulli(label_prob).long()
        
        self.time_step += 1
        return x, y
        
    def __iter__(self):
        while True:
            yield self._generate_sample()

online_ds = OnlineLearningDataset("normal")
online_dl = DataLoader(online_ds, batch_size=16, num_workers=0)

# Simulate online learning
model = nn.Linear(5, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("Simulating online learning...")
for i, (x, y) in enumerate(online_dl):
    if i >= 5:  # Just a few iterations for demo
        break
        
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    print(f"Step {i}: Loss = {loss.item():.4f}")

# 5. STREAMING WITH BACKPRESSURE
print("\n5. STREAMING WITH BACKPRESSURE")

class BackpressureStreamingDataset(IterableDataset):
    """Streaming dataset with backpressure handling"""
    
    def __init__(self, stream_source, max_queue_size=1000, backpressure_threshold=0.8):
        self.stream_source = stream_source
        self.max_queue_size = max_queue_size
        self.backpressure_threshold = backpressure_threshold
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.producer_thread = None
        self.stop_event = threading.Event()
        
    def _producer(self):
        """Producer with backpressure awareness"""
        stream_iter = self.stream_source()
        while not self.stop_event.is_set():
            try:
                # Check backpressure
                queue_utilization = self.data_queue.qsize() / self.max_queue_size
                
                if queue_utilization > self.backpressure_threshold:
                    # Apply backpressure - slow down production
                    time.sleep(0.01)
                    continue
                
                data = next(stream_iter)
                self.data_queue.put(data, timeout=0.1)
                
            except StopIteration:
                break
            except queue.Full:
                # Drop data if queue still full
                continue
                
    def __iter__(self):
        if self.producer_thread is None or not self.producer_thread.is_alive():
            self.producer_thread = threading.Thread(target=self._producer)
            self.producer_thread.daemon = True
            self.producer_thread.start()
            
        while True:
            try:
                data = self.data_queue.get(timeout=1.0)
                yield data
            except queue.Empty:
                if self.stop_event.is_set():
                    break

backpressure_ds = BackpressureStreamingDataset(synthetic_stream)
print("Backpressure streaming dataset created")

# 6. ADAPTIVE BATCH SIZE STREAMING
print("\n6. ADAPTIVE BATCH SIZE STREAMING")

class AdaptiveBatchStreamingDataset(IterableDataset):
    """Streaming dataset with adaptive batch sizing"""
    
    def __init__(self, stream_source, min_batch=16, max_batch=128, target_latency=0.1):
        self.stream_source = stream_source
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_latency = target_latency
        self.current_batch_size = min_batch
        self.latency_history = deque(maxlen=10)
        
    def _adjust_batch_size(self):
        """Adjust batch size based on latency"""
        if len(self.latency_history) < 3:
            return
            
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        
        if avg_latency > self.target_latency * 1.2:
            # Too slow, decrease batch size
            self.current_batch_size = max(self.min_batch, 
                                        int(self.current_batch_size * 0.9))
        elif avg_latency < self.target_latency * 0.8:
            # Too fast, increase batch size
            self.current_batch_size = min(self.max_batch, 
                                        int(self.current_batch_size * 1.1))
                                        
    def __iter__(self):
        stream_iter = self.stream_source()
        buffer = []
        
        while True:
            start_time = time.time()
            
            # Collect batch
            while len(buffer) < self.current_batch_size:
                try:
                    data = next(stream_iter)
                    buffer.append(data)
                except StopIteration:
                    break
                    
            if buffer:
                # Process batch
                batch_x, batch_y = zip(*buffer)
                batch_tensor_x = torch.stack(batch_x)
                batch_tensor_y = torch.stack(batch_y)
                
                end_time = time.time()
                latency = end_time - start_time
                self.latency_history.append(latency)
                
                # Adjust batch size for next iteration
                self._adjust_batch_size()
                
                buffer.clear()
                yield batch_tensor_x, batch_tensor_y

adaptive_ds = AdaptiveBatchStreamingDataset(synthetic_stream)
print(f"Adaptive batch streaming dataset created")

# 7. MEMORY-EFFICIENT STREAMING
print("\n7. MEMORY-EFFICIENT STREAMING")

class MemoryEfficientStreamingDataset(IterableDataset):
    """Memory-efficient streaming with automatic cleanup"""
    
    def __init__(self, stream_source, memory_limit_mb=100):
        self.stream_source = stream_source
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.memory_usage = 0
        self.cache = {}
        self.access_order = deque()
        
    def _estimate_memory(self, data):
        """Estimate memory usage of data"""
        x, y = data
        return x.numel() * x.element_size() + y.numel() * y.element_size()
        
    def _cleanup_memory(self):
        """Remove old data to stay under memory limit"""
        while self.memory_usage > self.memory_limit_bytes and self.access_order:
            old_key = self.access_order.popleft()
            if old_key in self.cache:
                data = self.cache.pop(old_key)
                self.memory_usage -= self._estimate_memory(data)
                
    def __iter__(self):
        stream_iter = self.stream_source()
        key_counter = 0
        
        for data in stream_iter:
            # Estimate and track memory
            data_size = self._estimate_memory(data)
            
            # Cleanup if needed
            if self.memory_usage + data_size > self.memory_limit_bytes:
                self._cleanup_memory()
                
            # Cache data
            key = key_counter
            self.cache[key] = data
            self.access_order.append(key)
            self.memory_usage += data_size
            key_counter += 1
            
            yield data

memory_efficient_ds = MemoryEfficientStreamingDataset(synthetic_stream, memory_limit_mb=50)
print("Memory-efficient streaming dataset created")

# 8. STREAMING PERFORMANCE MONITORING
print("\n8. STREAMING PERFORMANCE MONITORING")

class MonitoredStreamingDataset(IterableDataset):
    """Streaming dataset with performance monitoring"""
    
    def __init__(self, stream_source, monitor_interval=100):
        self.stream_source = stream_source
        self.monitor_interval = monitor_interval
        self.sample_count = 0
        self.total_time = 0
        self.throughput_history = []
        
    def _log_metrics(self):
        """Log performance metrics"""
        if self.sample_count > 0:
            avg_time_per_sample = self.total_time / self.sample_count
            throughput = self.sample_count / self.total_time if self.total_time > 0 else 0
            self.throughput_history.append(throughput)
            
            print(f"Samples: {self.sample_count}, "
                  f"Avg time/sample: {avg_time_per_sample:.4f}s, "
                  f"Throughput: {throughput:.2f} samples/s")
                  
    def __iter__(self):
        stream_iter = self.stream_source()
        start_time = time.time()
        
        for data in stream_iter:
            self.sample_count += 1
            current_time = time.time()
            self.total_time = current_time - start_time
            
            # Log metrics periodically
            if self.sample_count % self.monitor_interval == 0:
                self._log_metrics()
                
            yield data

monitored_ds = MonitoredStreamingDataset(synthetic_stream, monitor_interval=50)
print("Monitored streaming dataset created")

# 9. FAULT-TOLERANT STREAMING
print("\n9. FAULT-TOLERANT STREAMING")

class FaultTolerantStreamingDataset(IterableDataset):
    """Streaming dataset with fault tolerance and recovery"""
    
    def __init__(self, stream_sources, retry_attempts=3, failover_delay=1.0):
        self.stream_sources = stream_sources  # List of backup sources
        self.retry_attempts = retry_attempts
        self.failover_delay = failover_delay
        self.current_source_idx = 0
        self.failure_count = 0
        
    def _get_current_stream(self):
        """Get current active stream"""
        return self.stream_sources[self.current_source_idx]()
        
    def _failover_to_next_source(self):
        """Switch to next available stream source"""
        self.current_source_idx = (self.current_source_idx + 1) % len(self.stream_sources)
        self.failure_count = 0
        time.sleep(self.failover_delay)
        print(f"Failed over to source {self.current_source_idx}")
        
    def __iter__(self):
        stream_iter = self._get_current_stream()
        
        while True:
            try:
                data = next(stream_iter)
                self.failure_count = 0  # Reset on success
                yield data
                
            except StopIteration:
                # Stream ended, try next source
                if len(self.stream_sources) > 1:
                    self._failover_to_next_source()
                    stream_iter = self._get_current_stream()
                else:
                    break
                    
            except Exception as e:
                self.failure_count += 1
                print(f"Stream error: {e}, attempt {self.failure_count}")
                
                if self.failure_count >= self.retry_attempts:
                    if len(self.stream_sources) > 1:
                        self._failover_to_next_source()
                        stream_iter = self._get_current_stream()
                    else:
                        raise
                else:
                    time.sleep(0.1 * self.failure_count)  # Exponential backoff

# Multiple stream sources for fault tolerance
fault_tolerant_ds = FaultTolerantStreamingDataset([synthetic_stream, synthetic_stream])
print("Fault-tolerant streaming dataset created")

# 10. STREAMING BEST PRACTICES
print("\n10. STREAMING BEST PRACTICES")

class ProductionStreamingDataset(IterableDataset):
    """Production-ready streaming dataset with all best practices"""
    
    def __init__(self, 
                 stream_source,
                 buffer_size=10000,
                 batch_size=32,
                 num_workers=2,
                 memory_limit_mb=500,
                 monitor_interval=1000,
                 enable_checkpointing=True,
                 checkpoint_interval=10000):
        
        self.stream_source = stream_source
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.monitor_interval = monitor_interval
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize components
        self.buffer = deque(maxlen=buffer_size)
        self.metrics = {
            'samples_processed': 0,
            'bytes_processed': 0,
            'avg_latency': 0,
            'throughput': 0
        }
        
    def save_checkpoint(self, filepath):
        """Save streaming state checkpoint"""
        if self.enable_checkpointing:
            checkpoint = {
                'metrics': self.metrics,
                'buffer_size': len(self.buffer),
                'timestamp': time.time()
            }
            torch.save(checkpoint, filepath)
            
    def load_checkpoint(self, filepath):
        """Load streaming state checkpoint"""
        if self.enable_checkpointing:
            checkpoint = torch.load(filepath)
            self.metrics = checkpoint['metrics']
            
    def get_metrics(self):
        """Get current performance metrics"""
        return self.metrics.copy()
        
    def __iter__(self):
        stream_iter = self.stream_source()
        start_time = time.time()
        
        for data in stream_iter:
            self.buffer.append(data)
            self.metrics['samples_processed'] += 1
            
            # Batch processing
            if len(self.buffer) >= self.batch_size:
                batch_data = []
                for _ in range(self.batch_size):
                    if self.buffer:
                        batch_data.append(self.buffer.popleft())
                
                if batch_data:
                    xs, ys = zip(*batch_data)
                    batch_x = torch.stack(xs)
                    batch_y = torch.stack(ys)
                    
                    # Update metrics
                    current_time = time.time()
                    self.metrics['avg_latency'] = (current_time - start_time) / self.metrics['samples_processed']
                    self.metrics['throughput'] = self.metrics['samples_processed'] / (current_time - start_time)
                    
                    # Checkpoint
                    if (self.enable_checkpointing and 
                        self.metrics['samples_processed'] % self.checkpoint_interval == 0):
                        self.save_checkpoint(f"streaming_checkpoint_{self.metrics['samples_processed']}.pt")
                    
                    yield batch_x, batch_y

production_ds = ProductionStreamingDataset(synthetic_stream)
print("Production streaming dataset created")

print("\n=== STREAMING DATA LOADING COMPLETE ===")
print("Key concepts covered:")
print("- Basic streaming datasets")
print("- Buffered and threaded streaming") 
print("- Online learning with concept drift")
print("- Backpressure handling")
print("- Adaptive batch sizing")
print("- Memory-efficient streaming")
print("- Performance monitoring")
print("- Fault tolerance and recovery")
print("- Production best practices")