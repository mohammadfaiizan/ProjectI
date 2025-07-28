#!/usr/bin/env python3
"""PyTorch Tensor Serialization - Save, load, and pickling operations"""

import torch
import pickle
import os
import tempfile
import gzip
import json

print("=== Basic Tensor Serialization ===")

# Create sample tensors
tensor_data = torch.randn(3, 4, 5)
state_dict = {
    'weights': torch.randn(10, 5),
    'bias': torch.randn(10),
    'parameters': torch.randn(20, 20)
}

print(f"Tensor shape: {tensor_data.shape}")
print(f"State dict keys: {list(state_dict.keys())}")

# Save single tensor
torch.save(tensor_data, 'single_tensor.pt')
print("Saved single tensor to 'single_tensor.pt'")

# Load single tensor
loaded_tensor = torch.load('single_tensor.pt')
print(f"Loaded tensor shape: {loaded_tensor.shape}")
print(f"Tensors equal: {torch.equal(tensor_data, loaded_tensor)}")

# Save state dictionary
torch.save(state_dict, 'state_dict.pt')
print("Saved state dictionary")

# Load state dictionary
loaded_state_dict = torch.load('state_dict.pt')
print(f"Loaded state dict keys: {list(loaded_state_dict.keys())}")

print("\n=== File Formats and Compression ===")

# Save with different file extensions
torch.save(tensor_data, 'tensor.pth')  # Common PyTorch extension
torch.save(tensor_data, 'tensor.pkl')  # Pickle extension
torch.save(tensor_data, 'tensor.tar')  # Archive format

# Compressed saving
with gzip.open('tensor_compressed.gz', 'wb') as f:
    torch.save(tensor_data, f)
print("Saved compressed tensor")

# Load compressed
with gzip.open('tensor_compressed.gz', 'rb') as f:
    loaded_compressed = torch.load(f)
print(f"Compressed load successful: {torch.equal(tensor_data, loaded_compressed)}")

print("\n=== Device-Aware Loading ===")

# Save tensor from CPU
cpu_tensor = torch.randn(5, 5)
torch.save(cpu_tensor, 'cpu_tensor.pt')

# Load with device specification
loaded_cpu = torch.load('cpu_tensor.pt', map_location='cpu')
print(f"Loaded to CPU: {loaded_cpu.device}")

if torch.cuda.is_available():
    # Save GPU tensor
    gpu_tensor = torch.randn(5, 5, device='cuda')
    torch.save(gpu_tensor, 'gpu_tensor.pt')
    
    # Load GPU tensor to CPU
    gpu_to_cpu = torch.load('gpu_tensor.pt', map_location='cpu')
    print(f"GPU tensor loaded to CPU: {gpu_to_cpu.device}")
    
    # Load CPU tensor to GPU
    cpu_to_gpu = torch.load('cpu_tensor.pt', map_location='cuda')
    print(f"CPU tensor loaded to GPU: {cpu_to_gpu.device}")
    
    # Load with lambda mapping
    mapped_tensor = torch.load('gpu_tensor.pt', map_location=lambda storage, loc: storage)
    print(f"Lambda mapped tensor device: {mapped_tensor.device}")

print("\n=== Partial Loading and Filtering ===")

# Large state dict for partial loading
large_state_dict = {
    f'layer_{i}': torch.randn(100, 100) for i in range(10)
}
large_state_dict['metadata'] = {'version': 1.0, 'timestamp': '2024-01-01'}

torch.save(large_state_dict, 'large_state.pt')

# Load only specific keys
def load_specific_keys(filepath, keys_to_load):
    full_dict = torch.load(filepath)
    return {k: full_dict[k] for k in keys_to_load if k in full_dict}

partial_dict = load_specific_keys('large_state.pt', ['layer_0', 'layer_1', 'metadata'])
print(f"Partial loading keys: {list(partial_dict.keys())}")

# Memory-mapped loading for large files
with open('large_state.pt', 'rb') as f:
    # This would use memory mapping for very large files
    mmap_loaded = torch.load(f, map_location='cpu')
    print(f"Memory-mapped loading successful: {len(mmap_loaded)}")

print("\n=== Version Compatibility ===")

# Save with pickle protocol specification
torch.save(tensor_data, 'tensor_protocol2.pt', pickle_protocol=2)
torch.save(tensor_data, 'tensor_protocol4.pt', pickle_protocol=4)

# Load with different protocols
loaded_p2 = torch.load('tensor_protocol2.pt')
loaded_p4 = torch.load('tensor_protocol4.pt')

print(f"Protocol 2 loading: {torch.equal(tensor_data, loaded_p2)}")
print(f"Protocol 4 loading: {torch.equal(tensor_data, loaded_p4)}")

print("\n=== Custom Serialization ===")

class TensorContainer:
    def __init__(self, tensors, metadata=None):
        self.tensors = tensors
        self.metadata = metadata or {}
    
    def save(self, filepath):
        data = {
            'tensors': self.tensors,
            'metadata': self.metadata
        }
        torch.save(data, filepath)
    
    @classmethod
    def load(cls, filepath):
        data = torch.load(filepath)
        return cls(data['tensors'], data['metadata'])

# Create and save container
container = TensorContainer(
    tensors={'x': torch.randn(10, 10), 'y': torch.randn(5, 5)},
    metadata={'created_by': 'PyTorch', 'version': '2.0'}
)
container.save('tensor_container.pt')

# Load container
loaded_container = TensorContainer.load('tensor_container.pt')
print(f"Container tensors: {list(loaded_container.tensors.keys())}")
print(f"Container metadata: {loaded_container.metadata}")

print("\n=== Streaming and Buffered I/O ===")

# Save to buffer
import io

buffer = io.BytesIO()
torch.save(tensor_data, buffer)
buffer.seek(0)  # Reset buffer position

# Load from buffer
loaded_from_buffer = torch.load(buffer)
print(f"Buffer I/O successful: {torch.equal(tensor_data, loaded_from_buffer)}")

# Multiple tensors in single file
multiple_tensors = {
    'tensor1': torch.randn(50, 50),
    'tensor2': torch.randn(30, 30),
    'tensor3': torch.randn(40, 40)
}

# Save multiple tensors
torch.save(multiple_tensors, 'multiple_tensors.pt')

# Streaming load (process one at a time)
loaded_multiple = torch.load('multiple_tensors.pt')
for name, tensor in loaded_multiple.items():
    print(f"Streamed {name}: {tensor.shape}")

print("\n=== Error Handling and Validation ===")

# Safe loading with error handling
def safe_load(filepath, default=None):
    try:
        return torch.load(filepath)
    except FileNotFoundError:
        print(f"File {filepath} not found")
        return default
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return default

# Test safe loading
safe_result = safe_load('nonexistent.pt', default=torch.zeros(2, 2))
print(f"Safe load result shape: {safe_result.shape}")

# Validate loaded tensor
def validate_tensor(tensor, expected_shape=None, expected_dtype=None):
    if expected_shape and tensor.shape != expected_shape:
        raise ValueError(f"Shape mismatch: {tensor.shape} vs {expected_shape}")
    if expected_dtype and tensor.dtype != expected_dtype:
        raise ValueError(f"Dtype mismatch: {tensor.dtype} vs {expected_dtype}")
    return True

# Validation example
try:
    validate_tensor(loaded_tensor, expected_shape=(3, 4, 5), expected_dtype=torch.float32)
    print("Tensor validation passed")
except ValueError as e:
    print(f"Validation failed: {e}")

print("\n=== Cross-Platform Compatibility ===")

# Save with explicit endianness handling
big_endian_tensor = torch.randn(5, 5).to(torch.float32)
torch.save(big_endian_tensor, 'cross_platform.pt')

# Load and check
loaded_cross = torch.load('cross_platform.pt')
print(f"Cross-platform load: {torch.equal(big_endian_tensor, loaded_cross)}")

# Path handling for different OS
import pathlib

# Use pathlib for cross-platform paths
save_path = pathlib.Path('data') / 'tensor.pt'
save_path.parent.mkdir(exist_ok=True)
torch.save(tensor_data, save_path)
print(f"Saved to cross-platform path: {save_path}")

print("\n=== Incremental and Backup Strategies ===")

# Versioned saving
def save_with_version(obj, base_path, version=None):
    if version is None:
        # Auto-increment version
        version = 1
        while os.path.exists(f"{base_path}_v{version}.pt"):
            version += 1
    
    filepath = f"{base_path}_v{version}.pt"
    torch.save(obj, filepath)
    return filepath

# Save multiple versions
v1_path = save_with_version(tensor_data, 'versioned_tensor')
modified_tensor = tensor_data + 1
v2_path = save_with_version(modified_tensor, 'versioned_tensor')

print(f"Saved version 1: {v1_path}")
print(f"Saved version 2: {v2_path}")

# Backup with checksum
import hashlib

def save_with_checksum(obj, filepath):
    # Save object
    torch.save(obj, filepath)
    
    # Calculate checksum
    with open(filepath, 'rb') as f:
        checksum = hashlib.md5(f.read()).hexdigest()
    
    # Save checksum
    checksum_path = filepath + '.md5'
    with open(checksum_path, 'w') as f:
        f.write(checksum)
    
    return checksum

checksum = save_with_checksum(tensor_data, 'tensor_with_checksum.pt')
print(f"Saved with checksum: {checksum}")

def verify_checksum(filepath):
    checksum_path = filepath + '.md5'
    if not os.path.exists(checksum_path):
        return False
    
    # Read stored checksum
    with open(checksum_path, 'r') as f:
        stored_checksum = f.read().strip()
    
    # Calculate current checksum
    with open(filepath, 'rb') as f:
        current_checksum = hashlib.md5(f.read()).hexdigest()
    
    return stored_checksum == current_checksum

verified = verify_checksum('tensor_with_checksum.pt')
print(f"Checksum verification: {verified}")

print("\n=== Performance Optimization ===")

import time

# Performance comparison of different methods
large_tensor = torch.randn(1000, 1000)

# Standard save/load
start_time = time.time()
torch.save(large_tensor, 'perf_standard.pt')
standard_save_time = time.time() - start_time

start_time = time.time()
loaded_standard = torch.load('perf_standard.pt')
standard_load_time = time.time() - start_time

# Compressed save/load
start_time = time.time()
with gzip.open('perf_compressed.gz', 'wb') as f:
    torch.save(large_tensor, f)
compressed_save_time = time.time() - start_time

start_time = time.time()
with gzip.open('perf_compressed.gz', 'rb') as f:
    loaded_compressed = torch.load(f)
compressed_load_time = time.time() - start_time

print(f"Standard save: {standard_save_time:.4f}s, load: {standard_load_time:.4f}s")
print(f"Compressed save: {compressed_save_time:.4f}s, load: {compressed_load_time:.4f}s")

# File size comparison
standard_size = os.path.getsize('perf_standard.pt')
compressed_size = os.path.getsize('perf_compressed.gz')

print(f"Standard size: {standard_size / 1024:.2f} KB")
print(f"Compressed size: {compressed_size / 1024:.2f} KB")
print(f"Compression ratio: {standard_size / compressed_size:.2f}x")

print("\n=== Advanced Serialization Patterns ===")

# Lazy loading for large models
class LazyTensorLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            self._data = torch.load(self.filepath)
        return self._data
    
    def unload(self):
        self._data = None

# Create lazy loader
lazy_loader = LazyTensorLoader('large_state.pt')
print(f"Lazy loader created for: {lazy_loader.filepath}")

# Access data when needed
first_access = time.time()
data = lazy_loader.data
print(f"First access time: {time.time() - first_access:.4f}s")

# Second access (cached)
second_access = time.time()
data_cached = lazy_loader.data
print(f"Second access time: {time.time() - second_access:.6f}s")

# Memory cleanup
lazy_loader.unload()

print("\n=== Integration with Model Checkpoints ===")

# Simulate model checkpoint
model_checkpoint = {
    'epoch': 100,
    'model_state_dict': state_dict,
    'optimizer_state_dict': {'lr': 0.001, 'momentum': 0.9},
    'loss': 0.1234,
    'accuracy': 0.95
}

# Save checkpoint
torch.save(model_checkpoint, 'model_checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('model_checkpoint.pt')
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Checkpoint loss: {checkpoint['loss']}")

# Partial checkpoint loading
def load_model_only(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['model_state_dict']

model_state = load_model_only('model_checkpoint.pt')
print(f"Model state keys: {list(model_state.keys())}")

print("\n=== Cleanup ===")

# Clean up created files
files_to_remove = [
    'single_tensor.pt', 'state_dict.pt', 'tensor.pth', 'tensor.pkl', 'tensor.tar',
    'tensor_compressed.gz', 'cpu_tensor.pt', 'large_state.pt', 'tensor_protocol2.pt',
    'tensor_protocol4.pt', 'tensor_container.pt', 'multiple_tensors.pt',
    'cross_platform.pt', 'versioned_tensor_v1.pt', 'versioned_tensor_v2.pt',
    'tensor_with_checksum.pt', 'tensor_with_checksum.pt.md5', 'perf_standard.pt',
    'perf_compressed.gz', 'model_checkpoint.pt'
]

if torch.cuda.is_available():
    files_to_remove.append('gpu_tensor.pt')

# Remove files
for filepath in files_to_remove:
    if os.path.exists(filepath):
        os.remove(filepath)

# Remove directories
if os.path.exists('data'):
    os.rmdir('data')

print("Cleanup completed")

print("\n=== Serialization Best Practices ===")

print("Tensor Serialization Best Practices:")
print("1. Use .pt or .pth extensions for PyTorch files")
print("2. Include metadata with model checkpoints")
print("3. Use map_location for device-agnostic loading")
print("4. Implement version compatibility checks")
print("5. Validate loaded tensors for expected properties")
print("6. Use compression for storage efficiency")
print("7. Implement checksums for data integrity")
print("8. Consider lazy loading for large models")
print("9. Use pathlib for cross-platform file paths")
print("10. Implement proper error handling")

print("\n=== Tensor Serialization Complete ===") 