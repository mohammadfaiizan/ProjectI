#!/usr/bin/env python3
"""PyTorch Memory Mapped Datasets - Efficient large dataset handling"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import mmap
import struct
import pickle
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

print("=== Memory Mapped Datasets Overview ===")

print("Memory mapping topics covered:")
print("1. NumPy memory-mapped arrays")
print("2. HDF5 datasets for large files")
print("3. Custom binary format memory mapping")
print("4. PyTorch tensor memory mapping")
print("5. Hierarchical data storage")
print("6. Memory-efficient loading strategies")
print("7. Performance optimization")
print("8. Best practices and limitations")

print("\n=== NumPy Memory-Mapped Arrays ===")

class NumpyMemMapDataset(Dataset):
    """Dataset using NumPy memory-mapped arrays"""
    
    def __init__(self, data_file: str, labels_file: str, create_sample_data: bool = True):
        self.data_file = data_file
        self.labels_file = labels_file
        
        # Create sample data if files don't exist
        if create_sample_data and (not os.path.exists(data_file) or not os.path.exists(labels_file)):
            self._create_sample_data()
        
        # Open memory-mapped arrays
        self.data_mmap = np.load(data_file, mmap_mode='r')
        self.labels_mmap = np.load(labels_file, mmap_mode='r')
        
        print(f"  Loaded memory-mapped data: {self.data_mmap.shape}, {self.data_mmap.dtype}")
        print(f"  Loaded memory-mapped labels: {self.labels_mmap.shape}, {self.labels_mmap.dtype}")
    
    def _create_sample_data(self):
        """Create sample data files for demonstration"""
        print(f"  Creating sample memory-mapped files...")
        
        # Generate large dataset
        num_samples = 10000
        feature_dim = 128
        
        # Create data array
        data = np.random.randn(num_samples, feature_dim).astype(np.float32)
        labels = np.random.randint(0, 10, num_samples, dtype=np.int32)
        
        # Save as memory-mappable files
        np.save(self.data_file, data)
        np.save(self.labels_file, labels)
        
        print(f"  Created data file: {self.data_file} ({os.path.getsize(self.data_file)} bytes)")
        print(f"  Created labels file: {self.labels_file} ({os.path.getsize(self.labels_file)} bytes)")
    
    def __len__(self) -> int:
        return len(self.data_mmap)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Access memory-mapped data directly
        data = torch.from_numpy(self.data_mmap[idx].copy())
        label = torch.tensor(self.labels_mmap[idx], dtype=torch.long)
        
        return data, label
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics without loading all data"""
        return {
            'shape': self.data_mmap.shape,
            'dtype': str(self.data_mmap.dtype),
            'memory_usage_mb': self.data_mmap.nbytes / (1024 * 1024),
            'num_classes': len(np.unique(self.labels_mmap))
        }

# Test NumPy memory-mapped dataset
print("Testing NumPy memory-mapped dataset:")

temp_dir = tempfile.mkdtemp()
data_file = os.path.join(temp_dir, "data.npy")
labels_file = os.path.join(temp_dir, "labels.npy")

numpy_dataset = NumpyMemMapDataset(data_file, labels_file, create_sample_data=True)
print(f"  Dataset statistics: {numpy_dataset.get_statistics()}")

# Test with DataLoader
numpy_loader = DataLoader(numpy_dataset, batch_size=32, shuffle=True)
batch_data, batch_labels = next(iter(numpy_loader))
print(f"  Batch shape: {batch_data.shape}, labels shape: {batch_labels.shape}")

print("\n=== HDF5 Datasets for Large Files ===")

class HDF5Dataset(Dataset):
    """Dataset using HDF5 for efficient large file handling"""
    
    def __init__(self, hdf5_file: str, data_key: str = 'data', labels_key: str = 'labels', 
                 create_sample_data: bool = True):
        self.hdf5_file = hdf5_file
        self.data_key = data_key
        self.labels_key = labels_key
        
        # Create sample data if file doesn't exist
        if create_sample_data and not os.path.exists(hdf5_file):
            self._create_sample_hdf5()
        
        # Open HDF5 file in read-only mode
        self.h5_file = h5py.File(hdf5_file, 'r')
        self.data = self.h5_file[data_key]
        self.labels = self.h5_file[labels_key]
        
        print(f"  Loaded HDF5 dataset: {self.data.shape}, compression: {self.data.compression}")
        print(f"  Chunk size: {self.data.chunks}")
    
    def _create_sample_hdf5(self):
        """Create sample HDF5 file with compression"""
        print(f"  Creating sample HDF5 file: {self.hdf5_file}")
        
        num_samples = 5000
        feature_dim = 256
        
        with h5py.File(self.hdf5_file, 'w') as f:
            # Create datasets with compression and chunking
            data_chunk_size = (min(1000, num_samples), feature_dim)
            
            # Data with compression
            data_dataset = f.create_dataset(
                self.data_key,
                shape=(num_samples, feature_dim),
                dtype=np.float32,
                compression='gzip',
                compression_opts=9,
                chunks=data_chunk_size,
                shuffle=True,
                fletcher32=True  # Error detection
            )
            
            # Labels with compression
            labels_dataset = f.create_dataset(
                self.labels_key,
                shape=(num_samples,),
                dtype=np.int32,
                compression='gzip',
                chunks=(min(10000, num_samples),)
            )
            
            # Generate and write data in chunks to manage memory
            chunk_size = 1000
            for i in range(0, num_samples, chunk_size):
                end_idx = min(i + chunk_size, num_samples)
                
                # Generate chunk data
                chunk_data = np.random.randn(end_idx - i, feature_dim).astype(np.float32)
                chunk_labels = np.random.randint(0, 5, end_idx - i, dtype=np.int32)
                
                # Write to HDF5
                data_dataset[i:end_idx] = chunk_data
                labels_dataset[i:end_idx] = chunk_labels
            
            # Add metadata
            f.attrs['created_by'] = 'PyTorch Memory-Mapped Dataset'
            f.attrs['num_samples'] = num_samples
            f.attrs['feature_dim'] = feature_dim
            f.attrs['num_classes'] = 5
        
        file_size = os.path.getsize(self.hdf5_file)
        print(f"  Created HDF5 file: {file_size} bytes")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # HDF5 handles memory mapping internally
        data = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return data, label
    
    def get_slice(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a slice of data efficiently"""
        data_slice = torch.from_numpy(self.data[start:end])
        labels_slice = torch.from_numpy(self.labels[start:end])
        return data_slice, labels_slice
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        metadata = {}
        for key, value in self.h5_file.attrs.items():
            metadata[key] = value
        
        metadata.update({
            'data_shape': self.data.shape,
            'data_dtype': str(self.data.dtype),
            'compression': self.data.compression,
            'chunks': self.data.chunks
        })
        
        return metadata
    
    def __del__(self):
        """Close HDF5 file when dataset is destroyed"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

# Test HDF5 dataset
print("Testing HDF5 dataset:")

hdf5_file = os.path.join(temp_dir, "dataset.h5")
hdf5_dataset = HDF5Dataset(hdf5_file, create_sample_data=True)

print(f"  Metadata: {hdf5_dataset.get_metadata()}")

# Test efficient slicing
slice_data, slice_labels = hdf5_dataset.get_slice(100, 200)
print(f"  Slice [100:200]: data shape {slice_data.shape}, labels shape {slice_labels.shape}")

# Test with DataLoader
hdf5_loader = DataLoader(hdf5_dataset, batch_size=64, shuffle=False)
hdf5_batch = next(iter(hdf5_loader))
print(f"  HDF5 batch: data {hdf5_batch[0].shape}, labels {hdf5_batch[1].shape}")

print("\n=== Custom Binary Format Memory Mapping ===")

class BinaryMemMapDataset(Dataset):
    """Dataset using custom binary format with memory mapping"""
    
    def __init__(self, binary_file: str, index_file: str, create_sample_data: bool = True):
        self.binary_file = binary_file
        self.index_file = index_file
        
        # Create sample data if files don't exist
        if create_sample_data and not os.path.exists(binary_file):
            self._create_binary_dataset()
        
        # Load index for random access
        self.index = self._load_index()
        
        # Open binary file with memory mapping
        self.file_handle = open(binary_file, 'rb')
        self.mmap = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        
        print(f"  Loaded binary dataset: {len(self.index)} samples")
        print(f"  File size: {len(self.mmap)} bytes")
    
    def _create_binary_dataset(self):
        """Create custom binary format dataset"""
        print(f"  Creating binary dataset: {self.binary_file}")
        
        num_samples = 2000
        
        # Binary format: [sample_size(4 bytes)][feature_dim(4 bytes)][data][label(4 bytes)]
        index = []
        
        with open(self.binary_file, 'wb') as f:
            offset = 0
            
            for i in range(num_samples):
                # Variable size features (simulating real-world variability)
                feature_dim = np.random.randint(50, 200)
                data = np.random.randn(feature_dim).astype(np.float32)
                label = np.random.randint(0, 8)
                
                # Calculate sample size
                sample_size = 4 + 4 + (feature_dim * 4) + 4  # header + data + label
                
                # Write sample
                f.write(struct.pack('I', sample_size))  # Sample size
                f.write(struct.pack('I', feature_dim))  # Feature dimension
                f.write(data.tobytes())  # Data
                f.write(struct.pack('i', label))  # Label
                
                # Store in index
                index.append({
                    'offset': offset,
                    'sample_size': sample_size,
                    'feature_dim': feature_dim
                })
                
                offset += sample_size
        
        # Save index
        with open(self.index_file, 'wb') as f:
            pickle.dump(index, f)
        
        print(f"  Created {num_samples} samples, total size: {offset} bytes")
    
    def _load_index(self) -> List[Dict]:
        """Load sample index for random access"""
        with open(self.index_file, 'rb') as f:
            return pickle.load(f)
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sample info from index
        sample_info = self.index[idx]
        offset = sample_info['offset']
        feature_dim = sample_info['feature_dim']
        
        # Seek to sample position in memory map
        self.mmap.seek(offset)
        
        # Read sample header
        sample_size = struct.unpack('I', self.mmap.read(4))[0]
        read_feature_dim = struct.unpack('I', self.mmap.read(4))[0]
        
        # Validate
        assert read_feature_dim == feature_dim, f"Feature dim mismatch: {read_feature_dim} vs {feature_dim}"
        
        # Read data
        data_bytes = self.mmap.read(feature_dim * 4)
        data = np.frombuffer(data_bytes, dtype=np.float32)
        
        # Read label
        label = struct.unpack('i', self.mmap.read(4))[0]
        
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample"""
        return self.index[idx].copy()
    
    def __del__(self):
        """Cleanup memory map and file handle"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file_handle'):
            self.file_handle.close()

# Test binary memory-mapped dataset
print("Testing binary memory-mapped dataset:")

binary_file = os.path.join(temp_dir, "dataset.bin")
index_file = os.path.join(temp_dir, "dataset.idx")

binary_dataset = BinaryMemMapDataset(binary_file, index_file, create_sample_data=True)

# Test variable-size samples
for i in [0, 10, 100]:
    sample_info = binary_dataset.get_sample_info(i)
    data, label = binary_dataset[i]
    print(f"  Sample {i}: feature_dim={sample_info['feature_dim']}, "
          f"data_shape={data.shape}, label={label.item()}")

print("\n=== PyTorch Tensor Memory Mapping ===")

class TorchMemMapDataset(Dataset):
    """Dataset using PyTorch's built-in memory mapping"""
    
    def __init__(self, tensor_file: str, create_sample_data: bool = True):
        self.tensor_file = tensor_file
        
        # Create sample data if file doesn't exist
        if create_sample_data and not os.path.exists(tensor_file):
            self._create_torch_dataset()
        
        # Load memory-mapped tensor
        self.data_tensor = torch.load(tensor_file, map_location='cpu')
        
        # Make tensor memory-mapped if not already
        if not self.data_tensor.is_shared():
            self.data_tensor.share_memory_()
        
        print(f"  Loaded PyTorch memory-mapped tensor: {self.data_tensor.shape}")
        print(f"  Is shared: {self.data_tensor.is_shared()}")
        print(f"  Storage type: {type(self.data_tensor.storage())}")
    
    def _create_torch_dataset(self):
        """Create PyTorch tensor dataset"""
        print(f"  Creating PyTorch tensor dataset: {self.tensor_file}")
        
        num_samples = 3000
        feature_dim = 100
        
        # Create tensor data
        data = torch.randn(num_samples, feature_dim, dtype=torch.float32)
        labels = torch.randint(0, 7, (num_samples,), dtype=torch.long)
        
        # Combine into single tensor for easier memory mapping
        dataset_tensor = {
            'data': data,
            'labels': labels,
            'metadata': {
                'num_samples': num_samples,
                'feature_dim': feature_dim,
                'num_classes': 7
            }
        }
        
        # Save with memory mapping enabled
        torch.save(dataset_tensor, self.tensor_file)
        
        file_size = os.path.getsize(self.tensor_file)
        print(f"  Created tensor file: {file_size} bytes")
    
    def __len__(self) -> int:
        return len(self.data_tensor['data'])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Direct tensor access (memory-mapped)
        data = self.data_tensor['data'][idx]
        label = self.data_tensor['labels'][idx]
        
        return data, label
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self.data_tensor['metadata']

# Test PyTorch memory-mapped dataset
print("Testing PyTorch memory-mapped dataset:")

torch_file = os.path.join(temp_dir, "torch_dataset.pt")
torch_dataset = TorchMemMapDataset(torch_file, create_sample_data=True)

print(f"  Metadata: {torch_dataset.get_metadata()}")

# Test with DataLoader
torch_loader = DataLoader(torch_dataset, batch_size=50, shuffle=True)
torch_batch = next(iter(torch_loader))
print(f"  Torch batch: data {torch_batch[0].shape}, labels {torch_batch[1].shape}")

print("\n=== Hierarchical Data Storage ===")

class HierarchicalDataset(Dataset):
    """Dataset with hierarchical organization for complex data"""
    
    def __init__(self, hdf5_file: str, create_sample_data: bool = True):
        self.hdf5_file = hdf5_file
        
        # Create sample hierarchical data
        if create_sample_data and not os.path.exists(hdf5_file):
            self._create_hierarchical_data()
        
        # Open HDF5 file
        self.h5_file = h5py.File(hdf5_file, 'r')
        
        # Navigate hierarchical structure
        self.groups = list(self.h5_file.keys())
        self.sample_info = self._build_sample_index()
        
        print(f"  Loaded hierarchical dataset with {len(self.groups)} groups")
        print(f"  Total samples: {len(self.sample_info)}")
    
    def _create_hierarchical_data(self):
        """Create hierarchical HDF5 structure"""
        print(f"  Creating hierarchical dataset: {self.hdf5_file}")
        
        with h5py.File(self.hdf5_file, 'w') as f:
            # Create different data modalities
            modalities = ['images', 'text', 'audio', 'metadata']
            
            for modality in modalities:
                group = f.create_group(modality)
                
                if modality == 'images':
                    # Simulate image data
                    num_images = 1000
                    for i in range(num_images):
                        img_data = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                        group.create_dataset(f'image_{i:05d}', data=img_data, compression='gzip')
                
                elif modality == 'text':
                    # Simulate text embeddings
                    num_texts = 1000
                    for i in range(num_texts):
                        text_emb = np.random.randn(768).astype(np.float32)
                        group.create_dataset(f'text_{i:05d}', data=text_emb, compression='gzip')
                
                elif modality == 'audio':
                    # Simulate audio features
                    num_audio = 1000
                    for i in range(num_audio):
                        audio_feat = np.random.randn(128).astype(np.float32)
                        group.create_dataset(f'audio_{i:05d}', data=audio_feat, compression='gzip')
                
                elif modality == 'metadata':
                    # Create metadata
                    metadata = group.create_group('info')
                    metadata.attrs['total_samples'] = 1000
                    metadata.attrs['created_by'] = 'Hierarchical Dataset'
                    metadata.attrs['version'] = '1.0'
                    
                    # Sample-level metadata
                    labels = np.random.randint(0, 10, 1000)
                    categories = np.random.choice(['cat', 'dog', 'bird'], 1000)
                    
                    group.create_dataset('labels', data=labels)
                    
                    # String arrays need special handling
                    str_dtype = h5py.string_dtype(encoding='utf-8')
                    group.create_dataset('categories', data=categories, dtype=str_dtype)
        
        print(f"  Created hierarchical HDF5 file: {os.path.getsize(self.hdf5_file)} bytes")
    
    def _build_sample_index(self) -> List[Dict[str, Any]]:
        """Build index of all samples across modalities"""
        sample_info = []
        
        # Determine number of samples (assuming all modalities have same count)
        num_samples = len(self.h5_file['metadata']['labels'])
        
        for i in range(num_samples):
            info = {
                'index': i,
                'has_image': f'image_{i:05d}' in self.h5_file['images'],
                'has_text': f'text_{i:05d}' in self.h5_file['text'],
                'has_audio': f'audio_{i:05d}' in self.h5_file['audio'],
                'label': int(self.h5_file['metadata']['labels'][i]),
                'category': self.h5_file['metadata']['categories'][i].decode('utf-8')
            }
            sample_info.append(info)
        
        return sample_info
    
    def __len__(self) -> int:
        return len(self.sample_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get multi-modal sample"""
        info = self.sample_info[idx]
        sample = {}
        
        # Load available modalities
        if info['has_image']:
            img_data = self.h5_file['images'][f'image_{idx:05d}'][...]
            sample['image'] = torch.from_numpy(img_data).float() / 255.0
        
        if info['has_text']:
            text_data = self.h5_file['text'][f'text_{idx:05d}'][...]
            sample['text'] = torch.from_numpy(text_data)
        
        if info['has_audio']:
            audio_data = self.h5_file['audio'][f'audio_{idx:05d}'][...]
            sample['audio'] = torch.from_numpy(audio_data)
        
        # Add metadata
        sample['label'] = torch.tensor(info['label'], dtype=torch.long)
        sample['category'] = info['category']
        
        return sample
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get sample information"""
        return self.sample_info[idx].copy()
    
    def __del__(self):
        """Close HDF5 file"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

# Test hierarchical dataset
print("Testing hierarchical dataset:")

hierarchical_file = os.path.join(temp_dir, "hierarchical.h5")
hierarchical_dataset = HierarchicalDataset(hierarchical_file, create_sample_data=True)

# Test multi-modal sample
sample = hierarchical_dataset[0]
sample_info = hierarchical_dataset.get_sample_info(0)

print(f"  Sample info: {sample_info}")
print(f"  Sample keys: {list(sample.keys())}")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"    {key}: {value.shape}")
    else:
        print(f"    {key}: {value}")

print("\n=== Performance Comparison ===")

def benchmark_dataset_loading(dataset, name, num_samples=100):
    """Benchmark dataset loading performance"""
    
    # Random access benchmark
    indices = np.random.randint(0, len(dataset), num_samples)
    
    start_time = time.time()
    for idx in indices:
        _ = dataset[idx]
    random_access_time = time.time() - start_time
    
    # Sequential access benchmark
    start_time = time.time()
    for i in range(num_samples):
        _ = dataset[i]
    sequential_access_time = time.time() - start_time
    
    return {
        'random_access_time': random_access_time,
        'sequential_access_time': sequential_access_time,
        'random_access_rate': num_samples / random_access_time,
        'sequential_access_rate': num_samples / sequential_access_time
    }

print("Performance comparison of memory-mapped datasets:")

datasets_to_test = [
    ('NumPy MemMap', numpy_dataset),
    ('HDF5', hdf5_dataset),
    ('Binary MemMap', binary_dataset),
    ('PyTorch MemMap', torch_dataset)
]

for name, dataset in datasets_to_test:
    try:
        results = benchmark_dataset_loading(dataset, name, num_samples=50)
        print(f"  {name:15}: Random={results['random_access_rate']:.1f} samples/sec, "
              f"Sequential={results['sequential_access_rate']:.1f} samples/sec")
    except Exception as e:
        print(f"  {name:15}: Error - {e}")

print("\n=== Memory Usage Analysis ===")

def analyze_memory_usage():
    """Analyze memory usage of different approaches"""
    
    print("Memory usage analysis:")
    
    # Check file sizes
    files_to_check = [
        ('NumPy data', data_file),
        ('NumPy labels', labels_file),
        ('HDF5', hdf5_file),
        ('Binary', binary_file),
        ('Binary index', index_file),
        ('PyTorch', torch_file),
        ('Hierarchical', hierarchical_file)
    ]
    
    total_size = 0
    for name, filepath in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            print(f"  {name:15}: {size:>10,} bytes ({size/(1024**2):.2f} MB)")
    
    print(f"  {'Total':15}: {total_size:>10,} bytes ({total_size/(1024**2):.2f} MB)")
    
    # Memory mapping benefits
    print("\nMemory mapping benefits:")
    print("  1. Reduced RAM usage - data stays on disk")
    print("  2. Faster startup - no need to load entire dataset")
    print("  3. Shared memory across processes")
    print("  4. Virtual memory management by OS")
    print("  5. Efficient random access to large files")

analyze_memory_usage()

print("\n=== Best Practices and Limitations ===")

print("Memory Mapping Best Practices:")
print("1. Use for datasets larger than available RAM")
print("2. Ensure fast storage (SSD preferred)")
print("3. Consider data layout for access patterns")
print("4. Use compression for storage efficiency")
print("5. Implement proper error handling for I/O")

print("\nFormat Recommendations:")
print("1. NumPy: Simple numeric arrays, fast access")
print("2. HDF5: Complex hierarchical data, compression")
print("3. Binary: Custom formats, maximum control")
print("4. PyTorch: Native tensor support, GPU compatibility")

print("\nLimitations and Considerations:")
print("1. I/O bound performance on slow storage")
print("2. Network storage may have high latency")
print("3. Memory mapping overhead for small datasets")
print("4. File corruption risks with concurrent access")
print("5. Platform-specific memory mapping behavior")

print("\nTroubleshooting Tips:")
print("1. Monitor I/O wait times during training")
print("2. Use prefetching to hide I/O latency")
print("3. Consider data preprocessing and caching")
print("4. Profile different batch sizes and worker counts")
print("5. Test with representative access patterns")

print("\n=== Memory Mapped Datasets Complete ===")

# Cleanup temporary files
import shutil
try:
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary directory: {temp_dir}")
except Exception as e:
    print(f"Cleanup warning: {e}")

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None