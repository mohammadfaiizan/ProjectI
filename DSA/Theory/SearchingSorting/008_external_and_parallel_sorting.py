"""
External and Parallel Sorting Algorithms
========================================

Topics: External sorting, parallel algorithms, distributed sorting
Companies: Google, Facebook, Amazon, Microsoft (System Design)
Difficulty: Hard/Advanced
"""

import heapq
import tempfile
import os
from typing import List, Iterator, Optional
import threading
import concurrent.futures
from multiprocessing import Pool
import time

class ExternalAndParallelSorting:
    
    # ==========================================
    # 1. EXTERNAL SORTING ALGORITHMS
    # ==========================================
    
    def external_merge_sort(self, input_file: str, output_file: str, 
                          chunk_size: int = 1000, temp_dir: str = None) -> None:
        """External merge sort for files larger than memory
        Time: O(n log n), Space: O(chunk_size)
        """
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        # Phase 1: Sort chunks and create temporary files
        temp_files = self._create_sorted_chunks(input_file, chunk_size, temp_dir)
        
        # Phase 2: K-way merge of sorted chunks
        self._k_way_merge_files(temp_files, output_file)
        
        # Cleanup temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _create_sorted_chunks(self, input_file: str, chunk_size: int, 
                            temp_dir: str) -> List[str]:
        """Create sorted chunks from input file"""
        temp_files = []
        chunk = []
        
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    number = int(line.strip())
                    chunk.append(number)
                    
                    if len(chunk) >= chunk_size:
                        # Sort chunk and write to temp file
                        chunk.sort()
                        temp_file = self._write_chunk_to_file(chunk, temp_dir)
                        temp_files.append(temp_file)
                        chunk = []
                        
                except ValueError:
                    continue  # Skip invalid numbers
        
        # Handle remaining chunk
        if chunk:
            chunk.sort()
            temp_file = self._write_chunk_to_file(chunk, temp_dir)
            temp_files.append(temp_file)
        
        return temp_files
    
    def _write_chunk_to_file(self, chunk: List[int], temp_dir: str) -> str:
        """Write sorted chunk to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                              dir=temp_dir, suffix='.tmp')
        
        for number in chunk:
            temp_file.write(f"{number}\n")
        
        temp_file.close()
        return temp_file.name
    
    def _k_way_merge_files(self, temp_files: List[str], output_file: str) -> None:
        """K-way merge of sorted files using heap"""
        file_readers = []
        heap = []
        
        # Open all files and initialize heap
        for i, temp_file in enumerate(temp_files):
            f = open(temp_file, 'r')
            file_readers.append(f)
            
            # Read first number from each file
            line = f.readline().strip()
            if line:
                heapq.heappush(heap, (int(line), i))
        
        # Merge files
        with open(output_file, 'w') as out_f:
            while heap:
                value, file_index = heapq.heappop(heap)
                out_f.write(f"{value}\n")
                
                # Read next number from the same file
                line = file_readers[file_index].readline().strip()
                if line:
                    heapq.heappush(heap, (int(line), file_index))
        
        # Close all file readers
        for f in file_readers:
            f.close()
    
    def external_sort_with_replacement_selection(self, input_file: str, 
                                                output_file: str, 
                                                memory_size: int = 1000) -> None:
        """External sort using replacement selection for run generation
        Produces longer initial runs than fixed-size chunks
        """
        # Phase 1: Generate runs using replacement selection
        run_files = self._replacement_selection(input_file, memory_size)
        
        # Phase 2: Merge runs
        self._k_way_merge_files(run_files, output_file)
        
        # Cleanup
        for run_file in run_files:
            if os.path.exists(run_file):
                os.remove(run_file)
    
    def _replacement_selection(self, input_file: str, memory_size: int) -> List[str]:
        """Generate runs using replacement selection algorithm"""
        run_files = []
        heap = []
        
        with open(input_file, 'r') as f:
            # Initialize heap with first memory_size elements
            for _ in range(memory_size):
                line = f.readline().strip()
                if not line:
                    break
                heapq.heappush(heap, (int(line), 0))  # (value, run_number)
            
            current_run = 0
            current_run_file = None
            
            while heap:
                if current_run_file is None:
                    current_run_file = tempfile.NamedTemporaryFile(
                        mode='w', delete=False, suffix='.run')
                
                # Output minimum element
                min_val, run_num = heapq.heappop(heap)
                
                if run_num == current_run:
                    current_run_file.write(f"{min_val}\n")
                    
                    # Read next element
                    line = f.readline().strip()
                    if line:
                        next_val = int(line)
                        if next_val >= min_val:
                            heapq.heappush(heap, (next_val, current_run))
                        else:
                            heapq.heappush(heap, (next_val, current_run + 1))
                else:
                    # Start new run
                    current_run_file.close()
                    run_files.append(current_run_file.name)
                    current_run += 1
                    
                    current_run_file = tempfile.NamedTemporaryFile(
                        mode='w', delete=False, suffix='.run')
                    current_run_file.write(f"{min_val}\n")
            
            if current_run_file:
                current_run_file.close()
                run_files.append(current_run_file.name)
        
        return run_files
    
    # ==========================================
    # 2. PARALLEL SORTING ALGORITHMS
    # ==========================================
    
    def parallel_merge_sort(self, arr: List[int], num_threads: int = 4) -> List[int]:
        """Parallel merge sort using threading
        Time: O(n log n), Space: O(n)
        """
        def merge(left: List[int], right: List[int]) -> List[int]:
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        def parallel_merge_sort_helper(arr: List[int], depth: int = 0) -> List[int]:
            if len(arr) <= 1:
                return arr
            
            # Use sequential sort for small arrays or deep recursion
            if len(arr) < 1000 or depth > 3:
                return sorted(arr)
            
            mid = len(arr) // 2
            
            # Use threading for parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_left = executor.submit(parallel_merge_sort_helper, 
                                            arr[:mid], depth + 1)
                future_right = executor.submit(parallel_merge_sort_helper, 
                                             arr[mid:], depth + 1)
                
                left = future_left.result()
                right = future_right.result()
            
            return merge(left, right)
        
        return parallel_merge_sort_helper(arr)
    
    def parallel_quick_sort(self, arr: List[int], num_processes: int = 4) -> List[int]:
        """Parallel quick sort using multiprocessing
        Time: O(n log n) average, Space: O(log n)
        """
        def sequential_quick_sort(arr: List[int]) -> List[int]:
            if len(arr) <= 1:
                return arr
            
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            
            return sequential_quick_sort(left) + middle + sequential_quick_sort(right)
        
        if len(arr) <= 1000:  # Use sequential for small arrays
            return sequential_quick_sort(arr)
        
        # Parallel partitioning
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        # Parallel recursive calls
        with Pool(processes=min(num_processes, 2)) as pool:
            if len(left) > 1000:
                future_left = pool.apply_async(self.parallel_quick_sort, 
                                             (left, num_processes // 2))
            else:
                future_left = pool.apply_async(sequential_quick_sort, (left,))
            
            if len(right) > 1000:
                future_right = pool.apply_async(self.parallel_quick_sort, 
                                              (right, num_processes // 2))
            else:
                future_right = pool.apply_async(sequential_quick_sort, (right,))
            
            sorted_left = future_left.get()
            sorted_right = future_right.get()
        
        return sorted_left + middle + sorted_right
    
    def parallel_bucket_sort(self, arr: List[float], num_buckets: int = None, 
                           num_threads: int = 4) -> List[float]:
        """Parallel bucket sort
        Time: O(n + k) average, Space: O(n)
        """
        if not arr:
            return arr
        
        if num_buckets is None:
            num_buckets = len(arr)
        
        # Create buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute elements into buckets
        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val
        
        for num in arr:
            if range_val == 0:
                bucket_index = 0
            else:
                bucket_index = int((num - min_val) / range_val * (num_buckets - 1))
            buckets[bucket_index].append(num)
        
        # Sort buckets in parallel
        def sort_bucket(bucket):
            return sorted(bucket)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(sort_bucket, bucket) for bucket in buckets]
            sorted_buckets = [future.result() for future in futures]
        
        # Concatenate sorted buckets
        result = []
        for bucket in sorted_buckets:
            result.extend(bucket)
        
        return result
    
    # ==========================================
    # 3. DISTRIBUTED SORTING CONCEPTS
    # ==========================================
    
    def sample_sort_simulation(self, arr: List[int], num_processors: int = 4) -> List[int]:
        """Simulation of sample sort (distributed sorting algorithm)
        Used in parallel/distributed systems
        """
        if len(arr) <= 1:
            return arr
        
        # Step 1: Each processor sorts its local data
        chunk_size = len(arr) // num_processors
        chunks = []
        
        for i in range(num_processors):
            start = i * chunk_size
            end = start + chunk_size if i < num_processors - 1 else len(arr)
            chunk = sorted(arr[start:end])
            chunks.append(chunk)
        
        # Step 2: Sample selection
        sample_size = num_processors - 1
        samples = []
        
        for chunk in chunks:
            if len(chunk) > 0:
                # Select evenly spaced samples
                step = max(1, len(chunk) // sample_size) if sample_size > 0 else 1
                for j in range(0, len(chunk), step):
                    if len(samples) < sample_size * num_processors:
                        samples.append(chunk[j])
        
        # Step 3: Select splitters
        samples.sort()
        splitters = []
        if len(samples) > 0:
            step = len(samples) // (num_processors - 1) if num_processors > 1 else 1
            for i in range(step, len(samples), step):
                if len(splitters) < num_processors - 1:
                    splitters.append(samples[i])
        
        # Step 4: Redistribute data based on splitters
        buckets = [[] for _ in range(num_processors)]
        
        for chunk in chunks:
            for num in chunk:
                bucket_idx = 0
                for splitter in splitters:
                    if num <= splitter:
                        break
                    bucket_idx += 1
                buckets[bucket_idx].append(num)
        
        # Step 5: Sort each bucket and merge
        result = []
        for bucket in buckets:
            result.extend(sorted(bucket))
        
        return result
    
    # ==========================================
    # 4. UTILITY METHODS
    # ==========================================
    
    def generate_test_file(self, filename: str, size: int, max_value: int = 10000) -> None:
        """Generate a test file with random numbers"""
        import random
        
        with open(filename, 'w') as f:
            for _ in range(size):
                f.write(f"{random.randint(1, max_value)}\n")
    
    def verify_sorted_file(self, filename: str) -> bool:
        """Verify if a file contains sorted numbers"""
        prev_num = float('-inf')
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    num = int(line.strip())
                    if num < prev_num:
                        return False
                    prev_num = num
            return True
        except (ValueError, IOError):
            return False
    
    def benchmark_parallel_vs_sequential(self, arr: List[int]) -> dict:
        """Compare parallel vs sequential sorting performance"""
        results = {}
        
        # Sequential merge sort
        start_time = time.time()
        sequential_result = sorted(arr)
        sequential_time = time.time() - start_time
        
        # Parallel merge sort
        start_time = time.time()
        parallel_result = self.parallel_merge_sort(arr)
        parallel_time = time.time() - start_time
        
        results = {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': sequential_time / parallel_time if parallel_time > 0 else 0,
            'sequential_correct': sequential_result == sorted(arr),
            'parallel_correct': parallel_result == sorted(arr)
        }
        
        return results

# Test Examples
def run_examples():
    sorter = ExternalAndParallelSorting()
    
    print("=== EXTERNAL AND PARALLEL SORTING ===\n")
    
    # Test parallel sorting
    print("1. PARALLEL SORTING:")
    test_arr = [random.randint(1, 1000) for _ in range(1000)]
    
    start_time = time.time()
    sequential = sorted(test_arr)
    seq_time = time.time() - start_time
    
    start_time = time.time()
    parallel = sorter.parallel_merge_sort(test_arr)
    par_time = time.time() - start_time
    
    print(f"Sequential time: {seq_time:.4f}s")
    print(f"Parallel time: {par_time:.4f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x" if par_time > 0 else "N/A")
    print(f"Results match: {sequential == parallel}")
    
    # Test sample sort simulation
    print("\n2. SAMPLE SORT SIMULATION:")
    test_arr = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30, 15, 8]
    sample_sorted = sorter.sample_sort_simulation(test_arr, 4)
    print(f"Original: {test_arr}")
    print(f"Sample sorted: {sample_sorted}")
    print(f"Correctly sorted: {sample_sorted == sorted(test_arr)}")
    
    # Test parallel bucket sort
    print("\n3. PARALLEL BUCKET SORT:")
    float_arr = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12]
    bucket_sorted = sorter.parallel_bucket_sort(float_arr)
    print(f"Original: {float_arr}")
    print(f"Bucket sorted: {bucket_sorted}")
    print(f"Correctly sorted: {bucket_sorted == sorted(float_arr)}")
    
    print("\n4. EXTERNAL SORTING DEMO:")
    print("External sorting is designed for large files.")
    print("Demo would require creating temporary files.")
    print("Use generate_test_file() and external_merge_sort() for testing.")

if __name__ == "__main__":
    import random
    run_examples() 