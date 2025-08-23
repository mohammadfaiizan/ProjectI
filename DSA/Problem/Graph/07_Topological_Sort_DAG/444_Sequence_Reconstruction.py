"""
444. Sequence Reconstruction - Multiple Approaches
Difficulty: Medium

You are given an integer array nums of length n where nums is a permutation of the integers in the range [1, n]. You are also given a 2D integer array sequences where sequences[i] is a subsequence of nums.

Check if nums is the shortest supersequence of the sequences. A shortest supersequence is a sequence with the shortest length such that all sequences in sequences are subsequences of it.

A sequence a is a subsequence of a sequence b if a can be derived from b by deleting some or no elements without changing the order of the remaining elements.

Return true if nums is the only shortest supersequence of sequences, otherwise, return false.
"""

from typing import List, Set
from collections import defaultdict, deque

class SequenceReconstruction:
    """Multiple approaches to check sequence reconstruction"""
    
    def sequenceReconstruction_topological_sort(self, nums: List[int], sequences: List[List[int]]) -> bool:
        """
        Approach 1: Topological Sort with Unique Path Check
        
        Build graph from sequences and check if topological order is unique.
        
        Time: O(V + E), Space: O(V + E)
        """
        n = len(nums)
        
        # Build graph from sequences
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Initialize all numbers
        for num in nums:
            in_degree[num] = 0
        
        # Build dependencies from sequences
        for seq in sequences:
            for i in range(len(seq) - 1):
                if seq[i + 1] not in graph[seq[i]]:
                    graph[seq[i]].add(seq[i + 1])
                    in_degree[seq[i + 1]] += 1
        
        # Check if all numbers from nums appear in sequences
        seq_nums = set()
        for seq in sequences:
            seq_nums.update(seq)
        
        if seq_nums != set(nums):
            return False
        
        # Topological sort with uniqueness check
        queue = deque()
        for num in nums:
            if in_degree[num] == 0:
                queue.append(num)
        
        result = []
        
        while queue:
            # If more than one node has in-degree 0, order is not unique
            if len(queue) > 1:
                return False
            
            current = queue.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if result matches nums
        return result == nums
    
    def sequenceReconstruction_constraint_validation(self, nums: List[int], sequences: List[List[int]]) -> bool:
        """
        Approach 2: Direct Constraint Validation
        
        Validate that sequences provide enough constraints to uniquely determine nums.
        
        Time: O(V + E), Space: O(V + E)
        """
        n = len(nums)
        
        # Create position mapping for nums
        pos = {num: i for i, num in enumerate(nums)}
        
        # Check if all sequence elements are in nums
        for seq in sequences:
            for num in seq:
                if num not in pos:
                    return False
        
        # Build adjacency relationships
        edges = set()
        for seq in sequences:
            for i in range(len(seq) - 1):
                edges.add((seq[i], seq[i + 1]))
        
        # Check if sequences provide necessary constraints
        # For each adjacent pair in nums, there must be a sequence that establishes this order
        for i in range(n - 1):
            if (nums[i], nums[i + 1]) not in edges:
                return False
        
        # Check if any sequence violates the order in nums
        for seq in sequences:
            for i in range(len(seq) - 1):
                if pos[seq[i]] >= pos[seq[i + 1]]:
                    return False
        
        return True
    
    def sequenceReconstruction_graph_validation(self, nums: List[int], sequences: List[List[int]]) -> bool:
        """
        Approach 3: Graph-based Validation
        
        Build directed graph and validate against nums ordering.
        
        Time: O(V + E), Space: O(V + E)
        """
        if not sequences:
            return False
        
        # Collect all numbers from sequences
        seq_nums = set()
        for seq in sequences:
            seq_nums.update(seq)
        
        # Check if sequences contain exactly the numbers in nums
        if seq_nums != set(nums):
            return False
        
        # Build directed graph
        graph = defaultdict(set)
        in_degree = {num: 0 for num in nums}
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                if seq[i + 1] not in graph[seq[i]]:
                    graph[seq[i]].add(seq[i + 1])
                    in_degree[seq[i + 1]] += 1
        
        # Simulate topological sort
        available = deque()
        for num in nums:
            if in_degree[num] == 0:
                available.append(num)
        
        result = []
        
        while available:
            # Must have exactly one choice at each step
            if len(available) != 1:
                return False
            
            current = available.popleft()
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    available.append(neighbor)
        
        return result == nums
    
    def sequenceReconstruction_incremental_check(self, nums: List[int], sequences: List[List[int]]) -> bool:
        """
        Approach 4: Incremental Validation Check
        
        Check constraints incrementally as we build the sequence.
        
        Time: O(V + E), Space: O(V + E)
        """
        n = len(nums)
        
        # Build index mapping
        index = {num: i for i, num in enumerate(nums)}
        
        # Validate all sequence numbers are in nums
        for seq in sequences:
            for num in seq:
                if num not in index:
                    return False
        
        # Track which adjacent pairs are confirmed by sequences
        confirmed_pairs = set()
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                # Check if this pair respects the order in nums
                if index[seq[i]] >= index[seq[i + 1]]:
                    return False
                confirmed_pairs.add((seq[i], seq[i + 1]))
        
        # Check if all necessary adjacent pairs in nums are confirmed
        for i in range(n - 1):
            if (nums[i], nums[i + 1]) not in confirmed_pairs:
                return False
        
        # Build graph to check for unique topological order
        graph = defaultdict(list)
        in_degree = {num: 0 for num in nums}
        
        for a, b in confirmed_pairs:
            graph[a].append(b)
            in_degree[b] += 1
        
        # Check uniqueness of topological order
        queue = deque()
        for num in nums:
            if in_degree[num] == 0:
                queue.append(num)
        
        while queue:
            if len(queue) > 1:
                return False
            
            current = queue.popleft()
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return True
    
    def sequenceReconstruction_optimized_validation(self, nums: List[int], sequences: List[List[int]]) -> bool:
        """
        Approach 5: Optimized Single-Pass Validation
        
        Optimize validation with single-pass checking.
        
        Time: O(V + E), Space: O(V)
        """
        if not sequences:
            return False
        
        n = len(nums)
        pos = {num: i for i, num in enumerate(nums)}
        
        # Check all sequence numbers are valid
        seq_nums = set()
        for seq in sequences:
            for num in seq:
                if num not in pos:
                    return False
                seq_nums.add(num)
        
        if len(seq_nums) != n:
            return False
        
        # Track predecessors for each number
        predecessors = [set() for _ in range(n)]
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                curr_pos = pos[seq[i]]
                next_pos = pos[seq[i + 1]]
                
                # Check order consistency
                if curr_pos >= next_pos:
                    return False
                
                # Add predecessor relationship
                predecessors[next_pos].add(curr_pos)
        
        # Check if each position (except first) has exactly the previous position as predecessor
        for i in range(1, n):
            if i - 1 not in predecessors[i]:
                return False
        
        return True

def test_sequence_reconstruction():
    """Test sequence reconstruction algorithms"""
    solver = SequenceReconstruction()
    
    test_cases = [
        ([1,2,3], [[1,2],[1,3]], False, "Missing constraint 2->3"),
        ([1,2,3], [[1,2],[1,3],[2,3]], True, "All constraints present"),
        ([4,1,5,2,6,3], [[5,2,6,3],[4,1,5,2]], True, "Complex valid case"),
        ([1], [[1]], True, "Single element"),
        ([1,2,3], [[1,3],[2,3]], False, "Missing 1->2 constraint"),
    ]
    
    algorithms = [
        ("Topological Sort", solver.sequenceReconstruction_topological_sort),
        ("Constraint Validation", solver.sequenceReconstruction_constraint_validation),
        ("Graph Validation", solver.sequenceReconstruction_graph_validation),
        ("Incremental Check", solver.sequenceReconstruction_incremental_check),
        ("Optimized Validation", solver.sequenceReconstruction_optimized_validation),
    ]
    
    print("=== Testing Sequence Reconstruction ===")
    
    for nums, sequences, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"nums: {nums}, sequences: {sequences}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums, sequences)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:22} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_sequence_reconstruction()

"""
Sequence Reconstruction demonstrates topological sorting
with uniqueness validation and constraint satisfaction
for sequence ordering problems.
"""
