"""
421. Maximum XOR of Two Numbers in an Array - Multiple Approaches
Difficulty: Medium

Given an integer array nums, return the maximum result of nums[i] XOR nums[j], 
where 0 <= i <= j < n.

Examples:
Input: nums = [3,10,5,25,2,8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.

Input: nums = [14,70,53,83,49,91,36,80,92,51,66,70]
Output: 127

Approaches:
1. Brute Force O(n²)
2. Bit Manipulation with Trie
3. Optimized Trie with Bit Packing
4. Prefix Trie with Early Termination
5. Hash Set Approach
6. Divide and Conquer
7. Segment Tree Approach
"""

from typing import List, Optional, Tuple
import time
from collections import defaultdict

class TrieNode:
    """Trie node for binary representation"""
    def __init__(self):
        self.children = {}  # 0 or 1
        self.numbers = []   # Store numbers that pass through this node

class BitTrieNode:
    """Optimized trie node using bit manipulation"""
    __slots__ = ['left', 'right', 'count']
    
    def __init__(self):
        self.left = None   # 0 bit
        self.right = None  # 1 bit
        self.count = 0     # Count of numbers

class MaxXORFinder:
    
    def __init__(self):
        """Initialize Maximum XOR finder"""
        self.bit_length = 32  # Assuming 32-bit integers
    
    def find_maximum_xor_brute_force(self, nums: List[int]) -> int:
        """
        Approach 1: Brute Force
        
        Try all pairs and find maximum XOR.
        
        Time: O(n²)
        Space: O(1)
        """
        if not nums or len(nums) < 2:
            return 0
        
        max_xor = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                current_xor = nums[i] ^ nums[j]
                max_xor = max(max_xor, current_xor)
        
        return max_xor
    
    def find_maximum_xor_trie(self, nums: List[int]) -> int:
        """
        Approach 2: Bit Manipulation with Trie
        
        Build trie of binary representations and find maximum XOR.
        
        Time: O(n * 32) = O(n)
        Space: O(n * 32) = O(n)
        """
        if not nums or len(nums) < 2:
            return 0
        
        root = TrieNode()
        
        # Insert all numbers into trie
        def insert(num: int) -> None:
            """Insert number into binary trie"""
            node = root
            for i in range(31, -1, -1):  # Start from MSB
                bit = (num >> i) & 1
                
                if bit not in node.children:
                    node.children[bit] = TrieNode()
                
                node = node.children[bit]
                node.numbers.append(num)
        
        # Find maximum XOR for a number
        def find_max_xor_for_num(num: int) -> int:
            """Find maximum XOR for given number"""
            node = root
            max_xor = 0
            
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                # Try to go to opposite bit for maximum XOR
                toggled_bit = 1 - bit
                
                if toggled_bit in node.children:
                    max_xor |= (1 << i)
                    node = node.children[toggled_bit]
                else:
                    node = node.children[bit]
            
            return max_xor
        
        # Build trie
        for num in nums:
            insert(num)
        
        # Find maximum XOR
        max_xor = 0
        for num in nums:
            max_xor = max(max_xor, find_max_xor_for_num(num))
        
        return max_xor
    
    def find_maximum_xor_optimized_trie(self, nums: List[int]) -> int:
        """
        Approach 3: Optimized Trie with Bit Packing
        
        Use optimized trie structure for better performance.
        
        Time: O(n * 32) = O(n)
        Space: O(n * 32) = O(n)
        """
        if not nums or len(nums) < 2:
            return 0
        
        root = BitTrieNode()
        
        # Insert number into optimized trie
        def insert(num: int) -> None:
            """Insert number into bit trie"""
            node = root
            node.count += 1
            
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                
                if bit == 0:
                    if node.left is None:
                        node.left = BitTrieNode()
                    node = node.left
                else:
                    if node.right is None:
                        node.right = BitTrieNode()
                    node = node.right
                
                node.count += 1
        
        # Find maximum XOR
        def find_max_xor_for_num(num: int) -> int:
            """Find maximum XOR for given number"""
            node = root
            max_xor = 0
            
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                
                if bit == 0:
                    # Try to go right (1) for maximum XOR
                    if node.right is not None:
                        max_xor |= (1 << i)
                        node = node.right
                    else:
                        node = node.left
                else:
                    # Try to go left (0) for maximum XOR
                    if node.left is not None:
                        max_xor |= (1 << i)
                        node = node.left
                    else:
                        node = node.right
            
            return max_xor
        
        # Build trie
        for num in nums:
            insert(num)
        
        # Find maximum XOR
        max_xor = 0
        for num in nums:
            max_xor = max(max_xor, find_max_xor_for_num(num))
        
        return max_xor
    
    def find_maximum_xor_prefix_trie(self, nums: List[int]) -> int:
        """
        Approach 4: Prefix Trie with Early Termination
        
        Build trie incrementally with early termination optimization.
        
        Time: O(n * 32) = O(n)
        Space: O(n * 32) = O(n)
        """
        if not nums or len(nums) < 2:
            return 0
        
        root = BitTrieNode()
        max_xor = 0
        
        # Insert first number
        def insert(num: int) -> None:
            """Insert number into trie"""
            node = root
            
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                
                if bit == 0:
                    if node.left is None:
                        node.left = BitTrieNode()
                    node = node.left
                else:
                    if node.right is None:
                        node.right = BitTrieNode()
                    node = node.right
        
        # Find maximum XOR and insert simultaneously
        def find_and_insert(num: int) -> int:
            """Find max XOR and insert number"""
            # Find maximum XOR first
            node = root
            current_xor = 0
            
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                
                if bit == 0:
                    if node.right is not None:
                        current_xor |= (1 << i)
                        node = node.right
                    else:
                        node = node.left
                else:
                    if node.left is not None:
                        current_xor |= (1 << i)
                        node = node.left
                    else:
                        node = node.right
            
            # Now insert the number
            insert(num)
            
            return current_xor
        
        # Insert first number
        insert(nums[0])
        
        # Process remaining numbers
        for i in range(1, len(nums)):
            current_xor = find_and_insert(nums[i])
            max_xor = max(max_xor, current_xor)
        
        return max_xor
    
    def find_maximum_xor_hash_set(self, nums: List[int]) -> int:
        """
        Approach 5: Hash Set Approach
        
        Use hash set to find maximum XOR bit by bit.
        
        Time: O(32 * n) = O(n)
        Space: O(n)
        """
        if not nums or len(nums) < 2:
            return 0
        
        max_xor = 0
        mask = 0
        
        # Build answer bit by bit from left to right
        for i in range(31, -1, -1):
            mask |= (1 << i)  # Update mask to include current bit
            
            # Get prefixes of all numbers with current mask
            prefixes = {num & mask for num in nums}
            
            # Try to update max_xor by including current bit
            temp = max_xor | (1 << i)
            
            # Check if this max_xor is achievable
            # If a ^ b = temp, then a ^ temp = b
            for prefix in prefixes:
                if temp ^ prefix in prefixes:
                    max_xor = temp
                    break
        
        return max_xor
    
    def find_maximum_xor_divide_conquer(self, nums: List[int]) -> int:
        """
        Approach 6: Divide and Conquer
        
        Divide numbers based on most significant bit.
        
        Time: O(n * log(max_num))
        Space: O(log(max_num)) for recursion
        """
        if not nums or len(nums) < 2:
            return 0
        
        def solve(nums: List[int], bit_pos: int) -> int:
            """Solve for given bit position"""
            if bit_pos < 0 or len(nums) < 2:
                return 0
            
            # Divide numbers based on current bit
            left_nums = []   # Numbers with 0 at bit_pos
            right_nums = []  # Numbers with 1 at bit_pos
            
            for num in nums:
                if (num >> bit_pos) & 1:
                    right_nums.append(num)
                else:
                    left_nums.append(num)
            
            # If all numbers have same bit at this position
            if not left_nums or not right_nums:
                return solve(nums, bit_pos - 1)
            
            # Maximum XOR will either be from same group or across groups
            max_same_group = max(
                solve(left_nums, bit_pos - 1) if len(left_nums) > 1 else 0,
                solve(right_nums, bit_pos - 1) if len(right_nums) > 1 else 0
            )
            
            # Maximum XOR across groups (will have 1 at bit_pos)
            max_across_groups = (1 << bit_pos)
            
            # Find best pair across groups
            if left_nums and right_nums:
                # Use trie for remaining bits
                trie_root = BitTrieNode()
                
                # Insert left numbers
                for num in left_nums:
                    node = trie_root
                    for i in range(bit_pos - 1, -1, -1):
                        bit = (num >> i) & 1
                        if bit == 0:
                            if node.left is None:
                                node.left = BitTrieNode()
                            node = node.left
                        else:
                            if node.right is None:
                                node.right = BitTrieNode()
                            node = node.right
                
                # Find best match for each right number
                best_across = 0
                for num in right_nums:
                    node = trie_root
                    current_xor = 0
                    
                    for i in range(bit_pos - 1, -1, -1):
                        bit = (num >> i) & 1
                        
                        if bit == 0:
                            if node.right is not None:
                                current_xor |= (1 << i)
                                node = node.right
                            else:
                                node = node.left
                        else:
                            if node.left is not None:
                                current_xor |= (1 << i)
                                node = node.left
                            else:
                                node = node.right
                    
                    best_across = max(best_across, current_xor)
                
                max_across_groups += best_across
            
            return max(max_same_group, max_across_groups)
        
        return solve(nums, 31)
    
    def find_maximum_xor_segment_tree(self, nums: List[int]) -> int:
        """
        Approach 7: Segment Tree Approach
        
        Use segment tree to efficiently find maximum XOR in ranges.
        
        Time: O(n * log(max_num))
        Space: O(n * log(max_num))
        """
        if not nums or len(nums) < 2:
            return 0
        
        # Sort numbers for segment tree approach
        sorted_nums = sorted(set(nums))
        n = len(sorted_nums)
        
        if n < 2:
            return 0
        
        # Build segment tree of tries
        class SegmentTreeNode:
            def __init__(self):
                self.trie_root = BitTrieNode()
                self.left_child = None
                self.right_child = None
        
        def insert_into_trie(trie_root: BitTrieNode, num: int) -> None:
            """Insert number into trie"""
            node = trie_root
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                if bit == 0:
                    if node.left is None:
                        node.left = BitTrieNode()
                    node = node.left
                else:
                    if node.right is None:
                        node.right = BitTrieNode()
                    node = node.right
        
        def query_max_xor(trie_root: BitTrieNode, num: int) -> int:
            """Find maximum XOR with given number"""
            if trie_root.left is None and trie_root.right is None:
                return 0
            
            node = trie_root
            max_xor = 0
            
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                
                if bit == 0:
                    if node.right is not None:
                        max_xor |= (1 << i)
                        node = node.right
                    else:
                        node = node.left
                else:
                    if node.left is not None:
                        max_xor |= (1 << i)
                        node = node.left
                    else:
                        node = node.right
                
                if node is None:
                    break
            
            return max_xor
        
        def build_segment_tree(start: int, end: int) -> SegmentTreeNode:
            """Build segment tree of tries"""
            seg_node = SegmentTreeNode()
            
            if start == end:
                insert_into_trie(seg_node.trie_root, sorted_nums[start])
                return seg_node
            
            mid = (start + end) // 2
            seg_node.left_child = build_segment_tree(start, mid)
            seg_node.right_child = build_segment_tree(mid + 1, end)
            
            # Merge tries
            for i in range(start, end + 1):
                insert_into_trie(seg_node.trie_root, sorted_nums[i])
            
            return seg_node
        
        # Build segment tree
        root = build_segment_tree(0, n - 1)
        
        # Find maximum XOR
        max_xor = 0
        for num in sorted_nums:
            current_max = query_max_xor(root.trie_root, num)
            max_xor = max(max_xor, current_max)
        
        return max_xor


def test_basic_functionality():
    """Test basic maximum XOR functionality"""
    print("=== Testing Maximum XOR Functionality ===")
    
    finder = MaxXORFinder()
    
    test_cases = [
        ([3, 10, 5, 25, 2, 8], 28),
        ([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70], 127),
        ([1], 0),
        ([1, 2], 3),
        ([0, 0], 0),
        ([2, 4], 6),
        ([8, 10, 2], 10),
    ]
    
    approaches = [
        ("Brute Force", finder.find_maximum_xor_brute_force),
        ("Trie Basic", finder.find_maximum_xor_trie),
        ("Optimized Trie", finder.find_maximum_xor_optimized_trie),
        ("Prefix Trie", finder.find_maximum_xor_prefix_trie),
        ("Hash Set", finder.find_maximum_xor_hash_set),
        ("Divide & Conquer", finder.find_maximum_xor_divide_conquer),
        ("Segment Tree", finder.find_maximum_xor_segment_tree),
    ]
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(nums.copy())
                status = "✓" if result == expected else "✗"
                print(f"  {approach_name:20}: {result:3} {status}")
            except Exception as e:
                print(f"  {approach_name:20}: Error - {e}")


def benchmark_performance():
    """Benchmark performance of different approaches"""
    print("\n=== Performance Benchmark ===")
    
    finder = MaxXORFinder()
    
    import random
    
    # Generate test datasets
    def generate_test_data(size: int, max_val: int = 10**6) -> List[int]:
        return [random.randint(0, max_val) for _ in range(size)]
    
    test_sizes = [100, 500, 1000, 2000]
    
    approaches = [
        ("Brute Force", finder.find_maximum_xor_brute_force),
        ("Optimized Trie", finder.find_maximum_xor_optimized_trie),
        ("Hash Set", finder.find_maximum_xor_hash_set),
        ("Divide & Conquer", finder.find_maximum_xor_divide_conquer),
    ]
    
    print(f"{'Size':<8} {'Method':<20} {'Time(ms)':<12} {'Result':<10}")
    print("-" * 55)
    
    for size in test_sizes:
        nums = generate_test_data(size)
        
        for approach_name, approach_func in approaches:
            if approach_name == "Brute Force" and size > 1000:
                continue  # Skip brute force for large inputs
            
            try:
                start_time = time.time()
                result = approach_func(nums.copy())
                end_time = time.time()
                
                elapsed_ms = (end_time - start_time) * 1000
                
                print(f"{size:<8} {approach_name:<20} {elapsed_ms:<12.2f} {result:<10}")
                
            except Exception as e:
                print(f"{size:<8} {approach_name:<20} {'Error':<12} {str(e)[:10]:<10}")
        
        print()


def analyze_bit_patterns():
    """Analyze bit patterns for XOR optimization"""
    print("\n=== Bit Pattern Analysis ===")
    
    finder = MaxXORFinder()
    
    # Test with specific bit patterns
    test_patterns = [
        ("Powers of 2", [1, 2, 4, 8, 16, 32]),
        ("Sequential", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("Alternating bits", [0b10101010, 0b01010101, 0b11001100, 0b00110011]),
        ("Large numbers", [1000000, 2000000, 3000000, 4000000]),
        ("Mixed pattern", [1, 1000, 100000, 10000000]),
    ]
    
    for pattern_name, nums in test_patterns:
        print(f"\n{pattern_name}:")
        print(f"  Numbers: {nums}")
        
        # Show binary representations
        print(f"  Binary:")
        for num in nums:
            print(f"    {num:>8}: {bin(num)[2:]:>32}")
        
        # Find maximum XOR
        max_xor = finder.find_maximum_xor_optimized_trie(nums)
        print(f"  Max XOR: {max_xor} (binary: {bin(max_xor)[2:]})")
        
        # Find the pair that gives maximum XOR
        best_pair = None
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] ^ nums[j] == max_xor:
                    best_pair = (nums[i], nums[j])
                    break
            if best_pair:
                break
        
        if best_pair:
            print(f"  Best pair: {best_pair[0]} ^ {best_pair[1]} = {max_xor}")


def demonstrate_trie_construction():
    """Demonstrate trie construction process"""
    print("\n=== Trie Construction Demonstration ===")
    
    nums = [3, 10, 5, 25]
    print(f"Numbers: {nums}")
    print(f"Binary representations:")
    
    for num in nums:
        print(f"  {num:2}: {bin(num)[2:]:>8}")
    
    # Build trie step by step
    root = BitTrieNode()
    
    def insert_and_show(num: int) -> None:
        """Insert number and show trie state"""
        print(f"\nInserting {num} ({bin(num)[2:]:>8}):")
        
        node = root
        path = []
        
        for i in range(7, -1, -1):  # Show 8 bits for simplicity
            bit = (num >> i) & 1
            path.append(str(bit))
            
            if bit == 0:
                if node.left is None:
                    node.left = BitTrieNode()
                    print(f"    Created left child at level {7-i}")
                node = node.left
            else:
                if node.right is None:
                    node.right = BitTrieNode()
                    print(f"    Created right child at level {7-i}")
                node = node.right
        
        print(f"    Path: {' -> '.join(path)}")
    
    def find_max_xor_and_show(num: int) -> int:
        """Find max XOR and show process"""
        print(f"\nFinding max XOR for {num} ({bin(num)[2:]:>8}):")
        
        node = root
        max_xor = 0
        path = []
        
        for i in range(7, -1, -1):
            bit = (num >> i) & 1
            
            # Try opposite bit first
            if bit == 0:
                if node.right is not None:
                    max_xor |= (1 << i)
                    node = node.right
                    path.append("1")
                    print(f"    Level {7-i}: Want 1, found 1 ✓")
                else:
                    node = node.left
                    path.append("0")
                    print(f"    Level {7-i}: Want 1, found 0")
            else:
                if node.left is not None:
                    max_xor |= (1 << i)
                    node = node.left
                    path.append("0")
                    print(f"    Level {7-i}: Want 0, found 0 ✓")
                else:
                    node = node.right
                    path.append("1")
                    print(f"    Level {7-i}: Want 0, found 1")
        
        print(f"    Best path: {' -> '.join(path)}")
        print(f"    XOR result: {max_xor} ({bin(max_xor)[2:]:>8})")
        return max_xor
    
    # Build trie step by step
    max_xor = 0
    for i, num in enumerate(nums):
        insert_and_show(num)
        
        if i > 0:  # Can find XOR starting from second number
            current_max = find_max_xor_and_show(num)
            max_xor = max(max_xor, current_max)
    
    print(f"\nFinal maximum XOR: {max_xor}")


def demonstrate_optimization_techniques():
    """Demonstrate various optimization techniques"""
    print("\n=== Optimization Techniques ===")
    
    finder = MaxXORFinder()
    
    # Technique 1: Early termination
    print("1. Early Termination Optimization:")
    
    large_nums = list(range(1000, 2000, 100))
    
    def find_with_early_termination(nums: List[int]) -> Tuple[int, int]:
        """Find max XOR with early termination"""
        max_so_far = 0
        comparisons = 0
        
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                comparisons += 1
                current_xor = nums[i] ^ nums[j]
                
                if current_xor > max_so_far:
                    max_so_far = current_xor
                
                # Early termination: if we found theoretical maximum
                max_possible = (1 << 32) - 1  # All bits set
                if max_so_far == max_possible:
                    break
            else:
                continue
            break
        
        return max_so_far, comparisons
    
    result, comparisons = find_with_early_termination(large_nums[:10])
    total_possible = 10 * 9 // 2
    
    print(f"   Result: {result}")
    print(f"   Comparisons: {comparisons}/{total_possible} ({comparisons/total_possible*100:.1f}%)")
    
    # Technique 2: Bit-level optimization
    print(f"\n2. Bit-level Optimization:")
    
    def analyze_bit_distribution(nums: List[int]) -> None:
        """Analyze bit distribution for optimization"""
        bit_counts = [0] * 32
        
        for num in nums:
            for i in range(32):
                if (num >> i) & 1:
                    bit_counts[i] += 1
        
        print(f"   Bit distribution (position: count):")
        for i in range(31, -1, -1):
            if bit_counts[i] > 0:
                print(f"     Bit {i:2}: {bit_counts[i]:3} numbers")
        
        # Identify most significant bits
        msb_positions = []
        for i in range(31, -1, -1):
            if bit_counts[i] > 0:
                msb_positions.append(i)
                if len(msb_positions) >= 3:
                    break
        
        print(f"   Top MSB positions: {msb_positions}")
    
    analyze_bit_distribution([3, 10, 5, 25, 2, 8])
    
    # Technique 3: Memory optimization
    print(f"\n3. Memory Optimization:")
    
    import sys
    
    # Compare memory usage of different approaches
    test_nums = list(range(100))
    
    # Standard trie
    standard_trie = TrieNode()
    for num in test_nums:
        node = standard_trie
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
    
    # Optimized trie
    optimized_trie = BitTrieNode()
    for num in test_nums:
        node = optimized_trie
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit == 0:
                if node.left is None:
                    node.left = BitTrieNode()
                node = node.left
            else:
                if node.right is None:
                    node.right = BitTrieNode()
                node = node.right
    
    def calculate_trie_memory(node, visited=None) -> int:
        """Calculate memory usage of trie"""
        if visited is None:
            visited = set()
        
        if id(node) in visited:
            return 0
        
        visited.add(id(node))
        memory = sys.getsizeof(node)
        
        if hasattr(node, 'children'):
            memory += sum(calculate_trie_memory(child, visited) 
                         for child in node.children.values())
        else:
            if node.left:
                memory += calculate_trie_memory(node.left, visited)
            if node.right:
                memory += calculate_trie_memory(node.right, visited)
        
        return memory
    
    standard_memory = calculate_trie_memory(standard_trie)
    optimized_memory = calculate_trie_memory(optimized_trie)
    
    print(f"   Standard trie memory: {standard_memory} bytes")
    print(f"   Optimized trie memory: {optimized_memory} bytes")
    print(f"   Memory savings: {(standard_memory - optimized_memory) / standard_memory * 100:.1f}%")


if __name__ == "__main__":
    test_basic_functionality()
    benchmark_performance()
    analyze_bit_patterns()
    demonstrate_trie_construction()
    demonstrate_optimization_techniques()

"""
Maximum XOR of Two Numbers demonstrates bit manipulation with trie optimization:

Key Approaches:
1. Brute Force - O(n²) comparison of all pairs
2. Bit Trie - Build trie of binary representations for O(n) solution
3. Optimized Trie - Memory-efficient trie with bit-level operations
4. Prefix Trie - Incremental building with early termination
5. Hash Set - Bit-by-bit construction using hash lookups
6. Divide & Conquer - Recursive approach based on MSB
7. Segment Tree - Range-based optimization for complex scenarios

Optimization Techniques:
- Bit-level trie operations for memory efficiency
- Early termination when theoretical maximum is reached
- Incremental trie building to avoid rebuilding
- Hash-based approaches for specific bit patterns
- Memory-optimized node structures using __slots__

Real-world Applications:
- Cryptographic applications requiring maximum difference
- Network routing optimization
- Error detection and correction
- Database indexing with bit-level optimization
- Competitive programming and algorithm contests

The trie-based approach is most practical, offering O(n) time complexity
with reasonable space usage, making it suitable for large datasets.
"""
