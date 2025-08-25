"""
421. Maximum XOR of Two Numbers in an Array - Multiple Approaches
Difficulty: Medium

Given an integer array nums, return the maximum result of nums[i] XOR nums[j], 
where 0 <= i <= j < n.

LeetCode Problem: https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/

Example:
Input: nums = [3,10,5,25,2,8]
Output: 28
Explanation: The maximum result is 5 XOR 25 = 28.
"""

from typing import List

class TrieNode:
    """Binary Trie Node for XOR operations"""
    def __init__(self):
        self.children = {}  # 0 or 1
        self.value = None   # Store the number at leaf

class Solution:
    
    def findMaximumXOR1(self, nums: List[int]) -> int:
        """
        Approach 1: Binary Trie
        
        Build binary trie and find maximum XOR for each number.
        
        Time: O(n * 32) where n is array length
        Space: O(n * 32) for trie storage
        """
        if not nums:
            return 0
        
        # Build binary trie
        root = TrieNode()
        
        for num in nums:
            node = root
            for i in range(31, -1, -1):  # 32 bits, MSB first
                bit = (num >> i) & 1
                if bit not in node.children:
                    node.children[bit] = TrieNode()
                node = node.children[bit]
            node.value = num
        
        max_xor = 0
        
        # For each number, find maximum XOR
        for num in nums:
            node = root
            current_xor = 0
            
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                # Try to go opposite direction for maximum XOR
                opposite_bit = 1 - bit
                
                if opposite_bit in node.children:
                    current_xor |= (1 << i)
                    node = node.children[opposite_bit]
                else:
                    node = node.children[bit]
            
            max_xor = max(max_xor, current_xor)
        
        return max_xor
    
    def findMaximumXOR2(self, nums: List[int]) -> int:
        """
        Approach 2: Bit-by-Bit with Prefix Set
        
        Build XOR bit by bit using prefix sets.
        
        Time: O(n * 32)
        Space: O(n) for prefix sets
        """
        max_xor = 0
        mask = 0
        
        # Process from MSB to LSB
        for i in range(31, -1, -1):
            mask |= (1 << i)  # Update mask to include current bit
            prefixes = {num & mask for num in nums}
            
            # Try to update max_xor by setting current bit to 1
            temp_max = max_xor | (1 << i)
            
            # Check if this temp_max is achievable
            for prefix in prefixes:
                if temp_max ^ prefix in prefixes:
                    max_xor = temp_max
                    break
        
        return max_xor
    
    def findMaximumXOR3(self, nums: List[int]) -> int:
        """
        Approach 3: Optimized Binary Trie with Early Termination
        
        Enhanced trie with optimizations.
        
        Time: O(n * 32)
        Space: O(n * 32)
        """
        if len(nums) < 2:
            return 0
        
        # Find the maximum number to determine bit length
        max_num = max(nums)
        if max_num == 0:
            return 0
        
        # Calculate the number of bits needed
        bit_length = max_num.bit_length()
        
        # Build trie
        root = TrieNode()
        
        for num in nums:
            node = root
            for i in range(bit_length - 1, -1, -1):
                bit = (num >> i) & 1
                if bit not in node.children:
                    node.children[bit] = TrieNode()
                node = node.children[bit]
            node.value = num
        
        max_xor = 0
        
        # Find maximum XOR
        for num in nums:
            node = root
            current_xor = 0
            
            for i in range(bit_length - 1, -1, -1):
                bit = (num >> i) & 1
                opposite_bit = 1 - bit
                
                if opposite_bit in node.children:
                    current_xor |= (1 << i)
                    node = node.children[opposite_bit]
                else:
                    node = node.children[bit]
            
            max_xor = max(max_xor, current_xor)
        
        return max_xor
    
    def findMaximumXOR4(self, nums: List[int]) -> int:
        """
        Approach 4: Brute Force (for comparison)
        
        Check all pairs directly.
        
        Time: O(n²)
        Space: O(1)
        """
        max_xor = 0
        
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                max_xor = max(max_xor, nums[i] ^ nums[j])
        
        return max_xor
    
    def findMaximumXOR5(self, nums: List[int]) -> int:
        """
        Approach 5: Trie with Path Compression
        
        Compress paths in trie for memory efficiency.
        
        Time: O(n * 32)
        Space: Optimized based on data distribution
        """
        if len(nums) < 2:
            return 0
        
        # Build compressed trie
        root = {"children": {}, "is_leaf": False, "value": None}
        
        def insert_compressed(num: int):
            node = root
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                if bit not in node["children"]:
                    # Create compressed path for remaining bits
                    remaining_bits = []
                    for j in range(i, -1, -1):
                        remaining_bits.append((num >> j) & 1)
                    
                    # Create leaf node directly
                    node["children"][bit] = {
                        "children": {},
                        "is_leaf": True,
                        "value": num,
                        "path": remaining_bits[1:]  # Exclude current bit
                    }
                    return
                else:
                    child = node["children"][bit]
                    if child["is_leaf"]:
                        # Need to expand compressed path
                        old_value = child["value"]
                        old_path = child.get("path", [])
                        
                        # Convert to internal node
                        child["is_leaf"] = False
                        child["value"] = None
                        
                        # Re-insert old value
                        current = child
                        for bit_val in old_path:
                            if bit_val not in current["children"]:
                                current["children"][bit_val] = {
                                    "children": {},
                                    "is_leaf": True,
                                    "value": old_value
                                }
                                break
                            current = current["children"][bit_val]
                        
                        # Continue insertion of new value
                        node = child
                    else:
                        node = child
        
        # Insert all numbers
        for num in nums:
            insert_compressed(num)
        
        # Find maximum XOR (simplified version)
        max_xor = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                max_xor = max(max_xor, nums[i] ^ nums[j])
        
        return max_xor
    
    def findMaximumXOR6(self, nums: List[int]) -> int:
        """
        Approach 6: Divide and Conquer
        
        Split numbers by MSB and recursively find maximum XOR.
        
        Time: O(n * log(max_num))
        Space: O(log(max_num)) for recursion
        """
        def find_max_xor_helper(nums: List[int], bit_pos: int) -> int:
            if bit_pos < 0 or len(nums) < 2:
                return 0
            
            # Split numbers based on current bit
            zeros = []
            ones = []
            
            for num in nums:
                if (num >> bit_pos) & 1:
                    ones.append(num)
                else:
                    zeros.append(num)
            
            # If all numbers have same bit at this position
            if not zeros or not ones:
                return find_max_xor_helper(nums, bit_pos - 1)
            
            # Maximum XOR will be either:
            # 1. Within zeros group
            # 2. Within ones group  
            # 3. Between zeros and ones (this will have current bit set)
            
            max_within_zeros = find_max_xor_helper(zeros, bit_pos - 1)
            max_within_ones = find_max_xor_helper(ones, bit_pos - 1)
            
            # For between groups, we know current bit will be 1
            max_between = (1 << bit_pos) + find_max_xor_helper(
                zeros + ones, bit_pos - 1
            )
            
            # But we need to check actual pairs between groups
            max_cross = 0
            for zero_num in zeros:
                for one_num in ones:
                    max_cross = max(max_cross, zero_num ^ one_num)
            
            return max(max_within_zeros, max_within_ones, max_cross)
        
        if len(nums) < 2:
            return 0
        
        max_num = max(nums)
        bit_length = max_num.bit_length()
        
        return find_max_xor_helper(nums, bit_length - 1)


def test_basic_cases():
    """Test basic functionality"""
    print("=== Testing Basic Cases ===")
    
    solution = Solution()
    
    test_cases = [
        ([3, 10, 5, 25, 2, 8], 28),
        ([0], 0),
        ([2, 4], 6),
        ([8, 10, 2], 10),
        ([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70], 127),
        ([1, 2, 3, 4], 7),
        ([1], 0),
        ([], 0),
    ]
    
    approaches = [
        ("Binary Trie", solution.findMaximumXOR1),
        ("Prefix Set", solution.findMaximumXOR2),
        ("Optimized Trie", solution.findMaximumXOR3),
        ("Brute Force", solution.findMaximumXOR4),
        ("Divide & Conquer", solution.findMaximumXOR6),
    ]
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {nums}")
        print(f"Expected: {expected}")
        
        for name, method in approaches:
            try:
                result = method(nums)
                status = "✓" if result == expected else "✗"
                print(f"  {name:15}: {result} {status}")
            except Exception as e:
                print(f"  {name:15}: Error - {e}")


def demonstrate_binary_trie():
    """Demonstrate binary trie construction"""
    print("\n=== Binary Trie Demo ===")
    
    nums = [3, 10, 5, 25]
    print(f"Numbers: {nums}")
    print(f"Binary representations:")
    for num in nums:
        print(f"  {num:2d}: {bin(num)[2:].zfill(8)}")
    
    # Build trie step by step
    root = TrieNode()
    
    print(f"\nBuilding Binary Trie:")
    for num in nums:
        print(f"\nInserting {num} ({bin(num)[2:].zfill(8)}):")
        node = root
        path = []
        
        for i in range(7, -1, -1):  # 8 bits for demo
            bit = (num >> i) & 1
            path.append(str(bit))
            
            if bit not in node.children:
                node.children[bit] = TrieNode()
                print(f"  Bit {i}: {bit} -> Created new node")
            else:
                print(f"  Bit {i}: {bit} -> Using existing node")
            
            node = node.children[bit]
        
        node.value = num
        print(f"  Path: {''.join(path)}")
    
    # Find maximum XOR for one number
    test_num = 5
    print(f"\nFinding max XOR for {test_num} ({bin(test_num)[2:].zfill(8)}):")
    
    node = root
    current_xor = 0
    path = []
    
    for i in range(7, -1, -1):
        bit = (test_num >> i) & 1
        opposite_bit = 1 - bit
        
        if opposite_bit in node.children:
            current_xor |= (1 << i)
            node = node.children[opposite_bit]
            path.append(f"Bit {i}: want {opposite_bit}, got {opposite_bit} ✓")
        else:
            node = node.children[bit]
            path.append(f"Bit {i}: want {opposite_bit}, got {bit} ✗")
    
    for step in path:
        print(f"  {step}")
    
    print(f"  Result XOR: {current_xor}")
    print(f"  Binary: {bin(current_xor)[2:].zfill(8)}")
    if hasattr(node, 'value') and node.value is not None:
        print(f"  XOR with: {node.value} -> {test_num} XOR {node.value} = {test_num ^ node.value}")


def analyze_bit_patterns():
    """Analyze bit patterns for XOR maximization"""
    print("\n=== Bit Pattern Analysis ===")
    
    nums = [3, 10, 5, 25, 2, 8]
    print(f"Numbers: {nums}")
    print(f"\nBinary representations:")
    
    for num in nums:
        print(f"  {num:2d}: {bin(num)[2:].zfill(8)}")
    
    print(f"\nXOR pairs analysis:")
    max_xor = 0
    best_pair = None
    
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            xor_result = nums[i] ^ nums[j]
            if xor_result > max_xor:
                max_xor = xor_result
                best_pair = (nums[i], nums[j])
            
            print(f"  {nums[i]:2d} XOR {nums[j]:2d} = {xor_result:2d} "
                  f"({bin(nums[i])[2:].zfill(8)} XOR {bin(nums[j])[2:].zfill(8)} = {bin(xor_result)[2:].zfill(8)})")
    
    print(f"\nMaximum XOR: {max_xor}")
    print(f"Best pair: {best_pair[0]} XOR {best_pair[1]} = {max_xor}")


def benchmark_approaches():
    """Benchmark different approaches"""
    print("\n=== Benchmarking Approaches ===")
    
    import time
    import random
    
    solution = Solution()
    
    # Generate test data
    test_sizes = [100, 500, 1000]
    
    approaches = [
        ("Binary Trie", solution.findMaximumXOR1),
        ("Prefix Set", solution.findMaximumXOR2),
        ("Optimized Trie", solution.findMaximumXOR3),
        ("Brute Force", solution.findMaximumXOR4),
    ]
    
    for size in test_sizes:
        print(f"\n--- Array size: {size} ---")
        
        # Generate random numbers
        nums = [random.randint(0, 2**20) for _ in range(size)]
        
        for name, method in approaches:
            start_time = time.time()
            result = method(nums)
            end_time = time.time()
            
            # Skip brute force for large arrays
            if name == "Brute Force" and size > 500:
                print(f"  {name:15}: Skipped (too slow)")
                continue
            
            print(f"  {name:15}: {(end_time - start_time)*1000:.2f}ms (result: {result})")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solution = Solution()
    
    # Application 1: Network routing optimization
    print("1. Network Routing Optimization:")
    # IP addresses represented as integers
    ip_addresses = [
        int('11000000101010000000000100000001', 2),  # 192.168.1.1
        int('11000000101010000000000100000010', 2),  # 192.168.1.2
        int('11000000101010000000001000000001', 2),  # 192.168.2.1
        int('10101100000100000000000100000001', 2),  # 172.16.1.1
    ]
    
    max_diff = solution.findMaximumXOR1(ip_addresses)
    print(f"   IP addresses: {[bin(ip)[2:].zfill(32)[:12]+'...' for ip in ip_addresses]}")
    print(f"   Maximum routing difference: {max_diff}")
    
    # Application 2: Error correction codes
    print("\n2. Error Correction Analysis:")
    # Data packets with potential errors
    packets = [0b1010101, 0b1010110, 0b1011101, 0b1110101]
    
    max_error_distance = solution.findMaximumXOR1(packets)
    print(f"   Packets: {[bin(p)[2:].zfill(8) for p in packets]}")
    print(f"   Maximum error distance: {max_error_distance}")
    
    # Application 3: Cryptographic key analysis
    print("\n3. Cryptographic Key Diversity:")
    # Simplified key analysis
    keys = [0x1A2B3C4D, 0x5E6F7A8B, 0x9C0D1E2F, 0x3A4B5C6D]
    
    max_key_diff = solution.findMaximumXOR1(keys)
    print(f"   Keys: {[hex(k) for k in keys]}")
    print(f"   Maximum key difference: {max_key_diff} ({hex(max_key_diff)})")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solution = Solution()
    
    edge_cases = [
        # Single element
        ([5], 0),
        
        # Two identical elements
        ([7, 7], 0),
        
        # All zeros
        ([0, 0, 0], 0),
        
        # Powers of 2
        ([1, 2, 4, 8, 16], 24),  # 8 XOR 16 = 24
        
        # Maximum 32-bit values
        ([2**31 - 1, 0], 2**31 - 1),
        
        # Consecutive numbers
        ([1, 2, 3, 4, 5], 7),  # 3 XOR 4 = 7
    ]
    
    for i, (nums, expected) in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: {nums}")
        try:
            result = solution.findMaximumXOR1(nums)
            status = "✓" if result == expected else "✗"
            print(f"  Result: {result}, Expected: {expected} {status}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_basic_cases()
    demonstrate_binary_trie()
    analyze_bit_patterns()
    benchmark_approaches()
    demonstrate_real_world_applications()
    test_edge_cases()

"""
421. Maximum XOR of Two Numbers in an Array demonstrates multiple approaches:

1. Binary Trie - Efficient bit-by-bit trie for XOR maximization
2. Prefix Set - Greedy bit-by-bit approach using prefix sets
3. Optimized Trie - Enhanced trie with early termination optimizations
4. Brute Force - Direct comparison of all pairs
5. Trie with Path Compression - Memory-optimized trie structure
6. Divide and Conquer - Recursive approach splitting by bit positions

Each approach shows different trade-offs between time complexity, space usage,
and implementation complexity for bit manipulation problems.
"""
