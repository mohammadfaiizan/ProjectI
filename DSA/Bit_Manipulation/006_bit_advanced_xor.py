"""
🧠 ADVANCED XOR PROBLEMS
=======================

This module covers advanced XOR-based algorithms and data structures.
These problems showcase the power of XOR operations in competitive programming.

Topics Covered:
1. Maximum XOR of Two Numbers in an Array
2. Trie-Based XOR Queries
3. Subarrays with XOR Equal to K
4. Maximum XOR Subarray
5. XOR Basis for Vector Space

Author: Interview Preparation Collection
LeetCode Problems: 421, 1707, 1269, 1310, 1763, 1803
"""

class TrieNode:
    """Trie node for XOR operations."""
    
    def __init__(self):
        self.children = {}  # 0 and 1 children
        self.count = 0      # Number of elements ending here
        self.min_idx = float('inf')  # Minimum index (for range queries)
        self.max_idx = -1   # Maximum index


class XORTrie:
    """Trie data structure optimized for XOR operations."""
    
    def __init__(self, max_bits: int = 32):
        self.root = TrieNode()
        self.max_bits = max_bits
    
    def insert(self, num: int, idx: int = 0) -> None:
        """
        Insert number into XOR trie.
        
        Args:
            num: Number to insert
            idx: Index of the number (for range queries)
            
        Time: O(log max_num), Space: O(log max_num)
        """
        node = self.root
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            if bit not in node.children:
                node.children[bit] = TrieNode()
            
            node = node.children[bit]
            node.count += 1
            node.min_idx = min(node.min_idx, idx)
            node.max_idx = max(node.max_idx, idx)
        
        node.count += 1
    
    def remove(self, num: int, idx: int = 0) -> None:
        """Remove number from XOR trie."""
        node = self.root
        path = []
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            
            if bit not in node.children:
                return  # Number not in trie
            
            path.append((node, bit))
            node = node.children[bit]
            node.count -= 1
        
        # Clean up empty nodes
        for parent, bit in reversed(path):
            if parent.children[bit].count == 0:
                del parent.children[bit]
            else:
                break
    
    def find_max_xor(self, num: int) -> int:
        """
        Find number in trie that gives maximum XOR with given number.
        
        Args:
            num: Query number
            
        Returns:
            Maximum XOR value
            
        Time: O(log max_num), Space: O(1)
        """
        if not self.root.children:
            return 0
        
        node = self.root
        max_xor = 0
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            toggle_bit = 1 - bit  # Opposite bit for maximum XOR
            
            if toggle_bit in node.children and node.children[toggle_bit].count > 0:
                max_xor |= (1 << i)
                node = node.children[toggle_bit]
            elif bit in node.children and node.children[bit].count > 0:
                node = node.children[bit]
            else:
                break
        
        return max_xor
    
    def find_max_xor_in_range(self, num: int, left: int, right: int) -> int:
        """
        Find maximum XOR with numbers in given index range.
        
        Args:
            num: Query number
            left: Left index bound
            right: Right index bound
            
        Returns:
            Maximum XOR in range
            
        Time: O(log max_num), Space: O(1)
        """
        node = self.root
        max_xor = 0
        
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            toggle_bit = 1 - bit
            
            # Check if toggle bit path has valid indices
            if (toggle_bit in node.children and 
                node.children[toggle_bit].count > 0 and
                node.children[toggle_bit].min_idx <= right and
                node.children[toggle_bit].max_idx >= left):
                max_xor |= (1 << i)
                node = node.children[toggle_bit]
            elif (bit in node.children and 
                  node.children[bit].count > 0 and
                  node.children[bit].min_idx <= right and
                  node.children[bit].max_idx >= left):
                node = node.children[bit]
            else:
                return max_xor
        
        return max_xor


class MaximumXORProblems:
    """Problems involving finding maximum XOR values."""
    
    @staticmethod
    def find_maximum_xor_naive(nums: list) -> int:
        """
        Find maximum XOR of any two numbers (naive approach).
        
        Args:
            nums: List of numbers
            
        Returns:
            Maximum XOR value
            
        Time: O(n^2), Space: O(1)
        LeetCode: 421
        """
        max_xor = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                max_xor = max(max_xor, nums[i] ^ nums[j])
        
        return max_xor
    
    @staticmethod
    def find_maximum_xor_trie(nums: list) -> int:
        """
        Find maximum XOR using trie (optimized approach).
        
        Args:
            nums: List of numbers
            
        Returns:
            Maximum XOR value
            
        Time: O(n * log max_num), Space: O(n * log max_num)
        """
        if len(nums) < 2:
            return 0
        
        trie = XORTrie()
        max_xor = 0
        
        # Insert first number
        trie.insert(nums[0])
        
        # For each subsequent number, find max XOR and insert
        for i in range(1, len(nums)):
            max_xor = max(max_xor, trie.find_max_xor(nums[i]))
            trie.insert(nums[i])
        
        return max_xor
    
    @staticmethod
    def maximum_xor_with_queries(nums: list, queries: list) -> list:
        """
        Answer maximum XOR queries with constraints.
        
        Args:
            nums: List of numbers
            queries: List of (x, m) where we want max XOR of x with nums[i] where nums[i] <= m
            
        Returns:
            List of maximum XOR values for each query
            
        Time: O((n + q) * log max_num), Space: O(n * log max_num)
        LeetCode: 1707
        """
        # Sort numbers with their indices
        sorted_nums = sorted((num, i) for i, num in enumerate(nums))
        
        # Sort queries with their indices
        sorted_queries = sorted((m, x, i) for i, (x, m) in enumerate(queries))
        
        results = [0] * len(queries)
        trie = XORTrie()
        j = 0
        
        for m, x, query_idx in sorted_queries:
            # Insert all numbers <= m into trie
            while j < len(sorted_nums) and sorted_nums[j][0] <= m:
                trie.insert(sorted_nums[j][0])
                j += 1
            
            # Find maximum XOR
            if j > 0:  # Trie is not empty
                results[query_idx] = trie.find_max_xor(x)
            else:
                results[query_idx] = -1
        
        return results
    
    @staticmethod
    def maximum_xor_bit_manipulation(nums: list) -> int:
        """
        Find maximum XOR using bit manipulation technique.
        
        Args:
            nums: List of numbers
            
        Returns:
            Maximum XOR value
            
        Time: O(n * log max_num), Space: O(n)
        """
        max_xor = 0
        mask = 0
        
        for i in range(31, -1, -1):  # Check each bit position
            mask |= (1 << i)
            prefixes = {num & mask for num in nums}
            
            temp = max_xor | (1 << i)
            
            # Check if we can achieve this temp value
            for prefix in prefixes:
                if temp ^ prefix in prefixes:
                    max_xor = temp
                    break
        
        return max_xor


class XORSubarrayProblems:
    """Problems involving XOR operations on subarrays."""
    
    @staticmethod
    def subarrays_with_xor_k_naive(nums: list, k: int) -> int:
        """
        Count subarrays with XOR equal to k (naive approach).
        
        Args:
            nums: List of numbers
            k: Target XOR value
            
        Returns:
            Count of subarrays
            
        Time: O(n^2), Space: O(1)
        LeetCode: 1269
        """
        count = 0
        n = len(nums)
        
        for i in range(n):
            xor_val = 0
            for j in range(i, n):
                xor_val ^= nums[j]
                if xor_val == k:
                    count += 1
        
        return count
    
    @staticmethod
    def subarrays_with_xor_k_optimized(nums: list, k: int) -> int:
        """
        Count subarrays with XOR equal to k (optimized using prefix XOR).
        
        Args:
            nums: List of numbers
            k: Target XOR value
            
        Returns:
            Count of subarrays
            
        Time: O(n), Space: O(n)
        """
        count = 0
        prefix_xor = 0
        xor_count = {0: 1}  # Include empty prefix
        
        for num in nums:
            prefix_xor ^= num
            
            # If prefix_xor ^ target_xor = k, then target_xor = prefix_xor ^ k
            target_xor = prefix_xor ^ k
            
            if target_xor in xor_count:
                count += xor_count[target_xor]
            
            xor_count[prefix_xor] = xor_count.get(prefix_xor, 0) + 1
        
        return count
    
    @staticmethod
    def maximum_xor_subarray_naive(nums: list) -> int:
        """
        Find maximum XOR of any subarray (naive approach).
        
        Args:
            nums: List of numbers
            
        Returns:
            Maximum XOR value
            
        Time: O(n^2), Space: O(1)
        """
        max_xor = 0
        n = len(nums)
        
        for i in range(n):
            xor_val = 0
            for j in range(i, n):
                xor_val ^= nums[j]
                max_xor = max(max_xor, xor_val)
        
        return max_xor
    
    @staticmethod
    def maximum_xor_subarray_trie(nums: list) -> int:
        """
        Find maximum XOR subarray using trie optimization.
        
        Args:
            nums: List of numbers
            
        Returns:
            Maximum XOR value
            
        Time: O(n * log max_num), Space: O(n * log max_num)
        """
        trie = XORTrie()
        trie.insert(0)  # Empty prefix
        
        max_xor = 0
        prefix_xor = 0
        
        for num in nums:
            prefix_xor ^= num
            max_xor = max(max_xor, trie.find_max_xor(prefix_xor))
            trie.insert(prefix_xor)
        
        return max_xor
    
    @staticmethod
    def count_triplets_xor_zero(nums: list) -> int:
        """
        Count triplets (i, j, k) where i < j < k and nums[i] ^ nums[j] ^ nums[k] = 0.
        
        Args:
            nums: List of numbers
            
        Returns:
            Count of valid triplets
            
        Time: O(n^2), Space: O(n)
        """
        count = 0
        n = len(nums)
        
        # For each middle element
        for j in range(1, n - 1):
            left_xor = {}
            right_xor = {}
            
            # Count XOR values on left side
            xor_val = 0
            for i in range(j):
                xor_val ^= nums[i]
                left_xor[xor_val] = left_xor.get(xor_val, 0) + 1
            
            # Count XOR values on right side
            xor_val = 0
            for k in range(n - 1, j, -1):
                xor_val ^= nums[k]
                right_xor[xor_val] = right_xor.get(xor_val, 0) + 1
            
            # For triplet XOR to be 0: left_xor ^ nums[j] ^ right_xor = 0
            # So: left_xor ^ right_xor = nums[j]
            for left_val in left_xor:
                target = left_val ^ nums[j]
                if target in right_xor:
                    count += left_xor[left_val] * right_xor[target]
        
        return count


class XORBasis:
    """XOR Basis for vector space operations."""
    
    def __init__(self, max_bits: int = 32):
        self.basis = [0] * max_bits
        self.max_bits = max_bits
        self.size = 0
    
    def insert(self, num: int) -> bool:
        """
        Insert number into XOR basis.
        
        Args:
            num: Number to insert
            
        Returns:
            True if number was linearly independent, False otherwise
            
        Time: O(log max_num), Space: O(1)
        """
        for i in range(self.max_bits - 1, -1, -1):
            if not (num & (1 << i)):
                continue
            
            if not self.basis[i]:
                self.basis[i] = num
                self.size += 1
                return True
            
            num ^= self.basis[i]
        
        return False  # Number is linearly dependent
    
    def can_represent(self, num: int) -> bool:
        """
        Check if number can be represented using current basis.
        
        Args:
            num: Number to check
            
        Returns:
            True if representable, False otherwise
            
        Time: O(log max_num), Space: O(1)
        """
        for i in range(self.max_bits - 1, -1, -1):
            if not (num & (1 << i)):
                continue
            
            if not self.basis[i]:
                return False
            
            num ^= self.basis[i]
        
        return True
    
    def get_maximum(self) -> int:
        """
        Get maximum value that can be represented.
        
        Returns:
            Maximum representable value
            
        Time: O(log max_num), Space: O(1)
        """
        result = 0
        for i in range(self.max_bits - 1, -1, -1):
            result = max(result, result ^ self.basis[i])
        return result
    
    def get_minimum_nonzero(self) -> int:
        """
        Get minimum non-zero value that can be represented.
        
        Returns:
            Minimum non-zero representable value
            
        Time: O(log max_num), Space: O(1)
        """
        for i in range(self.max_bits):
            if self.basis[i]:
                return self.basis[i]
        return 0
    
    def count_representable(self) -> int:
        """
        Count total numbers that can be represented.
        
        Returns:
            Count of representable numbers
            
        Time: O(1), Space: O(1)
        """
        return 1 << self.size  # 2^size
    
    def kth_smallest(self, k: int) -> int:
        """
        Find k-th smallest representable number (0-indexed).
        
        Args:
            k: Index of desired number
            
        Returns:
            k-th smallest representable number
            
        Time: O(log max_num), Space: O(1)
        """
        if k >= (1 << self.size):
            return -1
        
        result = 0
        bit_pos = 0
        
        for i in range(self.max_bits):
            if self.basis[i]:
                if k & (1 << bit_pos):
                    result ^= self.basis[i]
                bit_pos += 1
        
        return result


class AdvancedXORDemo:
    """Demonstration of advanced XOR algorithms."""
    
    @staticmethod
    def demonstrate_maximum_xor():
        """Demonstrate maximum XOR finding algorithms."""
        print("=== MAXIMUM XOR PROBLEMS ===")
        
        nums = [3, 10, 5, 25, 2, 8]
        
        # Compare different approaches
        naive_max = MaximumXORProblems.find_maximum_xor_naive(nums)
        trie_max = MaximumXORProblems.find_maximum_xor_trie(nums)
        bit_max = MaximumXORProblems.maximum_xor_bit_manipulation(nums)
        
        print(f"Array: {nums}")
        print(f"Maximum XOR (naive): {naive_max}")
        print(f"Maximum XOR (trie): {trie_max}")
        print(f"Maximum XOR (bit manipulation): {bit_max}")
        
        # XOR queries
        queries = [(3, 20), (2, 15), (5, 10)]
        query_results = MaximumXORProblems.maximum_xor_with_queries(nums, queries)
        print(f"XOR queries {queries}: {query_results}")
    
    @staticmethod
    def demonstrate_xor_subarrays():
        """Demonstrate XOR subarray problems."""
        print("\n=== XOR SUBARRAY PROBLEMS ===")
        
        nums = [4, 2, 2, 6, 4]
        k = 6
        
        # Count subarrays with XOR = k
        naive_count = XORSubarrayProblems.subarrays_with_xor_k_naive(nums, k)
        optimized_count = XORSubarrayProblems.subarrays_with_xor_k_optimized(nums, k)
        
        print(f"Array: {nums}")
        print(f"Subarrays with XOR = {k} (naive): {naive_count}")
        print(f"Subarrays with XOR = {k} (optimized): {optimized_count}")
        
        # Maximum XOR subarray
        max_subarray_naive = XORSubarrayProblems.maximum_xor_subarray_naive(nums)
        max_subarray_trie = XORSubarrayProblems.maximum_xor_subarray_trie(nums)
        
        print(f"Maximum XOR subarray (naive): {max_subarray_naive}")
        print(f"Maximum XOR subarray (trie): {max_subarray_trie}")
        
        # Count triplets with XOR = 0
        triplet_count = XORSubarrayProblems.count_triplets_xor_zero(nums)
        print(f"Triplets with XOR = 0: {triplet_count}")
    
    @staticmethod
    def demonstrate_xor_trie():
        """Demonstrate XOR trie operations."""
        print("\n=== XOR TRIE OPERATIONS ===")
        
        trie = XORTrie()
        numbers = [3, 10, 5, 25, 2, 8]
        
        # Insert numbers
        for i, num in enumerate(numbers):
            trie.insert(num, i)
        
        # Test max XOR queries
        for query in [4, 15, 7]:
            max_xor = trie.find_max_xor(query)
            print(f"Max XOR with {query}: {max_xor}")
        
        # Range queries
        range_xor = trie.find_max_xor_in_range(4, 1, 4)
        print(f"Max XOR with 4 in range [1, 4]: {range_xor}")
    
    @staticmethod
    def demonstrate_xor_basis():
        """Demonstrate XOR basis operations."""
        print("\n=== XOR BASIS ===")
        
        basis = XORBasis()
        numbers = [6, 4, 2, 3, 1]
        
        print(f"Inserting numbers: {numbers}")
        for num in numbers:
            independent = basis.insert(num)
            print(f"  {num}: {'linearly independent' if independent else 'linearly dependent'}")
        
        print(f"Basis size: {basis.size}")
        print(f"Maximum representable: {basis.get_maximum()}")
        print(f"Minimum non-zero: {basis.get_minimum_nonzero()}")
        print(f"Total representable numbers: {basis.count_representable()}")
        
        # Check representability
        test_numbers = [7, 5, 9, 0]
        for num in test_numbers:
            can_rep = basis.can_represent(num)
            print(f"Can represent {num}: {can_rep}")
        
        # Find k-th smallest
        print("First few representable numbers:")
        for k in range(min(8, basis.count_representable())):
            kth = basis.kth_smallest(k)
            print(f"  {k}-th: {kth}")


def performance_comparison():
    """Compare performance of different XOR algorithms."""
    import time
    import random
    
    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Generate test data
    n = 1000
    nums = [random.randint(1, 100000) for _ in range(n)]
    
    # Compare maximum XOR algorithms
    start = time.time()
    naive_result = MaximumXORProblems.find_maximum_xor_naive(nums[:100])  # Smaller for naive
    naive_time = time.time() - start
    
    start = time.time()
    trie_result = MaximumXORProblems.find_maximum_xor_trie(nums)
    trie_time = time.time() - start
    
    start = time.time()
    bit_result = MaximumXORProblems.maximum_xor_bit_manipulation(nums)
    bit_time = time.time() - start
    
    print(f"Maximum XOR (100 numbers):")
    print(f"  Naive O(n²): {naive_time:.4f} seconds")
    print(f"  Trie O(n log n): {trie_time:.4f} seconds")
    print(f"  Bit manipulation: {bit_time:.4f} seconds")
    
    # Compare subarray XOR counting
    k = 5
    start = time.time()
    naive_count = XORSubarrayProblems.subarrays_with_xor_k_naive(nums[:100], k)
    naive_sub_time = time.time() - start
    
    start = time.time()
    opt_count = XORSubarrayProblems.subarrays_with_xor_k_optimized(nums, k)
    opt_sub_time = time.time() - start
    
    print(f"\nSubarray XOR counting:")
    print(f"  Naive O(n²): {naive_sub_time:.4f} seconds")
    print(f"  Optimized O(n): {opt_sub_time:.4f} seconds")


if __name__ == "__main__":
    # Run all demonstrations
    demo = AdvancedXORDemo()
    
    demo.demonstrate_maximum_xor()
    demo.demonstrate_xor_subarrays()
    demo.demonstrate_xor_trie()
    demo.demonstrate_xor_basis()
    
    performance_comparison()
    
    print("\n🎯 Key Advanced XOR Patterns:")
    print("1. XOR Trie: Efficient maximum XOR queries in O(log n) time")
    print("2. Prefix XOR: Convert subarray problems to prefix problems")
    print("3. XOR Basis: Linear algebra for XOR vector spaces")
    print("4. Bit manipulation: Build answer bit by bit for optimization")
    print("5. Range queries: Use additional indexing in trie nodes")
    print("6. Linear independence: XOR basis for spanning set problems") 