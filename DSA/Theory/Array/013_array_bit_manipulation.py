"""
Array Bit Manipulation and Math Operations
==========================================

Topics: Bitwise operations on arrays, XOR tricks, math problems
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List
import math

class ArrayBitManipulation:
    
    # ==========================================
    # 1. BASIC BIT OPERATIONS ON ARRAYS
    # ==========================================
    
    def single_number(self, nums: List[int]) -> int:
        """LC 136: Single Number - XOR approach
        Time: O(n), Space: O(1)
        """
        result = 0
        for num in nums:
            result ^= num
        return result
    
    def single_number_ii(self, nums: List[int]) -> int:
        """LC 137: Single Number II (appears once, others thrice)
        Time: O(n), Space: O(1)
        """
        ones = twos = 0
        
        for num in nums:
            # Update ones and twos
            ones = (ones ^ num) & ~twos
            twos = (twos ^ num) & ~ones
        
        return ones
    
    def single_number_iii(self, nums: List[int]) -> List[int]:
        """LC 260: Single Number III (two numbers appear once)
        Time: O(n), Space: O(1)
        """
        # XOR all numbers to get XOR of two unique numbers
        xor_result = 0
        for num in nums:
            xor_result ^= num
        
        # Find rightmost set bit
        rightmost_bit = xor_result & (-xor_result)
        
        # Divide numbers into two groups and XOR each group
        num1 = num2 = 0
        for num in nums:
            if num & rightmost_bit:
                num1 ^= num
            else:
                num2 ^= num
        
        return [num1, num2]
    
    def missing_number(self, nums: List[int]) -> int:
        """LC 268: Missing Number using XOR
        Time: O(n), Space: O(1)
        """
        result = len(nums)
        for i, num in enumerate(nums):
            result ^= i ^ num
        return result
    
    # ==========================================
    # 2. BITWISE ARRAY MANIPULATIONS
    # ==========================================
    
    def count_bits(self, n: int) -> List[int]:
        """LC 338: Counting Bits
        Time: O(n), Space: O(1) excluding output
        """
        result = [0] * (n + 1)
        
        for i in range(1, n + 1):
            result[i] = result[i >> 1] + (i & 1)
        
        return result
    
    def hamming_distance(self, x: int, y: int) -> int:
        """LC 461: Hamming Distance
        Time: O(1), Space: O(1)
        """
        xor = x ^ y
        count = 0
        
        while xor:
            count += xor & 1
            xor >>= 1
        
        return count
    
    def total_hamming_distance(self, nums: List[int]) -> int:
        """LC 477: Total Hamming Distance
        Time: O(n), Space: O(1)
        """
        total = 0
        n = len(nums)
        
        for i in range(32):  # 32 bits for integer
            count_ones = 0
            for num in nums:
                count_ones += (num >> i) & 1
            
            # Count pairs with different bits at position i
            total += count_ones * (n - count_ones)
        
        return total
    
    def find_complement(self, num: int) -> int:
        """LC 476: Number Complement
        Time: O(1), Space: O(1)
        """
        # Find number of bits
        bit_length = num.bit_length()
        
        # Create mask with all 1s of required length
        mask = (1 << bit_length) - 1
        
        return num ^ mask
    
    # ==========================================
    # 3. ARRAY SUBSET PROBLEMS USING BITS
    # ==========================================
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """LC 78: Subsets using bit manipulation
        Time: O(n * 2^n), Space: O(n * 2^n)
        """
        n = len(nums)
        result = []
        
        for i in range(1 << n):  # 2^n combinations
            subset = []
            for j in range(n):
                if i & (1 << j):
                    subset.append(nums[j])
            result.append(subset)
        
        return result
    
    def subsets_with_dup(self, nums: List[int]) -> List[List[int]]:
        """LC 90: Subsets II with duplicates
        Time: O(n * 2^n), Space: O(n * 2^n)
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(1 << n):
            subset = []
            valid = True
            
            for j in range(n):
                if i & (1 << j):
                    # Check for duplicates
                    if j > 0 and nums[j] == nums[j-1] and not (i & (1 << (j-1))):
                        valid = False
                        break
                    subset.append(nums[j])
            
            if valid:
                result.append(subset)
        
        return result
    
    # ==========================================
    # 4. MATHEMATICAL OPERATIONS ON ARRAYS
    # ==========================================
    
    def majority_element(self, nums: List[int]) -> int:
        """LC 169: Majority Element using bit manipulation
        Time: O(n), Space: O(1)
        """
        def get_bit(num: int, i: int) -> int:
            return (num >> i) & 1
        
        majority = 0
        n = len(nums)
        
        for i in range(32):
            count = sum(get_bit(num, i) for num in nums)
            
            if count > n // 2:
                majority |= (1 << i)
        
        return majority
    
    def find_duplicate_bitwise(self, nums: List[int]) -> int:
        """LC 287: Find Duplicate using bit manipulation
        Time: O(n), Space: O(1)
        """
        def count_bits_at_position(nums: List[int], bit: int) -> int:
            count = 0
            for num in nums:
                if num & (1 << bit):
                    count += 1
            return count
        
        duplicate = 0
        n = len(nums) - 1
        
        for i in range(32):
            expected_count = 0
            actual_count = count_bits_at_position(nums, i)
            
            # Count expected 1s at this bit position for numbers 1 to n
            for j in range(1, n + 1):
                if j & (1 << i):
                    expected_count += 1
            
            if actual_count > expected_count:
                duplicate |= (1 << i)
        
        return duplicate
    
    def range_bitwise_and(self, left: int, right: int) -> int:
        """LC 201: Bitwise AND of Numbers Range
        Time: O(1), Space: O(1)
        """
        shift = 0
        
        # Find common prefix
        while left != right:
            left >>= 1
            right >>= 1
            shift += 1
        
        return left << shift
    
    # ==========================================
    # 5. ADVANCED BIT PROBLEMS
    # ==========================================
    
    def max_xor_of_two_numbers(self, nums: List[int]) -> int:
        """LC 421: Maximum XOR of Two Numbers using Trie
        Time: O(n), Space: O(1)
        """
        class TrieNode:
            def __init__(self):
                self.children = {}
        
        def insert(root, num):
            node = root
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                if bit not in node.children:
                    node.children[bit] = TrieNode()
                node = node.children[bit]
        
        def find_max_xor(root, num):
            node = root
            max_xor = 0
            for i in range(31, -1, -1):
                bit = (num >> i) & 1
                toggled_bit = 1 - bit
                
                if toggled_bit in node.children:
                    max_xor |= (1 << i)
                    node = node.children[toggled_bit]
                else:
                    node = node.children[bit]
            
            return max_xor
        
        root = TrieNode()
        for num in nums:
            insert(root, num)
        
        max_result = 0
        for num in nums:
            max_result = max(max_result, find_max_xor(root, num))
        
        return max_result
    
    def utf8_validation(self, data: List[int]) -> bool:
        """LC 393: UTF-8 Validation
        Time: O(n), Space: O(1)
        """
        count = 0
        
        for num in data:
            if count == 0:
                if (num >> 5) == 0b110:
                    count = 1
                elif (num >> 4) == 0b1110:
                    count = 2
                elif (num >> 3) == 0b11110:
                    count = 3
                elif (num >> 7):
                    return False
            else:
                if (num >> 6) != 0b10:
                    return False
                count -= 1
        
        return count == 0
    
    def gray_code(self, n: int) -> List[int]:
        """LC 89: Gray Code
        Time: O(2^n), Space: O(2^n)
        """
        result = [0]
        
        for i in range(n):
            # Mirror the current sequence and add 2^i to each mirrored element
            for j in range(len(result) - 1, -1, -1):
                result.append(result[j] | (1 << i))
        
        return result

# Test Examples
def run_examples():
    abm = ArrayBitManipulation()
    
    print("=== ARRAY BIT MANIPULATION EXAMPLES ===\n")
    
    # Single number problems
    print("1. SINGLE NUMBER PROBLEMS:")
    nums = [2, 2, 1]
    single = abm.single_number(nums)
    print(f"Single number in {nums}: {single}")
    
    nums = [2, 2, 3, 2]
    single_ii = abm.single_number_ii(nums)
    print(f"Single number II in {nums}: {single_ii}")
    
    nums = [1, 2, 1, 3, 2, 5]
    single_iii = abm.single_number_iii(nums)
    print(f"Single number III in {nums}: {single_iii}")
    
    # Bit counting
    print("\n2. BIT COUNTING:")
    n = 5
    count_bits_result = abm.count_bits(n)
    print(f"Count bits for 0 to {n}: {count_bits_result}")
    
    # Hamming distance
    print("\n3. HAMMING DISTANCE:")
    x, y = 1, 4
    hamming = abm.hamming_distance(x, y)
    print(f"Hamming distance between {x} and {y}: {hamming}")
    
    # Subsets using bits
    print("\n4. SUBSETS USING BITS:")
    nums = [1, 2, 3]
    subsets = abm.subsets(nums)
    print(f"All subsets of {nums}: {subsets}")
    
    # Mathematical operations
    print("\n5. MATHEMATICAL OPERATIONS:")
    nums = [3, 2, 3]
    majority = abm.majority_element(nums)
    print(f"Majority element in {nums}: {majority}")
    
    # Advanced problems
    print("\n6. ADVANCED PROBLEMS:")
    nums = [3, 10, 5, 25, 2, 8]
    max_xor = abm.max_xor_of_two_numbers(nums)
    print(f"Maximum XOR in {nums}: {max_xor}")
    
    n = 2
    gray = abm.gray_code(n)
    print(f"Gray code for n={n}: {gray}")

if __name__ == "__main__":
    run_examples() 