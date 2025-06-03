"""
🔁 BIT MANIPULATION - COMMON PROBLEMS
====================================

This module covers frequently asked bit manipulation problems in interviews.
These problems demonstrate practical applications of bit manipulation techniques.

Topics Covered:
1. Count Total Set Bits (1 to N)
2. Sum of XOR of All Pairs
3. Hamming Distance (Pairwise and Total)
4. Find One Non-Repeating Number
5. Find Two Non-Repeating Numbers
6. Find Non-Repeating Number in Thrice Repeating Array

Author: Interview Preparation Collection
LeetCode Problems: 191, 201, 268, 136, 137, 260, 461, 477
"""

class CountSetBitsProblems:
    """Problems related to counting set bits in various scenarios."""
    
    @staticmethod
    def count_set_bits_1_to_n_naive(n: int) -> int:
        """
        Count total set bits from 1 to n (naive approach).
        
        Args:
            n: Upper limit
            
        Returns:
            Total count of set bits
            
        Time: O(n * log n), Space: O(1)
        """
        total = 0
        for i in range(1, n + 1):
            total += bin(i).count('1')
        return total
    
    @staticmethod
    def count_set_bits_1_to_n_optimized(n: int) -> int:
        """
        Count total set bits from 1 to n (optimized approach).
        
        Key Insight: For each bit position, count how many numbers
        have that bit set in range [1, n].
        
        Args:
            n: Upper limit
            
        Returns:
            Total count of set bits
            
        Time: O(log n), Space: O(1)
        """
        if n <= 0:
            return 0
        
        total = 0
        power_of_2 = 1
        
        while power_of_2 <= n:
            # Complete cycles of 0s and 1s
            complete_cycles = (n + 1) // (power_of_2 * 2)
            total += complete_cycles * power_of_2
            
            # Remaining numbers in incomplete cycle
            remainder = (n + 1) % (power_of_2 * 2)
            if remainder > power_of_2:
                total += remainder - power_of_2
            
            power_of_2 *= 2
        
        return total
    
    @staticmethod
    def count_set_bits_range(start: int, end: int) -> int:
        """
        Count set bits in range [start, end].
        
        Args:
            start: Start of range
            end: End of range
            
        Returns:
            Total set bits in range
            
        Time: O(log end), Space: O(1)
        """
        if start == 0:
            return CountSetBitsProblems.count_set_bits_1_to_n_optimized(end)
        
        return (CountSetBitsProblems.count_set_bits_1_to_n_optimized(end) - 
                CountSetBitsProblems.count_set_bits_1_to_n_optimized(start - 1))
    
    @staticmethod
    def count_set_bits_in_factorial(n: int) -> int:
        """
        Count set bits in n! (factorial).
        
        Args:
            n: Number for factorial
            
        Returns:
            Set bits in n!
            
        Time: O(n * log(n!)), Space: O(1)
        """
        factorial = 1
        for i in range(1, n + 1):
            factorial *= i
        
        return bin(factorial).count('1')


class XORProblems:
    """Problems involving XOR operations and properties."""
    
    @staticmethod
    def xor_of_all_pairs_naive(arr: list) -> int:
        """
        Calculate XOR of all pairs (naive approach).
        
        Args:
            arr: Input array
            
        Returns:
            XOR of all pairs
            
        Time: O(n²), Space: O(1)
        """
        result = 0
        n = len(arr)
        
        for i in range(n):
            for j in range(i + 1, n):
                result ^= (arr[i] ^ arr[j])
        
        return result
    
    @staticmethod
    def xor_of_all_pairs_optimized(arr: list) -> int:
        """
        Calculate XOR of all pairs (optimized).
        
        Key Insight: Each element appears in exactly (n-1) pairs.
        If (n-1) is even, XOR becomes 0. If odd, it's XOR of all elements.
        
        Args:
            arr: Input array
            
        Returns:
            XOR of all pairs
            
        Time: O(n), Space: O(1)
        """
        n = len(arr)
        
        # If n-1 is even, result is 0
        if (n - 1) % 2 == 0:
            return 0
        
        # If n-1 is odd, result is XOR of all elements
        result = 0
        for num in arr:
            result ^= num
        
        return result
    
    @staticmethod
    def sum_of_xor_all_pairs(arr: list) -> int:
        """
        Calculate sum of XOR of all pairs.
        
        Args:
            arr: Input array
            
        Returns:
            Sum of XOR of all pairs
            
        Time: O(n * log(max_element)), Space: O(1)
        """
        n = len(arr)
        result = 0
        
        # Check each bit position
        for i in range(32):  # Assuming 32-bit integers
            count_set = 0
            
            # Count numbers with ith bit set
            for num in arr:
                if num & (1 << i):
                    count_set += 1
            
            count_unset = n - count_set
            
            # Contribution of ith bit to final sum
            pairs_with_different_bits = count_set * count_unset
            result += pairs_with_different_bits * (1 << i)
        
        return result
    
    @staticmethod
    def xor_queries_range(arr: list, queries: list) -> list:
        """
        Answer XOR queries for given ranges.
        
        Args:
            arr: Input array
            queries: List of [left, right] ranges
            
        Returns:
            List of XOR results for each query
            
        Time: O(n + q), Space: O(n)
        """
        n = len(arr)
        prefix_xor = [0] * (n + 1)
        
        # Build prefix XOR array
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ arr[i]
        
        results = []
        for left, right in queries:
            # XOR of range [left, right] = prefix_xor[right+1] ^ prefix_xor[left]
            results.append(prefix_xor[right + 1] ^ prefix_xor[left])
        
        return results


class HammingDistanceProblems:
    """Problems related to Hamming distance calculations."""
    
    @staticmethod
    def hamming_distance(x: int, y: int) -> int:
        """
        Calculate Hamming distance between two numbers.
        
        Hamming distance = number of different bits.
        
        Args:
            x, y: Two numbers
            
        Returns:
            Hamming distance
            
        Time: O(log max(x,y)), Space: O(1)
        LeetCode: 461
        """
        xor_result = x ^ y
        count = 0
        
        # Count set bits in XOR result
        while xor_result:
            count += xor_result & 1
            xor_result >>= 1
        
        return count
    
    @staticmethod
    def hamming_distance_builtin(x: int, y: int) -> int:
        """Hamming distance using built-in function."""
        return bin(x ^ y).count('1')
    
    @staticmethod
    def total_hamming_distance_naive(nums: list) -> int:
        """
        Calculate total Hamming distance between all pairs (naive).
        
        Args:
            nums: List of numbers
            
        Returns:
            Total Hamming distance
            
        Time: O(n² * log(max_num)), Space: O(1)
        LeetCode: 477
        """
        total = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                total += HammingDistanceProblems.hamming_distance(nums[i], nums[j])
        
        return total
    
    @staticmethod
    def total_hamming_distance_optimized(nums: list) -> int:
        """
        Calculate total Hamming distance between all pairs (optimized).
        
        Key Insight: For each bit position, count numbers with bit set/unset.
        Pairs with different bits contribute to Hamming distance.
        
        Args:
            nums: List of numbers
            
        Returns:
            Total Hamming distance
            
        Time: O(n * log(max_num)), Space: O(1)
        """
        total = 0
        n = len(nums)
        
        # Check each bit position
        for i in range(32):  # Assuming 32-bit integers
            count_set = 0
            
            # Count numbers with ith bit set
            for num in nums:
                if num & (1 << i):
                    count_set += 1
            
            count_unset = n - count_set
            
            # Each pair with different ith bit contributes 1 to total distance
            total += count_set * count_unset
        
        return total


class SingleNumberProblems:
    """Problems involving finding non-repeating numbers."""
    
    @staticmethod
    def single_number_once(nums: list) -> int:
        """
        Find the number that appears once (others appear twice).
        
        Key Insight: XOR of identical numbers is 0.
        XOR of all numbers gives the unique number.
        
        Args:
            nums: List where one number appears once, others twice
            
        Returns:
            The unique number
            
        Time: O(n), Space: O(1)
        LeetCode: 136
        """
        result = 0
        for num in nums:
            result ^= num
        return result
    
    @staticmethod
    def single_number_twice(nums: list) -> int:
        """
        Find the number that appears once (others appear thrice).
        
        Key Insight: Use two variables to track bits that appear
        once and twice. Reset when a bit appears thrice.
        
        Args:
            nums: List where one number appears once, others thrice
            
        Returns:
            The unique number
            
        Time: O(n), Space: O(1)
        LeetCode: 137
        """
        ones = 0  # Bits that appeared once
        twos = 0  # Bits that appeared twice
        
        for num in nums:
            # Update twos: bits that were in ones and now appear again
            twos |= ones & num
            
            # Update ones: XOR with current number
            ones ^= num
            
            # Remove bits that appeared thrice (present in both ones and twos)
            common_bits = ones & twos
            ones &= ~common_bits
            twos &= ~common_bits
        
        return ones
    
    @staticmethod
    def single_number_twice_alternative(nums: list) -> int:
        """
        Alternative approach using bit counting.
        
        Args:
            nums: List where one number appears once, others thrice
            
        Returns:
            The unique number
            
        Time: O(n), Space: O(1)
        """
        result = 0
        
        # Check each bit position
        for i in range(32):
            count = 0
            
            # Count how many numbers have ith bit set
            for num in nums:
                if num & (1 << i):
                    count += 1
            
            # If count is not divisible by 3, unique number has this bit set
            if count % 3 != 0:
                result |= (1 << i)
        
        return result
    
    @staticmethod
    def two_single_numbers(nums: list) -> list:
        """
        Find two numbers that appear once (others appear twice).
        
        Key Insight: XOR all numbers to get XOR of two unique numbers.
        Find any set bit in this XOR to separate the two numbers.
        
        Args:
            nums: List where two numbers appear once, others twice
            
        Returns:
            List of two unique numbers
            
        Time: O(n), Space: O(1)
        LeetCode: 260
        """
        # XOR all numbers to get XOR of two unique numbers
        xor_all = 0
        for num in nums:
            xor_all ^= num
        
        # Find rightmost set bit in XOR result
        rightmost_set_bit = xor_all & (-xor_all)
        
        # Separate numbers into two groups based on this bit
        group1_xor = 0
        group2_xor = 0
        
        for num in nums:
            if num & rightmost_set_bit:
                group1_xor ^= num
            else:
                group2_xor ^= num
        
        return [group1_xor, group2_xor]
    
    @staticmethod
    def single_number_k_times(nums: list, k: int) -> int:
        """
        Find number that appears once (others appear k times).
        
        Args:
            nums: Input list
            k: Frequency of other numbers
            
        Returns:
            The unique number
            
        Time: O(n), Space: O(1)
        """
        result = 0
        
        # Check each bit position
        for i in range(32):
            count = 0
            
            # Count numbers with ith bit set
            for num in nums:
                if num & (1 << i):
                    count += 1
            
            # If count is not divisible by k, unique number has this bit set
            if count % k != 0:
                result |= (1 << i)
        
        return result


class MissingNumberProblems:
    """Problems involving finding missing numbers using bit manipulation."""
    
    @staticmethod
    def missing_number_xor(nums: list) -> int:
        """
        Find missing number in array [0, 1, 2, ..., n].
        
        Args:
            nums: Array with one missing number
            
        Returns:
            Missing number
            
        Time: O(n), Space: O(1)
        LeetCode: 268
        """
        n = len(nums)
        result = n  # Start with n (the potentially missing number)
        
        for i in range(n):
            result ^= i ^ nums[i]
        
        return result
    
    @staticmethod
    def missing_number_sum(nums: list) -> int:
        """Find missing number using sum formula."""
        n = len(nums)
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum
    
    @staticmethod
    def find_disappeared_numbers(nums: list) -> list:
        """
        Find all numbers disappeared from array [1, 2, ..., n].
        
        Args:
            nums: Array where numbers can appear multiple times
            
        Returns:
            List of missing numbers
            
        Time: O(n), Space: O(1) excluding output
        LeetCode: 448
        """
        n = len(nums)
        
        # Mark presence by making numbers negative
        for i in range(n):
            index = abs(nums[i]) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
        
        # Find positive numbers (missing indices)
        result = []
        for i in range(n):
            if nums[i] > 0:
                result.append(i + 1)
        
        # Restore original array
        for i in range(n):
            nums[i] = abs(nums[i])
        
        return result


class BitManipulationDemo:
    """Demonstration of all bit manipulation problems."""
    
    @staticmethod
    def demonstrate_count_set_bits():
        """Demonstrate set bit counting problems."""
        print("=== COUNT SET BITS PROBLEMS ===")
        
        n = 10
        naive_count = CountSetBitsProblems.count_set_bits_1_to_n_naive(n)
        optimized_count = CountSetBitsProblems.count_set_bits_1_to_n_optimized(n)
        
        print(f"Set bits from 1 to {n}:")
        print(f"  Naive approach: {naive_count}")
        print(f"  Optimized approach: {optimized_count}")
        print(f"  Results match: {naive_count == optimized_count}")
        
        # Range counting
        start, end = 5, 15
        range_count = CountSetBitsProblems.count_set_bits_range(start, end)
        print(f"Set bits from {start} to {end}: {range_count}")
    
    @staticmethod
    def demonstrate_xor_problems():
        """Demonstrate XOR-related problems."""
        print("\n=== XOR PROBLEMS ===")
        
        arr = [1, 2, 3, 4, 5]
        
        # XOR of all pairs
        naive_xor = XORProblems.xor_of_all_pairs_naive(arr)
        optimized_xor = XORProblems.xor_of_all_pairs_optimized(arr)
        print(f"XOR of all pairs in {arr}:")
        print(f"  Naive: {naive_xor}")
        print(f"  Optimized: {optimized_xor}")
        
        # Sum of XOR of all pairs
        sum_xor = XORProblems.sum_of_xor_all_pairs(arr)
        print(f"Sum of XOR of all pairs: {sum_xor}")
        
        # XOR queries
        queries = [[0, 2], [1, 3], [0, 4]]
        query_results = XORProblems.xor_queries_range(arr, queries)
        print(f"XOR queries {queries}: {query_results}")
    
    @staticmethod
    def demonstrate_hamming_distance():
        """Demonstrate Hamming distance problems."""
        print("\n=== HAMMING DISTANCE PROBLEMS ===")
        
        # Single pair
        x, y = 1, 4
        distance = HammingDistanceProblems.hamming_distance(x, y)
        print(f"Hamming distance between {x} and {y}: {distance}")
        
        # Total distance
        nums = [4, 14, 2]
        naive_total = HammingDistanceProblems.total_hamming_distance_naive(nums)
        optimized_total = HammingDistanceProblems.total_hamming_distance_optimized(nums)
        print(f"Total Hamming distance in {nums}:")
        print(f"  Naive: {naive_total}")
        print(f"  Optimized: {optimized_total}")
    
    @staticmethod
    def demonstrate_single_number_problems():
        """Demonstrate single number problems."""
        print("\n=== SINGLE NUMBER PROBLEMS ===")
        
        # Single number (appears once, others twice)
        nums1 = [2, 2, 1]
        single1 = SingleNumberProblems.single_number_once(nums1)
        print(f"Single number in {nums1}: {single1}")
        
        # Single number (appears once, others thrice)
        nums2 = [2, 2, 3, 2]
        single2 = SingleNumberProblems.single_number_twice(nums2)
        single2_alt = SingleNumberProblems.single_number_twice_alternative(nums2)
        print(f"Single number in {nums2}: {single2} (alternative: {single2_alt})")
        
        # Two single numbers
        nums3 = [1, 2, 1, 3, 2, 5]
        two_singles = SingleNumberProblems.two_single_numbers(nums3)
        print(f"Two single numbers in {nums3}: {two_singles}")
    
    @staticmethod
    def demonstrate_missing_number_problems():
        """Demonstrate missing number problems."""
        print("\n=== MISSING NUMBER PROBLEMS ===")
        
        # Missing number
        nums = [3, 0, 1]
        missing_xor = MissingNumberProblems.missing_number_xor(nums)
        missing_sum = MissingNumberProblems.missing_number_sum(nums)
        print(f"Missing number in {nums}: {missing_xor} (sum method: {missing_sum})")
        
        # Disappeared numbers
        nums_dup = [4, 3, 2, 7, 8, 2, 3, 1]
        disappeared = MissingNumberProblems.find_disappeared_numbers(nums_dup.copy())
        print(f"Disappeared numbers in {nums_dup}: {disappeared}")


def performance_comparison():
    """Compare performance of different approaches."""
    import time
    import random
    
    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Generate test data
    n = 1000
    test_array = [random.randint(1, 1000) for _ in range(n)]
    
    # Compare Hamming distance calculations
    start = time.time()
    naive_result = HammingDistanceProblems.total_hamming_distance_naive(test_array[:100])
    naive_time = time.time() - start
    
    start = time.time()
    optimized_result = HammingDistanceProblems.total_hamming_distance_optimized(test_array[:100])
    optimized_time = time.time() - start
    
    print(f"Hamming Distance (100 numbers):")
    print(f"  Naive O(n²): {naive_time:.4f} seconds")
    print(f"  Optimized O(n): {optimized_time:.4f} seconds")
    print(f"  Speedup: {naive_time/optimized_time:.2f}x")
    print(f"  Results match: {naive_result == optimized_result}")
    
    # Compare set bit counting
    n = 10000
    start = time.time()
    naive_count = CountSetBitsProblems.count_set_bits_1_to_n_naive(n)
    naive_time = time.time() - start
    
    start = time.time()
    optimized_count = CountSetBitsProblems.count_set_bits_1_to_n_optimized(n)
    optimized_time = time.time() - start
    
    print(f"\nSet Bits Count (1 to {n}):")
    print(f"  Naive O(n log n): {naive_time:.4f} seconds")
    print(f"  Optimized O(log n): {optimized_time:.4f} seconds")
    print(f"  Speedup: {naive_time/optimized_time:.2f}x")
    print(f"  Results match: {naive_count == optimized_count}")


if __name__ == "__main__":
    # Run all demonstrations
    demo = BitManipulationDemo()
    
    demo.demonstrate_count_set_bits()
    demo.demonstrate_xor_problems()
    demo.demonstrate_hamming_distance()
    demo.demonstrate_single_number_problems()
    demo.demonstrate_missing_number_problems()
    
    performance_comparison()
    
    print("\n🎯 Key Problem-Solving Patterns:")
    print("1. XOR properties: a^a=0, a^0=a (useful for finding unique elements)")
    print("2. Bit counting: Check each bit position separately")
    print("3. Hamming distance: XOR + count set bits")
    print("4. Missing numbers: XOR or sum-based approaches")
    print("5. Single number variants: Use XOR properties creatively")
    print("6. Optimization: O(n²) → O(n) by analyzing bit patterns") 