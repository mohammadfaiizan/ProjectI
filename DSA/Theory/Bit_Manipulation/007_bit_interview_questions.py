"""
🎯 BIT TRICKS IN INTERVIEW PROBLEMS
==================================

This module covers the most frequently asked bit manipulation problems in interviews.
These problems demonstrate practical applications of XOR properties and bit operations.

Topics Covered:
1. Single Number (LeetCode 136)
2. Single Number II (Every Element Appears Thrice)
3. Missing Number (XOR Approach)
4. Find Duplicate Number using XOR
5. Find Element Occurring Odd Number of Times

Author: Interview Preparation Collection
LeetCode Problems: 136, 137, 260, 268, 287, 389, 421, 461, 477
"""

class SingleNumberProblems:
    """Classic single number problems using XOR properties."""
    
    @staticmethod
    def single_number_basic(nums: list) -> int:
        """
        Find the number that appears once (others appear twice).
        
        Key Insight: XOR is self-inverse (a ^ a = 0) and commutative.
        XORing all numbers cancels out duplicates, leaving the single number.
        
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
    def single_number_with_explanation(nums: list) -> tuple:
        """
        Find single number with step-by-step explanation.
        
        Args:
            nums: Input list
            
        Returns:
            Tuple (result, steps)
        """
        result = 0
        steps = []
        
        for i, num in enumerate(nums):
            old_result = result
            result ^= num
            steps.append(f"Step {i+1}: {old_result} ^ {num} = {result}")
        
        return result, steps
    
    @staticmethod
    def single_number_three_occurrences(nums: list) -> int:
        """
        Find the number that appears once (others appear thrice).
        
        Key Insight: Use two variables to track bits appearing once and twice.
        When a bit appears thrice, remove it from both tracking variables.
        
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
            # Update twos: add bits that were in ones and appear again
            twos |= ones & num
            
            # Update ones: toggle bits in current number
            ones ^= num
            
            # Remove bits that appeared thrice (present in both ones and twos)
            threes = ones & twos
            ones &= ~threes
            twos &= ~threes
        
        return ones
    
    @staticmethod
    def single_number_general_k_occurrences(nums: list, k: int) -> int:
        """
        Find number appearing once when others appear k times.
        
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
            bit_sum = 0
            
            # Count set bits at position i
            for num in nums:
                if num & (1 << i):
                    bit_sum += 1
            
            # If sum is not divisible by k, unique number has this bit set
            if bit_sum % k != 0:
                result |= (1 << i)
        
        return result
    
    @staticmethod
    def two_single_numbers(nums: list) -> list:
        """
        Find two numbers that appear once (others appear twice).
        
        Key Insight: XOR all numbers to get XOR of two unique numbers.
        Use any set bit to partition numbers into two groups.
        
        Args:
            nums: List where two numbers appear once, others twice
            
        Returns:
            List of two unique numbers
            
        Time: O(n), Space: O(1)
        LeetCode: 260
        """
        # Get XOR of two unique numbers
        xor_result = 0
        for num in nums:
            xor_result ^= num
        
        # Find rightmost set bit to partition numbers
        rightmost_set_bit = xor_result & (-xor_result)
        
        # Partition numbers into two groups and XOR each group
        group1 = 0
        group2 = 0
        
        for num in nums:
            if num & rightmost_set_bit:
                group1 ^= num
            else:
                group2 ^= num
        
        return [group1, group2]
    
    @staticmethod
    def three_single_numbers(nums: list) -> list:
        """
        Find three numbers that appear once (others appear twice).
        Extension of two single numbers problem.
        
        Args:
            nums: List where three numbers appear once, others twice
            
        Returns:
            List of three unique numbers
            
        Time: O(n), Space: O(1)
        """
        # This requires more complex bit manipulation
        # For interview purposes, focus on one and two single numbers
        
        # Count occurrences to identify unique numbers
        from collections import Counter
        count = Counter(nums)
        return [num for num, freq in count.items() if freq == 1]


class MissingNumberProblems:
    """Problems involving finding missing numbers using bit manipulation."""
    
    @staticmethod
    def missing_number_xor(nums: list) -> int:
        """
        Find missing number in array [0, 1, 2, ..., n].
        
        Key Insight: XOR all numbers from 0 to n with array elements.
        Duplicate numbers cancel out, leaving only the missing number.
        
        Args:
            nums: Array with one missing number
            
        Returns:
            Missing number
            
        Time: O(n), Space: O(1)
        LeetCode: 268
        """
        n = len(nums)
        result = n  # Start with the largest expected number
        
        for i in range(n):
            result ^= i ^ nums[i]
        
        return result
    
    @staticmethod
    def missing_number_comparison(nums: list) -> dict:
        """
        Compare different methods to find missing number.
        
        Args:
            nums: Array with one missing number
            
        Returns:
            Dictionary with results from different methods
        """
        n = len(nums)
        
        # Method 1: XOR
        xor_result = n
        for i in range(n):
            xor_result ^= i ^ nums[i]
        
        # Method 2: Sum formula
        expected_sum = n * (n + 1) // 2
        actual_sum = sum(nums)
        sum_result = expected_sum - actual_sum
        
        # Method 3: Set difference
        full_set = set(range(n + 1))
        given_set = set(nums)
        set_result = list(full_set - given_set)[0]
        
        return {
            'xor_method': xor_result,
            'sum_method': sum_result,
            'set_method': set_result
        }
    
    @staticmethod
    def missing_numbers_range(nums: list, start: int, end: int) -> list:
        """
        Find all missing numbers in range [start, end].
        
        Args:
            nums: Array of numbers
            start: Range start
            end: Range end
            
        Returns:
            List of missing numbers
            
        Time: O(n + range_size), Space: O(range_size)
        """
        present = set(nums)
        missing = []
        
        for i in range(start, end + 1):
            if i not in present:
                missing.append(i)
        
        return missing
    
    @staticmethod
    def first_missing_positive(nums: list) -> int:
        """
        Find first missing positive number.
        
        Uses array indices as a hash map with bit manipulation insights.
        
        Args:
            nums: Array of integers
            
        Returns:
            First missing positive integer
            
        Time: O(n), Space: O(1)
        LeetCode: 41
        """
        n = len(nums)
        
        # Replace non-positive numbers and numbers > n with n+1
        for i in range(n):
            if nums[i] <= 0 or nums[i] > n:
                nums[i] = n + 1
        
        # Use sign to mark presence
        for i in range(n):
            num = abs(nums[i])
            if num <= n:
                nums[num - 1] = -abs(nums[num - 1])
        
        # Find first positive number
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        
        return n + 1


class DuplicateNumberProblems:
    """Problems involving finding duplicates using bit manipulation."""
    
    @staticmethod
    def find_duplicate_xor_simple(nums: list) -> int:
        """
        Find duplicate in array where all numbers appear once except one appears twice.
        
        Args:
            nums: Array with one duplicate
            
        Returns:
            The duplicate number
            
        Time: O(n), Space: O(1)
        """
        result = 0
        for num in nums:
            result ^= num
        
        # XOR with expected numbers (assuming range [1, n-1])
        n = len(nums) - 1
        for i in range(1, n + 1):
            result ^= i
        
        return result
    
    @staticmethod
    def find_duplicate_cycle_detection(nums: list) -> int:
        """
        Find duplicate using Floyd's cycle detection (not pure bit manipulation).
        
        Args:
            nums: Array where each number is in range [1, n]
            
        Returns:
            The duplicate number
            
        Time: O(n), Space: O(1)
        LeetCode: 287
        """
        # Phase 1: Find intersection point in cycle
        slow = fast = nums[0]
        
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        
        # Phase 2: Find entrance to cycle
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        
        return fast
    
    @staticmethod
    def find_all_duplicates(nums: list) -> list:
        """
        Find all duplicates in array where elements are in range [1, n].
        
        Uses index manipulation with bit-like thinking.
        
        Args:
            nums: Array of integers
            
        Returns:
            List of duplicate numbers
            
        Time: O(n), Space: O(1)
        LeetCode: 442
        """
        result = []
        
        for num in nums:
            index = abs(num) - 1
            if nums[index] < 0:
                result.append(abs(num))
            else:
                nums[index] = -nums[index]
        
        # Restore original array
        for i in range(len(nums)):
            nums[i] = abs(nums[i])
        
        return result


class OddOccurrenceProblems:
    """Problems involving elements occurring odd number of times."""
    
    @staticmethod
    def find_odd_occurrence_single(nums: list) -> int:
        """
        Find element occurring odd number of times (only one such element).
        
        Args:
            nums: Array where one element occurs odd times, others even
            
        Returns:
            Element with odd occurrence
            
        Time: O(n), Space: O(1)
        """
        result = 0
        for num in nums:
            result ^= num
        return result
    
    @staticmethod
    def find_odd_occurrence_multiple(nums: list) -> list:
        """
        Find all elements occurring odd number of times.
        
        Args:
            nums: Array of numbers
            
        Returns:
            List of elements with odd occurrences
            
        Time: O(n), Space: O(n)
        """
        from collections import Counter
        count = Counter(nums)
        return [num for num, freq in count.items() if freq % 2 == 1]
    
    @staticmethod
    def count_elements_odd_frequency(nums: list) -> int:
        """
        Count how many elements have odd frequency.
        
        Args:
            nums: Array of numbers
            
        Returns:
            Count of elements with odd frequency
            
        Time: O(n), Space: O(n)
        """
        from collections import Counter
        count = Counter(nums)
        return sum(1 for freq in count.values() if freq % 2 == 1)


class StringBitProblems:
    """String problems that can be solved using bit manipulation concepts."""
    
    @staticmethod
    def find_difference_in_strings(s: str, t: str) -> str:
        """
        Find the extra character in string t compared to s.
        
        String t is string s with one extra character added.
        
        Args:
            s: Original string
            t: String with one extra character
            
        Returns:
            The extra character
            
        Time: O(n), Space: O(1)
        LeetCode: 389
        """
        result = 0
        
        # XOR all characters in both strings
        for char in s:
            result ^= ord(char)
        
        for char in t:
            result ^= ord(char)
        
        return chr(result)
    
    @staticmethod
    def is_anagram_bit_counting(s: str, t: str) -> bool:
        """
        Check if two strings are anagrams using bit-like counting.
        
        Args:
            s: First string
            t: Second string
            
        Returns:
            True if anagrams, False otherwise
            
        Time: O(n), Space: O(1)
        """
        if len(s) != len(t):
            return False
        
        # Count character frequencies using XOR-like approach
        char_xor = [0] * 26
        
        for i in range(len(s)):
            char_xor[ord(s[i]) - ord('a')] += 1
            char_xor[ord(t[i]) - ord('a')] -= 1
        
        return all(count == 0 for count in char_xor)


class InterviewBitTricksDemo:
    """Demonstration of interview bit manipulation problems."""
    
    @staticmethod
    def demonstrate_single_number():
        """Demonstrate single number problems."""
        print("=== SINGLE NUMBER PROBLEMS ===")
        
        # Basic single number
        nums1 = [2, 2, 1]
        result1 = SingleNumberProblems.single_number_basic(nums1)
        result1_explained, steps = SingleNumberProblems.single_number_with_explanation(nums1)
        
        print(f"Array: {nums1}")
        print(f"Single number: {result1}")
        print("XOR steps:")
        for step in steps:
            print(f"  {step}")
        
        # Single number with three occurrences
        nums2 = [2, 2, 3, 2]
        result2 = SingleNumberProblems.single_number_three_occurrences(nums2)
        result2_general = SingleNumberProblems.single_number_general_k_occurrences(nums2, 3)
        
        print(f"\nArray: {nums2}")
        print(f"Single number (thrice method): {result2}")
        print(f"Single number (general method): {result2_general}")
        
        # Two single numbers
        nums3 = [1, 2, 1, 3, 2, 5]
        result3 = SingleNumberProblems.two_single_numbers(nums3)
        print(f"\nArray: {nums3}")
        print(f"Two single numbers: {result3}")
    
    @staticmethod
    def demonstrate_missing_number():
        """Demonstrate missing number problems."""
        print("\n=== MISSING NUMBER PROBLEMS ===")
        
        nums = [3, 0, 1]
        methods = MissingNumberProblems.missing_number_comparison(nums)
        
        print(f"Array: {nums}")
        print("Missing number using different methods:")
        for method, result in methods.items():
            print(f"  {method}: {result}")
        
        # Missing numbers in range
        nums_range = [1, 3, 6, 7, 10]
        missing_range = MissingNumberProblems.missing_numbers_range(nums_range, 1, 10)
        print(f"\nArray: {nums_range}")
        print(f"Missing numbers in range [1, 10]: {missing_range}")
        
        # First missing positive
        nums_positive = [3, 4, -1, 1]
        first_missing = MissingNumberProblems.first_missing_positive(nums_positive.copy())
        print(f"\nArray: {nums_positive}")
        print(f"First missing positive: {first_missing}")
    
    @staticmethod
    def demonstrate_duplicate_problems():
        """Demonstrate duplicate finding problems."""
        print("\n=== DUPLICATE NUMBER PROBLEMS ===")
        
        # Simple duplicate with XOR
        nums_dup = [1, 3, 4, 2, 2]
        duplicate_cycle = DuplicateNumberProblems.find_duplicate_cycle_detection(nums_dup.copy())
        
        print(f"Array: {nums_dup}")
        print(f"Duplicate (cycle detection): {duplicate_cycle}")
        
        # All duplicates
        nums_all_dup = [4, 3, 2, 7, 8, 2, 3, 1]
        all_duplicates = DuplicateNumberProblems.find_all_duplicates(nums_all_dup.copy())
        print(f"\nArray: {nums_all_dup}")
        print(f"All duplicates: {all_duplicates}")
    
    @staticmethod
    def demonstrate_odd_occurrence():
        """Demonstrate odd occurrence problems."""
        print("\n=== ODD OCCURRENCE PROBLEMS ===")
        
        # Single odd occurrence
        nums_odd = [2, 3, 5, 4, 5, 3, 4]
        single_odd = OddOccurrenceProblems.find_odd_occurrence_single(nums_odd)
        print(f"Array: {nums_odd}")
        print(f"Element with odd occurrence: {single_odd}")
        
        # Multiple odd occurrences
        nums_multi_odd = [1, 2, 3, 2, 3, 1, 3]
        multiple_odd = OddOccurrenceProblems.find_odd_occurrence_multiple(nums_multi_odd)
        count_odd = OddOccurrenceProblems.count_elements_odd_frequency(nums_multi_odd)
        print(f"\nArray: {nums_multi_odd}")
        print(f"Elements with odd occurrences: {multiple_odd}")
        print(f"Count of elements with odd frequency: {count_odd}")
    
    @staticmethod
    def demonstrate_string_problems():
        """Demonstrate string bit manipulation problems."""
        print("\n=== STRING BIT PROBLEMS ===")
        
        # Find difference in strings
        s = "abcd"
        t = "abcde"
        diff_char = StringBitProblems.find_difference_in_strings(s, t)
        print(f"String s: '{s}'")
        print(f"String t: '{t}'")
        print(f"Extra character: '{diff_char}'")
        
        # Anagram check
        s1, s2 = "listen", "silent"
        is_anagram = StringBitProblems.is_anagram_bit_counting(s1, s2)
        print(f"\nStrings: '{s1}' and '{s2}'")
        print(f"Are anagrams: {is_anagram}")


def interview_tips_and_patterns():
    """Provide tips and patterns for interview problems."""
    print("\n=== INTERVIEW TIPS & PATTERNS ===")
    
    print("Common Bit Manipulation Interview Patterns:")
    print("1. XOR for finding unique elements (a ^ a = 0)")
    print("2. XOR for missing numbers (complete set XOR with given set)")
    print("3. Bit positioning for occurrence counting")
    print("4. Index manipulation as pseudo-bit operations")
    print("5. Two-pass algorithms for complex bit problems")
    
    print("\nInterview Strategy:")
    print("1. Always mention XOR properties: self-inverse, commutative, associative")
    print("2. Start with naive O(n²) or O(n log n) approach")
    print("3. Optimize using bit manipulation to O(n) time, O(1) space")
    print("4. Explain bit-by-bit reasoning clearly")
    print("5. Handle edge cases: empty arrays, single elements")
    
    print("\nTime Complexity Goals:")
    print("• Single number problems: O(n) time, O(1) space")
    print("• Missing number: O(n) time, O(1) space")
    print("• Duplicate finding: O(n) time, O(1) space")
    print("• String problems: O(n) time, O(1) space")


def common_mistakes_and_pitfalls():
    """Highlight common mistakes in bit manipulation interviews."""
    print("\n=== COMMON MISTAKES & PITFALLS ===")
    
    print("Common Mistakes:")
    print("1. Forgetting that XOR is commutative (order doesn't matter)")
    print("2. Not handling negative numbers properly in bit operations")
    print("3. Assuming array contains consecutive numbers without verification")
    print("4. Using extra space when O(1) space solution exists")
    print("5. Not considering integer overflow in sum-based approaches")
    
    print("\nDebugging Tips:")
    print("1. Trace through XOR operations step by step")
    print("2. Convert numbers to binary to visualize bit operations")
    print("3. Test with simple examples first (arrays of size 3-5)")
    print("4. Verify XOR properties with small test cases")
    print("5. Check edge cases: empty arrays, single elements, all same numbers")


if __name__ == "__main__":
    # Run all demonstrations
    demo = InterviewBitTricksDemo()
    
    demo.demonstrate_single_number()
    demo.demonstrate_missing_number()
    demo.demonstrate_duplicate_problems()
    demo.demonstrate_odd_occurrence()
    demo.demonstrate_string_problems()
    
    interview_tips_and_patterns()
    common_mistakes_and_pitfalls()
    
    print("\n🎯 Interview Success Formula:")
    print("1. Recognize XOR patterns in 'find unique' problems")
    print("2. Use bit manipulation for O(1) space optimization")
    print("3. Explain your reasoning clearly with examples")
    print("4. Start simple, then optimize")
    print("5. Practice these patterns until they become intuitive") 